"""Task 3: measure the effect of scalar (int8) quantization on search precision and speed.

The collection is updated with a scalar quantization config (the original vectors
are kept), and the k-NN vs ANN comparison from Task 1 is run twice: once with
``rescore=True`` and once with ``rescore=False`` (both with ``oversampling=2.0``).
The exact k-NN baseline ignores the quantized vectors, so it always runs on the
original full-precision vectors.

Run from the repository root with::

    uv run python tasks/task_3.py

Connection settings default to ``localhost:6333`` and the ``arxiv_papers``
collection; they can be overridden with the ``QDRANT_HOST``, ``QDRANT_PORT``
and ``COLLECTION_NAME`` environment variables (or a ``.env`` file).
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.http.models import SearchParams, VectorParams

K: int = 10
QUERIES_FILE: Path = Path(__file__).resolve().parent.parent / "dataset" / "queries_embeddings.json"
CLIENT_TIMEOUT_SECONDS: int = 30
WARM_UP_QUERIES: int = 10
# The exact baseline must run on the original vectors, not on the quantized ones.
EXACT_SEARCH_PARAMS: SearchParams = SearchParams(
    exact=True,
    quantization=models.QuantizationSearchParams(ignore=True),
)
OVERSAMPLING: float = 2.0
RESCORE_VALUES: tuple[bool, ...] = (True, False)
QUANTIZATION_CONFIG: models.ScalarQuantization = models.ScalarQuantization(
    scalar=models.ScalarQuantizationConfig(
        type=models.ScalarType.INT8,
        quantile=0.99,
        always_ram=False,
    ),
)
# How long to wait for Qdrant to build the quantized vectors after the collection update.
OPTIMIZATION_TIMEOUT_SECONDS: float = 300.0
OPTIMIZATION_POLL_SECONDS: float = 1.0

DEFAULT_QDRANT_HOST: str = "localhost"
DEFAULT_QDRANT_PORT: int = 6333
DEFAULT_COLLECTION_NAME: str = "arxiv_papers"


class SetupError(RuntimeError):
    """Raised when the settings, the dataset or the Qdrant collection are not usable for the evaluation."""


@dataclass(frozen=True)
class Settings:
    """Qdrant connection settings."""

    host: str = DEFAULT_QDRANT_HOST
    port: int = DEFAULT_QDRANT_PORT
    collection_name: str = DEFAULT_COLLECTION_NAME


def load_settings() -> Settings:
    """Load settings from the environment (and an optional ``.env`` file).

    Raises:
        SetupError: if ``QDRANT_PORT`` is not a valid port number.
    """
    load_dotenv()
    port_value: str | None = os.getenv("QDRANT_PORT", str(DEFAULT_QDRANT_PORT))

    if port_value is None:
        raise SetupError("QDRANT_PORT is not set.")

    try:
        port: int = int(port_value)
    except ValueError as exc:
        raise SetupError(f"QDRANT_PORT must be an integer, got {port_value!r}.") from exc
    if not 0 < port < 65536:
        raise SetupError(f"QDRANT_PORT must be between 1 and 65535, got {port}.")
    return Settings(
        host=os.getenv("QDRANT_HOST", DEFAULT_QDRANT_HOST),
        port=port,
        collection_name=os.getenv("COLLECTION_NAME", DEFAULT_COLLECTION_NAME),
    )


def load_test_dataset(path: Path) -> dict[str, list[float]]:
    """Load the test dataset mapping query text to its embedding.

    Raises:
        SetupError: if the file is missing or unreadable, is not valid UTF-8 JSON, is empty,
            or contains embeddings that are not non-empty lists of numbers of the same dimension.
    """
    try:
        with open(path, "r", encoding="utf-8") as file:
            raw = json.load(file)
    except FileNotFoundError as exc:
        raise SetupError(f"Test dataset not found: {path}") from exc
    except UnicodeDecodeError as exc:
        raise SetupError(f"Test dataset {path} is not valid UTF-8: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SetupError(f"Test dataset {path} is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise SetupError(f"Cannot read test dataset {path}: {exc}") from exc

    if not isinstance(raw, dict) or not raw:
        raise SetupError(f"Test dataset {path} must be a non-empty JSON object mapping query -> embedding.")

    dataset: dict[str, list[float]] = {}
    for query, embedding in raw.items():
        if (
            not isinstance(embedding, list)
            or not embedding
            or not all(
                isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in embedding
            )
        ):
            raise SetupError(f"Embedding for query {query!r} must be a non-empty list of finite numbers.")
        dataset[query] = [float(x) for x in embedding]

    dataset_dimension(dataset)
    return dataset


def dataset_dimension(dataset: dict[str, list[float]]) -> int:
    """Return the common embedding dimension of the dataset.

    Raises:
        SetupError: if the dataset is empty or embeddings have different dimensions.
    """
    dims: set[int] = {len(embedding) for embedding in dataset.values()}
    if len(dims) != 1:
        raise SetupError(f"Test dataset embeddings must share one dimension, found: {sorted(dims)}")
    return dims.pop()


def validate_collection(client: QdrantClient, collection_name: str, dim: int) -> None:
    """Check that the collection exists and has a single unnamed dense vector of size ``dim``.

    Raises:
        SetupError: if the collection is missing, uses named (or no dense) vectors,
            or its vector size differs from ``dim``.
    """
    if not client.collection_exists(collection_name):
        raise SetupError(
            f"Collection '{collection_name}' does not exist. "
            "Create and populate it first (see the 'Vector database with Qdrant' project)."
        )

    vectors: VectorParams | dict[str, VectorParams] | None = client.get_collection(collection_name).config.params.vectors
    if isinstance(vectors, dict) and vectors:
        raise SetupError(
            f"Collection '{collection_name}' uses named vectors ({', '.join(sorted(vectors))}); "
            "this script expects a single unnamed dense vector."
        )
    if not isinstance(vectors, models.VectorParams):
        raise SetupError(
            f"Collection '{collection_name}' has no dense vector configuration "
            "(e.g. it only has sparse vectors); this script expects a single unnamed dense vector."
        )

    if vectors.size != dim:
        raise SetupError(
            f"Vector size mismatch: collection '{collection_name}' has size {vectors.size}, "
            f"but the test dataset embeddings have dimension {dim}."
        )


def create_client(settings: Settings) -> QdrantClient:
    """Create a Qdrant client for the configured host and port."""
    return QdrantClient(host=settings.host, port=settings.port, timeout=CLIENT_TIMEOUT_SECONDS)


def search_ids(
    client: QdrantClient,
    collection_name: str,
    embedding: Sequence[float],
    k: int,
    search_params: models.SearchParams | None = None,
) -> tuple[set[int | str | UUID], float | int]:
    """Run a single query and return the IDs of the top-``k`` points and the elapsed time in seconds.

    Only the ``query_points`` call is timed; payloads and vectors are not requested,
    so the measurement reflects the search cost rather than serialization.
    Pass ``search_params=models.SearchParams(exact=True)`` for an exact k-NN search.
    """
    query = embedding if isinstance(embedding, list) else list(embedding)
    start = time.perf_counter()
    points = client.query_points(
        collection_name=collection_name,
        query=query,
        limit=k,
        search_params=search_params,
        with_payload=False,
        with_vectors=False,
    ).points
    elapsed = time.perf_counter() - start
    return {point.id for point in points}, elapsed


def precision_at_k(ann_ids: set[int | str], knn_ids: set[int | str], k: int) -> float:
    """Return precision@k of the ANN result against the exact k-NN result (the golden set)."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}.")
    return len(ann_ids & knn_ids) / k


def quantized_search_params(rescore: bool, oversampling: float = OVERSAMPLING) -> models.SearchParams:
    """Return search params for the approximate search over the quantized vectors."""
    return models.SearchParams(
        quantization=models.QuantizationSearchParams(
            rescore=rescore,
            oversampling=oversampling,
        )
    )


def wait_for_optimization(
    client: QdrantClient,
    collection_name: str,
    timeout: float = OPTIMIZATION_TIMEOUT_SECONDS,
    poll_interval: float = OPTIMIZATION_POLL_SECONDS,
) -> None:
    """Block until the collection is ``green``, i.e. the optimizers have finished building the quantized vectors.

    A ``grey`` collection has pending optimizations that were not started yet; an empty
    optimizer config update is sent to trigger them.

    Raises:
        SetupError: if the collection turns ``red`` or is still not ``green`` after ``timeout`` seconds.
    """
    deadline = time.monotonic() + timeout
    while True:
        status = client.get_collection(collection_name).status
        if status == models.CollectionStatus.GREEN:
            return
        if status == models.CollectionStatus.RED:
            raise SetupError(f"Collection '{collection_name}' is red: an optimization failed, check the Qdrant logs.")
        if status == models.CollectionStatus.GREY:
            client.update_collection(collection_name=collection_name, optimizer_config=models.OptimizersConfigDiff())
        if time.monotonic() >= deadline:
            raise SetupError(
                f"Collection '{collection_name}' is still {status.value} after {timeout:.0f} s; "
                "the quantized vectors are not ready yet, try again later."
            )
        time.sleep(poll_interval)


def apply_scalar_quantization(client: QdrantClient, collection_name: str) -> bool:
    """Enable int8 scalar quantization on the collection and wait until the quantized vectors are built.

    The original vectors are kept, so the exact baseline can still use them. The update is
    skipped when the collection already has the same quantization config.

    Returns:
        ``True`` if the collection was updated, ``False`` if it was already quantized this way.
    """
    current = client.get_collection(collection_name).config.quantization_config
    updated = current != QUANTIZATION_CONFIG
    if updated:
        client.update_collection(
            collection_name=collection_name,
            optimizer_config=models.OptimizersConfigDiff(),
            quantization_config=QUANTIZATION_CONFIG,
        )
    wait_for_optimization(client, collection_name)
    return updated


def warm_up(
    client: QdrantClient,
    collection_name: str,
    embeddings: Iterable[Sequence[float]],
    k: int,
    ann_search_params: models.SearchParams,
    n: int = WARM_UP_QUERIES,
) -> None:
    """Run up to ``n`` untimed approximate and exact queries so cold caches do not skew the measurements."""
    for i, embedding in enumerate(embeddings):
        if i >= n:
            break
        search_ids(client, collection_name, embedding, k, ann_search_params)
        search_ids(client, collection_name, embedding, k, EXACT_SEARCH_PARAMS)


def evaluate_ann_vs_knn(
    client: QdrantClient,
    collection_name: str,
    test_dataset: dict[str, list[float]],
    k: int,
    ann_search_params: models.SearchParams,
    warm_up_queries: int = WARM_UP_QUERIES,
) -> tuple[float, float, float]:
    """Compare the approximate search (with ``ann_search_params``) with the exact k-NN search.

    The exact search ignores quantization, so it is a full-precision baseline. ANN and exact
    queries are interleaved per embedding so both see the same cache state.

    Returns:
        ``(avg_precision, avg_ann_time, avg_knn_time)``, with times in seconds.

    Raises:
        ValueError: if the test dataset is empty.
    """
    if not test_dataset:
        raise ValueError("Test dataset is empty.")

    warm_up(client, collection_name, test_dataset.values(), k, ann_search_params, warm_up_queries)

    precisions: list[float] = []
    ann_times: list[float] = []
    knn_times: list[float] = []
    for _query, embedding in test_dataset.items():
        ann_ids, ann_time = search_ids(client, collection_name, embedding, k, ann_search_params)
        knn_ids, knn_time = search_ids(client, collection_name, embedding, k, EXACT_SEARCH_PARAMS)
        precisions.append(precision_at_k(ann_ids, knn_ids, k))
        ann_times.append(ann_time)
        knn_times.append(knn_time)

    return statistics.mean(precisions), statistics.mean(ann_times), statistics.mean(knn_times)


def result_formatting(k, avg_precision, avg_ann_time, avg_knn_time):
    print(f'Average precision@{k}: {avg_precision:.4f}')
    print(f'Average ANN query time: {avg_ann_time * 1000:.2f} ms')
    print(f'Average exact k-NN query time: {avg_knn_time * 1000:.2f} ms')


def main() -> None:
    try:
        settings = load_settings()
    except SetupError as exc:
        sys.exit(f"Error: {exc}")

    client = create_client(settings)
    try:
        test_dataset = load_test_dataset(QUERIES_FILE)
        dim = dataset_dimension(test_dataset)
        validate_collection(client, settings.collection_name, dim)
        apply_scalar_quantization(client, settings.collection_name)
        for i, rescore in enumerate(RESCORE_VALUES):
            if i:
                print()
            print(f"Scalar quantization (int8), rescore={rescore}, oversampling={OVERSAMPLING}:")
            avg_precision, avg_ann_time, avg_knn_time = evaluate_ann_vs_knn(
                client, settings.collection_name, test_dataset, K, quantized_search_params(rescore)
            )
            result_formatting(K, avg_precision, avg_ann_time, avg_knn_time)
    except SetupError as exc:
        sys.exit(f"Error: {exc}")
    except (ResponseHandlingException, ConnectionError) as exc:
        sys.exit(
            f"Error: cannot reach Qdrant at {settings.host}:{settings.port} ({exc}).\n"
            "Hint: start the Qdrant Docker container, e.g. "
            "`docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant`."
        )
    except UnexpectedResponse as exc:
        sys.exit(
            f"Error: Qdrant at {settings.host}:{settings.port} returned HTTP {exc.status_code} "
            f"({exc.reason_phrase}).\n"
            "Hint: check that QDRANT_HOST/QDRANT_PORT point to a Qdrant REST endpoint "
            "and that no API key is required."
        )
    finally:
        client.close()


if __name__ == "__main__":
    main()
