"""Task 2: balance the search by sweeping the HNSW search parameter ``hnsw_ef``.

The exact k-NN results are computed once and used as the ground truth; the
approximate search is then run for every ``hnsw_ef`` value to measure its
average precision@k and query time. Only search-time parameters change: the
index and the stored embeddings are left untouched.

Run from the repository root with::

    uv run python tasks/task_2.py

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
EXACT_SEARCH_PARAMS: SearchParams = SearchParams(exact=True)
HNSW_EF_VALUES: list[int] = [10, 20, 50, 100, 200]
# Each hnsw_ef value is timed over this many passes of the test dataset; the first
# pass is discarded as an extra warm-up, the remaining ones are averaged.
TIMED_REPEATS: int = 3

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


def warm_up(
    client: QdrantClient,
    collection_name: str,
    embeddings: Iterable[Sequence[float]],
    k: int,
    hnsw_ef_values: Sequence[int],
    n: int = WARM_UP_QUERIES,
) -> None:
    """Run up to ``n`` untimed approximate queries per ``hnsw_ef`` value to warm Qdrant and the OS page cache."""
    for i, embedding in enumerate(embeddings):
        if i >= n:
            break
        for hnsw_ef in hnsw_ef_values:
            search_ids(client, collection_name, embedding, k, models.SearchParams(hnsw_ef=hnsw_ef))


def compute_ground_truth(
    client: QdrantClient,
    collection_name: str,
    test_dataset: dict[str, list[float]],
    k: int,
) -> dict[str, set[int | str | UUID]]:
    """Run the exact k-NN search for every query and return its top-``k`` IDs keyed by the query text."""
    ground_truth: dict[str, set[int | str | UUID]] = {}
    for query, embedding in test_dataset.items():
        ground_truth[query], _elapsed = search_ids(client, collection_name, embedding, k, EXACT_SEARCH_PARAMS)
    return ground_truth


def evaluate_hnsw_ef(
    k: int,
    hnsw_ef_values: Sequence[int],
    test_dataset: dict[str, list[float]],
    *,
    client: QdrantClient,
    collection_name: str,
    repeats: int = TIMED_REPEATS,
    warm_up_queries: int = WARM_UP_QUERIES,
) -> list[dict[str, float | int]]:
    """Measure precision@``k`` and query time of the approximate search for each ``hnsw_ef`` value.

    The exact k-NN results are the ground truth. Before timing, a few throwaway approximate
    queries warm the caches; each ``hnsw_ef`` value is then run ``repeats`` times over the whole
    test dataset and the first pass is discarded, so the order of the values does not bias the timings.

    Returns:
        One dict per ``hnsw_ef`` value: ``{"hnsw_ef", "avg_precision", "avg_query_time_ms"}``.

    Raises:
        ValueError: if the test dataset is empty, ``hnsw_ef_values`` contains a non-positive value,
            or ``repeats`` is lower than 2.
    """
    if not test_dataset:
        raise ValueError("Test dataset is empty.")
    if any(hnsw_ef <= 0 for hnsw_ef in hnsw_ef_values):
        raise ValueError(f"hnsw_ef values must be positive, got {list(hnsw_ef_values)}.")
    if repeats < 2:
        raise ValueError(f"repeats must be at least 2 (the first pass is discarded), got {repeats}.")

    ground_truth = compute_ground_truth(client, collection_name, test_dataset, k)
    warm_up(client, collection_name, test_dataset.values(), k, hnsw_ef_values, warm_up_queries)

    results: list[dict[str, float | int]] = []
    for hnsw_ef in hnsw_ef_values:
        search_params = models.SearchParams(hnsw_ef=hnsw_ef)
        precisions: list[float] = []
        query_times: list[float] = []
        for repeat in range(repeats):
            for query, embedding in test_dataset.items():
                ann_ids, elapsed = search_ids(client, collection_name, embedding, k, search_params)
                if repeat == 0:
                    continue
                precisions.append(precision_at_k(ann_ids, ground_truth[query], k))
                query_times.append(elapsed)
        results.append(
            {
                "hnsw_ef": hnsw_ef,
                "avg_precision": statistics.mean(precisions),
                "avg_query_time_ms": statistics.mean(query_times) * 1000,
            }
        )
    return results


def display_results(k: int, results: list[dict[str, float | int]]) -> None:
    """Print the results list and a readable table of precision@``k`` and query time per ``hnsw_ef``."""
    print("results = [")
    for result in results:
        print(f"    {result},")
    print("]")
    print()
    print(f"{'hnsw_ef':>8} | {f'avg precision@{k}':>17} | {'avg query time':>14}")
    print(f"{'-' * 8}-+-{'-' * 17}-+-{'-' * 14}")
    for result in results:
        print(
            f"{result['hnsw_ef']:>8} | {result['avg_precision']:>17.4f} | "
            f"{result['avg_query_time_ms']:>11.2f} ms"
        )


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
        results = evaluate_hnsw_ef(
            K, HNSW_EF_VALUES, test_dataset, client=client, collection_name=settings.collection_name
        )
        display_results(K, results)
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
