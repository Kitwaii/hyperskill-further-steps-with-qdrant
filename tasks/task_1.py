"""Task 1: compare Qdrant's approximate (HNSW) search with exact k-NN search.

Run from the repository root with::

    uv run python tasks/task_1.py

If the ``arxiv_papers`` collection does not exist yet (only the Qdrant container is
running, see ``docker-compose.yml``), the script first creates it and loads
``ml-arxiv-embeddings.json`` into it **in batches**, then waits for the collection to be
green before measuring anything.

The loading is designed for a machine with ~1 GB of free RAM:

* the 12 GB JSON Lines file is streamed line by line and upserted in batches of
  ``BATCH_SIZE`` points with ``wait=True``, so neither the script nor Qdrant ever holds
  more than one batch in flight;
* vectors (~2.5 GB) and the HNSW graph are stored on disk (memory-mapped), payloads too
  (Qdrant default), so the OS page cache keeps only what fits in RAM;
* HNSW indexing is deferred (``indexing_threshold=0``) during the upload and enabled once
  every point is in, so the graph is built once instead of competing with the ingestion.

``indexing_threshold == 0`` therefore also marks an unfinished load: if the script is
interrupted, running it again re-upserts the file (point IDs are UUIDv5 of the arXiv ID,
so nothing is duplicated) and then enables indexing.

Connection settings default to ``localhost:6333`` and the ``arxiv_papers``
collection; they can be overridden with the ``QDRANT_HOST``, ``QDRANT_PORT``,
``COLLECTION_NAME`` and ``EMBEDDING_FILE_PATH`` environment variables (or a ``.env`` file).
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time
import uuid
from collections.abc import Generator, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.http.models import SearchParams, VectorParams

K: int = 10
QUERIES_FILE: Path = Path(__file__).resolve().parent.parent / "dataset" / "queries_embeddings.json"
CLIENT_TIMEOUT_SECONDS: int = 60
WARM_UP_QUERIES: int = 5
EXACT_SEARCH_PARAMS: SearchParams = SearchParams(exact=True)

DEFAULT_QDRANT_HOST: str = "localhost"
DEFAULT_QDRANT_PORT: int = 6333
DEFAULT_COLLECTION_NAME: str = "arxiv_papers"
DEFAULT_EMBEDDING_FILE: Path = Path(__file__).resolve().parent.parent / "ml-arxiv-embeddings.json"

VECTOR_DIMENSION: int = 1536  # text-embedding-ada-002
BATCH_SIZE: int = 500
DEFERRED_INDEXING_THRESHOLD: int = 0  # no HNSW build while uploading
DEFAULT_INDEXING_THRESHOLD: int = 10_000  # Qdrant default (KB), restored after the upload
MAX_INDEXING_THREADS: int = 2  # fewer threads -> lower peak RAM while building HNSW
GREEN_POLL_SECONDS: float = 10.0


class SetupError(RuntimeError):
    """Raised when the settings, the dataset or the Qdrant collection are not usable for the evaluation."""


@dataclass(frozen=True)
class Settings:
    """Qdrant connection settings."""

    host: str = DEFAULT_QDRANT_HOST
    port: int = DEFAULT_QDRANT_PORT
    collection_name: str = DEFAULT_COLLECTION_NAME
    embedding_file: Path = DEFAULT_EMBEDDING_FILE


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
        embedding_file=Path(os.getenv("EMBEDDING_FILE_PATH", str(DEFAULT_EMBEDDING_FILE))),
    )


def create_collection(client: QdrantClient, collection_name: str) -> None:
    """Create the collection for the arXiv embeddings with a low-RAM, deferred-indexing setup.

    Distance and size are the ones of the 'Vector database with Qdrant' project (cosine, 1536);
    HNSW parameters (``m``, ``ef_construct``...) stay at their defaults.
    """
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=VECTOR_DIMENSION, distance=models.Distance.COSINE, on_disk=True),
        hnsw_config=models.HnswConfigDiff(on_disk=True, max_indexing_threads=MAX_INDEXING_THREADS),
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=DEFERRED_INDEXING_THRESHOLD),
        on_disk_payload=True,
    )
    print(f"Collection '{collection_name}' created (vectors on disk, indexing deferred).")


def stream_json(file_path: Path) -> Generator[Any, Any, None]:
    """Yield one record per non-empty line, without ever loading the whole file into memory."""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_vectors_to_qdrant(
    client: QdrantClient, collection_name: str, file_path: Path, batch_size: int = BATCH_SIZE
) -> None:
    """Upsert the file into the collection in batches of ``batch_size`` points.

    The embedding becomes the vector, every other field goes to the payload; records
    without an embedding are skipped. ``wait=True`` makes each upsert block until Qdrant
    has applied the batch, so data never piles up on the server side.
    """
    batch: list[models.PointStruct] = []
    total_records = 0
    skipped = 0
    start = time.perf_counter()

    for record in stream_json(file_path):
        if not record.get("embedding"):
            skipped += 1
            continue

        payload = {k: v for k, v in record.items() if k != "embedding"}
        point = models.PointStruct(
            id=str(uuid.uuid5(namespace=uuid.NAMESPACE_DNS, name=str(record["id"]))),
            vector=record["embedding"],
            payload=payload,
        )
        batch.append(point)

        if len(batch) >= batch_size:
            client.upsert(collection_name=collection_name, points=batch, wait=True)
            total_records += len(batch)
            batch = []
            if total_records % (batch_size * 20) == 0:
                print(f"Inserted {total_records} records... ({time.perf_counter() - start:.0f} s)")

    if batch:
        client.upsert(collection_name=collection_name, points=batch, wait=True)
        total_records += len(batch)
        batch = []

    print(
        f"Total records inserted: {total_records} ({skipped} skipped without embedding) "
        f"in {(time.perf_counter() - start) / 60:.1f} min."
    )


def enable_indexing(client: QdrantClient, collection_name: str) -> None:
    """Restore the default indexing threshold so Qdrant builds the HNSW graph."""
    client.update_collection(
        collection_name=collection_name,
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=DEFAULT_INDEXING_THRESHOLD),
    )
    print("Upload finished, HNSW indexing enabled.")


def wait_until_green(client: QdrantClient, collection_name: str) -> None:
    """Block until the collection is green: no operation should run while it is ingesting or indexing."""
    while True:
        info = client.get_collection(collection_name)
        if info.status == models.CollectionStatus.GREEN:
            print(
                f"Collection '{collection_name}' is green: {info.points_count} points, "
                f"{info.indexed_vectors_count} indexed vectors."
            )
            return
        print(
            f"Collection is {info.status.value}: "
            f"{info.indexed_vectors_count}/{info.points_count} vectors indexed, waiting..."
        )
        time.sleep(GREEN_POLL_SECONDS)


def ensure_collection_loaded(client: QdrantClient, settings: Settings) -> None:
    """Create and load the collection if needed, then wait for it to be green.

    A collection whose indexing threshold is still 0 was not fully loaded (interrupted run):
    the file is upserted again before indexing is enabled.
    """
    name = settings.collection_name
    if not client.collection_exists(name):
        create_collection(client, name)
    elif client.get_collection(name).config.optimizer_config.indexing_threshold != DEFERRED_INDEXING_THRESHOLD:
        wait_until_green(client, name)
        return
    else:
        print(f"Collection '{name}' was not fully loaded: resuming the upload.")

    if not settings.embedding_file.is_file():
        raise SetupError(
            f"Embedding file not found: {settings.embedding_file} "
            "(set EMBEDDING_FILE_PATH to the ml-arxiv-embeddings.json location)."
        )
    load_vectors_to_qdrant(client, name, settings.embedding_file)
    enable_indexing(client, name)
    wait_until_green(client, name)


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

    info = client.get_collection(collection_name)
    if info.status != models.CollectionStatus.GREEN:
        raise SetupError(
            f"Collection '{collection_name}' is {info.status.value} (still ingesting or indexing: "
            f"{info.indexed_vectors_count}/{info.points_count} vectors indexed). "
            "Wait until it is green before measuring."
        )

    vectors: VectorParams | dict[str, VectorParams] | None = info.config.params.vectors
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
    n: int = WARM_UP_QUERIES,
) -> None:
    """Run up to ``n`` untimed ANN and exact queries so cold caches do not skew the measurements."""
    for i, embedding in enumerate(embeddings):
        if i >= n:
            break
        search_ids(client, collection_name, embedding, k)
        search_ids(client, collection_name, embedding, k, EXACT_SEARCH_PARAMS)


def evaluate_ann_vs_knn(
    client: QdrantClient,
    collection_name: str,
    test_dataset: dict[str, list[float]],
    k: int,
) -> tuple[float, float, float]:
    """Compare the default ANN (HNSW) search with exact k-NN search over the test dataset.

    ANN and exact queries are interleaved per embedding so both see the same cache state.

    Returns:
        ``(avg_precision, avg_ann_time, avg_knn_time)``, with times in seconds.

    Raises:
        ValueError: if the test dataset is empty.
    """
    if not test_dataset:
        raise ValueError("Test dataset is empty.")

    precisions: list[float] = []
    ann_times: list[float] = []
    knn_times: list[float] = []
    for _query, embedding in test_dataset.items():
        ann_ids, ann_time = search_ids(client, collection_name, embedding, k)
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
        ensure_collection_loaded(client, settings)
        validate_collection(client, settings.collection_name, dim)
        warm_up(client, settings.collection_name, test_dataset.values(), K)
        avg_precision, avg_ann_time, avg_knn_time = evaluate_ann_vs_knn(
            client, settings.collection_name, test_dataset, K
        )
        result_formatting(K, avg_precision, avg_ann_time, avg_knn_time)
    except SetupError as exc:
        sys.exit(f"Error: {exc}")
    except (ResponseHandlingException, ConnectionError) as exc:
        sys.exit(
            f"Error: cannot reach Qdrant at {settings.host}:{settings.port} ({exc}).\n"
            "Hint: start the Qdrant Docker container with `docker compose up -d`."
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
