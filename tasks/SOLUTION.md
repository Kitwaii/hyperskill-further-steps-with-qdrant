### Observed values 

```
results = [
    {'hnsw_ef': 10, 'avg_precision': 0.98, 'avg_query_time_ms': 13.85786572122015},
    {'hnsw_ef': 20, 'avg_precision': 0.994, 'avg_query_time_ms': 18.56782881775871},
    {'hnsw_ef': 50, 'avg_precision': 0.999, 'avg_query_time_ms': 30.718583804555237},
    {'hnsw_ef': 100, 'avg_precision': 0.999, 'avg_query_time_ms': 46.10799675807357},
    {'hnsw_ef': 200, 'avg_precision': 1.0, 'avg_query_time_ms': 80.20917290821671},
]

 hnsw_ef |  avg precision@10 | avg query time
---------+-------------------+---------------
      10 |            0.9800 |       13.86 ms
      20 |            0.9940 |       18.57 ms
      50 |            0.9990 |       30.72 ms
     100 |            0.9990 |       46.11 ms
     200 |            1.0000 |       80.21 ms
```

### Reflection

**Setup.** The values above were measured against a local Qdrant 1.18.0 Docker container on a 16 GB machine with about 1 GB of free RAM. The `arxiv_papers` collection held 411,120 points: 1536-dimensional `text-embedding-ada-002` embeddings with cosine distance and default HNSW parameters (`m=16`, `ef_construct=100`, `full_scan_threshold=10000` KB). The collection was green, with all vectors indexed and spread over 9 segments. It was loaded by the script in batches of 500 points with HNSW indexing deferred until the end of the upload, and vectors, the HNSW graph and payloads are stored on disk (memory-mapped). The 100 queries from `dataset/queries_embeddings.json` were used with `k=10`. The exact k-NN top-10 (`SearchParams(exact=True)`) was computed once per query and used as the ground truth. The approximate search was then run with `SearchParams(hnsw_ef=...)` for `hnsw_ef` in `[10, 20, 50, 100, 200]`. Following the note on cache warming, 10 untimed warm-up queries were run for every `hnsw_ef` value, and each value was then run over the whole dataset 3 times. The first pass was discarded and the other two were averaged, so the order of the values does not bias the timings. Only search-time parameters changed: the index and the stored embeddings were left untouched.

**Pattern 1: precision@10 rises with `hnsw_ef` and saturates quickly.** Precision goes from 0.980 (`hnsw_ef=10`) to 0.994 (20), 0.999 (50 and 100) and 1.000 (200). This is the expected behaviour of HNSW:
- `hnsw_ef` is the size of the candidate list kept during graph traversal. A small list makes the search greedy and it can miss true neighbours. A larger list explores more of the graph and recovers them.
- With `hnsw_ef=10` and `limit=10`, the list is exactly as long as the number of results, so there is no margin: 20 of the 1,000 true neighbours (100 queries × 10) are lost.
- The gains shrink quickly: +1.4 points from 10 to 20, +0.5 from 20 to 50, nothing from 50 to 100, and +0.1 (a single neighbour) from 100 to 200.
- `hnsw_ef=100` gives 0.9990, exactly the precision measured in task 1 with the default search parameters. This is consistent: when `hnsw_ef` is not set, Qdrant uses `ef_construct` (100) as the search-time `ef`.
- The segments (about 45k points, ~280 MB of vectors each) are far larger than `full_scan_threshold`, so the HNSW graph is really used. The numbers therefore measure a real ANN accuracy cost, not a hidden full scan.

Because the exact top-k is the golden set, precision@10 equals recall@10 here.

**Pattern 2: the query time grows with `hnsw_ef`, from 13.86 ms to 80.21 ms (about 5.8x).** A larger candidate list means more nodes visited and more distance computations on 1536-dimensional vectors. The growth is steady, and above `hnsw_ef=50` it is close to linear (+15 ms from 50 to 100, +34 ms from 100 to 200). On this machine, each visited node is likely to require a read from the memory-mapped vectors and graph on disk, since only a fraction of the ~2.5 GB of vectors fits in the page cache. The search work, rather than the fixed HTTP and serialization overhead, therefore dominates the latency. The default `hnsw_ef=100` takes 46.11 ms here, against 97.63 ms in task 1. The difference comes from the measurement protocol rather than the search: in task 1 each approximate query was interleaved with an exact scan of the whole collection, which evicts the cache, while here each value is repeated and the first pass is discarded.

**Caveats.**
- 100 queries (200 timed queries per value) is a small sample, and timings depend heavily on the page cache state, disk speed and background activity, so the averages are only indicative. A precision difference of 0.001 corresponds to a single missed neighbour over 1,000 results.
- Only IDs were requested (`with_payload=False`, `with_vectors=False`) and only the `query_points` call was timed; network and serialization costs are still included.
- The results come from a single on-disk, memory-constrained setup and a single index built with default parameters. An in-memory deployment, or a different `m`/`ef_construct`, would give much lower absolute latencies and a different curve.

**Conclusion.** `hnsw_ef` is a real speed/accuracy knob on this collection. `hnsw_ef=50` is the best balance: 99.9% precision at 30.72 ms, the same precision as the default `hnsw_ef=100` while being about 33% faster (46.11 ms). `hnsw_ef=200` is the only value with perfect precision, but it costs 80.21 ms (+74% over 100) to recover a single neighbour. `hnsw_ef=10` is the fastest (13.86 ms) but loses 2% of the true neighbours, which is only acceptable if latency matters much more than recall. Here, a value between 20 and 50 is the sensible range.
