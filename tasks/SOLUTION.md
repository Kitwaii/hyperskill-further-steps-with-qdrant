### Observed values 

```
results = [
    {'hnsw_ef': 10, 'avg_precision': 1.0, 'avg_query_time_ms': 6.255421860009847},
    {'hnsw_ef': 20, 'avg_precision': 1.0, 'avg_query_time_ms': 7.0283229500500966},
    {'hnsw_ef': 50, 'avg_precision': 1.0, 'avg_query_time_ms': 6.294976239969401},
    {'hnsw_ef': 100, 'avg_precision': 1.0, 'avg_query_time_ms': 6.243683294997027},
    {'hnsw_ef': 200, 'avg_precision': 1.0, 'avg_query_time_ms': 6.231272044974503},
]

 hnsw_ef |  avg precision@10 | avg query time
---------+-------------------+---------------
      10 |            1.0000 |        6.26 ms
      20 |            1.0000 |        7.03 ms
      50 |            1.0000 |        6.29 ms
     100 |            1.0000 |        6.24 ms
     200 |            1.0000 |        6.23 ms
```

### Reflection

**Setup.** The values above were measured against a local Qdrant Docker container. The `arxiv_papers` collection held 4,540 points: 1536-dimensional `text-embedding-ada-002` embeddings with cosine distance, default HNSW parameters (`m=16`, `ef_construct=100`, `full_scan_threshold=10000` KB), status `green`, spread over 8 segments. The 100 queries from `dataset/queries_embeddings.json` were used with `k=10`. The exact k-NN top-10 (`SearchParams(exact=True)`) was computed once per query and used as the ground truth. The approximate search was then run with `SearchParams(hnsw_ef=...)` for `hnsw_ef` in `[10, 20, 50, 100, 200]`. Following the note on cache warming, 10 untimed warm-up queries were run for every `hnsw_ef` value, and each value was then run over the whole dataset 3 times. The first pass was discarded and the other two were averaged, so the order of the values does not bias the timings. Only search-time parameters changed: the index and the stored embeddings were left untouched.

**Pattern 1: precision@10 is 1.0000 for every `hnsw_ef`, even `hnsw_ef=10`.** In theory, a lower `hnsw_ef` (a smaller candidate list during graph traversal) should lose some true neighbours, and precision should rise as `hnsw_ef` grows. Here, the curve is flat at the maximum. Two things explain this:
- **The HNSW graph is not used at all on this collection.** Qdrant's telemetry for the run reported about 12,500 `unfiltered_plain` (full-scan) searches, about 700 `unfiltered_exact` searches and no HNSW searches. Each segment holds about 570 vectors × 1536 floats × 4 bytes ≈ 3.5 MB, which is well below `full_scan_threshold=10000` KB. For segments that small, Qdrant's query planner decides that a brute-force scan is cheaper than walking the graph. It scans every vector, so the "approximate" search returns exactly the exact top-10, and `hnsw_ef` is simply ignored.
- **Even if HNSW were used, `hnsw_ef=10` would not be a real stress test.** The candidate list can never be smaller than the number of results requested, so with `limit=10` the effective `ef` is at least 10. And graphs of a few hundred nodes per segment are small enough that even a modest `ef` explores most of them.

Because the exact top-k is the golden set, precision@10 equals recall@10 here: no true neighbour was lost for any setting.

**Pattern 2: the query time does not depend on `hnsw_ef` (6.23–6.29 ms, with one 7.03 ms outlier at `hnsw_ef=20`).** If the graph were traversed, a higher `hnsw_ef` would mean more candidates, more distance computations and a higher latency. The timings show no trend, for two reasons:
- Since every query is a full scan, the search work is identical for all `hnsw_ef` values.
- The latency is dominated by fixed overhead rather than by the search itself: an HTTP/REST round trip, JSON serialisation of the 1536-float query vector, and the Python client. Scoring about 4.5k vectors takes well under a millisecond.

The 7.03 ms value at `hnsw_ef=20` doesn't fit any trend (both lower and higher values are faster). It is measurement noise, such as OS scheduling, background activity or a garbage-collection pause, rather than an effect of the parameter.

**Caveats.**
- 100 queries (200 timed queries per value) is a small sample, and timings of a few milliseconds are sensitive to noise, so the averages are only indicative.
- Only IDs were requested (`with_payload=False`, `with_vectors=False`) and only the `query_points` call was timed, but network and serialisation cost are still included.
- The results describe this particular small collection and should not be extrapolated to production scale.

**Conclusion.** On this collection, tuning `hnsw_ef` has no visible effect: Qdrant falls back to exact full scans because the segments are below `full_scan_threshold`, so precision is always perfect and latency is constant. The expected speed/accuracy trade-off (lower `hnsw_ef` → faster but less precise, higher `hnsw_ef` → more precise but slower) would only appear once the HNSW index is actually used. That happens with a much larger collection (hundreds of thousands to millions of points), or with a lower `full_scan_threshold`, which was not changed here because the task asks to leave the collection configuration untouched. For a collection of this size, the default settings are already optimal: there is no accuracy to gain from a higher `hnsw_ef` and no speed to gain from a lower one.