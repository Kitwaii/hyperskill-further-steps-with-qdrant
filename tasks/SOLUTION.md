### Observed values 

```
Average precision@10: 1.0000
Average ANN query time: 6.42 ms
Average exact k-NN query time: 6.40 ms
```

### Reflection

**Setup.** The values above were measured against a local Qdrant Docker container. The `arxiv_papers` collection held 3,420 points: 1536-dimensional `text-embedding-ada-002` embeddings with cosine distance, default HNSW parameters (`m=16`, `ef_construct=100`, `full_scan_threshold=10000` KB), status `green`, all vectors indexed and spread over 8 segments. The 100 queries from `dataset/queries_embeddings.json` were used with `k=10`. Five untimed warm-up query pairs ran before measuring, and the ANN and exact queries were interleaved for each embedding.

**Pattern 1: the approximate search matches exact search perfectly (precision@10 = 1.0000).** For every one of the 100 queries, the HNSW top-10 was identical to the exact top-10. This is expected at this scale:
- The collection is small: about 3.4k vectors, or roughly 21 MB of raw floats, split into segments of only a few hundred points each. With the default search-time `ef` (at least 100, far above `k=10`), the HNSW traversal visits a large share of each small graph, so it almost never misses a true neighbour.
- `full_scan_threshold` is 10,000 KB, and every segment is well below that size (about 2.6 MB). The query planner may therefore skip the graph and scan these segments brute-force, which makes the "approximate" search exact in practice.
- The defaults (`m=16`, `ef_construct=100`) are tuned for high recall. Missed neighbours usually only appear with much larger collections (hundreds of thousands to millions of points), a lower `hnsw_ef`, or quantization.

Because the exact top-k is the golden set, precision@10 equals recall@10 here. A score of 1.0 means the ANN search lost no true neighbours.

**Pattern 2: ANN and exact k-NN take the same time (6.42 ms vs 6.40 ms).** The 0.02 ms difference is well within measurement noise, so neither mode is faster. The latency is dominated by fixed overhead rather than by the search itself:
- Each query is an HTTP/REST round trip, and the 1536-float query vector is serialised to JSON.
- The Python client adds its own processing.

Scoring 3,420 vectors by brute force takes well under a millisecond, so the work HNSW saves is hidden behind several milliseconds of overhead. Exact search costs O(N) per query, while HNSW costs roughly O(log N). The gap only becomes visible once a full scan is expensive, i.e. on much larger collections.

**Caveats.**
- 100 queries is a small sample, and individual timings vary with caching, OS scheduling and background activity, so the averages are only indicative.
- Only IDs were requested (`with_payload=False`, `with_vectors=False`) and only the `query_points` call was timed. The measurements therefore exclude payload transfer, but they still include network and serialisation cost.
- The results describe this particular small collection and should not be extrapolated to production scale.

**Conclusion.** At this collection size, the speed/accuracy trade-off of ANN does not show: HNSW is as accurate as exact k-NN and no faster. The trade-off would appear with many more points, or by lowering `hnsw_ef` (and/or `m`). This would reduce latency at the cost of precision, which is explored in the next task.