### Observed values 

```
Average precision@10: 1.0000
Average ANN query time: 6.32 ms
Average exact k-NN query time: 72.50 ms
```

### Reflection

**Setup.** The values above were measured against a local Qdrant (v1.20.0) Docker container. The `arxiv_papers` collection held 411,220 points: 1536-dimensional `text-embedding-ada-002` embeddings with cosine distance, default HNSW parameters (`m=16`, `ef_construct=100`, `full_scan_threshold=10000` KB), status `green`, 410,624 vectors indexed and spread over 8 segments. The collection was loaded from the full `ml-arxiv-embeddings.json` file (411,323 lines, of which 103 have no embedding and were skipped; the file also contains about 100 repeated arXiv ids, so the count is within ~100 of the 411,120 reference). The 100 queries from `dataset/queries_embeddings.json` were used with `k=10`. Five untimed warm-up query pairs ran before measuring, and the ANN and exact queries were interleaved for each embedding. The measurement was repeated three times, and the results were stable (precision 1.0000 every time, ANN 6.0-6.4 ms, exact 70-72.5 ms).

**Pattern 1: the approximate search still matches exact search perfectly (precision@10 = 1.0000).** For every one of the 100 queries, the HNSW top-10 was identical to the exact top-10, even with 411k points. Several factors explain this:
- The collection is large enough that HNSW is really used: about 411k vectors, or roughly 2.5 GB of raw floats, split into 8 segments of about 50k points (~300 MB each). This is far above `full_scan_threshold` (10,000 KB), so the planner traverses the graph instead of scanning brute-force. The perfect score is therefore a property of the index and not of a hidden full scan.
- The default search-time `ef` is at least 100, ten times `k=10`, so the graph search explores many more candidates than it returns. With `m=16` and `ef_construct=100`, the graph is built with high-recall settings.
- Each query is searched in 8 segments, and each segment returns its own top candidates that are merged afterwards. This gives further safety margin.
- `text-embedding-ada-002` vectors of related paper abstracts are well-clustered, and the queries are topical, so the true neighbours are close to each other and easy to reach in the graph.
- 100 queries is a small sample: a few missed neighbours would be visible only as a precision slightly below 1.0, and none showed up. Lower `hnsw_ef`, a smaller `m` or quantization are needed to expose the accuracy cost of ANN.

Because the exact top-k is the golden set, precision@10 equals recall@10 here. A score of 1.0 means the ANN search lost no true neighbours.

**Pattern 2: ANN is about 11x faster than exact k-NN (6.32 ms vs 72.50 ms).** This is the speed/accuracy trade-off that was not visible on a small collection:
- Exact search must score every one of the 411k 1536-dimensional vectors, which is O(N) work per query. That is about 630 million multiply-adds per query, and it dominates the 72 ms.
- HNSW only visits a few thousand vectors per segment by following graph edges (roughly O(log N)), so the search itself takes a small fraction of a millisecond to a few milliseconds.
- The ANN latency of 6 ms is mostly fixed overhead rather than search: an HTTP/REST round trip, serialising the 1536-float query vector to JSON, and Python client processing. The exact search has the same overhead, so the real difference in search work is even larger than 11x.

The cost of exact search grows linearly with the collection size, while the cost of HNSW grows slowly, so the gap would widen further with millions of points.

**Caveats.**
- 100 queries is a small sample, and individual timings vary with caching, OS scheduling and background activity, so the averages are only indicative.
- Only IDs were requested (`with_payload=False`, `with_vectors=False`) and only the `query_points` call was timed. The measurements therefore exclude payload transfer, but they still include network and serialisation cost.
- The measurements were run on a single local machine with the data in the page cache and the HNSW graph in RAM; a remote or disk-bound deployment will show different absolute numbers.
- Precision@10 = 1.0 only says that default settings were accurate for these queries; it does not guarantee that harder queries or tuned-down settings behave the same way.

**Conclusion.** At 411k points, ANN keeps the accuracy of exact k-NN (precision@10 = 1.0000) while being around 11x faster (6.32 ms vs 72.50 ms). The default HNSW configuration is therefore a good fit for this collection: it gives a large speed-up at no measurable accuracy cost. The trade-off can be pushed further, by lowering `hnsw_ef` (and/or `m`), which would reduce latency at the cost of precision. This is explored in the next task.
