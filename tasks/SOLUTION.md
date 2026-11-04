### Observed values 

```
Average precision@10: 0.9990
Average ANN query time: 97.63 ms
Average exact k-NN query time: 440.81 ms
```

### Reflection

**Setup.** The values above were measured against a local Qdrant 1.18.0 Docker container on a 16 GB machine with about 1 GB of free RAM. The `arxiv_papers` collection held 411,120 points: 1536-dimensional `text-embedding-ada-002` embeddings with cosine distance and default HNSW parameters (`m=16`, `ef_construct=100`, `full_scan_threshold=10000` KB). The collection was green, with all vectors indexed and spread over 9 segments. To fit in memory, the collection was loaded by the script in batches of 500 points (`wait=True`), with HNSW indexing deferred until the upload was complete (`indexing_threshold=0`, then restored to 10000). Vectors, the HNSW graph and payloads are stored on disk (memory-mapped). The 100 queries from `dataset/queries_embeddings.json` were used with `k=10`. Five untimed warm-up query pairs ran before measuring, and the ANN and exact queries were interleaved for each embedding.

**Pattern 1: the approximate search is almost exact (precision@10 = 0.9990).** Across 100 queries × 10 results, HNSW returned 999 of the 1,000 true nearest neighbours: a single neighbour was missed in total. This is the first sign of the "approximate" in ANN. At this size, segments hold about 45k points (~280 MB of vectors), far above `full_scan_threshold`, so every query goes through the HNSW graph instead of a brute-force scan. The default parameters (`m=16`, `ef_construct=100`, search-time `ef` of at least 100 for `k=10`) still keep the recall very high. Because the exact top-k is the golden set, precision@10 equals recall@10 here.

**Pattern 2: ANN is about 4.5× faster than exact k-NN (97.63 ms vs 440.81 ms).** Exact search has to score all 411k vectors (~2.5 GB of floats) for every query, i.e. O(N) work, while HNSW only visits a small part of the graph (roughly O(log N)). Unlike the small collection, the saved work is now much larger than the fixed HTTP and serialization overhead, so the trade-off becomes visible.

Both latencies are nevertheless high in absolute terms, and the speed-up is smaller than the algorithmic difference suggests. Vectors and the graph live on disk, and only a fraction of them fits in the ~1 GB page cache. The exact scan is bound by reading gigabytes per query, and the HNSW traversal causes random disk reads (page faults) instead of memory accesses. With the vectors in RAM (`on_disk=False`), both searches would be much faster, but the collection could not have been loaded on this machine.

**Caveats.**
- 100 queries is a small sample, and timings depend heavily on the page cache state, disk speed and background activity, so the averages are only indicative.
- Only IDs were requested (`with_payload=False`, `with_vectors=False`) and only the `query_points` call was timed; network and serialization costs are still included.
- The latencies reflect an on-disk, memory-constrained setup and should not be compared with an in-memory deployment.

**Conclusion.** At about 400k points, the ANN trade-off shows up: HNSW is about 4.5× faster than exact k-NN while losing only 0.1% of the true neighbours. On this machine, storage (disk vs RAM) weighs as much on latency as the choice of algorithm. Lowering `hnsw_ef` should reduce ANN latency further at the cost of precision, which is explored in the next task.
