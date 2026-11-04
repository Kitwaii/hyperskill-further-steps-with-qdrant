### Observed values 

```
Scalar quantization (int8), rescore=True, oversampling=2.0:
Average precision@10: 1.0000
Average ANN query time: 984.23 ms
Average exact k-NN query time: 658.35 ms

Scalar quantization (int8), rescore=False, oversampling=2.0:
Average precision@10: 0.8360
Average ANN query time: 1048.21 ms
Average exact k-NN query time: 619.57 ms
```

### Reflection

**Setup.** The values above were measured against a local Qdrant 1.18.0 Docker container on a 16 GB machine with about 1 GB of free RAM. The `arxiv_papers` collection held 411,120 points: 1536-dimensional `text-embedding-ada-002` embeddings with cosine distance and default HNSW parameters (`m=16`, `ef_construct=100`). It was loaded by the script in batches of 500 points with HNSW indexing deferred until the end of the upload, and vectors, the HNSW graph and payloads are stored on disk (memory-mapped). Scalar quantization was then enabled (`ScalarType.INT8`, `quantile=0.99`, `always_ram=False`, original vectors kept), and the script waited for the collection to be green again: all 411,120 vectors indexed, 9 segments. The task 1 comparison was then run twice, with `rescore=True` and `rescore=False` (`oversampling=2.0`, `k=10`, the 100 queries of `dataset/queries_embeddings.json`, 10 untimed warm-up queries, ANN and exact queries interleaved per embedding). The exact baseline ignored quantization (`ignore=True`), so it scores the original float32 vectors.

**Pattern 1: rescoring keeps the precision, skipping it loses about 16% of the true neighbours.** With `rescore=True`, precision@10 is 1.000: all 1,000 true neighbours (100 queries × 10) were found, as good as the 0.999 of task 1 without quantization. With `rescore=False`, it drops to 0.836, i.e. 164 neighbours lost. Int8 scores are a coarse approximation of the cosine similarity: 1536 values squeezed into 256 levels cannot separate neighbours whose scores are very close, so the quantized ranking reorders them. With `rescore=True`, the `limit × oversampling = 20` candidates found with the quantized vectors are re-scored with the original float32 vectors before keeping the top 10, which repairs this ordering. The quantized stage is good at finding the right candidates, but not at ranking them precisely.

**Pattern 2: on this setup, the quantized search is slower than the exact search (984 ms and 1,048 ms vs 658 ms and 620 ms).** This is the opposite of the expected speed-up, and much slower than the 97.63 ms ANN of task 1 without quantization. Two observations explain it:
- `rescore=False` is not faster than `rescore=True` (1,048 vs 984 ms). Re-scoring 20 candidates is negligible, so the cost is in the HNSW traversal over the quantized vectors itself.
- With `always_ram=False`, the quantized vectors (~630 MB) are memory-mapped like everything else. The approximate search now does random reads in three on-disk structures (the graph, the quantized vectors, and the original vectors for rescoring), while only about 1 GB of page cache is available. In addition, the interleaved exact query scans the 2.5 GB of original vectors after every approximate query and evicts these pages again. Each approximate query therefore pays many page faults (random disk reads), while the exact scan reads the vectors sequentially, which disks handle much better. The exact search itself is also slower than in task 1 (440.81 ms), consistent with the extra pressure on the page cache.

**Caveats.**
- 100 queries is a small sample, and timings depend heavily on the page cache state, disk speed and background activity, so the averages are only indicative.
- The interleaving of ANN and exact queries, required to compare them on the same cache state, is harsh for a memory-constrained setup: in a real application, approximate queries would not be followed by full scans, and their working set would stay in cache.
- The results describe a single on-disk setup with `always_ram=False`; they do not reflect the speed-up quantization gives when the quantized vectors are in RAM. Quantization also changes the collection permanently: later searches use it unless `ignore=True` is passed.

**Conclusion.** Quantization without rescoring is not acceptable here: it loses 16% of the true neighbours and is not even faster. With `rescore=True` and `oversampling=2.0`, int8 quantization keeps a perfect precision, so the accuracy side of the trade-off is solved. The speed side is not: with `always_ram=False` on a machine where nothing fits in RAM, quantization adds disk reads instead of saving them. The production pattern described in the task, `always_ram=True` with the original vectors on disk, is the one that fits this machine: the ~630 MB of int8 vectors would stay in RAM, the graph traversal would no longer hit the disk, and only the 20 rescored candidates per query would be read from the original vectors.
