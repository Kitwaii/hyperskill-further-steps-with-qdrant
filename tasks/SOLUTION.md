### Observed values 

```
Scalar quantization (int8), rescore=True, oversampling=2.0:
Average precision@10: 0.9990
Average ANN query time: 5.21 ms
Average exact k-NN query time: 71.24 ms

Scalar quantization (int8), rescore=False, oversampling=2.0:
Average precision@10: 0.8300
Average ANN query time: 5.18 ms
Average exact k-NN query time: 70.47 ms
```

### Reflection

**Setup.** Local Qdrant, collection `arxiv_papers` with 411,220 points (1536-dim, cosine, default HNSW, 8 segments of about 50k points each). Scalar quantization was enabled (`ScalarType.INT8`, `quantile=0.99`, `always_ram=False`, original vectors kept) and the script waited for the collection to be `green`. The task 1 comparison was then run with `rescore=True` and `rescore=False` (`oversampling=2.0`, `k=10`, 100 queries). The exact baseline ignored quantization (`ignore=True`), so it scores the original float32 vectors.

**Pattern 1: rescoring keeps precision, skipping it costs 17 points.** With `rescore=True`, precision is 0.999. With `rescore=False`, it drops to 0.83. Int8 scores are approximate and, combined with the HNSW approximation, they reorder close neighbours. With `rescore=True`, the oversampled candidates (`2.0 × 10 = 20`) are re-ranked with the original vectors, which recovers almost everything. Segments are far above `full_scan_threshold`, so HNSW is really used.

**Pattern 2: both quantized runs have the same latency (5.21 vs 5.18 ms), about 13.7x faster than exact search (about 71 ms).** Re-ranking 20 candidates costs almost nothing, so the difference between the two runs is noise. The exact baseline has to score all 411k float32 vectors (about 2.5 GB), which explains the 71 ms. Compared with task 1 (6.32 ms without quantization), the gain from quantization itself is small (about 1 ms): the original vectors are kept and the data fits in the OS page cache, so the RAM and disk benefits of quantization are not visible here.

**Caveats.** 100 queries is a small sample and timings of a few ms are noisy. One local machine, with `always_ram=False` and the data in the page cache. Quantization changes the collection permanently, and later searches use it unless `ignore=True` is passed.

**Conclusion.** Use `rescore=True` with oversampling: 99.9% precision at about the same speed. `rescore=False` saves almost no time here and loses 17% of the true neighbours, so it is not worth it on this setup. The memory savings of quantization would matter on a larger collection, or with the quantized vectors in RAM (`always_ram=True`) and the originals on disk.
