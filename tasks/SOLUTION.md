### Observed values 

```
Scalar quantization (int8), rescore=True, oversampling=2.0:
Average precision@10: 1.0000
Average ANN query time: 5.73 ms
Average exact k-NN query time: 5.68 ms

Scalar quantization (int8), rescore=False, oversampling=2.0:
Average precision@10: 1.0000
Average ANN query time: 5.52 ms
Average exact k-NN query time: 5.47 ms
```

### Reflection

**Setup.** The values above were measured against a local Qdrant Docker container. The `arxiv_papers` collection held 3,580 points: 1536-dimensional `text-embedding-ada-002` embeddings of arXiv titles and abstracts, with cosine distance, default HNSW parameters (`m=16`, `ef_construct=100`, `full_scan_threshold=10000` KB), status `green` and 8 segments. The script first enabled scalar quantization on the collection (`ScalarType.INT8`, `quantile=0.99`, `always_ram=False`, original vectors kept) and waited until the optimizer had built the quantized vectors and the collection was `green` again. It then ran the Task 1 comparison twice, with `QuantizationSearchParams(rescore=True, oversampling=2.0)` and with `rescore=False, oversampling=2.0`. The exact k-NN baseline used `SearchParams(exact=True, quantization=QuantizationSearchParams(ignore=True))`, so it always scored the original float32 vectors and is a true full-precision ground truth. Each run used the 100 test queries with `k=10`, 10 untimed warm-up query pairs, and ANN and exact queries interleaved for each embedding.

**Pattern 1: quantization costs no precision here, with or without rescoring (precision@10 = 1.0000 in both runs).** In theory, int8 quantization maps each float32 component to one of 256 levels, so distances computed on quantized vectors are slightly off and can reorder close neighbours. Rescoring corrects this by re-ranking the oversampled candidates (`2.0 × 10 = 20`) with the original vectors, so `rescore=False` was expected to show a visible drop. It didn't, for three reasons:
- **The quantization error is small compared with the gaps between neighbours.** With `quantile=0.99`, the int8 range is fitted to the bulk of the value distribution of ada-002 embeddings, and the error in a cosine score is typically far below 1%. On this real, diverse set of abstracts, the scores of the top-10 papers for a query are usually separated by more than that error, so the order of the top-10 doesn't change even without rescoring. (In an earlier test on synthetic near-duplicate vectors with tiny score gaps, `rescore=False` did drop to about 0.82, which shows the effect does exist when neighbours are very close.)
- **The collection is small and mostly searched by full scan.** Each segment holds about 450 vectors × 1536 floats × 4 bytes ≈ 2.7 MB, well below `full_scan_threshold=10000` KB. As in Tasks 1 and 2, Qdrant's query planner prefers a brute-force scan of these segments over the HNSW graph. There is no graph approximation on top of the quantization error, and 160 vectors had not yet been indexed (`indexed_vectors_count=3420` of 3,580).
- **Oversampling gives extra safety margin.** Each segment returns 20 candidates for a top-10, so a true neighbour slightly underestimated by the int8 scoring still makes it into the merged result.

Because the exact top-k is the golden set, precision@10 equals recall@10 here: no true neighbour was lost in either configuration.

**Pattern 2: quantized and exact searches take the same time (5.73 vs 5.68 ms with rescoring, 5.52 vs 5.47 ms without).** Quantization is meant to speed up search: int8 vectors are 4× smaller (about 1.5 KB instead of 6 KB per vector), so more of them fit in CPU cache and RAM, and integer distance computations are cheaper. Here, though:
- The search itself takes well under a millisecond for about 3.6k vectors. The latency is dominated by fixed overhead: an HTTP/REST round trip, JSON serialisation of the 1536-float query vector, and the Python client. The time saved by int8 scoring, or spent on rescoring 20 candidates, is hidden behind that overhead.
- The whole collection (about 22 MB of float32 vectors) easily fits in RAM and page cache, so the memory-bandwidth and disk-I/O benefits of quantization, which matter most on large or on-disk collections, can't show.
- `rescore=False` is about 0.2 ms faster than `rescore=True`, which is consistent with skipping the full-precision re-ranking. But the exact baseline measured in the same run is faster by the same amount (5.47 vs 5.68 ms), so the difference is mostly run-to-run noise (cache state, OS scheduling) rather than a measurable effect of rescoring.

**Caveats.**
- 100 queries is a small sample, and timings of a few milliseconds are sensitive to noise, so the averages are only indicative.
- Only IDs were requested (`with_payload=False`, `with_vectors=False`) and only the `query_points` call was timed, but network and serialisation cost are still included.
- Enabling quantization changes the collection permanently (it keeps the extra int8 vectors), and later searches use them by default unless `ignore=True` is passed.
- The results describe this particular small collection and should not be extrapolated to production scale.

**Conclusion.** On this collection, int8 scalar quantization is effectively free in accuracy: both `rescore=True` and `rescore=False` reproduce the exact top-10 perfectly, and there is no measurable speed difference because the search cost is negligible next to the request overhead. The benefits of quantization would show on a much larger collection, where the 4× smaller vectors reduce RAM usage and memory or disk I/O. There, keeping the original vectors on disk while the quantized ones stay in RAM (`always_ram=True`) is the usual setup. On such a collection, `rescore=True` with some oversampling is the safe default to recover any precision lost to quantization, and `rescore=False` trades a little precision for lower latency.