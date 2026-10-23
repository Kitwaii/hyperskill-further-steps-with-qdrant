### Observed values 

```
results = [
    {'hnsw_ef': 10, 'avg_precision': 0.968, 'avg_query_time_ms': 3.4665813750279995},
    {'hnsw_ef': 20, 'avg_precision': 0.988, 'avg_query_time_ms': 3.758014529994398},
    {'hnsw_ef': 50, 'avg_precision': 0.998, 'avg_query_time_ms': 4.539852460002294},
    {'hnsw_ef': 100, 'avg_precision': 1.0, 'avg_query_time_ms': 5.672465160014326},
    {'hnsw_ef': 200, 'avg_precision': 1.0, 'avg_query_time_ms': 7.87662929495582},
]

 hnsw_ef |  avg precision@10 | avg query time
---------+-------------------+---------------
      10 |            0.9680 |        3.47 ms
      20 |            0.9880 |        3.76 ms
      50 |            0.9980 |        4.54 ms
     100 |            1.0000 |        5.67 ms
     200 |            1.0000 |        7.88 ms
```

### Reflection

**Setup.** The values above were measured against a local Qdrant Docker container. The `arxiv_papers` collection held 411,220 points: 1536-dimensional `text-embedding-ada-002` embeddings with cosine distance, default HNSW parameters (`m=16`, `ef_construct=100`, `full_scan_threshold=10000` KB), status `green`, spread over 8 segments of about 50k points each. The 100 queries from `dataset/queries_embeddings.json` were used with `k=10`. The exact k-NN top-10 (`SearchParams(exact=True)`) was computed once per query and used as the ground truth. The approximate search was then run with `SearchParams(hnsw_ef=...)` for `hnsw_ef` in `[10, 20, 50, 100, 200]`. Following the note on cache warming, 10 untimed warm-up queries were run for every `hnsw_ef` value, and each value was then run over the whole dataset 3 times. The first pass was discarded and the other two were averaged, so the order of the values does not bias the timings. Only search-time parameters changed: the index and the stored embeddings were left untouched.

**Pattern 1: precision@10 rises with `hnsw_ef` and saturates at 1.0 from `hnsw_ef=100`.** Precision goes from 0.968 (`hnsw_ef=10`) to 0.988 (20), 0.998 (50) and 1.000 (100 and 200). This is the expected behaviour of HNSW:
- `hnsw_ef` is the size of the candidate list kept during graph traversal. A small list makes the search greedy and it can miss true neighbours. A larger list explores more of the graph and recovers them.
- With `hnsw_ef=10` and `limit=10`, the list is exactly as long as the number of results, so there is no margin and about 3% of the true top-10 neighbours are lost.
- The gains shrink quickly: +2.0 points from 10 to 20, +1.0 from 20 to 50, +0.2 from 50 to 100, and nothing after that. The curve saturates at `hnsw_ef=100`, which is also the default search-time `ef`. This explains why the task-1 comparison with default settings gave a precision of 1.0.
- On this collection, the segments are far larger than `full_scan_threshold` (about 300 MB each against 10,000 KB), so the HNSW graph is really used. The numbers above therefore measure a real ANN accuracy cost, not a hidden full scan.

Because the exact top-k is the golden set, precision@10 equals recall@10 here: the missing 3.2% at `hnsw_ef=10` are true neighbours that the graph search did not reach.

**Pattern 2: the query time grows with `hnsw_ef`, from 3.47 ms to 7.88 ms (about 2.3x).** A larger candidate list means more nodes visited and more distance computations on 1536-dimensional vectors, so each query does more work. The growth is steady (3.47, 3.76, 4.54, 5.67, 7.88 ms) and there is no outlier. Two remarks:
- The increase is sub-linear: `hnsw_ef` is multiplied by 20, but the latency by only 2.3. A fixed part of every query does not depend on `hnsw_ef`: the HTTP/REST round trip, the JSON serialisation of the 1536-float query vector, and the Python client. That overhead is a few milliseconds, so it hides part of the search cost.
- The two largest values cost the most for no gain. Going from `hnsw_ef=100` to 200 adds about 2.2 ms (+39%) while precision stays at 1.0.

**Caveats.**
- 100 queries (200 timed queries per value) is a small sample, and timings of a few milliseconds are sensitive to noise, so the averages are only indicative. A precision difference of 0.002 (as between `hnsw_ef=50` and 100) corresponds to 2 missed neighbours over 1,000 results.
- Only IDs were requested (`with_payload=False`, `with_vectors=False`) and only the `query_points` call was timed, but network and serialisation cost are still included.
- The results come from a single local machine, with the data in the page cache and the graph in RAM, and from a single index built with default parameters. A remote or disk-bound deployment, or a different `m`/`ef_construct`, would give different absolute numbers and a different curve.

**Conclusion.** `hnsw_ef` is a real speed/accuracy knob on this collection: lower values are faster but less precise, higher values are more precise but slower. `hnsw_ef=50` is a good balance: 99.8% precision at 4.54 ms, about 20% faster than `hnsw_ef=100` (5.67 ms) while losing only 0.2% of the true neighbours. `hnsw_ef=100` (the default) is the smallest value that gives perfect precision, at 5.67 ms. `hnsw_ef=200` only costs more time for no gain, and `hnsw_ef=10` is the fastest (3.47 ms) but loses about 3% of the true neighbours, which is only acceptable if latency matters much more than recall. If the application needs perfect recall, keep `hnsw_ef=100`. Otherwise, `hnsw_ef` between 50 and 100 is the sensible range.
