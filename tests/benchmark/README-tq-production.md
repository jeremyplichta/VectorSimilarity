# TurboQuant production benchmark

`bm_tq_production_fp32` is the native synthetic decision-record benchmark from TurboQuant Story
05. Its smoke mode is deliberately tiny; use `--tq-mode=list` to emit sharded full-matrix
commands and run those sequentially. Do not run the full matrix alongside `cargo`, `make`,
`build.sh`, or another benchmark.

For a bounded production-dimension diagnostic, add `--tq-corpus-size=N`,
`--tq-query-count=N`, and/or `--tq-repetitions=N` to a filtered `--tq-mode=full` invocation.
Those records are labeled `bounded_production_dimension_diagnostic`; only the unmodified full
configuration is labeled `production_gate`.

The separate MS MARCO quality diagnostic is not part of this executable. It must use the prepared,
unchanged inputs at:

```text
/Users/jeremy.plichta/git/redis-turboquant-bench/data/cache/prepared/corpus_raw.npy
/Users/jeremy.plichta/git/redis-turboquant-bench/data/cache/prepared/queries_raw.npy
```

That Python diagnostic compares exact FP32, historical pairwise-polar, the dense paper reference,
and each production candidate on identical vectors and seeds. Preserve every existing change in its
repository; this benchmark neither reads nor modifies that checkout.
