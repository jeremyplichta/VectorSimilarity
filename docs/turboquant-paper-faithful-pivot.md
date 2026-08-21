# Paper-faithful TurboQuant pivot

Status: implementation design for `codex/tq-paper-faithful-pivot`.

Primary source: Zandieh, Daliri, Hadian, and Mirrokni, “TurboQuant: Online Vector
Quantization with Near-optimal Distortion Rate,” arXiv:2504.19874v1, Algorithms 1 and 2.
The existing pairwise-polar implementation and its Rust oracle are historical references only.

## Scope and invariants

The public compression names remain `TQ2`, `TQ4`, and `TQ8`. A name denotes the total
Algorithm 2 bit budget `b`: `b - 1` Lloyd–Max bits and one QJL sign bit per coordinate. The
paper-faithful codec supports FP32 `COSINE` and `IP`. `L2` remains rejected until it has a
separate estimator and product contract.

The first implementation is scalar and auditable. SIMD, lookup-table scoring, and compressed
construction shortcuts are deferred until scalar conformance, storage, and statistical tests pass.

## Algorithms and equations

Let `d` be the dimension, `b` the advertised total bits per coordinate, `m = d` the number of
Gaussian QJL rows, `Pi` a deterministic random orthogonal matrix, and `S` an `m x d` matrix with
i.i.d. `N(0, 1)` entries. Stored nonzero vectors are reduced to a unit vector `u`; the metric-aware
scale `alpha` is the original norm for `IP` and one for `COSINE`. A zero vector has `alpha = 0`.

Algorithm 1 (`TurboQuant_mse`) uses the coordinate density induced by a random rotation of a unit
vector:

```text
f_d(t) = Gamma(d/2) / (sqrt(pi) Gamma((d-1)/2)) * (1 - t^2)^((d-3)/2), t in [-1, 1]

C(f_d, p) = min_{c_1 <= ... <= c_(2^p)}
              sum_k integral_{(c_(k-1)+c_k)/2}^{(c_k+c_(k+1))/2}
                    (t - c_k)^2 f_d(t) dt
```

For `p = b - 1`, setup computes a shared deterministic Lloyd–Max codebook `c`. Encoding and
decoding are:

```text
z          = Pi u
idx_j      = argmin_k |z_j - c_k|
z_tilde_j  = c_(idx_j)
u_mse      = Pi^T z_tilde
```

Algorithm 2 (`TurboQuant_prod`) adds QJL on the MSE residual:

```text
r      = u - u_mse
gamma  = ||r||_2
r_unit = r / gamma                       if gamma > 0
q_i    = sign(S_i r_unit), i in [1, m]  if gamma > 0
q_i    = +1                              if gamma = 0 (ignored because gamma is zero)

u_tilde = u_mse + gamma * sqrt(pi/2) / m * S^T q
x_tilde = alpha * u_tilde
```

Normalizing `r` before sketching is explicit even though the sign map is scale invariant. It makes
the ownership of residual magnitude unambiguous: `gamma` is the only stored residual magnitude.

For an uncompressed query `y`, define `y_metric = normalize(y)` for `COSINE` (with the zero query
handled explicitly) and `y_metric = y` for `IP`. The asymmetric estimator is evaluated without
materializing `u_tilde`:

```text
y_rot = Pi y_metric
y_jl  = S y_metric

IP_hat(y, x) = alpha * (
    sum_j y_rot_j * c_(idx_j)
    + gamma * sqrt(pi/2) / m * sum_i y_jl_i * q_i
)
```

This is exactly `<y_metric, x_tilde>`. For `COSINE`, `alpha` is one for every nonzero stored vector;
for `IP`, storing `alpha = ||x||_2` extends the unit-sphere algorithm as prescribed by the paper.
Queries are not normalized for `IP`, so non-unit query magnitude is preserved. VecSim distance is
`1 - IP_hat`. The estimator is not clamped, because clamping would introduce bias.

## Shared model state and determinism

Each index owns one model state containing:

- the dimension, total bit budget, and codec version;
- a deterministic `Pi` generated from the persisted seed by Gaussian sampling and QR/orthogonal
  construction;
- the exact-dimension Lloyd–Max centroids and midpoint decision boundaries for `f_d`;
- the deterministic `m x d` Gaussian QJL matrix with `m = d`.

The scalar implementation will use a repository-owned deterministic PRNG and Gaussian sampler,
not `std::normal_distribution`, whose output is not portable. Lloyd–Max initialization, numerical
integration, stopping tolerance, and maximum iteration count are fixed by the codec version.
Codebook tests enforce ordering, symmetry, midpoint boundaries, convergence, and deterministic
golden values.

## Stored and query representations

The fixed stored layout is:

```text
| packed Lloyd–Max indices | packed QJL signs | alpha: FP32 | gamma: FP32 |
```

Both packed regions are LSB-first and independently byte-rounded. Unused tail bits must be zero.
`alpha = 0` is the zero-vector marker. A zero vector scores exactly zero and never consults its
otherwise canonical all-zero index/sign payload. `gamma = 0` skips the residual correction but does
not imply that the source vector is zero.

The query representation contains rotated FP32 coordinates and projected FP32 QJL values. It is
temporary and is not included in stored-vector accounting.

At `d = 1024`, where both packed regions are byte-aligned:

| Mode | Lloyd–Max bits | Lloyd–Max bytes | QJL bytes | Metadata | Total | Actual bits/dim |
|---|---:|---:|---:|---:|---:|---:|
| TQ2 | 1 | 128 | 128 | 8 | 264 | 2.0625 |
| TQ4 | 3 | 384 | 128 | 8 | 520 | 4.0625 |
| TQ8 | 7 | 896 | 128 | 8 | 1,032 | 8.0625 |

The first FP32 metadata word is required by the paper's extension to non-unit inputs and also gives
zero vectors an exact representation. The second is the Algorithm 2 residual norm. Shared model
state, allocator overhead, and HNSW graph links are reported separately from this payload.

## HNSW construction and search

The paper defines asymmetric uncompressed-query-to-compressed-vector scoring. It does **not**
define a compressed-code-to-compressed-code estimator. The pairwise-polar sign-dot estimator must
therefore be removed rather than translated to the new layout.

VecSim already has separate `StoredToQuery` and `StoredToStored` dispatches:

- insertion traversal and ordinary search use the paper estimator above, with the incoming raw
  vector/query as the asymmetric operand;
- HNSW neighbor-diversity heuristics, reciprocal-link updates, and deletion repairs require
  stored-to-stored comparisons.

The correctness-first stored-to-stored implementation decodes each operand with Algorithm 2 and
then applies the exact requested metric to the decoded FP32 vectors. `COSINE` normalizes the two
decoded vectors before comparing them; `IP` preserves their stored `alpha` scales. This is not a
new estimator: it is exact metric evaluation on the paper-defined dequantization. Tests compare the
calculator with an explicit scalar decode and exercise insertion, reciprocal-link replacement,
and deletion repair. Its construction cost is intentionally accepted for the scalar milestone.

No raw-vector sidecar is retained: that would erase the advertised memory saving. Any later direct
compressed construction kernel must be algebraically equivalent to the explicit decode result and
must pass scalar parity before use.

## Persistence and compatibility

The stored bytes have no compatibility with the historical pairwise-polar blob and require a new
codec version. Standalone VecSim TQ-HNSW file serialization remains unsupported until that version
is represented in its serializer.

RediSearch RDB persists vector-field parameters and source Redis documents, then rebuilds the
in-memory index. Encoding version 29 stores paper codec marker 2 explicitly. The historical
version-28 pairwise-polar encoding is rejected: its parameter shape looks compatible, but its
`d/2` projection value and vector-byte contract must never be reinterpreted as paper-faithful
settings. Older non-TQ encodings remain loadable. Round-trip tests compare configuration,
ordering, and scores before and after reload.

Golden fixtures are regenerated from an independent, paper-derived scalar implementation. The
pairwise-polar Rust oracle is removed from conformance authority and retained only if a historical
comparison test still needs it.

## Correctness and optimization sequence

1. Add the independent scalar reference and failing Algorithm 1/2 conformance tests.
2. Implement exact-density Lloyd–Max generation, packed indices/signs, zero handling, and byte
   accounting.
3. Add normalized QJL encoding and statistical unbiasedness tests over dimensions, seeds, signs,
   and residual magnitudes.
4. Add asymmetric `COSINE` and non-unit `IP` scoring tests.
5. Add explicit-decode stored-to-stored scoring and HNSW construction/repair tests.
6. Update VecSim and RediSearch size estimates, RDB versioning, integration tests, and HLD.
7. Establish benchmark/error baselines with the scalar implementation.
8. Only then add packed scalar lookup optimizations and architecture-specific SIMD. Every optimized
   kernel must match scalar results across TQ2/TQ4/TQ8, randomized values, tails, and alignment.

The quantizer diagnostic reports exact FP32, historical pairwise-polar, paper scalar, and the
independent Python implementation on identical vectors and seeds. It reports bias, MAE, RMSE, p95
absolute error, recall@10, payload bytes, actual bits/dimension, encoding throughput, and scoring
throughput. It is not evidence of end-to-end HNSW correctness.
