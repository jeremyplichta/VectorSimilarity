/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

#include "VecSim/algorithms/tq/tq_model.h"

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace TQFlatDetails {

// This internal version identifies the meaning of the stored-distance choices below. It is
// deliberately separate from the public payload/model version: changing HNSW construction
// geometry does not change stored bytes or query scores.
inline constexpr uint8_t kTQStoredDistanceModeVersion = 1;

enum class TQStoredDistanceMode : uint8_t {
    FullDecodeReference = 1,
    CoarseMse = 2,
};

struct TQCoarseMseAccumulators {
    float dot;
    float lhs_norm_sq;
    float rhs_norm_sq;
};

#ifdef BUILD_TESTS
inline std::atomic_size_t coarse_mse_stored_distance_calls{0};
inline std::atomic_size_t zero_coarse_norm_events{0};

inline void ResetCoarseMseDiagnostics() {
    coarse_mse_stored_distance_calls.store(0);
    zero_coarse_norm_events.store(0);
}

inline size_t GetCoarseMseStoredDistanceCalls() { return coarse_mse_stored_distance_calls.load(); }

inline size_t GetZeroCoarseNormEvents() { return zero_coarse_norm_events.load(); }
#endif

// Accumulate the exact inner product and norms of the two Algorithm 1 reconstructions in rotated
// coordinates. Orthogonality makes inverse rotation unnecessary. QJL signs and gamma are not part
// of Algorithm 1 and are therefore intentionally never read here.
inline TQCoarseMseAccumulators AccumulateCoarseMse(const TQModelState &state,
                                                   const StorageView &lhs, const StorageView &rhs) {
#ifdef BUILD_TESTS
    coarse_mse_stored_distance_calls.fetch_add(1);
#endif
    float dot = 0.0f;
    float lhs_norm_sq = 0.0f;
    float rhs_norm_sq = 0.0f;
    for (size_t coordinate = 0; coordinate < state.dim; ++coordinate) {
        const float lhs_centroid = state.centroids[state.mseIndexAt(lhs, coordinate)];
        const float rhs_centroid = state.centroids[state.mseIndexAt(rhs, coordinate)];
        dot += lhs_centroid * rhs_centroid;
        lhs_norm_sq += lhs_centroid * lhs_centroid;
        rhs_norm_sq += rhs_centroid * rhs_centroid;
    }
    return {.dot = dot, .lhs_norm_sq = lhs_norm_sq, .rhs_norm_sq = rhs_norm_sq};
}

inline void RecordZeroCoarseNorm() {
#ifdef BUILD_TESTS
    zero_coarse_norm_events.fetch_add(1);
#endif
}

} // namespace TQFlatDetails
