/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>

namespace tq_paper_reference {

inline size_t PackedBytes(size_t count, size_t bits_per_value) {
    return (count * bits_per_value + 7) / 8;
}

// Algorithm 2 stores (b - 1)d centroid-index bits, d QJL sign bits, the source
// scale required for non-unit inputs, and the residual norm.
inline size_t StorageBytes(size_t dim, size_t total_bits) {
    if (dim == 0 || total_bits < 2) {
        throw std::invalid_argument("invalid paper TurboQuant dimensions or bit width");
    }
    return PackedBytes(dim, total_bits - 1) + PackedBytes(dim, 1) + 2 * sizeof(float);
}

inline float QjlScale(size_t projections) {
    if (projections == 0) {
        throw std::invalid_argument("QJL requires at least one projection");
    }
    return std::sqrt(std::acos(-1.0f) / 2.0f) / static_cast<float>(projections);
}

inline float QjlCorrection(float residual_norm, std::span<const float> projected_query,
                           std::span<const int8_t> residual_signs) {
    if (projected_query.size() != residual_signs.size()) {
        throw std::invalid_argument("QJL query and sign dimensions differ");
    }
    float signed_projection_sum = 0.0f;
    for (size_t i = 0; i < projected_query.size(); ++i) {
        signed_projection_sum += projected_query[i] * static_cast<float>(residual_signs[i]);
    }
    return residual_norm * QjlScale(projected_query.size()) * signed_projection_sum;
}

// For a one-bit Lloyd-Max quantizer of the exact sphere-coordinate density,
// symmetry fixes the boundary at zero. The positive centroid is E[X | X > 0].
inline std::pair<float, float> ExactOneBitCentroids(size_t dim) {
    if (dim < 2) {
        throw std::invalid_argument("sphere-coordinate density requires dimension >= 2");
    }
    const double normalizer =
        std::exp(std::lgamma(static_cast<double>(dim) / 2.0) - 0.5 * std::log(std::acos(-1.0)) -
                 std::lgamma(static_cast<double>(dim - 1) / 2.0));
    const float positive = static_cast<float>(2.0 * normalizer / static_cast<double>(dim - 1));
    return {-positive, positive};
}

} // namespace tq_paper_reference
