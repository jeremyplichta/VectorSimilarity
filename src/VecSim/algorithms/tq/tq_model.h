/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#endif

namespace TQFlatDetails {

inline constexpr uint64_t kQjlSeedOffset = 0xCAFEBABE00000001ULL;
inline constexpr size_t kLloydMaxIntegrationPoints = 1u << 17;
inline constexpr size_t kLloydMaxMaxIterations = 256;

inline bool IsSupportedTotalBits(size_t total_bits) {
    return total_bits == 2 || total_bits == 4 || total_bits == 8;
}

inline size_t MseBits(size_t total_bits) {
    if (!IsSupportedTotalBits(total_bits)) {
        throw std::invalid_argument("TurboQuant total bits must be 2, 4, or 8");
    }
    return total_bits - 1;
}

inline size_t PackedBytes(size_t count, size_t bits_per_value) {
    return (count * bits_per_value + 7) / 8;
}

inline float QjlScale(size_t projections) {
    if (projections == 0) {
        throw std::invalid_argument("TurboQuant requires at least one QJL projection");
    }
    return std::sqrt(std::acos(-1.0f) / 2.0f) / static_cast<float>(projections);
}

inline float DotProductScalar(const float *lhs, const float *rhs, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        sum += lhs[i] * rhs[i];
    }
    return sum;
}

inline float SumSquaresScalar(const float *values, size_t dim) {
    return DotProductScalar(values, values, dim);
}

inline constexpr bool HasPaperTqSimd() {
#if defined(__aarch64__) || defined(__SSE2__)
    return true;
#else
    return false;
#endif
}

// SplitMix64 plus an explicit Box-Muller transform makes model generation independent of the
// standard library's unspecified normal_distribution algorithm.
class DeterministicGaussianRng {
public:
    explicit DeterministicGaussianRng(uint64_t seed) : state(seed), spare(0.0), has_spare(false) {}

    float normal() {
        if (has_spare) {
            has_spare = false;
            return static_cast<float>(spare);
        }

        const double u1 = uniformOpen();
        const double u2 = uniformOpen();
        const double radius = std::sqrt(-2.0 * std::log(u1));
        const double theta = 2.0 * std::acos(-1.0) * u2;
        spare = radius * std::sin(theta);
        has_spare = true;
        return static_cast<float>(radius * std::cos(theta));
    }

private:
    uint64_t next() {
        uint64_t value = (state += 0x9E3779B97F4A7C15ULL);
        value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
        value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
        return value ^ (value >> 31);
    }

    double uniformOpen() {
        constexpr double kInv53 = 1.0 / static_cast<double>(uint64_t{1} << 53);
        return (static_cast<double>(next() >> 11) + 0.5) * kInv53;
    }

    uint64_t state;
    double spare;
    bool has_spare;
};

struct LloydMaxCodebook {
    std::vector<float> centroids;
    std::vector<float> boundaries;
};

// Solve Equation 4 against the exact coordinate density f_d. Midpoint quadrature avoids the
// integrable endpoint singularity at d=2 and fixed grid/iteration parameters version the result.
inline LloydMaxCodebook BuildLloydMaxCodebook(size_t dim, size_t mse_bits) {
    if (dim < 2) {
        throw std::invalid_argument("TurboQuant requires dimension >= 2");
    }
    if (mse_bits == 0 || mse_bits > 7) {
        throw std::invalid_argument("TurboQuant MSE bits must be between 1 and 7");
    }

    const size_t levels = size_t{1} << mse_bits;
    const double step = 2.0 / static_cast<double>(kLloydMaxIntegrationPoints);
    const double exponent = (static_cast<double>(dim) - 3.0) / 2.0;

    std::vector<double> coordinates(kLloydMaxIntegrationPoints);
    std::vector<double> mass_prefix(kLloydMaxIntegrationPoints + 1, 0.0);
    std::vector<double> moment_prefix(kLloydMaxIntegrationPoints + 1, 0.0);
    for (size_t i = 0; i < kLloydMaxIntegrationPoints; ++i) {
        const double x = -1.0 + (static_cast<double>(i) + 0.5) * step;
        const double log_base = std::log1p(-(x * x));
        const double weight = std::exp(exponent * log_base);
        coordinates[i] = x;
        mass_prefix[i + 1] = mass_prefix[i] + weight;
        moment_prefix[i + 1] = moment_prefix[i] + x * weight;
    }

    const double total_mass = mass_prefix.back();
    std::vector<double> centroids(levels);
    for (size_t level = 0; level < levels; ++level) {
        const double target =
            (static_cast<double>(level) + 0.5) * total_mass / static_cast<double>(levels);
        const auto it = std::lower_bound(mass_prefix.begin(), mass_prefix.end(), target);
        const size_t index =
            std::clamp<size_t>(static_cast<size_t>(std::distance(mass_prefix.begin(), it)), 1,
                               kLloydMaxIntegrationPoints);
        centroids[level] = coordinates[index - 1];
    }

    std::vector<double> boundaries(levels - 1);
    std::vector<double> updated(levels);
    const double tolerance = 1e-8 / std::sqrt(static_cast<double>(dim));
    for (size_t iteration = 0; iteration < kLloydMaxMaxIterations; ++iteration) {
        for (size_t i = 0; i + 1 < levels; ++i) {
            boundaries[i] = (centroids[i] + centroids[i + 1]) / 2.0;
        }

        size_t lower_index = 0;
        double max_delta = 0.0;
        for (size_t level = 0; level < levels; ++level) {
            const size_t upper_index =
                level + 1 == levels
                    ? kLloydMaxIntegrationPoints
                    : static_cast<size_t>(std::lower_bound(coordinates.begin(), coordinates.end(),
                                                           boundaries[level]) -
                                          coordinates.begin());
            const double mass = mass_prefix[upper_index] - mass_prefix[lower_index];
            const double moment = moment_prefix[upper_index] - moment_prefix[lower_index];
            updated[level] =
                mass > std::numeric_limits<double>::epsilon() ? moment / mass : centroids[level];
            lower_index = upper_index;
        }

        // Equation 4's density is symmetric. Enforcing that invariant removes numerical drift.
        for (size_t i = 0; i < levels / 2; ++i) {
            const size_t mirror = levels - 1 - i;
            const double magnitude = (std::abs(updated[i]) + std::abs(updated[mirror])) / 2.0;
            updated[i] = -magnitude;
            updated[mirror] = magnitude;
        }

        for (size_t i = 0; i < levels; ++i) {
            max_delta = std::max(max_delta, std::abs(updated[i] - centroids[i]));
        }
        centroids.swap(updated);
        if (max_delta <= tolerance) {
            break;
        }
    }

    LloydMaxCodebook result;
    result.centroids.reserve(levels);
    result.boundaries.reserve(levels - 1);
    for (double centroid : centroids) {
        result.centroids.push_back(static_cast<float>(centroid));
    }
    for (size_t i = 0; i + 1 < levels; ++i) {
        result.boundaries.push_back(static_cast<float>((centroids[i] + centroids[i + 1]) / 2.0));
    }
    return result;
}

struct StorageView {
    const uint8_t *mse_indices;
    const uint8_t *residual_signs;
    float source_scale;
    float residual_norm;
};

struct QueryView {
    const float *rotated;
    const float *qjl_projection;
    float norm_sq;
};

class TQModelState {
public:
    TQModelState(size_t dim, size_t total_bits, size_t projections, size_t seed, bool use_rotation)
        : dim(dim), total_bits(total_bits), mse_bits(MseBits(total_bits)), projections(projections),
          seed(seed), use_rotation(use_rotation), levels(size_t{1} << mse_bits),
          qjl_scale(QjlScale(projections)), packed_index_bytes(PackedBytes(dim, mse_bits)),
          packed_qjl_bytes(PackedBytes(dim, 1)), rotation_columns(dim * dim, 0.0f),
          rotation_rows(dim * dim, 0.0f), qjl_projection_rows(projections * dim, 0.0f) {
        if (dim < 2) {
            throw std::invalid_argument("TurboQuant requires dimension >= 2");
        }
        if (projections != dim) {
            throw std::invalid_argument("Paper-faithful TurboQuant requires projections == dim");
        }

        const auto codebook = BuildLloydMaxCodebook(dim, mse_bits);
        centroids = codebook.centroids;
        boundaries = codebook.boundaries;
        initializeRotation();
        initializeQjlProjectionRows();
    }

    size_t storageBlobSize() const {
        return packed_index_bytes + packed_qjl_bytes + 2 * sizeof(float);
    }

    size_t queryBlobSize() const { return (dim + projections + 1) * sizeof(float); }
    size_t packedIndexBytes() const { return packed_index_bytes; }
    size_t packedQjlBytes() const { return packed_qjl_bytes; }

    StorageView storageView(const void *blob) const {
        const auto *bytes = static_cast<const uint8_t *>(blob);
        const auto *indices = bytes;
        bytes += packedIndexBytes();
        const auto *signs = bytes;
        bytes += packedQjlBytes();
        float source_scale;
        float residual_norm;
        std::memcpy(&source_scale, bytes, sizeof(float));
        std::memcpy(&residual_norm, bytes + sizeof(float), sizeof(float));
        return {.mse_indices = indices,
                .residual_signs = signs,
                .source_scale = source_scale,
                .residual_norm = residual_norm};
    }

    QueryView queryView(const void *blob) const {
        const auto *words = static_cast<const float *>(blob);
        const auto *rotated = words;
        words += dim;
        const auto *qjl_projection = words;
        words += projections;
        return {.rotated = rotated, .qjl_projection = qjl_projection, .norm_sq = *words};
    }

    void writeMetadata(void *blob, float source_scale, float residual_norm) const {
        auto *bytes = static_cast<uint8_t *>(blob) + packedIndexBytes() + packedQjlBytes();
        std::memcpy(bytes, &source_scale, sizeof(float));
        std::memcpy(bytes + sizeof(float), &residual_norm, sizeof(float));
    }

    void applyRotation(const float *input, float *output) const {
        if (!use_rotation) {
            std::memcpy(output, input, dim * sizeof(float));
            return;
        }
        for (size_t row = 0; row < dim; ++row) {
            output[row] = DotProductScalar(rotation_rows.data() + row * dim, input, dim);
        }
    }

    void applyInverseRotation(const float *input, float *output) const {
        if (!use_rotation) {
            std::memcpy(output, input, dim * sizeof(float));
            return;
        }
        for (size_t col = 0; col < dim; ++col) {
            output[col] = DotProductScalar(rotation_columns.data() + col * dim, input, dim);
        }
    }

    void projectQjl(const float *input, float *output) const {
        for (size_t row = 0; row < projections; ++row) {
            output[row] = DotProductScalar(qjl_projection_rows.data() + row * dim, input, dim);
        }
    }

    uint16_t quantizeCoordinate(float value) const {
        return static_cast<uint16_t>(std::upper_bound(boundaries.begin(), boundaries.end(), value) -
                                     boundaries.begin());
    }

    void writeMseIndex(uint8_t *packed, size_t coordinate, uint16_t value) const {
        const size_t first_bit = coordinate * mse_bits;
        for (size_t bit = 0; bit < mse_bits; ++bit) {
            if ((value & (uint16_t{1} << bit)) != 0) {
                const size_t target = first_bit + bit;
                packed[target / 8] |= static_cast<uint8_t>(1u << (target % 8));
            }
        }
    }

    uint16_t mseIndexAt(const StorageView &storage, size_t coordinate) const {
        const size_t first_bit = coordinate * mse_bits;
        uint16_t value = 0;
        for (size_t bit = 0; bit < mse_bits; ++bit) {
            const size_t source = first_bit + bit;
            if ((storage.mse_indices[source / 8] & static_cast<uint8_t>(1u << (source % 8))) != 0) {
                value |= static_cast<uint16_t>(uint16_t{1} << bit);
            }
        }
        return value;
    }

    bool residualSignAt(const StorageView &storage, size_t projection) const {
        return (storage.residual_signs[projection / 8] &
                static_cast<uint8_t>(1u << (projection % 8))) != 0;
    }

    void encodeMse(const float *unit_vector, uint8_t *packed_indices,
                   float *reconstructed_unit) const {
        std::memset(packed_indices, 0, packedIndexBytes());
        std::vector<float> rotated(dim);
        std::vector<float> reconstructed_rotated(dim);
        applyRotation(unit_vector, rotated.data());
        for (size_t coordinate = 0; coordinate < dim; ++coordinate) {
            const uint16_t index = quantizeCoordinate(rotated[coordinate]);
            writeMseIndex(packed_indices, coordinate, index);
            reconstructed_rotated[coordinate] = centroids[index];
        }
        applyInverseRotation(reconstructed_rotated.data(), reconstructed_unit);
    }

    void packResidualSigns(const float *unit_residual, uint8_t *packed_signs) const {
        std::memset(packed_signs, 0, packedQjlBytes());
        std::vector<float> projected(projections);
        projectQjl(unit_residual, projected.data());
        for (size_t i = 0; i < projections; ++i) {
            if (projected[i] >= 0.0f) {
                packed_signs[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
            }
        }
    }

    float estimateInnerProductScalar(const StorageView &storage, const QueryView &query) const {
        if (storage.source_scale == 0.0f) {
            return 0.0f;
        }

        float coarse = 0.0f;
        for (size_t coordinate = 0; coordinate < dim; ++coordinate) {
            coarse += query.rotated[coordinate] * centroids[mseIndexAt(storage, coordinate)];
        }

        float residual = 0.0f;
        if (storage.residual_norm != 0.0f) {
            for (size_t projection = 0; projection < projections; ++projection) {
                const float sign = residualSignAt(storage, projection) ? 1.0f : -1.0f;
                residual += query.qjl_projection[projection] * sign;
            }
            residual *= storage.residual_norm * qjl_scale;
        }
        return storage.source_scale * (coarse + residual);
    }

    float estimateInnerProduct(const StorageView &storage, const QueryView &query) const {
        if (storage.source_scale == 0.0f) {
            return 0.0f;
        }

        const float coarse = packedCentroidDotSimd(storage, query.rotated);
        float residual = 0.0f;
        if (storage.residual_norm != 0.0f) {
            residual = packedSignDotSimd(storage, query.qjl_projection) * storage.residual_norm *
                       qjl_scale;
        }
        return storage.source_scale * (coarse + residual);
    }

    // Decode exactly as Algorithm 2. This is intentionally scalar and is used only for the
    // correctness-first stored-to-stored HNSW path.
    void decode(const StorageView &storage, float *output) const {
        if (storage.source_scale == 0.0f) {
            std::fill(output, output + dim, 0.0f);
            return;
        }

        std::vector<float> reconstructed_rotated(dim);
        for (size_t coordinate = 0; coordinate < dim; ++coordinate) {
            reconstructed_rotated[coordinate] = centroids[mseIndexAt(storage, coordinate)];
        }
        applyInverseRotation(reconstructed_rotated.data(), output);

        if (storage.residual_norm != 0.0f) {
            for (size_t coordinate = 0; coordinate < dim; ++coordinate) {
                float projected_sign_sum = 0.0f;
                for (size_t projection = 0; projection < projections; ++projection) {
                    const float sign = residualSignAt(storage, projection) ? 1.0f : -1.0f;
                    projected_sign_sum += qjl_projection_rows[projection * dim + coordinate] * sign;
                }
                output[coordinate] += storage.residual_norm * qjl_scale * projected_sign_sum;
            }
        }

        for (size_t coordinate = 0; coordinate < dim; ++coordinate) {
            output[coordinate] *= storage.source_scale;
        }
    }

    size_t dim;
    size_t total_bits;
    size_t mse_bits;
    size_t projections;
    size_t seed;
    bool use_rotation;
    size_t levels;
    float qjl_scale;
    std::vector<float> centroids;
    std::vector<float> boundaries;

private:
    float packedCentroidDotSimd(const StorageView &storage, const float *query) const {
#if defined(__aarch64__)
        float32x4_t accumulator = vdupq_n_f32(0.0f);
        size_t coordinate = 0;
        alignas(16) float centroid_lanes[4];
        for (; coordinate + 4 <= dim; coordinate += 4) {
            for (size_t lane = 0; lane < 4; ++lane) {
                centroid_lanes[lane] = centroids[mseIndexAt(storage, coordinate + lane)];
            }
            accumulator = vaddq_f32(
                accumulator, vmulq_f32(vld1q_f32(query + coordinate), vld1q_f32(centroid_lanes)));
        }
        float sum = vaddvq_f32(accumulator);
#elif defined(__SSE2__)
        __m128 accumulator = _mm_setzero_ps();
        size_t coordinate = 0;
        alignas(16) float centroid_lanes[4];
        for (; coordinate + 4 <= dim; coordinate += 4) {
            for (size_t lane = 0; lane < 4; ++lane) {
                centroid_lanes[lane] = centroids[mseIndexAt(storage, coordinate + lane)];
            }
            accumulator = _mm_add_ps(accumulator, _mm_mul_ps(_mm_loadu_ps(query + coordinate),
                                                             _mm_load_ps(centroid_lanes)));
        }
        alignas(16) float partial[4];
        _mm_store_ps(partial, accumulator);
        float sum = partial[0] + partial[1] + partial[2] + partial[3];
#else
        size_t coordinate = 0;
        float sum = 0.0f;
#endif
        for (; coordinate < dim; ++coordinate) {
            sum += query[coordinate] * centroids[mseIndexAt(storage, coordinate)];
        }
        return sum;
    }

    float packedSignDotSimd(const StorageView &storage, const float *query) const {
#if defined(__aarch64__)
        float32x4_t accumulator = vdupq_n_f32(0.0f);
        size_t projection = 0;
        alignas(16) float sign_lanes[4];
        for (; projection + 4 <= projections; projection += 4) {
            for (size_t lane = 0; lane < 4; ++lane) {
                sign_lanes[lane] = residualSignAt(storage, projection + lane) ? 1.0f : -1.0f;
            }
            accumulator = vaddq_f32(
                accumulator, vmulq_f32(vld1q_f32(query + projection), vld1q_f32(sign_lanes)));
        }
        float sum = vaddvq_f32(accumulator);
#elif defined(__SSE2__)
        __m128 accumulator = _mm_setzero_ps();
        size_t projection = 0;
        alignas(16) float sign_lanes[4];
        for (; projection + 4 <= projections; projection += 4) {
            for (size_t lane = 0; lane < 4; ++lane) {
                sign_lanes[lane] = residualSignAt(storage, projection + lane) ? 1.0f : -1.0f;
            }
            accumulator = _mm_add_ps(
                accumulator, _mm_mul_ps(_mm_loadu_ps(query + projection), _mm_load_ps(sign_lanes)));
        }
        alignas(16) float partial[4];
        _mm_store_ps(partial, accumulator);
        float sum = partial[0] + partial[1] + partial[2] + partial[3];
#else
        size_t projection = 0;
        float sum = 0.0f;
#endif
        for (; projection < projections; ++projection) {
            const float sign = residualSignAt(storage, projection) ? 1.0f : -1.0f;
            sum += query[projection] * sign;
        }
        return sum;
    }

    void initializeRotation() {
        if (!use_rotation) {
            for (size_t i = 0; i < dim; ++i) {
                rotation_columns[i * dim + i] = 1.0f;
                rotation_rows[i * dim + i] = 1.0f;
            }
            return;
        }

        DeterministicGaussianRng rng(seed);
        std::vector<float> candidate(dim);
        for (size_t col = 0; col < dim; ++col) {
            bool accepted = false;
            for (size_t attempt = 0; attempt < 16 && !accepted; ++attempt) {
                for (float &value : candidate) {
                    value = rng.normal();
                }

                // Reorthogonalize to keep the dense scalar QR stable at larger dimensions.
                for (size_t pass = 0; pass < 2; ++pass) {
                    for (size_t previous = 0; previous < col; ++previous) {
                        const float *previous_column = rotation_columns.data() + previous * dim;
                        const float projection =
                            DotProductScalar(candidate.data(), previous_column, dim);
                        for (size_t row = 0; row < dim; ++row) {
                            candidate[row] -= projection * previous_column[row];
                        }
                    }
                }

                const float norm_sq = SumSquaresScalar(candidate.data(), dim);
                if (norm_sq > 1e-12f) {
                    const float inverse_norm = 1.0f / std::sqrt(norm_sq);
                    float *column = rotation_columns.data() + col * dim;
                    for (size_t row = 0; row < dim; ++row) {
                        column[row] = candidate[row] * inverse_norm;
                    }
                    accepted = true;
                }
            }
            if (!accepted) {
                throw std::runtime_error("Failed to construct TurboQuant rotation");
            }
        }

        for (size_t row = 0; row < dim; ++row) {
            for (size_t col = 0; col < dim; ++col) {
                rotation_rows[row * dim + col] = rotation_columns[col * dim + row];
            }
        }
    }

    void initializeQjlProjectionRows() {
        DeterministicGaussianRng rng(seed + kQjlSeedOffset);
        for (float &value : qjl_projection_rows) {
            value = rng.normal();
        }
    }

    size_t packed_index_bytes;
    size_t packed_qjl_bytes;
    std::vector<float> rotation_columns;
    std::vector<float> rotation_rows;
    std::vector<float> qjl_projection_rows;
};

} // namespace TQFlatDetails
