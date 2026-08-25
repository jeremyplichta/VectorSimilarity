/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

#include "VecSim/algorithms/tq/tq_fast_structured_rotation.h"
#include "VecSim/memory/vecsim_malloc.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>
#include <variant>
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

enum class TQPayloadLayoutVersion : uint8_t { PaperV1 = 1 };
enum class TQCodecVersion : uint8_t { PaperEstimatorV1 = 1 };
enum class TQModelTransformVersion : uint8_t {
    DenseReferenceV1 = 1,
    FastStructuredRotationV1 = 2,
    FastStructuredV1 = 3,
};
enum class TQRotationBackendVersion : uint8_t {
    DenseHaarV1 = 1,
    FastStructuredV1 = 2,
};
enum class TQQjlBackendVersion : uint8_t {
    DenseGaussianV1 = 1,
    CirculantGaussianV1 = 2,
};
enum class TQMetricContract : uint8_t { CosineOrInnerProductV1 = 1 };

inline size_t CheckedAdd(size_t lhs, size_t rhs, const char *what) {
    if (rhs > std::numeric_limits<size_t>::max() - lhs) {
        throw std::overflow_error(what);
    }
    return lhs + rhs;
}

inline size_t CheckedMultiply(size_t lhs, size_t rhs, const char *what) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        throw std::overflow_error(what);
    }
    return lhs * rhs;
}

inline size_t CheckedBytes(size_t count, size_t element_size, const char *what) {
    return CheckedMultiply(count, element_size, what);
}

inline size_t CheckedRoundUpCapacity(size_t capacity, size_t block_size) {
    const size_t effective_block_size = block_size ? block_size : DEFAULT_BLOCK_SIZE;
    const size_t remainder = capacity % effective_block_size;
    return remainder == 0 ? capacity
                          : CheckedAdd(capacity, effective_block_size - remainder,
                                       "TurboQuant initial-capacity rounding overflow");
}

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
    const size_t bit_count =
        CheckedMultiply(count, bits_per_value, "TurboQuant packed-bit count overflow");
    return CheckedAdd(bit_count, size_t{7}, "TurboQuant packed-byte rounding overflow") / 8;
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

struct TQCodecConfig {
    size_t dimension;
    size_t total_bits;
    size_t mse_bits;
    size_t projections;
    TQMetricContract metric_contract;
    TQCodecVersion codec_version;
    TQPayloadLayoutVersion payload_layout_version;
    TQModelTransformVersion model_transform_version;
    TQRotationBackendVersion rotation_backend_version;
    TQQjlBackendVersion qjl_backend_version;
    uint64_t rotation_seed;
    uint64_t qjl_seed;
    bool use_rotation;

    static TQCodecConfig DenseReference(size_t dimension, size_t total_bits, size_t projections,
                                        uint64_t seed, bool use_rotation = true) {
        return {.dimension = dimension,
                .total_bits = total_bits,
                .mse_bits = MseBits(total_bits),
                .projections = projections,
                .metric_contract = TQMetricContract::CosineOrInnerProductV1,
                .codec_version = TQCodecVersion::PaperEstimatorV1,
                .payload_layout_version = TQPayloadLayoutVersion::PaperV1,
                .model_transform_version = TQModelTransformVersion::DenseReferenceV1,
                .rotation_backend_version = TQRotationBackendVersion::DenseHaarV1,
                .qjl_backend_version = TQQjlBackendVersion::DenseGaussianV1,
                .rotation_seed = seed,
                .qjl_seed = seed + kQjlSeedOffset,
                .use_rotation = use_rotation};
    }

    static TQCodecConfig FastStructuredRotation(size_t dimension, size_t total_bits,
                                                size_t projections, uint64_t seed) {
        return {.dimension = dimension,
                .total_bits = total_bits,
                .mse_bits = MseBits(total_bits),
                .projections = projections,
                .metric_contract = TQMetricContract::CosineOrInnerProductV1,
                .codec_version = TQCodecVersion::PaperEstimatorV1,
                .payload_layout_version = TQPayloadLayoutVersion::PaperV1,
                .model_transform_version = TQModelTransformVersion::FastStructuredRotationV1,
                .rotation_backend_version = TQRotationBackendVersion::FastStructuredV1,
                .qjl_backend_version = TQQjlBackendVersion::DenseGaussianV1,
                .rotation_seed = seed,
                .qjl_seed = seed + kQjlSeedOffset,
                .use_rotation = true};
    }
};

struct TQModelIdentity {
    size_t dimension;
    size_t total_bits;
    size_t projections;
    TQMetricContract metric_contract;
    TQCodecVersion codec_version;
    TQPayloadLayoutVersion payload_layout_version;
    TQModelTransformVersion model_transform_version;
    TQRotationBackendVersion rotation_backend_version;
    TQQjlBackendVersion qjl_backend_version;
    uint64_t rotation_seed;
    uint64_t qjl_seed;
    bool use_rotation;

    bool operator==(const TQModelIdentity &other) const {
        return dimension == other.dimension && total_bits == other.total_bits &&
               projections == other.projections && metric_contract == other.metric_contract &&
               codec_version == other.codec_version &&
               payload_layout_version == other.payload_layout_version &&
               model_transform_version == other.model_transform_version &&
               rotation_backend_version == other.rotation_backend_version &&
               qjl_backend_version == other.qjl_backend_version &&
               rotation_seed == other.rotation_seed && qjl_seed == other.qjl_seed &&
               use_rotation == other.use_rotation;
    }
};

inline void ValidateTQCodecConfig(const TQCodecConfig &config) {
    if (config.dimension < 2) {
        throw std::invalid_argument("TurboQuant requires dimension >= 2");
    }
    if (!IsSupportedTotalBits(config.total_bits) || config.mse_bits != MseBits(config.total_bits)) {
        throw std::invalid_argument("TurboQuant total bits must be 2, 4, or 8");
    }
    if (config.projections != config.dimension) {
        throw std::invalid_argument("Paper-faithful TurboQuant requires projections == dim");
    }
    switch (config.metric_contract) {
    case TQMetricContract::CosineOrInnerProductV1:
        break;
    default:
        throw std::invalid_argument("Unsupported TurboQuant metric contract");
    }
    switch (config.codec_version) {
    case TQCodecVersion::PaperEstimatorV1:
        break;
    default:
        throw std::invalid_argument("Unsupported TurboQuant codec version");
    }
    switch (config.payload_layout_version) {
    case TQPayloadLayoutVersion::PaperV1:
        break;
    default:
        throw std::invalid_argument("Unsupported TurboQuant payload layout version");
    }
    switch (config.model_transform_version) {
    case TQModelTransformVersion::DenseReferenceV1:
        if (config.rotation_backend_version != TQRotationBackendVersion::DenseHaarV1 ||
            config.qjl_backend_version != TQQjlBackendVersion::DenseGaussianV1) {
            throw std::invalid_argument("DenseReferenceV1 requires dense rotation and dense QJL");
        }
        break;
    case TQModelTransformVersion::FastStructuredRotationV1:
        if (config.rotation_backend_version != TQRotationBackendVersion::FastStructuredV1 ||
            config.qjl_backend_version != TQQjlBackendVersion::DenseGaussianV1 ||
            !config.use_rotation) {
            throw std::invalid_argument(
                "FastStructuredRotationV1 requires fast rotation and dense QJL");
        }
        break;
    case TQModelTransformVersion::FastStructuredV1:
        throw std::invalid_argument("FastStructuredV1 is not implemented yet");
    default:
        throw std::invalid_argument("Unsupported TurboQuant model transform version");
    }
    switch (config.rotation_backend_version) {
    case TQRotationBackendVersion::DenseHaarV1:
        break;
    case TQRotationBackendVersion::FastStructuredV1:
        break;
    default:
        throw std::invalid_argument("Unsupported TurboQuant rotation backend version");
    }
    switch (config.qjl_backend_version) {
    case TQQjlBackendVersion::DenseGaussianV1:
        break;
    case TQQjlBackendVersion::CirculantGaussianV1:
        throw std::invalid_argument("Circulant Gaussian TurboQuant QJL is not implemented yet");
    default:
        throw std::invalid_argument("Unsupported TurboQuant QJL backend version");
    }

    if (config.rotation_backend_version == TQRotationBackendVersion::DenseHaarV1) {
        CheckedMultiply(config.dimension, config.dimension,
                        "TurboQuant dense rotation size overflow");
    } else {
        CheckedBytes(config.dimension, sizeof(int8_t),
                     "TurboQuant structured rotation sign size overflow");
        CheckedBytes(config.dimension, sizeof(size_t),
                     "TurboQuant structured rotation permutation size overflow");
    }
    CheckedMultiply(config.projections, config.dimension, "TurboQuant dense QJL size overflow");
    PackedBytes(config.dimension, config.mse_bits);
    PackedBytes(config.projections, 1);
    const size_t query_words = CheckedAdd(
        CheckedAdd(config.dimension, config.projections, "TurboQuant query layout overflow"),
        size_t{1}, "TurboQuant query layout overflow");
    CheckedBytes(query_words, sizeof(float), "TurboQuant query byte size overflow");
}

inline void ValidatePublicTQParams(const TQFlatParams &params) {
    if (params.type != VecSimType_FLOAT32) {
        throw std::invalid_argument("TurboQuant currently supports FLOAT32 input only");
    }
    if (params.metric == VecSimMetric_L2) {
        throw std::invalid_argument("TurboQuant supports COSINE and IP metrics only");
    }
    if (params.metric != VecSimMetric_IP && params.metric != VecSimMetric_Cosine) {
        throw std::invalid_argument("Unsupported TurboQuant distance metric");
    }
    if (!params.useRotation) {
        throw std::invalid_argument("TurboQuant production configuration requires rotation");
    }
    ValidateTQCodecConfig(TQCodecConfig::DenseReference(params.dim, params.bits, params.projections,
                                                        static_cast<uint64_t>(params.seed), true));
    CheckedBytes(params.dim, sizeof(float), "TurboQuant input byte size overflow");
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

template <typename T>
using TQPersistentVector = std::vector<T, VecsimSTLAllocator<T>>;

struct TQCodebookState {
    TQCodebookState(const std::shared_ptr<VecSimAllocator> &allocator, size_t dimension,
                    size_t mse_bits)
        : centroids(size_t{1} << mse_bits, 0.0f, VecsimSTLAllocator<float>(allocator)),
          boundaries((size_t{1} << mse_bits) - 1, 0.0f, VecsimSTLAllocator<float>(allocator)) {
        const auto codebook = BuildLloydMaxCodebook(dimension, mse_bits);
        std::copy(codebook.centroids.begin(), codebook.centroids.end(), centroids.begin());
        std::copy(codebook.boundaries.begin(), codebook.boundaries.end(), boundaries.begin());
    }

    TQPersistentVector<float> centroids;
    TQPersistentVector<float> boundaries;
};

class DenseHaarRotationV1State {
public:
    DenseHaarRotationV1State(const std::shared_ptr<VecSimAllocator> &allocator,
                             const TQCodecConfig &config)
        : dimension(config.dimension), use_rotation(config.use_rotation),
          rotation_seed(config.rotation_seed),
          rotation_columns(
              CheckedMultiply(dimension, dimension, "TurboQuant dense rotation size overflow"),
              0.0f, VecsimSTLAllocator<float>(allocator)),
          rotation_rows(
              CheckedMultiply(dimension, dimension, "TurboQuant dense rotation size overflow"),
              0.0f, VecsimSTLAllocator<float>(allocator)) {
        initializeRotation();
    }

    void forwardRotate(const float *input, float *output, float *scratch) const {
        (void)scratch;
        if (!use_rotation) {
            std::memcpy(output, input, dimension * sizeof(float));
            return;
        }
        for (size_t row = 0; row < dimension; ++row) {
            output[row] =
                DotProductScalar(rotation_rows.data() + row * dimension, input, dimension);
        }
    }

    void inverseRotate(const float *input, float *output, float *scratch) const {
        (void)scratch;
        if (!use_rotation) {
            std::memcpy(output, input, dimension * sizeof(float));
            return;
        }
        for (size_t col = 0; col < dimension; ++col) {
            output[col] =
                DotProductScalar(rotation_columns.data() + col * dimension, input, dimension);
        }
    }

private:
    void initializeRotation() {
        if (!use_rotation) {
            for (size_t i = 0; i < dimension; ++i) {
                rotation_columns[i * dimension + i] = 1.0f;
                rotation_rows[i * dimension + i] = 1.0f;
            }
            return;
        }

        DeterministicGaussianRng rng(rotation_seed);
        std::vector<float> candidate(dimension);
        for (size_t col = 0; col < dimension; ++col) {
            bool accepted = false;
            for (size_t attempt = 0; attempt < 16 && !accepted; ++attempt) {
                for (float &value : candidate) {
                    value = rng.normal();
                }

                for (size_t pass = 0; pass < 2; ++pass) {
                    for (size_t previous = 0; previous < col; ++previous) {
                        const float *previous_column =
                            rotation_columns.data() + previous * dimension;
                        const float projection =
                            DotProductScalar(candidate.data(), previous_column, dimension);
                        for (size_t row = 0; row < dimension; ++row) {
                            candidate[row] -= projection * previous_column[row];
                        }
                    }
                }

                const float norm_sq = SumSquaresScalar(candidate.data(), dimension);
                if (norm_sq > 1e-12f) {
                    const float inverse_norm = 1.0f / std::sqrt(norm_sq);
                    float *column = rotation_columns.data() + col * dimension;
                    for (size_t row = 0; row < dimension; ++row) {
                        column[row] = candidate[row] * inverse_norm;
                    }
                    accepted = true;
                }
            }
            if (!accepted) {
                throw std::runtime_error("Failed to construct TurboQuant rotation");
            }
        }

        for (size_t row = 0; row < dimension; ++row) {
            for (size_t col = 0; col < dimension; ++col) {
                rotation_rows[row * dimension + col] = rotation_columns[col * dimension + row];
            }
        }
    }

    size_t dimension;
    bool use_rotation;
    uint64_t rotation_seed;
    TQPersistentVector<float> rotation_columns;
    TQPersistentVector<float> rotation_rows;
};

class DenseGaussianQjlV1State {
public:
    DenseGaussianQjlV1State(const std::shared_ptr<VecSimAllocator> &allocator,
                            const TQCodecConfig &config)
        : dimension(config.dimension), projections(config.projections), qjl_seed(config.qjl_seed),
          qjl_projection_rows(
              CheckedMultiply(projections, dimension, "TurboQuant dense QJL size overflow"), 0.0f,
              VecsimSTLAllocator<float>(allocator)) {
        DeterministicGaussianRng rng(qjl_seed);
        for (float &value : qjl_projection_rows) {
            value = rng.normal();
        }
    }

    void project(const float *input, float *output, float *scratch) const {
        (void)scratch;
        for (size_t row = 0; row < projections; ++row) {
            output[row] =
                DotProductScalar(qjl_projection_rows.data() + row * dimension, input, dimension);
        }
    }

    float coefficient(size_t projection, size_t coordinate) const {
        return qjl_projection_rows[projection * dimension + coordinate];
    }

private:
    size_t dimension;
    size_t projections;
    uint64_t qjl_seed;
    TQPersistentVector<float> qjl_projection_rows;
};

class TQRotationBackend {
public:
    TQRotationBackend(const std::shared_ptr<VecSimAllocator> &allocator,
                      const TQCodecConfig &config)
        : backend(createBackend(allocator, config)) {}

    void forwardRotate(const float *input, float *output, float *scratch) const {
        std::visit([input, output, scratch](
                       const auto &selected) { selected.forwardRotate(input, output, scratch); },
                   backend);
    }

    void inverseRotate(const float *input, float *output, float *scratch) const {
        std::visit([input, output, scratch](
                       const auto &selected) { selected.inverseRotate(input, output, scratch); },
                   backend);
    }

private:
    using BackendState = std::variant<DenseHaarRotationV1State, FastStructuredRotationV1>;

    static BackendState createBackend(const std::shared_ptr<VecSimAllocator> &allocator,
                                      const TQCodecConfig &config) {
        switch (config.rotation_backend_version) {
        case TQRotationBackendVersion::DenseHaarV1:
            return BackendState(std::in_place_type<DenseHaarRotationV1State>, allocator, config);
        case TQRotationBackendVersion::FastStructuredV1:
            return BackendState(std::in_place_type<FastStructuredRotationV1>, allocator,
                                config.dimension, config.rotation_seed);
        }
        throw std::logic_error("Unsupported TurboQuant rotation backend");
    }

    BackendState backend;
};

class TQQjlBackend {
public:
    TQQjlBackend(const std::shared_ptr<VecSimAllocator> &allocator, const TQCodecConfig &config)
        : version(config.qjl_backend_version), dense_gaussian(allocator, config) {}

    void project(const float *input, float *output, float *scratch) const {
        switch (version) {
        case TQQjlBackendVersion::DenseGaussianV1:
            dense_gaussian.project(input, output, scratch);
            return;
        case TQQjlBackendVersion::CirculantGaussianV1:
            break;
        }
        throw std::logic_error("Unsupported TurboQuant QJL backend");
    }

    float coefficient(size_t projection, size_t coordinate) const {
        switch (version) {
        case TQQjlBackendVersion::DenseGaussianV1:
            return dense_gaussian.coefficient(projection, coordinate);
        case TQQjlBackendVersion::CirculantGaussianV1:
            break;
        }
        throw std::logic_error("Unsupported TurboQuant inverse-QJL backend");
    }

private:
    TQQjlBackendVersion version;
    DenseGaussianQjlV1State dense_gaussian;
};

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
private:
    std::shared_ptr<VecSimAllocator> allocator_;

public:
    TQModelState(std::shared_ptr<VecSimAllocator> allocator, TQCodecConfig requested_config)
        : allocator_(std::move(allocator)), config(validateAndReturn(requested_config)),
          codebook(allocator_, config.dimension, config.mse_bits),
          rotation_backend(allocator_, config), qjl_backend(allocator_, config),
          dim(config.dimension), total_bits(config.total_bits), mse_bits(config.mse_bits),
          projections(config.projections), seed(config.rotation_seed),
          use_rotation(config.use_rotation), levels(size_t{1} << config.mse_bits),
          qjl_scale(QjlScale(config.projections)), centroids(codebook.centroids),
          boundaries(codebook.boundaries),
          packed_index_bytes(PackedBytes(config.dimension, config.mse_bits)),
          packed_qjl_bytes(PackedBytes(config.projections, 1)),
          storage_blob_size(CheckedAdd(CheckedAdd(packed_index_bytes, packed_qjl_bytes,
                                                  "TurboQuant payload byte size overflow"),
                                       2 * sizeof(float), "TurboQuant payload byte size overflow")),
          query_blob_size(CheckedBytes(CheckedAdd(CheckedAdd(config.dimension, config.projections,
                                                             "TurboQuant query layout overflow"),
                                                  size_t{1}, "TurboQuant query layout overflow"),
                                       sizeof(float), "TurboQuant query byte size overflow")) {}

#ifdef BUILD_TESTS
    // Identity rotation is intentionally available only to unit tests. Public factories reject it.
    TQModelState(size_t dim, size_t total_bits, size_t projections, size_t seed, bool use_rotation)
        : TQModelState(
              VecSimAllocator::newVecsimAllocator(),
              TQCodecConfig::DenseReference(dim, total_bits, projections, seed, use_rotation)) {}
#endif

    TQModelState(const TQModelState &) = delete;
    TQModelState &operator=(const TQModelState &) = delete;

    size_t storageBlobSize() const { return storage_blob_size; }
    size_t queryBlobSize() const { return query_blob_size; }
    size_t packedIndexBytes() const { return packed_index_bytes; }
    size_t packedQjlBytes() const { return packed_qjl_bytes; }
    std::shared_ptr<VecSimAllocator> getAllocator() const { return allocator_; }

    TQModelIdentity modelIdentity() const {
        return {.dimension = config.dimension,
                .total_bits = config.total_bits,
                .projections = config.projections,
                .metric_contract = config.metric_contract,
                .codec_version = config.codec_version,
                .payload_layout_version = config.payload_layout_version,
                .model_transform_version = config.model_transform_version,
                .rotation_backend_version = config.rotation_backend_version,
                .qjl_backend_version = config.qjl_backend_version,
                .rotation_seed = config.rotation_seed,
                .qjl_seed = config.qjl_seed,
                .use_rotation = config.use_rotation};
    }

    TQPayloadLayoutVersion payloadLayoutVersion() const { return config.payload_layout_version; }
    TQModelTransformVersion modelTransformVersion() const { return config.model_transform_version; }

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

    void applyRotation(const float *input, float *output, float *scratch = nullptr) const {
        rotation_backend.forwardRotate(input, output, scratch);
    }

    void applyInverseRotation(const float *input, float *output, float *scratch = nullptr) const {
        rotation_backend.inverseRotate(input, output, scratch);
    }

    void projectQjl(const float *input, float *output, float *scratch = nullptr) const {
        qjl_backend.project(input, output, scratch);
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

    void encodeMse(const float *unit_vector, uint8_t *packed_indices, float *reconstructed_unit,
                   float *rotated, float *reconstructed_rotated, float *transform_scratch) const {
        std::memset(packed_indices, 0, packedIndexBytes());
        applyRotation(unit_vector, rotated, transform_scratch);
        for (size_t coordinate = 0; coordinate < dim; ++coordinate) {
            const uint16_t index = quantizeCoordinate(rotated[coordinate]);
            writeMseIndex(packed_indices, coordinate, index);
            reconstructed_rotated[coordinate] = centroids[index];
        }
        applyInverseRotation(reconstructed_rotated, reconstructed_unit, transform_scratch);
    }

    void packResidualSigns(const float *unit_residual, uint8_t *packed_signs, float *projected,
                           float *qjl_scratch) const {
        std::memset(packed_signs, 0, packedQjlBytes());
        projectQjl(unit_residual, projected, qjl_scratch);
        for (size_t i = 0; i < projections; ++i) {
            if (projected[i] >= 0.0f) {
                packed_signs[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
            }
        }
    }

#ifdef BUILD_TESTS
    void packResidualSigns(const float *unit_residual, uint8_t *packed_signs) const {
        auto scratch = allocator_->allocate_unique(
            CheckedBytes(CheckedMultiply(dim, size_t{2}, "TurboQuant test scratch overflow"),
                         sizeof(float), "TurboQuant test scratch overflow"));
        if (!scratch) {
            throw std::bad_alloc();
        }
        auto *projected = static_cast<float *>(scratch.get());
        packResidualSigns(unit_residual, packed_signs, projected, projected + dim);
    }
#endif

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
    void decode(const StorageView &storage, float *output, float *reconstructed_rotated,
                float *transform_scratch) const {
        if (storage.source_scale == 0.0f) {
            std::fill(output, output + dim, 0.0f);
            return;
        }
        for (size_t coordinate = 0; coordinate < dim; ++coordinate) {
            reconstructed_rotated[coordinate] = centroids[mseIndexAt(storage, coordinate)];
        }
        applyInverseRotation(reconstructed_rotated, output, transform_scratch);
        if (storage.residual_norm != 0.0f) {
            for (size_t coordinate = 0; coordinate < dim; ++coordinate) {
                float projected_sign_sum = 0.0f;
                for (size_t projection = 0; projection < projections; ++projection) {
                    const float sign = residualSignAt(storage, projection) ? 1.0f : -1.0f;
                    projected_sign_sum += qjl_backend.coefficient(projection, coordinate) * sign;
                }
                output[coordinate] += storage.residual_norm * qjl_scale * projected_sign_sum;
            }
        }
        for (size_t coordinate = 0; coordinate < dim; ++coordinate) {
            output[coordinate] *= storage.source_scale;
        }
    }

#ifdef BUILD_TESTS
    void decode(const StorageView &storage, float *output) const {
        auto scratch = allocator_->allocate_unique(
            CheckedBytes(CheckedMultiply(dim, size_t{2}, "TurboQuant test scratch overflow"),
                         sizeof(float), "TurboQuant test scratch overflow"));
        if (!scratch) {
            throw std::bad_alloc();
        }
        auto *reconstructed_rotated = static_cast<float *>(scratch.get());
        decode(storage, output, reconstructed_rotated, reconstructed_rotated + dim);
    }
#endif

    const TQCodecConfig config;
    const TQCodebookState codebook;
    const TQRotationBackend rotation_backend;
    const TQQjlBackend qjl_backend;
    const size_t dim;
    const size_t total_bits;
    const size_t mse_bits;
    const size_t projections;
    const uint64_t seed;
    const bool use_rotation;
    const size_t levels;
    const float qjl_scale;
    const TQPersistentVector<float> &centroids;
    const TQPersistentVector<float> &boundaries;

private:
    static TQCodecConfig validateAndReturn(const TQCodecConfig &config) {
        ValidateTQCodecConfig(config);
        return config;
    }

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

    size_t packed_index_bytes;
    size_t packed_qjl_bytes;
    size_t storage_blob_size;
    size_t query_blob_size;
};

inline std::shared_ptr<const TQModelState>
AllocateTQModelState(const std::shared_ptr<VecSimAllocator> &allocator, TQCodecConfig config) {
    std::shared_ptr<TQModelState> state = std::allocate_shared<TQModelState>(
        VecsimSTLAllocator<TQModelState>(allocator), allocator, std::move(config));
    return state;
}

inline std::shared_ptr<const TQModelState>
AllocateDenseReferenceTQModelState(const std::shared_ptr<VecSimAllocator> &allocator, size_t dim,
                                   size_t total_bits, size_t projections, uint64_t seed,
                                   bool use_rotation = true) {
    return AllocateTQModelState(
        allocator, TQCodecConfig::DenseReference(dim, total_bits, projections, seed, use_rotation));
}

inline std::shared_ptr<const TQModelState>
AllocateFastStructuredRotationTQModelState(const std::shared_ptr<VecSimAllocator> &allocator,
                                           size_t dim, size_t total_bits, size_t projections,
                                           uint64_t seed) {
    return AllocateTQModelState(
        allocator, TQCodecConfig::FastStructuredRotation(dim, total_bits, projections, seed));
}

template <typename T>
struct AllocateSharedSizeProbe {
    alignas(T) std::byte bytes[sizeof(T)];
};

inline size_t GetTQModelSharedAllocationSize() {
    static const size_t size = []() {
        auto allocator = VecSimAllocator::newVecsimAllocator();
        const size_t before = allocator->getAllocationSize();
        [[maybe_unused]] auto probe = std::allocate_shared<AllocateSharedSizeProbe<TQModelState>>(
            VecsimSTLAllocator<AllocateSharedSizeProbe<TQModelState>>(allocator));
        return static_cast<size_t>(allocator->getAllocationSize() - before);
    }();
    return size;
}

inline size_t EstimateTrackedAllocation(size_t bytes) {
    return CheckedAdd(bytes, VecSimAllocator::getAllocationOverheadSize(),
                      "TurboQuant allocation estimate overflow");
}

inline size_t EstimateTQModelAllocationSize(const TQCodecConfig &config) {
    ValidateTQCodecConfig(config);
    size_t estimate = GetTQModelSharedAllocationSize();
    const size_t levels = size_t{1} << config.mse_bits;
    const auto add_allocation = [&estimate](size_t count, size_t element_size) {
        const size_t allocation = EstimateTrackedAllocation(
            CheckedBytes(count, element_size, "TurboQuant model allocation estimate overflow"));
        estimate =
            CheckedAdd(estimate, allocation, "TurboQuant model allocation estimate overflow");
    };

    add_allocation(levels, sizeof(float));
    add_allocation(levels - 1, sizeof(float));
    switch (config.rotation_backend_version) {
    case TQRotationBackendVersion::DenseHaarV1: {
        const size_t matrix_elements = CheckedMultiply(
            config.dimension, config.dimension, "TurboQuant dense rotation estimate overflow");
        add_allocation(matrix_elements, sizeof(float));
        add_allocation(matrix_elements, sizeof(float));
        break;
    }
    case TQRotationBackendVersion::FastStructuredV1:
        for (size_t stage = 0; stage < 3; ++stage) {
            add_allocation(config.dimension, sizeof(int8_t));
            add_allocation(config.dimension, sizeof(size_t));
            add_allocation(config.dimension, sizeof(size_t));
        }
        break;
    }

    const size_t qjl_elements = CheckedMultiply(config.projections, config.dimension,
                                                "TurboQuant dense QJL estimate overflow");
    add_allocation(qjl_elements, sizeof(float));
    return estimate;
}

inline size_t EstimateDenseReferenceTQModelAllocationSize(size_t dim, size_t total_bits,
                                                          size_t projections, uint64_t seed = 0) {
    return EstimateTQModelAllocationSize(
        TQCodecConfig::DenseReference(dim, total_bits, projections, seed));
}

inline size_t EstimateFastStructuredRotationTQModelAllocationSize(size_t dim, size_t total_bits,
                                                                  size_t projections,
                                                                  uint64_t seed = 0) {
    return EstimateTQModelAllocationSize(
        TQCodecConfig::FastStructuredRotation(dim, total_bits, projections, seed));
}

} // namespace TQFlatDetails
