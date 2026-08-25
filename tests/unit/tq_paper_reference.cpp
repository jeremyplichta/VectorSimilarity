/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#include "tq_paper_reference.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace tq_paper_reference {
namespace {

double pi() { return std::acos(-1.0); }

void validateShape(size_t dim, size_t total_bits) {
    if (dim < 2 || (total_bits != 2 && total_bits != 4 && total_bits != 8)) {
        throw std::invalid_argument("invalid paper TurboQuant dimensions or bit width");
    }
}

} // namespace

size_t PackedBytes(size_t count, size_t bits_per_value) {
    if (bits_per_value != 0 && count > (std::numeric_limits<size_t>::max() - 7) / bits_per_value) {
        throw std::overflow_error("packed TurboQuant size overflow");
    }
    return (count * bits_per_value + 7) / 8;
}

size_t StorageBytes(size_t dim, size_t total_bits) {
    validateShape(dim, total_bits);
    return PackedBytes(dim, total_bits - 1) + PackedBytes(dim, 1) + 2 * sizeof(float);
}

float QjlScale(size_t projections) {
    if (projections == 0) {
        throw std::invalid_argument("QJL requires at least one projection");
    }
    return std::sqrt(std::acos(-1.0f) / 2.0f) / static_cast<float>(projections);
}

float QjlCorrection(float residual_norm, std::span<const float> projected_query,
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

std::pair<float, float> ExactOneBitCentroids(size_t dim) {
    if (dim < 2) {
        throw std::invalid_argument("sphere-coordinate density requires dimension >= 2");
    }
    const double normalizer =
        std::exp(std::lgamma(static_cast<double>(dim) / 2.0) - 0.5 * std::log(pi()) -
                 std::lgamma(static_cast<double>(dim - 1) / 2.0));
    const float positive = static_cast<float>(2.0 * normalizer / static_cast<double>(dim - 1));
    return {-positive, positive};
}

ReferenceGaussianRng::ReferenceGaussianRng(uint64_t seed) : state_(seed) {}

uint64_t ReferenceGaussianRng::next() {
    // SplitMix64 is specified here rather than shared with production code. Combined with the
    // Box-Muller endpoint rule below, it is the RNG contract for DenseReferenceV1 fixtures.
    state_ += 0x9E3779B97F4A7C15ULL;
    uint64_t value = state_;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

double ReferenceGaussianRng::uniformOpen() {
    constexpr double kInv53 = 1.0 / static_cast<double>(uint64_t{1} << 53);
    return (static_cast<double>(next() >> 11) + 0.5) * kInv53;
}

double ReferenceGaussianRng::normal() {
    if (has_spare_) {
        has_spare_ = false;
        return spare_;
    }
    const double radius = std::sqrt(-2.0 * std::log(uniformOpen()));
    const double theta = 2.0 * pi() * uniformOpen();
    spare_ = radius * std::sin(theta);
    has_spare_ = true;
    return radius * std::cos(theta);
}

ReferenceLloydMaxCodebook BuildLloydMaxCodebook(size_t dim, size_t mse_bits) {
    if (dim < 2 || mse_bits == 0 || mse_bits > 7) {
        throw std::invalid_argument("invalid reference Lloyd-Max configuration");
    }

    // Integrate after x = sin(theta). This removes the d=2 endpoint singularity and is
    // intentionally different from production's x-domain midpoint-prefix implementation.
    const size_t levels = size_t{1} << mse_bits;
    std::vector<double> coordinates(kIntegrationPoints);
    std::vector<double> weights(kIntegrationPoints);
    const double theta_step = pi() / static_cast<double>(kIntegrationPoints);
    double total_mass = 0.0;
    for (size_t i = 0; i < kIntegrationPoints; ++i) {
        const double theta = -pi() / 2.0 + (static_cast<double>(i) + 0.5) * theta_step;
        const double cosine = std::max(0.0, std::cos(theta));
        coordinates[i] = std::sin(theta);
        weights[i] = std::pow(cosine, static_cast<double>(dim - 2));
        total_mass += weights[i];
    }

    std::vector<double> centroids(levels);
    double cumulative = 0.0;
    size_t sample = 0;
    for (size_t level = 0; level < levels; ++level) {
        const double target =
            (static_cast<double>(level) + 0.5) * total_mass / static_cast<double>(levels);
        while (sample + 1 < kIntegrationPoints && cumulative + weights[sample] < target) {
            cumulative += weights[sample++];
        }
        centroids[level] = coordinates[sample];
    }

    std::vector<double> boundaries(levels - 1);
    std::vector<double> updated(levels);
    std::vector<double> mass(levels);
    std::vector<double> moment(levels);
    for (size_t iteration = 0; iteration < kLloydMaxIterations; ++iteration) {
        for (size_t level = 0; level + 1 < levels; ++level) {
            boundaries[level] = (centroids[level] + centroids[level + 1]) / 2.0;
        }
        std::fill(mass.begin(), mass.end(), 0.0);
        std::fill(moment.begin(), moment.end(), 0.0);
        size_t level = 0;
        for (size_t i = 0; i < kIntegrationPoints; ++i) {
            while (level + 1 < levels && coordinates[i] > boundaries[level]) {
                ++level;
            }
            mass[level] += weights[i];
            moment[level] += weights[i] * coordinates[i];
        }
        for (size_t i = 0; i < levels; ++i) {
            updated[i] =
                mass[i] > std::numeric_limits<double>::min() ? moment[i] / mass[i] : centroids[i];
        }
        for (size_t i = 0; i < levels / 2; ++i) {
            const size_t mirror = levels - 1 - i;
            const double magnitude = (std::abs(updated[i]) + std::abs(updated[mirror])) / 2.0;
            updated[i] = -magnitude;
            updated[mirror] = magnitude;
        }
        double max_delta = 0.0;
        for (size_t i = 0; i < levels; ++i) {
            max_delta = std::max(max_delta, std::abs(updated[i] - centroids[i]));
        }
        centroids.swap(updated);
        if (max_delta <= kLloydMaxTolerance / std::sqrt(static_cast<double>(dim))) {
            break;
        }
    }
    for (size_t level = 0; level + 1 < levels; ++level) {
        boundaries[level] = (centroids[level] + centroids[level + 1]) / 2.0;
    }
    return {.centroids = std::move(centroids), .boundaries = std::move(boundaries)};
}

ReferenceDenseModel::ReferenceDenseModel(size_t dim, size_t total_bits, uint64_t rotation_seed,
                                         uint64_t qjl_seed)
    : dim_(dim), total_bits_(total_bits), mse_bits_(total_bits - 1),
      index_bytes_(PackedBytes(dim, mse_bits_)), sign_bytes_(PackedBytes(dim, 1)),
      codebook_(BuildLloydMaxCodebook(dim, mse_bits_)), rotation_columns_(dim * dim, 0.0),
      rotation_rows_(dim * dim, 0.0), qjl_rows_(dim * dim, 0.0) {
    validateShape(dim, total_bits);

    ReferenceGaussianRng rotation_rng(rotation_seed);
    std::vector<double> candidate(dim_);
    for (size_t column = 0; column < dim_; ++column) {
        for (double &value : candidate) {
            // DenseReferenceV1 stores FP32 Gaussian draws. The QR itself remains FP64 here.
            value = static_cast<float>(rotation_rng.normal());
        }
        for (size_t pass = 0; pass < 2; ++pass) {
            for (size_t previous = 0; previous < column; ++previous) {
                const auto previous_column =
                    std::span<const double>(rotation_columns_.data() + previous * dim_, dim_);
                const double projection = dot(candidate, previous_column);
                for (size_t row = 0; row < dim_; ++row) {
                    candidate[row] -= projection * previous_column[row];
                }
            }
        }
        const double candidate_norm = norm(candidate);
        if (!(candidate_norm > 1e-12)) {
            throw std::runtime_error("reference dense QR produced a degenerate column");
        }
        // Orientation convention: preserve the sampled Gaussian column's sign; no diagonal sign
        // flip is applied after normalization. This matches the versioned model generator.
        for (size_t row = 0; row < dim_; ++row) {
            rotation_columns_[column * dim_ + row] = candidate[row] / candidate_norm;
        }
    }
    for (size_t row = 0; row < dim_; ++row) {
        for (size_t column = 0; column < dim_; ++column) {
            rotation_rows_[row * dim_ + column] = rotation_columns_[column * dim_ + row];
        }
    }

    ReferenceGaussianRng qjl_rng(qjl_seed);
    for (double &value : qjl_rows_) {
        value = static_cast<float>(qjl_rng.normal());
    }
}

double ReferenceDenseModel::dot(std::span<const double> lhs, std::span<const double> rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::invalid_argument("reference dot-product dimensions differ");
    }
    double result = 0.0;
    for (size_t i = 0; i < lhs.size(); ++i) {
        result += lhs[i] * rhs[i];
    }
    return result;
}

double ReferenceDenseModel::norm(std::span<const double> values) {
    return std::sqrt(dot(values, values));
}

std::vector<double> ReferenceDenseModel::forwardRotate(std::span<const double> input) const {
    if (input.size() != dim_) {
        throw std::invalid_argument("reference rotation dimension differs");
    }
    std::vector<double> output(dim_);
    for (size_t row = 0; row < dim_; ++row) {
        output[row] = dot(std::span<const double>(rotation_rows_.data() + row * dim_, dim_), input);
    }
    return output;
}

std::vector<double> ReferenceDenseModel::inverseRotate(std::span<const double> input) const {
    if (input.size() != dim_) {
        throw std::invalid_argument("reference inverse-rotation dimension differs");
    }
    std::vector<double> output(dim_);
    for (size_t column = 0; column < dim_; ++column) {
        output[column] =
            dot(std::span<const double>(rotation_columns_.data() + column * dim_, dim_), input);
    }
    return output;
}

std::vector<double> ReferenceDenseModel::projectQjl(std::span<const double> input) const {
    if (input.size() != dim_) {
        throw std::invalid_argument("reference QJL dimension differs");
    }
    std::vector<double> output(dim_);
    for (size_t row = 0; row < dim_; ++row) {
        output[row] = dot(std::span<const double>(qjl_rows_.data() + row * dim_, dim_), input);
    }
    return output;
}

std::vector<double> ReferenceDenseModel::transposeQjl(std::span<const double> input) const {
    if (input.size() != dim_) {
        throw std::invalid_argument("reference transpose-QJL dimension differs");
    }
    std::vector<double> output(dim_, 0.0);
    for (size_t row = 0; row < dim_; ++row) {
        for (size_t column = 0; column < dim_; ++column) {
            output[column] += qjl_rows_[row * dim_ + column] * input[row];
        }
    }
    return output;
}

uint16_t ReferenceDenseModel::quantize(double value) const {
    return static_cast<uint16_t>(
        std::upper_bound(codebook_.boundaries.begin(), codebook_.boundaries.end(), value) -
        codebook_.boundaries.begin());
}

void ReferenceDenseModel::packIndex(std::vector<uint8_t> &bytes, size_t coordinate,
                                    uint16_t value) const {
    const size_t first_bit = coordinate * mse_bits_;
    for (size_t bit = 0; bit < mse_bits_; ++bit) {
        if ((value & (uint16_t{1} << bit)) != 0) {
            const size_t target = first_bit + bit;
            bytes[target / 8] |= static_cast<uint8_t>(1u << (target % 8));
        }
    }
}

void ReferenceDenseModel::packSign(std::vector<uint8_t> &bytes, size_t projection,
                                   bool positive) const {
    if (positive) {
        bytes[index_bytes_ + projection / 8] |= static_cast<uint8_t>(1u << (projection % 8));
    }
}

void ReferenceDenseModel::writeMetadata(std::vector<uint8_t> &bytes, float alpha,
                                        float gamma) const {
    std::memcpy(bytes.data() + index_bytes_ + sign_bytes_, &alpha, sizeof(float));
    std::memcpy(bytes.data() + index_bytes_ + sign_bytes_ + sizeof(float), &gamma, sizeof(float));
}

ReferenceEncoded ReferenceDenseModel::encode(std::span<const float> input, Metric metric) const {
    if (input.size() != dim_) {
        throw std::invalid_argument("reference encode dimension differs");
    }
    ReferenceEncoded result;
    result.bytes.assign(StorageBytes(dim_, total_bits_), 0);
    result.indices.resize(dim_);
    result.signs.resize(dim_);

    std::vector<double> unit(input.begin(), input.end());
    const double source_norm = norm(unit);
    if (source_norm == 0.0) {
        result.alpha = 0.0;
        result.gamma = 0.0;
        writeMetadata(result.bytes, 0.0f, 0.0f);
        return result;
    }
    for (double &value : unit) {
        value /= source_norm;
    }
    result.alpha = metric == Metric::Cosine ? 1.0 : source_norm;

    const auto rotated = forwardRotate(unit);
    std::vector<double> coarse_rotated(dim_);
    for (size_t coordinate = 0; coordinate < dim_; ++coordinate) {
        const uint16_t index = quantize(rotated[coordinate]);
        result.indices[coordinate] = index;
        coarse_rotated[coordinate] = codebook_.centroids[index];
        packIndex(result.bytes, coordinate, index);
    }
    const auto coarse = inverseRotate(coarse_rotated);
    std::vector<double> residual(dim_);
    for (size_t i = 0; i < dim_; ++i) {
        residual[i] = unit[i] - coarse[i];
    }
    result.gamma = norm(residual);
    if (result.gamma != 0.0) {
        for (double &value : residual) {
            value /= result.gamma;
        }
    }
    const auto projected = projectQjl(residual);
    for (size_t i = 0; i < dim_; ++i) {
        // sign(0) = +1. A zero residual therefore has a canonical positive sign payload.
        result.signs[i] = projected[i] >= 0.0 ? int8_t{1} : int8_t{-1};
        packSign(result.bytes, i, result.signs[i] > 0);
    }
    writeMetadata(result.bytes, static_cast<float>(result.alpha), static_cast<float>(result.gamma));
    return result;
}

ReferenceQuery ReferenceDenseModel::preprocessQuery(std::span<const float> query,
                                                    Metric metric) const {
    if (query.size() != dim_) {
        throw std::invalid_argument("reference query dimension differs");
    }
    std::vector<double> prepared(query.begin(), query.end());
    if (metric == Metric::Cosine) {
        const double query_norm = norm(prepared);
        if (query_norm != 0.0) {
            for (double &value : prepared) {
                value /= query_norm;
            }
        }
    }
    ReferenceQuery result;
    result.rotated = forwardRotate(prepared);
    result.qjl_projection = projectQjl(prepared);
    result.norm_sq = dot(prepared, prepared);
    return result;
}

double ReferenceDenseModel::estimateIp(const ReferenceEncoded &encoded,
                                       const ReferenceQuery &query) const {
    if (encoded.alpha == 0.0) {
        return 0.0;
    }
    double coarse = 0.0;
    for (size_t i = 0; i < dim_; ++i) {
        coarse += query.rotated[i] * codebook_.centroids[encoded.indices[i]];
    }
    double residual_sum = 0.0;
    if (encoded.gamma != 0.0) {
        for (size_t i = 0; i < dim_; ++i) {
            residual_sum += query.qjl_projection[i] * static_cast<double>(encoded.signs[i]);
        }
    }
    return encoded.alpha * (coarse + encoded.gamma * std::sqrt(pi() / 2.0) /
                                         static_cast<double>(dim_) * residual_sum);
}

std::vector<double> ReferenceDenseModel::decode(const ReferenceEncoded &encoded) const {
    if (encoded.alpha == 0.0) {
        return std::vector<double>(dim_, 0.0);
    }
    std::vector<double> coarse_rotated(dim_);
    for (size_t i = 0; i < dim_; ++i) {
        coarse_rotated[i] = codebook_.centroids[encoded.indices[i]];
    }
    auto result = inverseRotate(coarse_rotated);
    if (encoded.gamma != 0.0) {
        std::vector<double> signs(encoded.signs.begin(), encoded.signs.end());
        const auto correction = transposeQjl(signs);
        const double scale = encoded.gamma * std::sqrt(pi() / 2.0) / static_cast<double>(dim_);
        for (size_t i = 0; i < dim_; ++i) {
            result[i] += scale * correction[i];
        }
    }
    for (double &value : result) {
        value *= encoded.alpha;
    }
    return result;
}

} // namespace tq_paper_reference
