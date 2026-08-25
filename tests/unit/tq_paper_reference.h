/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace tq_paper_reference {

inline constexpr uint32_t kFixtureVersion = 1;
inline constexpr size_t kIntegrationPoints = 1u << 17;
inline constexpr size_t kLloydMaxIterations = 256;
inline constexpr double kLloydMaxTolerance = 1e-11;

enum class Metric {
    Cosine,
    InnerProduct,
};

size_t PackedBytes(size_t count, size_t bits_per_value);
size_t StorageBytes(size_t dim, size_t total_bits);
float QjlScale(size_t projections);
float QjlCorrection(float residual_norm, std::span<const float> projected_query,
                    std::span<const int8_t> residual_signs);
std::pair<float, float> ExactOneBitCentroids(size_t dim);

class ReferenceGaussianRng {
public:
    explicit ReferenceGaussianRng(uint64_t seed);
    double normal();

private:
    uint64_t next();
    double uniformOpen();

    uint64_t state_;
    double spare_ = 0.0;
    bool has_spare_ = false;
};

struct ReferenceLloydMaxCodebook {
    std::vector<double> centroids;
    std::vector<double> boundaries;
};

ReferenceLloydMaxCodebook BuildLloydMaxCodebook(size_t dim, size_t mse_bits);

struct ReferenceEncoded {
    std::vector<uint8_t> bytes;
    std::vector<uint16_t> indices;
    std::vector<int8_t> signs;
    double alpha = 0.0;
    double gamma = 0.0;
};

struct ReferenceQuery {
    std::vector<double> rotated;
    std::vector<double> qjl_projection;
    double norm_sq = 0.0;
};

// A test-only literal implementation of the paper backend. It intentionally has no production
// includes and owns independent implementations of model generation, packing, and scoring.
class ReferenceDenseModel {
public:
    ReferenceDenseModel(size_t dim, size_t total_bits, uint64_t rotation_seed, uint64_t qjl_seed);

    ReferenceEncoded encode(std::span<const float> input, Metric metric) const;
    ReferenceQuery preprocessQuery(std::span<const float> query, Metric metric) const;
    double estimateIp(const ReferenceEncoded &encoded, const ReferenceQuery &query) const;
    std::vector<double> decode(const ReferenceEncoded &encoded) const;
    std::vector<double> forwardRotate(std::span<const double> input) const;
    std::vector<double> inverseRotate(std::span<const double> input) const;
    std::vector<double> projectQjl(std::span<const double> input) const;
    std::vector<double> transposeQjl(std::span<const double> input) const;

    size_t dim() const { return dim_; }
    size_t totalBits() const { return total_bits_; }
    size_t mseBits() const { return mse_bits_; }
    const ReferenceLloydMaxCodebook &codebook() const { return codebook_; }

private:
    static double norm(std::span<const double> values);
    static double dot(std::span<const double> lhs, std::span<const double> rhs);
    uint16_t quantize(double value) const;
    void packIndex(std::vector<uint8_t> &bytes, size_t coordinate, uint16_t value) const;
    void packSign(std::vector<uint8_t> &bytes, size_t projection, bool positive) const;
    void writeMetadata(std::vector<uint8_t> &bytes, float alpha, float gamma) const;

    size_t dim_;
    size_t total_bits_;
    size_t mse_bits_;
    size_t index_bytes_;
    size_t sign_bytes_;
    ReferenceLloydMaxCodebook codebook_;
    // Columns of Q followed by rows of Q. Both are kept in the oracle to make orientation explicit.
    std::vector<double> rotation_columns_;
    std::vector<double> rotation_rows_;
    std::vector<double> qjl_rows_;
};

} // namespace tq_paper_reference
