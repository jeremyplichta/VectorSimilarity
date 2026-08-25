/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#include "gtest/gtest.h"

#include "VecSim/algorithms/tq/tq_flat.h"
#include "tq_paper_golden_fixture.h"
#include "tq_paper_reference.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <span>
#include <vector>

namespace {

using tq_paper_reference::Metric;
using tq_paper_reference::ReferenceDenseModel;

constexpr uint64_t kQjlSeedOffset = 0xCAFEBABE00000001ULL;

std::vector<float> diagnosticVector(size_t dim, size_t salt, float scale = 1.0f) {
    std::vector<float> values(dim);
    for (size_t i = 0; i < dim; ++i) {
        values[i] = scale * (0.73f * std::sin(static_cast<float>((i + 1) * (salt + 3)) * 0.317f) +
                             0.21f * std::cos(static_cast<float>((i + 2) * (salt + 7)) * 0.193f));
    }
    return values;
}

template <VecSimMetric ProductionMetric>
void compareProductionWithReference(size_t dim, size_t bits, size_t seed) {
    const Metric reference_metric =
        ProductionMetric == VecSimMetric_Cosine ? Metric::Cosine : Metric::InnerProduct;
    auto input = diagnosticVector(dim, bits + seed, 2.75f);
    auto query = diagnosticVector(dim, bits * 3 + seed, 1.6f);
    ReferenceDenseModel reference(dim, bits, seed, seed + kQjlSeedOffset);
    const auto expected = reference.encode(input, reference_metric);
    const auto expected_query = reference.preprocessQuery(query, reference_metric);

    auto allocator = VecSimAllocator::newVecsimAllocator();
    auto state = std::make_shared<TQFlatDetails::TQModelState>(dim, bits, dim, seed, true);
    TQFlatDetails::TQPreprocessor<ProductionMetric> preprocessor(allocator, state);
    void *storage_blob = nullptr;
    size_t storage_size = dim * sizeof(float);
    preprocessor.preprocessForStorage(input.data(), storage_blob, storage_size, 0);
    void *query_blob = nullptr;
    size_t query_size = dim * sizeof(float);
    preprocessor.preprocessQuery(query.data(), query_blob, query_size, 0);

    ASSERT_EQ(storage_size, expected.bytes.size());
    const auto storage = state->storageView(storage_blob);
    size_t differing_indices = 0;
    size_t differing_signs = 0;
    for (size_t i = 0; i < dim; ++i) {
        const auto production_index = state->mseIndexAt(storage, i);
        const auto reference_index = expected.indices[i];
        EXPECT_LE(std::abs(static_cast<int>(production_index) - static_cast<int>(reference_index)),
                  1);
        differing_indices += production_index != reference_index;
        differing_signs += state->residualSignAt(storage, i) != (expected.signs[i] > 0);
    }
    // The production backend performs QR and Lloyd-Max updates in FP32, while the independent
    // oracle uses FP64 and a theta-domain quadrature. Near a boundary that can move one code or
    // residual sign, but a material layout divergence is not acceptable.
    EXPECT_LE(differing_indices, std::max<size_t>(1, dim / 16));
    EXPECT_LE(differing_signs, std::max<size_t>(1, dim / 8));
    EXPECT_NEAR(storage.source_scale, expected.alpha, 2e-5 * std::max(1.0, expected.alpha));
    EXPECT_NEAR(storage.residual_norm, expected.gamma, 5e-4);
    const auto query_view = state->queryView(query_blob);
    for (size_t i = 0; i < dim; ++i) {
        EXPECT_NEAR(query_view.rotated[i], expected_query.rotated[i], 2e-5);
        EXPECT_NEAR(query_view.qjl_projection[i], expected_query.qjl_projection[i], 5e-5);
    }
    EXPECT_NEAR(query_view.norm_sq, expected_query.norm_sq,
                3e-5 * std::max(1.0, expected_query.norm_sq));
    EXPECT_NEAR(state->estimateInnerProductScalar(storage, query_view),
                reference.estimateIp(expected, expected_query),
                2e-2 * std::max(1.0, std::abs(reference.estimateIp(expected, expected_query))));

    std::vector<float> production_decoded(dim);
    state->decode(storage, production_decoded.data());
    const auto reference_decoded = reference.decode(expected);
    for (size_t i = 0; i < dim; ++i) {
        EXPECT_NEAR(production_decoded[i], reference_decoded[i],
                    5e-3 * std::max(1.0, std::abs(reference_decoded[i])));
    }

    allocator->free_allocation(storage_blob);
    allocator->free_allocation(query_blob);
}

struct BiasSummary {
    double mean;
    double standard_error;
};

BiasSummary residualBias(size_t dim, size_t samples, double gamma, double query_sign) {
    std::vector<double> residual(dim);
    std::vector<double> query(dim);
    for (size_t i = 0; i < dim; ++i) {
        residual[i] = std::sin(static_cast<double>(i + 1) * 0.37) + 0.2;
        query[i] = query_sign * (0.7 * residual[i] + std::cos(static_cast<double>(i + 3) * 0.19));
    }
    const double residual_norm =
        std::sqrt(std::inner_product(residual.begin(), residual.end(), residual.begin(), 0.0));
    for (double &value : residual) {
        value /= residual_norm;
    }
    const double exact_unit =
        std::inner_product(residual.begin(), residual.end(), query.begin(), 0.0);
    const double scale = std::sqrt(std::acos(-1.0) / 2.0) / static_cast<double>(dim);

    std::vector<double> errors;
    errors.reserve(samples);
    for (size_t sample = 0; sample < samples; ++sample) {
        tq_paper_reference::ReferenceGaussianRng rng(0xA53D19u + sample * 0x9E3779B9u + dim);
        double signed_sum = 0.0;
        for (size_t row = 0; row < dim; ++row) {
            double residual_projection = 0.0;
            double query_projection = 0.0;
            for (size_t column = 0; column < dim; ++column) {
                const double gaussian = rng.normal();
                residual_projection += gaussian * residual[column];
                query_projection += gaussian * query[column];
            }
            signed_sum += (residual_projection >= 0.0 ? 1.0 : -1.0) * query_projection;
        }
        errors.push_back(gamma * (scale * signed_sum - exact_unit));
    }
    const double mean =
        std::accumulate(errors.begin(), errors.end(), 0.0) / static_cast<double>(errors.size());
    double squared_deviation = 0.0;
    for (double error : errors) {
        squared_deviation += (error - mean) * (error - mean);
    }
    const double variance = squared_deviation / static_cast<double>(errors.size() - 1);
    return {.mean = mean,
            .standard_error = std::sqrt(variance / static_cast<double>(errors.size()))};
}

TEST(TQIndependentPaperReferenceTest, complete_codec_matches_dense_reference_backend) {
    for (size_t dim : {size_t{3}, size_t{8}, size_t{15}, size_t{31}}) {
        for (size_t bits : {size_t{2}, size_t{4}, size_t{8}}) {
            SCOPED_TRACE(::testing::Message() << "dim=" << dim << " bits=" << bits);
            compareProductionWithReference<VecSimMetric_IP>(dim, bits, 17);
            compareProductionWithReference<VecSimMetric_Cosine>(dim, bits, 31);
        }
    }
}

TEST(TQIndependentPaperReferenceTest, frozen_dense_fixtures_are_stable) {
    for (const auto &fixture : tq_paper_reference::kGoldenFixtures) {
        SCOPED_TRACE(fixture.name);
        ASSERT_EQ(fixture.fixture_version, tq_paper_reference::kFixtureVersion);
        const auto input =
            diagnosticVector(fixture.dimension, fixture.input_salt, fixture.input_scale);
        const auto query =
            diagnosticVector(fixture.dimension, fixture.query_salt, fixture.query_scale);
        ReferenceDenseModel reference(fixture.dimension, fixture.total_bits, fixture.rotation_seed,
                                      fixture.qjl_seed);
        const auto encoded = reference.encode(input, fixture.metric);
        const auto prepared = reference.preprocessQuery(query, fixture.metric);
        const auto decoded = reference.decode(encoded);
        const size_t packed_size =
            tq_paper_reference::PackedBytes(fixture.dimension, fixture.total_bits - 1) +
            tq_paper_reference::PackedBytes(fixture.dimension, 1);
        ASSERT_EQ(fixture.packed_bytes.size(), packed_size);
        EXPECT_TRUE(std::equal(fixture.packed_bytes.begin(), fixture.packed_bytes.end(),
                               encoded.bytes.begin()));
        EXPECT_NEAR(encoded.alpha, fixture.alpha, 1e-12);
        EXPECT_NEAR(encoded.gamma, fixture.gamma, 1e-12);
        ASSERT_EQ(prepared.rotated.size(), fixture.rotated_query.size());
        ASSERT_EQ(prepared.qjl_projection.size(), fixture.qjl_query.size());
        ASSERT_EQ(decoded.size(), fixture.decoded.size());
        for (size_t i = 0; i < fixture.dimension; ++i) {
            EXPECT_NEAR(prepared.rotated[i], fixture.rotated_query[i], 1e-12);
            EXPECT_NEAR(prepared.qjl_projection[i], fixture.qjl_query[i], 1e-12);
            EXPECT_NEAR(decoded[i], fixture.decoded[i], 1e-12);
        }
        const double estimate = reference.estimateIp(encoded, prepared);
        EXPECT_NEAR(estimate, fixture.estimated_ip, 1e-12);
        EXPECT_NEAR(1.0 - estimate, fixture.distance, 1e-12);

        const auto &codebook = reference.codebook();
        const size_t levels = codebook.centroids.size();
        const std::array<double, 4> centroid_signature = {
            codebook.centroids.front(), codebook.centroids[levels / 2 - 1],
            codebook.centroids[levels / 2], codebook.centroids.back()};
        const size_t boundary_center = codebook.boundaries.size() / 2;
        const std::array<double, 3> boundary_signature = {codebook.boundaries.front(),
                                                          codebook.boundaries[boundary_center],
                                                          codebook.boundaries.back()};
        for (size_t i = 0; i < centroid_signature.size(); ++i) {
            EXPECT_NEAR(centroid_signature[i], fixture.centroid_signature[i], 1e-12);
        }
        for (size_t i = 0; i < boundary_signature.size(); ++i) {
            EXPECT_NEAR(boundary_signature[i], fixture.boundary_signature[i], 1e-12);
        }
    }
}

TEST(TQIndependentPaperReferenceTest, zero_source_and_zero_residual_are_canonical) {
    constexpr size_t dim = 8;
    ReferenceDenseModel reference(dim, 4, 7, 7 + kQjlSeedOffset);
    const std::array<float, dim> zero{};
    const auto encoded = reference.encode(zero, Metric::InnerProduct);
    EXPECT_EQ(encoded.bytes, std::vector<uint8_t>(encoded.bytes.size(), 0));
    EXPECT_EQ(encoded.alpha, 0.0);
    EXPECT_EQ(encoded.gamma, 0.0);

    const std::vector<double> zero_residual(dim, 0.0);
    const auto projected = reference.projectQjl(zero_residual);
    EXPECT_TRUE(
        std::all_of(projected.begin(), projected.end(), [](double value) { return value == 0.0; }));
    EXPECT_TRUE(
        std::all_of(projected.begin(), projected.end(), [](double value) { return value >= 0.0; }));
}

TEST(TQIndependentPaperReferenceTest, dense_qjl_signed_bias_confidence_interval_contains_zero) {
    for (size_t dim : {size_t{8}, size_t{31}, size_t{128}}) {
        for (double gamma : {0.01, 0.1, 1.0, 10.0}) {
            for (double query_sign : {-1.0, 1.0}) {
                SCOPED_TRACE(::testing::Message() << "dim=" << dim << " gamma=" << gamma
                                                  << " query_sign=" << query_sign);
                const auto summary = residualBias(dim, 4096, gamma, query_sign);
                const double confidence_radius = 1.96 * summary.standard_error;
                EXPECT_LE(std::abs(summary.mean), confidence_radius + 1e-12);
                EXPECT_LT(std::abs(summary.mean),
                          gamma * (0.12 / std::sqrt(static_cast<double>(dim)) + 0.01));
            }
        }
    }
}

TEST(TQIndependentPaperReferenceTest, historical_formula_mutations_are_detected) {
    constexpr size_t dim = 31;
    constexpr double gamma = 0.37;
    const auto projected_query = diagnosticVector(dim, 9, 1.0f);
    std::vector<int8_t> signs(dim);
    for (size_t i = 0; i < dim; ++i) {
        signs[i] = i % 3 == 0 ? int8_t{-1} : int8_t{1};
    }
    const double signed_sum =
        std::inner_product(projected_query.begin(), projected_query.end(), signs.begin(), 0.0);
    const double correct = gamma * std::sqrt(std::acos(-1.0) / 2.0) / dim * signed_sum;
    const double wrong_scale = gamma * std::acos(-1.0) / (2.0 * dim) * signed_sum;
    const double omitted_gamma = std::sqrt(std::acos(-1.0) / 2.0) / dim * signed_sum;
    const double half_rows =
        gamma * std::sqrt(std::acos(-1.0) / 2.0) / (dim / 2.0) *
        std::inner_product(projected_query.begin(), projected_query.begin() + dim / 2,
                           signs.begin(), 0.0);
    EXPECT_GT(std::abs(correct - wrong_scale), 1e-3);
    EXPECT_GT(std::abs(correct - omitted_gamma), 1e-3);
    EXPECT_GT(std::abs(correct - half_rows), 1e-3);

    const auto codebook = tq_paper_reference::BuildLloydMaxCodebook(15, 3);
    ASSERT_FALSE(codebook.boundaries.empty());
    ASSERT_EQ(codebook.centroids.size(), codebook.boundaries.size() + 1);
    EXPECT_GT(std::abs(codebook.centroids[2] - codebook.boundaries[2]), 1e-3);
    EXPECT_NE(tq_paper_reference::StorageBytes(15, 4),
              tq_paper_reference::PackedBytes(15 / 2, 4) + sizeof(float) * (15 / 2));
}

// Kept out of the normal unit target: it exercises the specified production dimension with 64
// independent dense d-by-d sketches and is intended for the statistical-conformance job.
TEST(TQIndependentPaperReferenceSlowTest, DISABLED_qjl_unbiasedness_at_dimension_1024) {
    const auto summary = residualBias(1024, 64, 1.0, 1.0);
    EXPECT_LE(std::abs(summary.mean), 1.96 * summary.standard_error + 1e-12);
}

} // namespace
