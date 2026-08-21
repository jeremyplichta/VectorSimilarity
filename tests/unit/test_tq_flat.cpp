/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#include "gtest/gtest.h"

#include "VecSim/algorithms/hnsw/hnsw_serializer.h"
#include "VecSim/algorithms/tq/tq_flat.h"
#include "VecSim/vec_sim.h"
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
#include <utility>
#include <vector>

namespace {

VecSimParams CreateTQParams(size_t dim, VecSimMetric metric, size_t seed = 7,
                            bool use_rotation = true, size_t bits = 8) {
    TQFlatParams tq_params = {.type = VecSimType_FLOAT32,
                              .dim = dim,
                              .metric = metric,
                              .multi = false,
                              .initialCapacity = 0,
                              .blockSize = 4,
                              .bits = bits,
                              .projections = dim,
                              .seed = seed,
                              .useRotation = use_rotation};
    return VecSimParams{.algo = VecSimAlgo_TQ, .algoParams = {.tqFlatParams = tq_params}};
}

VecSimParams CreateTQHNSWParams(size_t dim, VecSimMetric metric, size_t seed = 7,
                                bool use_rotation = true, size_t bits = 8, size_t m = 16,
                                size_t ef_construction = 200, size_t ef_runtime = 50) {
    TQHNSWParams tq_params = {
        .type = VecSimType_FLOAT32,
        .dim = dim,
        .metric = metric,
        .multi = false,
        .initialCapacity = 0,
        .blockSize = 4,
        .bits = bits,
        .projections = dim,
        .seed = seed,
        .useRotation = use_rotation,
        .M = m,
        .efConstruction = ef_construction,
        .efRuntime = ef_runtime,
        .epsilon = 0.01,
    };
    return VecSimParams{.algo = VecSimAlgo_TQ_HNSW, .algoParams = {.tqHnswParams = tq_params}};
}

std::vector<std::pair<size_t, double>> TopK(VecSimIndex *index, const float *query, size_t k) {
    auto *reply = VecSimIndex_TopKQuery(index, query, k, nullptr, BY_SCORE);
    auto *iterator = VecSimQueryReply_GetIterator(reply);
    std::vector<std::pair<size_t, double>> results;
    for (size_t i = 0; i < VecSimQueryReply_Len(reply); ++i) {
        auto *result = VecSimQueryReply_IteratorNext(iterator);
        results.emplace_back(VecSimQueryResult_GetId(result), VecSimQueryResult_GetScore(result));
    }
    VecSimQueryReply_IteratorFree(iterator);
    VecSimQueryReply_Free(reply);
    return results;
}

template <VecSimMetric Metric>
struct EncodedPair {
    std::shared_ptr<VecSimAllocator> allocator;
    std::shared_ptr<TQFlatDetails::TQModelState> state;
    TQFlatDetails::TQPreprocessor<Metric> preprocessor;
    void *storage{nullptr};
    void *query{nullptr};

    EncodedPair(size_t dim, size_t bits, size_t seed, bool use_rotation, const float *vector,
                const float *query_vector)
        : allocator(VecSimAllocator::newVecsimAllocator()),
          state(std::make_shared<TQFlatDetails::TQModelState>(dim, bits, dim, seed, use_rotation)),
          preprocessor(allocator, state) {
        size_t storage_size = dim * sizeof(float);
        preprocessor.preprocessForStorage(vector, storage, storage_size, 0);
        size_t query_size = dim * sizeof(float);
        preprocessor.preprocessQuery(query_vector, query, query_size, 0);
    }

    ~EncodedPair() {
        allocator->free_allocation(storage);
        allocator->free_allocation(query);
    }
};

TEST(TQPaperReferenceTest, exact_one_bit_codebook_is_symmetric) {
    const auto [negative, positive] = tq_paper_reference::ExactOneBitCentroids(8);
    EXPECT_LT(negative, 0.0f);
    EXPECT_GT(positive, 0.0f);
    EXPECT_FLOAT_EQ(negative, -positive);
    EXPECT_FLOAT_EQ((negative + positive) / 2.0f, 0.0f);
}

TEST(TQPaperReferenceTest, qjl_correction_owns_residual_magnitude) {
    const std::vector<float> projected_query = {1.0f, -2.0f, 0.5f, 3.0f};
    const std::vector<int8_t> residual_signs = {1, -1, -1, 1};
    const float unit = tq_paper_reference::QjlCorrection(1.0f, projected_query, residual_signs);
    const float scaled = tq_paper_reference::QjlCorrection(0.125f, projected_query, residual_signs);
    EXPECT_FLOAT_EQ(scaled, unit * 0.125f);
}

TEST(TQPaperConformanceTest, stored_payload_matches_advertised_total_bit_budget) {
    EXPECT_EQ(tq_paper_reference::StorageBytes(1024, 2), 264);
    EXPECT_EQ(tq_paper_reference::StorageBytes(1024, 4), 520);
    EXPECT_EQ(tq_paper_reference::StorageBytes(1024, 8), 1032);

    for (size_t bits : {size_t{2}, size_t{4}, size_t{8}}) {
        auto params = CreateTQParams(1024, VecSimMetric_Cosine, 7, true, bits);
        EXPECT_EQ(
            TQFlatDetails::GetStorageDataSize<VecSimMetric_Cosine>(&params.algoParams.tqFlatParams),
            tq_paper_reference::StorageBytes(1024, bits));
    }
}

TEST(TQPaperConformanceTest, lloyd_max_codebook_uses_exact_sphere_density) {
    TQFlatDetails::TQModelState state(8, 2, 8, 7, false);
    const auto [expected_negative, expected_positive] = tq_paper_reference::ExactOneBitCentroids(8);
    ASSERT_EQ(state.centroids.size(), 2);
    ASSERT_EQ(state.boundaries.size(), 1);
    EXPECT_NEAR(state.centroids[0], expected_negative, 2e-5f);
    EXPECT_NEAR(state.centroids[1], expected_positive, 2e-5f);
    EXPECT_NEAR(state.boundaries[0], 0.0f, 1e-7f);
}

TEST(TQPaperConformanceTest, codebooks_are_symmetric_ordered_and_deterministic) {
    for (size_t bits : {size_t{2}, size_t{4}, size_t{8}}) {
        TQFlatDetails::TQModelState lhs(16, bits, 16, 3, false);
        TQFlatDetails::TQModelState rhs(16, bits, 16, 99, false);
        EXPECT_EQ(lhs.centroids, rhs.centroids);
        EXPECT_EQ(lhs.boundaries, rhs.boundaries);
        EXPECT_TRUE(std::is_sorted(lhs.centroids.begin(), lhs.centroids.end()));
        EXPECT_TRUE(std::is_sorted(lhs.boundaries.begin(), lhs.boundaries.end()));
        for (size_t i = 0; i < lhs.centroids.size(); ++i) {
            EXPECT_NEAR(lhs.centroids[i], -lhs.centroids[lhs.centroids.size() - 1 - i], 1e-7f);
        }
        for (size_t i = 0; i < lhs.boundaries.size(); ++i) {
            EXPECT_NEAR(lhs.boundaries[i], -lhs.boundaries[lhs.boundaries.size() - 1 - i], 1e-7f);
        }
    }
}

TEST(TQPaperConformanceTest, packed_layout_round_trips_indices_and_clears_tail_bits) {
    constexpr size_t dim = 10;
    const std::array<float, dim> vector = {0.8f,  -0.4f, 0.2f,  -0.1f, 0.05f,
                                           -0.7f, 0.3f,  -0.2f, 0.11f, -0.6f};
    for (size_t bits : {size_t{2}, size_t{4}, size_t{8}}) {
        EncodedPair<VecSimMetric_IP> encoded(dim, bits, 5, false, vector.data(), vector.data());
        const auto storage = encoded.state->storageView(encoded.storage);
        for (size_t i = 0; i < dim; ++i) {
            EXPECT_LT(encoded.state->mseIndexAt(storage, i), encoded.state->levels);
        }

        const size_t used_index_bits = dim * (bits - 1);
        const size_t unused_index_bits = encoded.state->packedIndexBytes() * 8 - used_index_bits;
        if (unused_index_bits != 0) {
            const uint8_t tail = storage.mse_indices[encoded.state->packedIndexBytes() - 1];
            const uint8_t unused_mask = static_cast<uint8_t>(0xFFu << (8 - unused_index_bits));
            EXPECT_EQ(tail & unused_mask, 0);
        }
        const size_t unused_sign_bits = encoded.state->packedQjlBytes() * 8 - dim;
        if (unused_sign_bits != 0) {
            const uint8_t tail = storage.residual_signs[encoded.state->packedQjlBytes() - 1];
            const uint8_t unused_mask = static_cast<uint8_t>(0xFFu << (8 - unused_sign_bits));
            EXPECT_EQ(tail & unused_mask, 0);
        }
    }
}

TEST(TQPaperConformanceTest, zero_vector_has_exact_zero_score_and_metadata) {
    constexpr size_t dim = 8;
    const std::array<float, dim> zero = {};
    const std::array<float, dim> query = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    EncodedPair<VecSimMetric_IP> encoded(dim, 4, 9, true, zero.data(), query.data());
    const auto storage = encoded.state->storageView(encoded.storage);
    const auto query_view = encoded.state->queryView(encoded.query);
    EXPECT_FLOAT_EQ(storage.source_scale, 0.0f);
    EXPECT_FLOAT_EQ(storage.residual_norm, 0.0f);
    EXPECT_FLOAT_EQ(encoded.state->estimateInnerProduct(storage, query_view), 0.0f);
}

TEST(TQPaperConformanceTest, zero_residual_has_positive_signs_and_clear_tail_bits) {
    constexpr size_t dim = 10;
    TQFlatDetails::TQModelState state(dim, 4, dim, 9, false);
    const std::array<float, dim> zero_residual = {};
    std::array<uint8_t, 2> signs = {0xFF, 0xFF};
    state.packResidualSigns(zero_residual.data(), signs.data());

    EXPECT_EQ(signs[0], 0xFF);
    EXPECT_EQ(signs[1], 0x03);
}

TEST(TQPaperConformanceTest, asymmetric_estimator_matches_algorithm_two_equation) {
    constexpr size_t dim = 8;
    const std::array<float, dim> vector = {1.2f, -0.4f, 0.2f, 0.7f, 0.1f, 0.3f, -0.8f, 0.5f};
    const std::array<float, dim> query = {-0.3f, 0.4f, 0.9f, -0.1f, 0.7f, 0.2f, 0.5f, -0.6f};
    EncodedPair<VecSimMetric_IP> encoded(dim, 4, 17, true, vector.data(), query.data());
    const auto storage = encoded.state->storageView(encoded.storage);
    const auto query_view = encoded.state->queryView(encoded.query);

    float coarse = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        coarse +=
            query_view.rotated[i] * encoded.state->centroids[encoded.state->mseIndexAt(storage, i)];
    }
    std::vector<int8_t> signs(dim);
    for (size_t i = 0; i < dim; ++i) {
        signs[i] = encoded.state->residualSignAt(storage, i) ? int8_t{1} : int8_t{-1};
    }
    const float residual = tq_paper_reference::QjlCorrection(
        storage.residual_norm, std::span<const float>(query_view.qjl_projection, dim), signs);
    const float expected = storage.source_scale * (coarse + residual);
    EXPECT_NEAR(encoded.state->estimateInnerProduct(storage, query_view), expected, 1e-6f);
}

TEST(TQPaperConformanceTest, qjl_estimator_is_statistically_unbiased_across_scales_and_signs) {
    constexpr size_t dim = 8;
    const std::array<float, dim> unit_storage = {0.72f, -0.31f, 0.18f, 0.41f,
                                                 -0.2f, 0.32f,  0.11f, -0.14f};
    const std::array<float, dim> query = {0.4f, -0.2f, 0.7f, 0.1f, -0.5f, 0.3f, 0.25f, -0.6f};

    for (float storage_scale : {0.25f, 1.0f, 3.0f}) {
        for (float query_sign : {-1.0f, 1.0f}) {
            std::array<float, dim> storage{};
            std::array<float, dim> signed_query{};
            for (size_t i = 0; i < dim; ++i) {
                storage[i] = storage_scale * unit_storage[i];
                signed_query[i] = query_sign * query[i];
            }
            const float exact =
                TQFlatDetails::DotProductScalar(storage.data(), signed_query.data(), dim);
            double estimate_sum = 0.0;
            constexpr size_t samples = 192;
            for (size_t seed = 1; seed <= samples; ++seed) {
                EncodedPair<VecSimMetric_IP> encoded(dim, 2, seed, false, storage.data(),
                                                     signed_query.data());
                estimate_sum +=
                    encoded.state->estimateInnerProduct(encoded.state->storageView(encoded.storage),
                                                        encoded.state->queryView(encoded.query));
            }
            EXPECT_NEAR(static_cast<float>(estimate_sum / samples), exact,
                        0.12f * std::max(1.0f, std::abs(exact)));
        }
    }
}

TEST(TQPaperConformanceTest, stored_to_stored_distance_matches_explicit_algorithm_two_decode) {
    constexpr size_t dim = 8;
    const std::array<float, dim> lhs = {1.0f, 0.2f, -0.3f, 0.4f, 0.1f, -0.2f, 0.6f, 0.7f};
    const std::array<float, dim> rhs = {-0.4f, 0.3f, 0.2f, 0.1f, 0.9f, -0.8f, 0.5f, 0.2f};
    auto allocator = VecSimAllocator::newVecsimAllocator();
    auto state = std::make_shared<TQFlatDetails::TQModelState>(dim, 4, dim, 23, true);
    TQFlatDetails::TQPreprocessor<VecSimMetric_IP> preprocessor(allocator, state);
    TQFlatDetails::TQDistanceCalculator<VecSimMetric_IP> calculator(allocator, state);

    void *lhs_blob = nullptr;
    void *rhs_blob = nullptr;
    size_t lhs_size = dim * sizeof(float);
    size_t rhs_size = dim * sizeof(float);
    preprocessor.preprocessForStorage(lhs.data(), lhs_blob, lhs_size, 0);
    preprocessor.preprocessForStorage(rhs.data(), rhs_blob, rhs_size, 0);

    std::vector<float> decoded_lhs(dim);
    std::vector<float> decoded_rhs(dim);
    state->decode(state->storageView(lhs_blob), decoded_lhs.data());
    state->decode(state->storageView(rhs_blob), decoded_rhs.data());
    const float expected =
        1.0f - TQFlatDetails::DotProductScalar(decoded_lhs.data(), decoded_rhs.data(), dim);
    EXPECT_NEAR(calculator.calcDistance(lhs_blob, rhs_blob, dim), expected, 1e-6f);
    allocator->free_allocation(lhs_blob);
    allocator->free_allocation(rhs_blob);
}

TEST(TQPaperConformanceTest, simd_matches_scalar_for_bit_widths_tails_and_unaligned_storage) {
    for (size_t dim :
         {size_t{2}, size_t{3}, size_t{7}, size_t{8}, size_t{15}, size_t{16}, size_t{31}}) {
        for (size_t bits : {size_t{2}, size_t{4}, size_t{8}}) {
            for (size_t seed = 1; seed <= 12; ++seed) {
                SCOPED_TRACE(::testing::Message()
                             << "dim=" << dim << " bits=" << bits << " seed=" << seed);
                std::vector<float> vector(dim);
                std::vector<float> query(dim);
                for (size_t i = 0; i < dim; ++i) {
                    vector[i] = std::sin(static_cast<float>((i + 1) * (seed + 3)) * 0.271f) *
                                static_cast<float>(seed + 1);
                    query[i] = std::cos(static_cast<float>((i + 2) * (seed + 5)) * 0.193f);
                }

                EncodedPair<VecSimMetric_IP> encoded(dim, bits, seed, true, vector.data(),
                                                     query.data());
                std::vector<uint8_t> unaligned(encoded.state->storageBlobSize() + 1);
                std::memcpy(unaligned.data() + 1, encoded.storage,
                            encoded.state->storageBlobSize());
                const auto storage = encoded.state->storageView(unaligned.data() + 1);
                const auto query_view = encoded.state->queryView(encoded.query);
                const float scalar = encoded.state->estimateInnerProductScalar(storage, query_view);
                const float simd = encoded.state->estimateInnerProduct(storage, query_view);
                EXPECT_NEAR(simd, scalar, 2e-5f * std::max(std::abs(scalar), 1.0f));
            }
        }
    }
}

TEST(TQFlatTest, non_unit_inner_product_vectors_preserve_magnitude) {
    constexpr size_t dim = 8;
    auto params = CreateTQParams(dim, VecSimMetric_IP, 31, true, 8);
    std::unique_ptr<VecSimIndex, decltype(&VecSimIndex_Free)> index(VecSimIndex_New(&params),
                                                                    VecSimIndex_Free);
    std::array<float, dim> query = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    auto small = query;
    auto large = query;
    small[0] = 0.5f;
    large[0] = 3.0f;
    ASSERT_EQ(VecSimIndex_AddVector(index.get(), small.data(), 1), 1);
    ASSERT_EQ(VecSimIndex_AddVector(index.get(), large.data(), 2), 1);
    const auto results = TopK(index.get(), query.data(), 2);
    ASSERT_EQ(results.size(), 2);
    EXPECT_EQ(results[0].first, 2);
    EXPECT_EQ(results[1].first, 1);
}

TEST(TQFlatTest, cosine_search_prefers_exact_direction) {
    constexpr size_t dim = 8;
    auto params = CreateTQParams(dim, VecSimMetric_Cosine, 37, true, 8);
    std::unique_ptr<VecSimIndex, decltype(&VecSimIndex_Free)> index(VecSimIndex_New(&params),
                                                                    VecSimIndex_Free);
    std::array<float, dim> query = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::array<float, dim> orthogonal = {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::array<float, dim> opposite = {-1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    ASSERT_EQ(VecSimIndex_AddVector(index.get(), orthogonal.data(), 2), 1);
    ASSERT_EQ(VecSimIndex_AddVector(index.get(), opposite.data(), 3), 1);
    ASSERT_EQ(VecSimIndex_AddVector(index.get(), query.data(), 1), 1);
    const auto results = TopK(index.get(), query.data(), 3);
    ASSERT_EQ(results.size(), 3);
    EXPECT_EQ(results[0].first, 1);
    EXPECT_EQ(results[2].first, 3);

    auto *range_reply = VecSimIndex_RangeQuery(index.get(), query.data(), 0.5, nullptr, BY_SCORE);
    ASSERT_EQ(VecSimQueryReply_Len(range_reply), 1);
    auto *range_iterator = VecSimQueryReply_GetIterator(range_reply);
    EXPECT_EQ(VecSimQueryResult_GetId(VecSimQueryReply_IteratorNext(range_iterator)), 1);
    VecSimQueryReply_IteratorFree(range_iterator);
    VecSimQueryReply_Free(range_reply);

    auto *batch_iterator = VecSimBatchIterator_New(index.get(), query.data(), nullptr);
    auto *batch_reply = VecSimBatchIterator_Next(batch_iterator, 3, BY_SCORE);
    ASSERT_EQ(VecSimQueryReply_Len(batch_reply), 3);
    auto *batch_reply_iterator = VecSimQueryReply_GetIterator(batch_reply);
    EXPECT_EQ(VecSimQueryResult_GetId(VecSimQueryReply_IteratorNext(batch_reply_iterator)), 1);
    VecSimQueryReply_IteratorFree(batch_reply_iterator);
    VecSimQueryReply_Free(batch_reply);
    VecSimBatchIterator_Free(batch_iterator);
}

TEST(TQFlatTest, hnsw_uses_paper_asymmetric_search_and_explicit_decode_maintenance) {
    constexpr size_t dim = 8;
    auto params = CreateTQHNSWParams(dim, VecSimMetric_Cosine, 41, true, 8, 8, 40, 40);
    std::unique_ptr<VecSimIndex, decltype(&VecSimIndex_Free)> index(VecSimIndex_New(&params),
                                                                    VecSimIndex_Free);
    std::array<float, dim> query = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (size_t label = 0; label < 24; ++label) {
        std::array<float, dim> vector{};
        for (size_t i = 0; i < dim; ++i) {
            vector[i] = std::sin(static_cast<float>((label + 1) * (i + 2)) * 0.37f);
        }
        ASSERT_EQ(VecSimIndex_AddVector(index.get(), vector.data(), label + 10), 1);
    }
    ASSERT_EQ(VecSimIndex_AddVector(index.get(), query.data(), 1), 1);
    const auto results = TopK(index.get(), query.data(), 5);
    ASSERT_FALSE(results.empty());
    EXPECT_EQ(results[0].first, 1);
}

TEST(TQFlatTest, rejects_non_paper_parameters) {
    EXPECT_THROW(TQFlatDetails::TQModelState(1, 4, 1, 7, true), std::invalid_argument);
    EXPECT_THROW(TQFlatDetails::TQModelState(8, 3, 8, 7, true), std::invalid_argument);
    EXPECT_THROW(TQFlatDetails::TQModelState(8, 4, 4, 7, true), std::invalid_argument);
}

TEST(TQFlatTest, tq_hnsw_rejects_l2_and_standalone_serialization) {
    auto params = CreateTQHNSWParams(8, VecSimMetric_L2);
    EXPECT_EQ(VecSimIndex_New(&params), nullptr);

    params = CreateTQHNSWParams(8, VecSimMetric_Cosine);
    std::unique_ptr<VecSimIndex, decltype(&VecSimIndex_Free)> index(VecSimIndex_New(&params),
                                                                    VecSimIndex_Free);
    auto *serializer = dynamic_cast<HNSWSerializer *>(index.get());
    ASSERT_NE(serializer, nullptr);
    EXPECT_THROW(serializer->saveIndex("unused-tq-hnsw-index"), std::runtime_error);
}

} // namespace
