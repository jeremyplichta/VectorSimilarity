/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#include "VecSim/algorithms/tq/tq_fast_structured_rotation.h"
#include "VecSim/algorithms/tq/tq_flat.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

namespace {

using TQFlatDetails::FastStructuredRotationV1;

constexpr std::array<size_t, 13> kRequiredDimensions = {1,   2,    3,    7,    255,  256, 257,
                                                        768, 1000, 1024, 1536, 3000, 3072};

std::vector<float> MakeInput(size_t dim, uint64_t salt = 0) {
    std::vector<float> result(dim);
    for (size_t i = 0; i < dim; ++i) {
        const double coordinate = static_cast<double>(i + 1);
        result[i] = static_cast<float>(std::sin(coordinate * 0.173 + salt * 0.019) +
                                       0.5 * std::cos(coordinate * 0.071 - salt * 0.013));
    }
    return result;
}

double Dot(const std::vector<float> &lhs, const std::vector<float> &rhs) {
    double result = 0.0;
    for (size_t i = 0; i < lhs.size(); ++i) {
        result += static_cast<double>(lhs[i]) * rhs[i];
    }
    return result;
}

double MaxMagnitude(const std::vector<float> &values) {
    double result = 0.0;
    for (float value : values) {
        result = std::max(result, std::abs(static_cast<double>(value)));
    }
    return result;
}

double RoundTripTolerance(size_t dim, double scale) {
    // Six block-FWHT passes contribute at most eight butterfly levels each. Scale by sqrt(d) to
    // keep the bound explicit for the full production-dimension matrix without hiding systematic
    // dimension-specific error.
    return 128.0 * std::numeric_limits<float>::epsilon() * std::sqrt(static_cast<double>(dim)) *
           std::max(1.0, scale);
}

uint64_t StateHash(const FastStructuredRotationV1 &rotation) {
    // Canonical FNV-1a over signed bytes and little-endian uint64 permutation entries. This is
    // independent of size_t width and locks every generated sign/permutation coordinate.
    uint64_t hash = 14695981039346656037ULL;
    const auto add_byte = [&hash](uint8_t byte) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    };

    for (size_t stage = 0; stage < 3; ++stage) {
        for (size_t coordinate = 0; coordinate < rotation.dimension(); ++coordinate) {
            add_byte(static_cast<uint8_t>(rotation.signAt(stage, coordinate)));
        }
        for (size_t destination = 0; destination < rotation.dimension(); ++destination) {
            const uint64_t source = rotation.permutationAt(stage, destination);
            for (size_t shift = 0; shift < 64; shift += 8) {
                add_byte(static_cast<uint8_t>(source >> shift));
            }
        }
    }
    return hash;
}

size_t NextBlockSize(size_t remaining) {
    size_t block_size = 1;
    const size_t limit = std::min<size_t>(remaining, 256);
    while (block_size <= limit / 2) {
        block_size *= 2;
    }
    return block_size;
}

void ReferenceBlockFwht(std::vector<double> &values) {
    size_t block_offset = 0;
    while (block_offset < values.size()) {
        const size_t block_size = NextBlockSize(values.size() - block_offset);
        for (size_t half_width = 1; half_width < block_size; half_width *= 2) {
            for (size_t base = 0; base < block_size; base += 2 * half_width) {
                for (size_t offset = 0; offset < half_width; ++offset) {
                    const size_t lhs_index = block_offset + base + offset;
                    const size_t rhs_index = lhs_index + half_width;
                    const double lhs = values[lhs_index];
                    const double rhs = values[rhs_index];
                    values[lhs_index] = lhs + rhs;
                    values[rhs_index] = lhs - rhs;
                }
            }
        }
        const double normalization = 1.0 / std::sqrt(static_cast<double>(block_size));
        for (size_t i = 0; i < block_size; ++i) {
            values[block_offset + i] *= normalization;
        }
        block_offset += block_size;
    }
}

std::vector<double> ReferenceForward(const FastStructuredRotationV1 &rotation,
                                     const std::vector<double> &input) {
    std::vector<double> current = input;
    std::vector<double> next(input.size());
    for (size_t stage = 0; stage < 3; ++stage) {
        for (size_t destination = 0; destination < input.size(); ++destination) {
            const size_t source = rotation.permutationAt(stage, destination);
            next[destination] = rotation.signAt(stage, source) * current[source];
        }
        ReferenceBlockFwht(next);
        current.swap(next);
    }
    return current;
}

TEST(FastStructuredRotationV1Test, RejectsInvalidInternalConstruction) {
    auto allocator = VecSimAllocator::newVecsimAllocator();
    EXPECT_THROW(FastStructuredRotationV1(nullptr, 1, 17), std::invalid_argument);
    EXPECT_THROW(FastStructuredRotationV1(allocator, 0, 17), std::invalid_argument);
}

TEST(FastStructuredRotationV1Test, ExactGenerationGolden) {
    auto allocator = VecSimAllocator::newVecsimAllocator();
    FastStructuredRotationV1 small(allocator, 7, 17);
    constexpr std::array<std::array<int, 7>, 3> expected_signs = {
        std::array<int, 7>{1, -1, -1, 1, -1, 1, -1}, std::array<int, 7>{1, -1, -1, 1, -1, -1, 1},
        std::array<int, 7>{1, -1, 1, 1, -1, 1, 1}};
    constexpr std::array<std::array<size_t, 7>, 3> expected_permutations = {
        std::array<size_t, 7>{2, 0, 5, 6, 4, 3, 1}, std::array<size_t, 7>{2, 4, 0, 5, 3, 1, 6},
        std::array<size_t, 7>{3, 4, 5, 1, 6, 2, 0}};

    for (size_t stage = 0; stage < 3; ++stage) {
        for (size_t destination = 0; destination < small.dimension(); ++destination) {
            EXPECT_EQ(small.signAt(stage, destination), expected_signs[stage][destination]);
            EXPECT_EQ(small.permutationAt(stage, destination),
                      expected_permutations[stage][destination]);
        }
    }
    EXPECT_EQ(StateHash(small), 0xE3C3806B8A340A45ULL);

    FastStructuredRotationV1 non_power_of_two(allocator, 257, 99);
    EXPECT_EQ(StateHash(non_power_of_two), 0x3A2AEA082911A753ULL);
    FastStructuredRotationV1 production_dimension(allocator, 3072, 17);
    EXPECT_EQ(StateHash(production_dimension), 0x11A001987408802DULL);
}

TEST(FastStructuredRotationV1Test, TransformOutputGolden) {
    auto allocator = VecSimAllocator::newVecsimAllocator();
    FastStructuredRotationV1 small(allocator, 7, 17);
    std::vector<float> input = {0.25f, -0.5f, 0.75f, -1.0f, 1.25f, -1.5f, 1.75f};
    std::vector<float> output(input.size());
    std::vector<float> scratch(input.size());
    small.forwardRotate(input.data(), output.data(), scratch.data());
    constexpr std::array<float, 7> expected = {0.760723352f, -1.92493689f, -1.0410533f,
                                               1.29105341f,  1.03033006f,  -0.323223293f,
                                               -0.741116583f};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(output[i], expected[i], 2e-6f) << "coordinate " << i;
    }

    FastStructuredRotationV1 non_power_of_two(allocator, 257, 99);
    input.resize(257);
    output.resize(257);
    scratch.resize(257);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 9.0f;
    }
    non_power_of_two.forwardRotate(input.data(), output.data(), scratch.data());
    constexpr std::array<size_t, 11> coordinates = {0, 1, 2, 63, 64, 127, 128, 129, 254, 255, 256};
    constexpr std::array<float, 11> expected_non_power_of_two = {
        -0.240071639f, -0.205566406f, 0.54671222f,  0.282497764f,  -0.135796458f, 0.423231423f,
        0.136122063f,  0.0143771693f, 0.243001401f, -0.248535097f, 1.40364599f};
    for (size_t i = 0; i < coordinates.size(); ++i) {
        EXPECT_NEAR(output[coordinates[i]], expected_non_power_of_two[i], 3e-6f)
            << "coordinate " << coordinates[i];
    }
}

TEST(FastStructuredRotationV1Test, DeterministicStateUsesIndependentStageDomains) {
    for (size_t dim : kRequiredDimensions) {
        auto allocator = VecSimAllocator::newVecsimAllocator();
        FastStructuredRotationV1 lhs(allocator, dim, 31);
        FastStructuredRotationV1 rhs(allocator, dim, 31);
        EXPECT_EQ(StateHash(lhs), StateHash(rhs)) << "dimension " << dim;
        for (size_t stage = 0; stage < 3; ++stage) {
            for (size_t destination = 0; destination < dim; ++destination) {
                const size_t source = lhs.permutationAt(stage, destination);
                ASSERT_LT(source, dim);
                EXPECT_EQ(lhs.inversePermutationAt(stage, source), destination);
            }
        }

        if (dim > 1) {
            EXPECT_NE(StateHash(lhs), StateHash(FastStructuredRotationV1(allocator, dim, 32)))
                << "dimension " << dim;
            bool stage_zero_differs_from_one = false;
            bool stage_zero_differs_from_two = false;
            for (size_t coordinate = 0; coordinate < dim; ++coordinate) {
                stage_zero_differs_from_one |=
                    lhs.signAt(0, coordinate) != lhs.signAt(1, coordinate) ||
                    lhs.permutationAt(0, coordinate) != lhs.permutationAt(1, coordinate);
                stage_zero_differs_from_two |=
                    lhs.signAt(0, coordinate) != lhs.signAt(2, coordinate) ||
                    lhs.permutationAt(0, coordinate) != lhs.permutationAt(2, coordinate);
            }
            EXPECT_TRUE(stage_zero_differs_from_one) << "dimension " << dim;
            EXPECT_TRUE(stage_zero_differs_from_two) << "dimension " << dim;
        }
    }
}

TEST(FastStructuredRotationV1Test, RoundTripNormAndInnerProductProperties) {
    for (size_t dim : kRequiredDimensions) {
        auto allocator = VecSimAllocator::newVecsimAllocator();
        FastStructuredRotationV1 rotation(allocator, dim, 17);
        const auto input = MakeInput(dim, 1);
        const auto other = MakeInput(dim, 2);
        std::vector<float> rotated(dim);
        std::vector<float> rotated_other(dim);
        std::vector<float> reconstructed(dim);
        std::vector<float> scratch(dim);

        rotation.forwardRotate(input.data(), rotated.data(), scratch.data());
        rotation.forwardRotate(other.data(), rotated_other.data(), scratch.data());
        rotation.inverseRotate(rotated.data(), reconstructed.data(), scratch.data());

        const double coordinate_tolerance = RoundTripTolerance(dim, MaxMagnitude(input));
        for (size_t i = 0; i < dim; ++i) {
            EXPECT_NEAR(reconstructed[i], input[i], coordinate_tolerance)
                << "dimension " << dim << ", coordinate " << i;
        }

        const double input_norm_sq = Dot(input, input);
        const double rotated_norm_sq = Dot(rotated, rotated);
        const double norm_tolerance = RoundTripTolerance(dim, input_norm_sq);
        EXPECT_NEAR(rotated_norm_sq, input_norm_sq, norm_tolerance) << "dimension " << dim;

        const double input_dot = Dot(input, other);
        const double rotated_dot = Dot(rotated, rotated_other);
        const double dot_tolerance =
            RoundTripTolerance(dim, std::sqrt(Dot(input, input) * Dot(other, other)));
        EXPECT_NEAR(rotated_dot, input_dot, dot_tolerance) << "dimension " << dim;

        auto in_place = input;
        rotation.forwardRotateInPlace(in_place.data(), scratch.data());
        EXPECT_EQ(std::memcmp(in_place.data(), rotated.data(), dim * sizeof(float)), 0)
            << "dimension " << dim;
        rotation.inverseRotateInPlace(in_place.data(), scratch.data());
        for (size_t i = 0; i < dim; ++i) {
            EXPECT_NEAR(in_place[i], input[i], coordinate_tolerance)
                << "in-place dimension " << dim << ", coordinate " << i;
        }
    }
}

TEST(FastStructuredRotationV1Test, MatchesIndependentFp64Transform) {
    for (size_t dim : kRequiredDimensions) {
        auto allocator = VecSimAllocator::newVecsimAllocator();
        FastStructuredRotationV1 rotation(allocator, dim, 7);
        const auto input = MakeInput(dim, 3);
        std::vector<double> double_input(input.begin(), input.end());
        const auto reference = ReferenceForward(rotation, double_input);
        std::vector<float> output(dim);
        std::vector<float> scratch(dim);
        rotation.forwardRotate(input.data(), output.data(), scratch.data());

        const double tolerance = RoundTripTolerance(dim, MaxMagnitude(input));
        for (size_t coordinate = 0; coordinate < dim; ++coordinate) {
            EXPECT_NEAR(output[coordinate], reference[coordinate], tolerance)
                << "dimension " << dim << ", coordinate " << coordinate;
        }
    }
}

TEST(FastStructuredRotationV1Test, ExplicitFp64MatricesAreOrthogonal) {
    for (size_t dim : {size_t{1}, size_t{2}, size_t{3}, size_t{7}}) {
        auto allocator = VecSimAllocator::newVecsimAllocator();
        FastStructuredRotationV1 rotation(allocator, dim, 99);
        std::vector<std::vector<double>> columns(dim, std::vector<double>(dim));
        for (size_t column = 0; column < dim; ++column) {
            std::vector<double> basis(dim);
            basis[column] = 1.0;
            const auto rotated = ReferenceForward(rotation, basis);
            for (size_t row = 0; row < dim; ++row) {
                columns[column][row] = rotated[row];
            }
        }

        for (size_t lhs = 0; lhs < dim; ++lhs) {
            for (size_t rhs = 0; rhs < dim; ++rhs) {
                const double inner_product = std::inner_product(
                    columns[lhs].begin(), columns[lhs].end(), columns[rhs].begin(), 0.0);
                EXPECT_NEAR(inner_product, lhs == rhs ? 1.0 : 0.0, 2e-12)
                    << "dimension " << dim << ", columns " << lhs << ", " << rhs;
            }
        }
    }
}

TEST(FastStructuredRotationV1Test, SignedZeroIsFiniteAndDeterministic) {
    for (size_t dim : {size_t{7}, size_t{257}}) {
        auto allocator = VecSimAllocator::newVecsimAllocator();
        FastStructuredRotationV1 rotation(allocator, dim, 17);
        std::vector<float> input(dim);
        for (size_t i = 0; i < dim; ++i) {
            input[i] = i % 2 == 0 ? 0.0f : -0.0f;
        }
        std::vector<float> first(dim);
        std::vector<float> second(dim);
        std::vector<float> scratch(dim);
        rotation.forwardRotate(input.data(), first.data(), scratch.data());
        rotation.forwardRotate(input.data(), second.data(), scratch.data());
        EXPECT_EQ(std::memcmp(first.data(), second.data(), dim * sizeof(float)), 0);
        for (float value : first) {
            EXPECT_TRUE(std::isfinite(value));
        }
        rotation.inverseRotateInPlace(first.data(), scratch.data());
        for (float value : first) {
            EXPECT_TRUE(std::isfinite(value));
            EXPECT_EQ(value, 0.0f);
        }
    }
}

TEST(FastStructuredRotationV1Test, PersistentStateIsLinearAndCallsDoNotAllocate) {
    for (size_t dim : kRequiredDimensions) {
        auto allocator = VecSimAllocator::newVecsimAllocator();
        const uint64_t baseline_bytes = allocator->getAllocationSize();
        {
            FastStructuredRotationV1 rotation(allocator, dim, 17);
            const size_t payload_bytes = 3 * dim * (sizeof(int8_t) + 2 * sizeof(size_t));
            const size_t allocation_overhead = 9 * VecSimAllocator::getAllocationOverheadSize();
            EXPECT_EQ(allocator->getAllocationSize() - baseline_bytes,
                      payload_bytes + allocation_overhead)
                << "dimension " << dim;

            std::vector<float, VecsimSTLAllocator<float>> output(
                dim, 0.0f, VecsimSTLAllocator<float>(allocator));
            std::vector<float, VecsimSTLAllocator<float>> scratch(
                dim, 0.0f, VecsimSTLAllocator<float>(allocator));
            const auto input = MakeInput(dim);
            const uint64_t allocation_count = allocator->getAllocationCount();
            rotation.forwardRotate(input.data(), output.data(), scratch.data());
            rotation.inverseRotateInPlace(output.data(), scratch.data());
            EXPECT_EQ(allocator->getAllocationCount(), allocation_count) << "dimension " << dim;
        }
        EXPECT_EQ(allocator->getAllocationSize(), baseline_bytes) << "dimension " << dim;
    }
}

TEST(FastStructuredRotationV1ModelTest, ProfileIdentityIsDistinctAndDenseQjlIsUnchanged) {
    constexpr size_t dim = 7;
    constexpr uint64_t seed = 17;
    auto allocator = VecSimAllocator::newVecsimAllocator();
    auto fast =
        TQFlatDetails::AllocateFastStructuredRotationTQModelState(allocator, dim, 4, dim, seed);
    auto dense =
        TQFlatDetails::AllocateDenseReferenceTQModelState(allocator, dim, 4, dim, seed, true);

    EXPECT_EQ(fast->modelTransformVersion(),
              TQFlatDetails::TQModelTransformVersion::FastStructuredRotationV1);
    EXPECT_EQ(fast->config.rotation_backend_version,
              TQFlatDetails::TQRotationBackendVersion::FastStructuredV1);
    EXPECT_EQ(fast->config.qjl_backend_version,
              TQFlatDetails::TQQjlBackendVersion::DenseGaussianV1);
    EXPECT_FALSE(fast->modelIdentity() == dense->modelIdentity());
    EXPECT_EQ(fast->storageBlobSize(), dense->storageBlobSize());
    EXPECT_EQ(fast->queryBlobSize(), dense->queryBlobSize());

    const auto input = MakeInput(dim, 11);
    std::vector<float> fast_rotation(dim);
    std::vector<float> direct_rotation(dim);
    std::vector<float> dense_rotation(dim);
    std::vector<float> fast_qjl(dim);
    std::vector<float> dense_qjl(dim);
    std::vector<float> scratch(dim);
    fast->applyRotation(input.data(), fast_rotation.data(), scratch.data());
    dense->applyRotation(input.data(), dense_rotation.data(), scratch.data());
    FastStructuredRotationV1 direct(allocator, dim, seed);
    direct.forwardRotate(input.data(), direct_rotation.data(), scratch.data());
    EXPECT_EQ(std::memcmp(fast_rotation.data(), direct_rotation.data(), dim * sizeof(float)), 0);
    EXPECT_NE(std::memcmp(fast_rotation.data(), dense_rotation.data(), dim * sizeof(float)), 0);

    fast->projectQjl(input.data(), fast_qjl.data(), scratch.data());
    dense->projectQjl(input.data(), dense_qjl.data(), scratch.data());
    EXPECT_EQ(std::memcmp(fast_qjl.data(), dense_qjl.data(), dim * sizeof(float)), 0);
}

TEST(FastStructuredRotationV1ModelTest, VersionValidationRejectsMismatchedOrFutureProfiles) {
    auto fast = TQFlatDetails::TQCodecConfig::FastStructuredRotation(8, 4, 8, 17);
    EXPECT_NO_THROW(TQFlatDetails::ValidateTQCodecConfig(fast));

    auto mismatched_rotation = fast;
    mismatched_rotation.rotation_backend_version =
        TQFlatDetails::TQRotationBackendVersion::DenseHaarV1;
    EXPECT_THROW(TQFlatDetails::ValidateTQCodecConfig(mismatched_rotation), std::invalid_argument);

    auto mismatched_qjl = fast;
    mismatched_qjl.qjl_backend_version = TQFlatDetails::TQQjlBackendVersion::CirculantGaussianV1;
    EXPECT_THROW(TQFlatDetails::ValidateTQCodecConfig(mismatched_qjl), std::invalid_argument);

    auto disabled_rotation = fast;
    disabled_rotation.use_rotation = false;
    EXPECT_THROW(TQFlatDetails::ValidateTQCodecConfig(disabled_rotation), std::invalid_argument);

    auto full_profile = TQFlatDetails::TQCodecConfig::FastStructured(8, 4, 8, 17);
    EXPECT_NO_THROW(TQFlatDetails::ValidateTQCodecConfig(full_profile));
}

TEST(FastStructuredRotationV1ModelTest, AllocatorEstimateIncludesOnlySelectedRotationState) {
    constexpr size_t dim = 1024;
    auto allocator = VecSimAllocator::newVecsimAllocator();
    const uint64_t baseline = allocator->getAllocationSize();
    auto model =
        TQFlatDetails::AllocateFastStructuredRotationTQModelState(allocator, dim, 4, dim, 17);
    const size_t actual = allocator->getAllocationSize() - baseline;
    const size_t fast_estimate =
        TQFlatDetails::EstimateFastStructuredRotationTQModelAllocationSize(dim, 4, dim, 17);
    const size_t dense_estimate =
        TQFlatDetails::EstimateDenseReferenceTQModelAllocationSize(dim, 4, dim, 17);
    EXPECT_EQ(actual, fast_estimate);
    EXPECT_LT(fast_estimate, dense_estimate);

    const size_t overhead = VecSimAllocator::getAllocationOverheadSize();
    const size_t dense_rotation_bytes = 2 * (dim * dim * sizeof(float) + overhead);
    const size_t structured_rotation_bytes =
        3 * (dim * sizeof(int8_t) + overhead + 2 * (dim * sizeof(size_t) + overhead));
    EXPECT_EQ(dense_estimate - fast_estimate, dense_rotation_bytes - structured_rotation_bytes);

    model.reset();
    EXPECT_EQ(allocator->getAllocationSize(), baseline);
}

TEST(FastStructuredRotationV1ModelTest, CodecRoundTripIsDeterministicForAllBitWidths) {
    constexpr size_t dim = 7;
    const auto source = MakeInput(dim, 21);
    const auto query = MakeInput(dim, 22);
    for (size_t bits : {size_t{2}, size_t{4}, size_t{8}}) {
        auto allocator = VecSimAllocator::newVecsimAllocator();
        auto first = TQFlatDetails::AllocateFastStructuredRotationTQModelState(allocator, dim, bits,
                                                                               dim, 31);
        auto second = TQFlatDetails::AllocateFastStructuredRotationTQModelState(allocator, dim,
                                                                                bits, dim, 31);
        EXPECT_EQ(first->modelIdentity(), second->modelIdentity());
        TQFlatDetails::TQPreprocessor<VecSimMetric_IP> first_preprocessor(allocator, first);
        TQFlatDetails::TQPreprocessor<VecSimMetric_IP> second_preprocessor(allocator, second);

        void *first_storage = nullptr;
        void *second_storage = nullptr;
        void *first_query = nullptr;
        void *second_query = nullptr;
        size_t first_storage_size = dim * sizeof(float);
        size_t second_storage_size = dim * sizeof(float);
        size_t first_query_size = dim * sizeof(float);
        size_t second_query_size = dim * sizeof(float);
        first_preprocessor.preprocessForStorage(source.data(), first_storage, first_storage_size,
                                                0);
        second_preprocessor.preprocessForStorage(source.data(), second_storage, second_storage_size,
                                                 0);
        first_preprocessor.preprocessQuery(query.data(), first_query, first_query_size, 0);
        second_preprocessor.preprocessQuery(query.data(), second_query, second_query_size, 0);

        EXPECT_EQ(first_storage_size, first->storageBlobSize());
        EXPECT_EQ(first_query_size, first->queryBlobSize());
        EXPECT_EQ(first_storage_size, second_storage_size);
        EXPECT_EQ(first_query_size, second_query_size);
        EXPECT_EQ(std::memcmp(first_storage, second_storage, first_storage_size), 0);
        EXPECT_EQ(std::memcmp(first_query, second_query, first_query_size), 0);
        TQFlatDetails::TQDistanceCalculator<VecSimMetric_IP> calculator(allocator, first);
        EXPECT_TRUE(
            std::isfinite(calculator.calcDistanceForQuery(first_storage, first_query, first->dim)));

        allocator->free_allocation(first_storage);
        allocator->free_allocation(second_storage);
        allocator->free_allocation(first_query);
        allocator->free_allocation(second_query);
    }
}

} // namespace
