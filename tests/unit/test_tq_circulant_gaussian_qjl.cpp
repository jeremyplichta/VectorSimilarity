/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#include "VecSim/algorithms/tq/tq_circulant_gaussian_qjl.h"
#include "VecSim/algorithms/tq/tq_flat.h"
#include "VecSim/algorithms/tq/tq_model.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using TQFlatDetails::CirculantGaussianQjlV1;
using TQFlatDetails::DenseGaussianQjlV1State;
using TQFlatDetails::TQCodecConfig;

constexpr std::array<size_t, 10> kRequiredDimensions = {7,    255,  256,  257,  768,
                                                        1000, 1024, 1536, 3000, 3072};
constexpr double kSqrtPiOverTwo = 1.2533141373155002512;

uint64_t SplitMixFinalizer(uint64_t value) {
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

std::vector<float> MakeInput(size_t dim, uint64_t salt = 0, float scale = 1.0f) {
    std::vector<float> result(dim);
    for (size_t coordinate = 0; coordinate < dim; ++coordinate) {
        const double index = static_cast<double>(coordinate + 1);
        result[coordinate] =
            scale * static_cast<float>(std::sin(index * 0.173 + salt * 0.019) +
                                       0.5 * std::cos(index * 0.071 - salt * 0.013));
    }
    return result;
}

std::vector<float> MakeGaussianInput(size_t dim, uint64_t seed) {
    constexpr uint64_t kIncrement = 0x9E3779B97F4A7C15ULL;
    const auto mix = [](uint64_t value) {
        value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
        value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
        return value ^ (value >> 31);
    };
    const auto next = [&seed, &mix]() {
        seed += kIncrement;
        return mix(seed);
    };
    const auto uniform_open = [&next]() {
        constexpr double kInv53 = 1.0 / static_cast<double>(uint64_t{1} << 53);
        return (static_cast<double>(next() >> 11) + 0.5) * kInv53;
    };

    std::vector<float> result(dim);
    size_t coordinate = 0;
    while (coordinate < dim) {
        const double radius = std::sqrt(-2.0 * std::log(uniform_open()));
        const double theta = 2.0 * std::acos(-1.0) * uniform_open();
        result[coordinate++] = static_cast<float>(radius * std::cos(theta));
        if (coordinate < dim) {
            result[coordinate++] = static_cast<float>(radius * std::sin(theta));
        }
    }
    return result;
}

double Dot(const float *lhs, const float *rhs, size_t dim) {
    double result = 0.0;
    for (size_t coordinate = 0; coordinate < dim; ++coordinate) {
        result += static_cast<double>(lhs[coordinate]) * rhs[coordinate];
    }
    return result;
}

double Norm(const std::vector<float> &values) {
    return std::sqrt(Dot(values.data(), values.data(), values.size()));
}

void Normalize(std::vector<float> &values) {
    const double norm = Norm(values);
    ASSERT_GT(norm, 0.0);
    for (float &value : values) {
        value = static_cast<float>(static_cast<double>(value) / norm);
    }
}

double MaxMagnitude(const float *values, size_t dim) {
    double result = 0.0;
    for (size_t coordinate = 0; coordinate < dim; ++coordinate) {
        result = std::max(result, std::abs(static_cast<double>(values[coordinate])));
    }
    return result;
}

double KernelTolerance(size_t dim, const float *input) {
    const double scale = MaxMagnitude(input, dim);
    return std::max(1e-37, 32.0 * std::numeric_limits<float>::epsilon() *
                               std::sqrt(static_cast<double>(dim)) * scale);
}

std::vector<double> DirectForward(const CirculantGaussianQjlV1 &qjl, const float *input) {
    const size_t dim = qjl.dimension();
    std::vector<double> output(dim);
    for (size_t row = 0; row < dim; ++row) {
        double value = 0.0;
        for (size_t coordinate = 0; coordinate < dim; ++coordinate) {
            const size_t generator_index = (row + dim - coordinate) % dim;
            value += static_cast<double>(qjl.gaussianAt(generator_index)) *
                     static_cast<double>(qjl.signAt(coordinate)) * input[coordinate];
        }
        output[row] = value;
    }
    return output;
}

std::vector<double> DirectAdjoint(const CirculantGaussianQjlV1 &qjl, const float *input) {
    const size_t dim = qjl.dimension();
    std::vector<double> output(dim);
    for (size_t coordinate = 0; coordinate < dim; ++coordinate) {
        double value = 0.0;
        for (size_t row = 0; row < dim; ++row) {
            const size_t generator_index = (row + dim - coordinate) % dim;
            value += static_cast<double>(qjl.gaussianAt(generator_index)) * input[row];
        }
        output[coordinate] = static_cast<double>(qjl.signAt(coordinate)) * value;
    }
    return output;
}

std::vector<double> WrongReversedForward(const CirculantGaussianQjlV1 &qjl, const float *input) {
    const size_t dim = qjl.dimension();
    std::vector<double> output(dim);
    for (size_t row = 0; row < dim; ++row) {
        for (size_t coordinate = 0; coordinate < dim; ++coordinate) {
            output[row] += static_cast<double>(qjl.gaussianAt((coordinate + dim - row) % dim)) *
                           static_cast<double>(qjl.signAt(coordinate)) * input[coordinate];
        }
    }
    return output;
}

std::vector<uint8_t> PackSigns(const float *projections, size_t count) {
    std::vector<uint8_t> packed((count + 7) / 8, uint8_t{0});
    for (size_t projection = 0; projection < count; ++projection) {
        if (projections[projection] >= 0.0f) {
            packed[projection / 8] |= static_cast<uint8_t>(uint8_t{1} << (projection % 8));
        }
    }
    return packed;
}

double Correction(const std::vector<uint8_t> &packed_signs, const float *projected_query,
                  size_t dim, double residual_norm) {
    double signed_sum = 0.0;
    for (size_t projection = 0; projection < dim; ++projection) {
        const bool positive = (packed_signs[projection / 8] &
                               static_cast<uint8_t>(uint8_t{1} << (projection % 8))) != 0;
        signed_sum += (positive ? 1.0 : -1.0) * projected_query[projection];
    }
    return residual_norm * kSqrtPiOverTwo * signed_sum / static_cast<double>(dim);
}

TEST(CirculantGaussianQjlV1Test, RejectsInvalidConstructionAndScratch) {
    auto allocator = VecSimAllocator::newVecsimAllocator();
    EXPECT_THROW(CirculantGaussianQjlV1(nullptr, 7, 17), std::invalid_argument);
    EXPECT_THROW(CirculantGaussianQjlV1(allocator, 0, 17), std::invalid_argument);
    EXPECT_THROW(CirculantGaussianQjlV1(allocator, 1, 17), std::invalid_argument);
    EXPECT_THROW(
        CirculantGaussianQjlV1(allocator, std::numeric_limits<size_t>::max() / 2 + size_t{1}, 17),
        std::overflow_error);

    CirculantGaussianQjlV1 small(allocator, 7, 17);
    CirculantGaussianQjlV1 larger(allocator, 9, 17);
    auto wrong_scratch = larger.createScratch();
    std::array<float, 7> input{};
    std::array<float, 7> output{};
    EXPECT_THROW(small.projectForward(input.data(), output.data(), wrong_scratch),
                 std::invalid_argument);
    auto scratch = small.createScratch();
    EXPECT_THROW(small.projectForward(nullptr, output.data(), scratch), std::invalid_argument);
    EXPECT_THROW(small.projectForward(input.data(), nullptr, scratch), std::invalid_argument);
    EXPECT_THROW(small.encodeSigns(input.data(), nullptr, output.data(), scratch),
                 std::invalid_argument);
    EXPECT_THROW(small.asymmetricCorrection(nullptr, nullptr, -1.0f), std::invalid_argument);
    EXPECT_THROW(
        small.asymmetricCorrection(nullptr, nullptr, std::numeric_limits<float>::quiet_NaN()),
        std::invalid_argument);
    EXPECT_FLOAT_EQ(small.asymmetricCorrection(nullptr, nullptr, 0.0f), 0.0f);
}

TEST(CirculantGaussianQjlV1Test, ExactDeterministicGenerationAndGoldenOutputs) {
    auto allocator = VecSimAllocator::newVecsimAllocator();
    CirculantGaussianQjlV1 qjl(allocator, 7, 17);
    EXPECT_EQ(qjl.dimension(), 7);
    EXPECT_EQ(qjl.projections(), 7);
    EXPECT_EQ(qjl.modelSeed(), 17);
    EXPECT_EQ(qjl.gaussianSeed(), 0xCAA337A6AB072736ULL);
    EXPECT_EQ(qjl.signSeed(), 0xC29F8BBDC5ECAD0BULL);
    EXPECT_EQ(qjl.gaussianGenerationVersion(), 1);
    EXPECT_EQ(qjl.signGenerationVersion(), 1);
    EXPECT_EQ(qjl.convolutionVersion(), 1);
    EXPECT_EQ(qjl.fftLength(), 16);

    constexpr std::array<float, 7> expected_gaussian = {
        1.031725883f,  0.1596310288f, 0.02967279218f, -0.2975956202f,
        -1.418772101f, 0.676810205f,  0.2644668221f};
    constexpr std::array<int, 7> expected_signs = {1, 1, -1, -1, -1, -1, -1};
    for (size_t coordinate = 0; coordinate < qjl.dimension(); ++coordinate) {
        EXPECT_NEAR(qjl.gaussianAt(coordinate), expected_gaussian[coordinate], 2e-7f);
        EXPECT_EQ(qjl.signAt(coordinate), expected_signs[coordinate]);
    }

    CirculantGaussianQjlV1 duplicate(allocator, 7, 17);
    CirculantGaussianQjlV1 different_seed(allocator, 7, 18);
    bool seed_changes_state = false;
    for (size_t coordinate = 0; coordinate < qjl.dimension(); ++coordinate) {
        EXPECT_EQ(qjl.gaussianAt(coordinate), duplicate.gaussianAt(coordinate));
        EXPECT_EQ(qjl.signAt(coordinate), duplicate.signAt(coordinate));
        seed_changes_state |= qjl.gaussianAt(coordinate) != different_seed.gaussianAt(coordinate) ||
                              qjl.signAt(coordinate) != different_seed.signAt(coordinate);
    }
    EXPECT_TRUE(seed_changes_state);

    std::array<float, 7> residual = {0.25f, -0.5f, 0.75f, -1.0f, 1.25f, -1.5f, 1.75f};
    std::array<float, 7> query = {-0.75f, 0.125f, 1.5f, 0.5f, -0.25f, 2.0f, -1.25f};
    std::array<float, 7> forward{};
    std::array<float, 7> adjoint{};
    std::array<float, 7> projected_query{};
    auto scratch = qjl.createScratch();
    qjl.projectForward(residual.data(), forward.data(), scratch);
    qjl.projectAdjoint(query.data(), adjoint.data(), scratch);
    qjl.projectQuery(query.data(), projected_query.data(), scratch);

    constexpr std::array<float, 7> expected_forward = {-1.663532257f, 1.277649164f,  -3.035103559f,
                                                       3.990250349f,  -2.145893812f, 2.016691208f,
                                                       -1.108969688f};
    constexpr std::array<float, 7> expected_adjoint = {0.519600749f,  -3.424259424f, -2.323710680f,
                                                       -2.452675819f, -1.217538476f, 0.05141082034f,
                                                       2.201719999f};
    constexpr std::array<float, 7> expected_query = {-0.9807710052f, -0.4482736588f, 0.9526292682f,
                                                     -3.589372396f,  1.477565527f,   -1.946357608f,
                                                     3.141020536f};
    for (size_t coordinate = 0; coordinate < qjl.dimension(); ++coordinate) {
        EXPECT_NEAR(forward[coordinate], expected_forward[coordinate], 2e-6f);
        EXPECT_NEAR(adjoint[coordinate], expected_adjoint[coordinate], 2e-6f);
        EXPECT_NEAR(projected_query[coordinate], expected_query[coordinate], 2e-6f);
    }

    std::array<uint8_t, 1> packed{};
    qjl.encodeSigns(residual.data(), packed.data(), forward.data(), scratch);
    EXPECT_EQ(packed[0], 0x2A);
    EXPECT_EQ(packed[0] & 0x80, 0);
    EXPECT_NEAR(qjl.asymmetricCorrection(packed.data(), projected_query.data(), 2.5f),
                -4.733252048f, 4e-6f);
}

TEST(CirculantGaussianQjlV1Test, DirectEquationsMatchExplicitSmallMatrix) {
    auto allocator = VecSimAllocator::newVecsimAllocator();
    CirculantGaussianQjlV1 qjl(allocator, 7, 99);
    const auto input = MakeInput(qjl.dimension(), 5);
    std::vector<double> matrix(qjl.dimension() * qjl.dimension());
    for (size_t row = 0; row < qjl.dimension(); ++row) {
        for (size_t coordinate = 0; coordinate < qjl.dimension(); ++coordinate) {
            matrix[row * qjl.dimension() + coordinate] =
                static_cast<double>(
                    qjl.gaussianAt((row + qjl.dimension() - coordinate) % qjl.dimension())) *
                qjl.signAt(coordinate);
        }
    }

    const auto forward = DirectForward(qjl, input.data());
    const auto adjoint = DirectAdjoint(qjl, input.data());
    for (size_t row = 0; row < qjl.dimension(); ++row) {
        double explicit_forward = 0.0;
        double explicit_adjoint = 0.0;
        for (size_t coordinate = 0; coordinate < qjl.dimension(); ++coordinate) {
            explicit_forward += matrix[row * qjl.dimension() + coordinate] * input[coordinate];
            explicit_adjoint += matrix[coordinate * qjl.dimension() + row] * input[coordinate];
        }
        EXPECT_DOUBLE_EQ(forward[row], explicit_forward);
        EXPECT_DOUBLE_EQ(adjoint[row], explicit_adjoint);
    }
}

TEST(CirculantGaussianQjlV1Test, FftForwardAndAdjointMatchDirectCompleteDimensionMatrix) {
    for (size_t dim : kRequiredDimensions) {
        auto allocator = VecSimAllocator::newVecsimAllocator();
        CirculantGaussianQjlV1 qjl(allocator, dim, 0x123456789ABCDEF0ULL + dim);
        const auto input = MakeInput(dim, dim);
        const auto query = MakeInput(dim, dim + 11);
        const auto direct_forward = DirectForward(qjl, input.data());
        const auto direct_adjoint = DirectAdjoint(qjl, query.data());
        std::vector<float> forward(dim);
        std::vector<float> adjoint(dim);
        auto scratch = qjl.createScratch();
        qjl.projectForward(input.data(), forward.data(), scratch);
        qjl.projectAdjoint(query.data(), adjoint.data(), scratch);

        const double forward_tolerance = KernelTolerance(dim, input.data());
        const double adjoint_tolerance = KernelTolerance(dim, query.data());
        for (size_t coordinate = 0; coordinate < dim; ++coordinate) {
            EXPECT_NEAR(forward[coordinate], direct_forward[coordinate], forward_tolerance)
                << "forward dimension " << dim << ", coordinate " << coordinate;
            EXPECT_NEAR(adjoint[coordinate], direct_adjoint[coordinate], adjoint_tolerance)
                << "adjoint dimension " << dim << ", coordinate " << coordinate;
        }

        const double lhs = Dot(forward.data(), query.data(), dim);
        const double rhs = Dot(input.data(), adjoint.data(), dim);
        const double identity_tolerance = 32.0 * std::numeric_limits<float>::epsilon() *
                                          std::max({1.0, std::abs(lhs), std::abs(rhs)});
        EXPECT_NEAR(lhs, rhs, identity_tolerance) << "adjoint identity dimension " << dim;
    }
}

TEST(CirculantGaussianQjlV1Test, QueryProjectionUsesForwardRatherThanAdjoint) {
    auto allocator = VecSimAllocator::newVecsimAllocator();
    CirculantGaussianQjlV1 qjl(allocator, 7, 17);
    const auto query = MakeInput(qjl.dimension(), 71);
    const auto direct_forward = DirectForward(qjl, query.data());
    const auto direct_adjoint = DirectAdjoint(qjl, query.data());
    std::vector<float> projected(qjl.dimension());
    auto scratch = qjl.createScratch();
    qjl.projectQuery(query.data(), projected.data(), scratch);

    double maximum_adjoint_difference = 0.0;
    for (size_t coordinate = 0; coordinate < qjl.dimension(); ++coordinate) {
        EXPECT_NEAR(projected[coordinate], direct_forward[coordinate],
                    KernelTolerance(qjl.dimension(), query.data()));
        maximum_adjoint_difference = std::max(
            maximum_adjoint_difference,
            std::abs(static_cast<double>(projected[coordinate]) - direct_adjoint[coordinate]));
    }
    EXPECT_GT(maximum_adjoint_difference, 0.25)
        << "this fixture must detect an accidental S^Tq query projection";
}

TEST(CirculantGaussianQjlV1Test, CircularIndexAndFftNormalizationAreMutationSensitive) {
    auto allocator = VecSimAllocator::newVecsimAllocator();
    CirculantGaussianQjlV1 qjl(allocator, 7, 31337);
    const auto input = MakeInput(qjl.dimension(), 9);
    const auto direct = DirectForward(qjl, input.data());
    const auto reversed = WrongReversedForward(qjl, input.data());
    std::vector<float> output(qjl.dimension());
    auto scratch = qjl.createScratch();
    qjl.projectForward(input.data(), output.data(), scratch);

    double maximum_reversed_difference = 0.0;
    for (size_t coordinate = 0; coordinate < qjl.dimension(); ++coordinate) {
        EXPECT_NEAR(output[coordinate], direct[coordinate],
                    KernelTolerance(qjl.dimension(), input.data()));
        EXPECT_NEAR(output[coordinate], direct[coordinate], 2e-6)
            << "an extra FFT or JL normalization must not be introduced";
        maximum_reversed_difference = std::max(maximum_reversed_difference,
                                               std::abs(direct[coordinate] - reversed[coordinate]));
    }
    EXPECT_GT(maximum_reversed_difference, 0.5)
        << "this fixture must detect reversing the circular index";
}

TEST(CirculantGaussianQjlV1Test, HandlesZeroTinyLargeInPlaceUnalignedAndTailInputs) {
    for (size_t dim : {size_t{7}, size_t{255}, size_t{257}, size_t{1000}, size_t{3000}}) {
        auto allocator = VecSimAllocator::newVecsimAllocator();
        CirculantGaussianQjlV1 qjl(allocator, dim, 47 + dim);
        auto scratch = qjl.createScratch();

        for (float scale : {0.0f, 1e-30f, 1.0f, 1e20f}) {
            const auto input = MakeInput(dim, 3, scale);
            const auto direct = DirectForward(qjl, input.data());
            std::vector<float> output(dim);
            qjl.projectForward(input.data(), output.data(), scratch);
            for (size_t coordinate = 0; coordinate < dim; ++coordinate) {
                EXPECT_TRUE(std::isfinite(output[coordinate]))
                    << "dimension " << dim << ", scale " << scale;
                EXPECT_NEAR(output[coordinate], direct[coordinate],
                            KernelTolerance(dim, input.data()))
                    << "dimension " << dim << ", scale " << scale << ", coordinate " << coordinate;
            }

            auto in_place = input;
            qjl.projectForward(in_place.data(), in_place.data(), scratch);
            EXPECT_EQ(std::memcmp(in_place.data(), output.data(), dim * sizeof(float)), 0)
                << "in-place dimension " << dim << ", scale " << scale;

            if (scale == 0.0f) {
                std::vector<uint8_t> zero_signs(qjl.packedSignBytes(), uint8_t{0});
                qjl.encodeSigns(input.data(), zero_signs.data(), output.data(), scratch);
                for (size_t projection = 0; projection < dim; ++projection) {
                    EXPECT_NE(zero_signs[projection / 8] &
                                  static_cast<uint8_t>(uint8_t{1} << (projection % 8)),
                              0)
                        << "sign(0) tie dimension/projection " << dim << "/" << projection;
                }
            }
        }

        std::vector<float> unaligned_input(dim + 1);
        std::vector<float> unaligned_output(dim + 1);
        const auto source = MakeInput(dim, 101);
        std::copy(source.begin(), source.end(), unaligned_input.begin() + 1);
        qjl.projectForward(unaligned_input.data() + 1, unaligned_output.data() + 1, scratch);
        const auto direct = DirectForward(qjl, unaligned_input.data() + 1);
        for (size_t coordinate = 0; coordinate < dim; ++coordinate) {
            EXPECT_NEAR(unaligned_output[coordinate + 1], direct[coordinate],
                        KernelTolerance(dim, source.data()));
        }

        std::vector<uint8_t> packed(qjl.packedSignBytes(), 0xFF);
        qjl.encodeSigns(source.data(), packed.data(), unaligned_output.data() + 1, scratch);
        if (dim % 8 != 0) {
            const uint8_t used_mask = static_cast<uint8_t>((uint16_t{1} << (dim % 8)) - 1);
            EXPECT_EQ(packed.back() & static_cast<uint8_t>(~used_mask), 0)
                << "tail bits dimension " << dim;
        }
    }
}

TEST(CirculantGaussianQjlV1Test, ResidualDirectionAndCorrectionScaleAcrossMagnitudes) {
    auto allocator = VecSimAllocator::newVecsimAllocator();
    CirculantGaussianQjlV1 qjl(allocator, 257, 71);
    auto direction = MakeInput(qjl.dimension(), 17);
    auto query = MakeInput(qjl.dimension(), 19);
    Normalize(direction);
    Normalize(query);
    auto scratch = qjl.createScratch();
    std::vector<float> projected_residual(qjl.dimension());
    std::vector<float> projected_query(qjl.dimension());
    std::vector<uint8_t> reference_signs(qjl.packedSignBytes());
    qjl.encodeSigns(direction.data(), reference_signs.data(), projected_residual.data(), scratch);
    qjl.projectQuery(query.data(), projected_query.data(), scratch);
    const float unit_correction =
        qjl.asymmetricCorrection(reference_signs.data(), projected_query.data(), 1.0f);

    for (float magnitude : {1e-20f, 0.125f, 1.0f, 7.5f, 1e20f}) {
        std::vector<float> scaled(direction.size());
        std::transform(direction.begin(), direction.end(), scaled.begin(),
                       [magnitude](float value) { return value * magnitude; });
        std::vector<uint8_t> scaled_signs(qjl.packedSignBytes());
        qjl.encodeSigns(scaled.data(), scaled_signs.data(), projected_residual.data(), scratch);
        EXPECT_EQ(scaled_signs, reference_signs)
            << "positive residual scaling must not change sign(Sr), magnitude " << magnitude;
        EXPECT_NEAR(
            qjl.asymmetricCorrection(reference_signs.data(), projected_query.data(), magnitude),
            static_cast<double>(unit_correction) * magnitude,
            std::max(1e-30, std::abs(static_cast<double>(unit_correction) * magnitude) * 4.0 *
                                std::numeric_limits<float>::epsilon()));
    }

    const double signed_sum_correction =
        Correction(reference_signs, projected_query.data(), qjl.dimension(), 1.0);
    EXPECT_NEAR(unit_correction, signed_sum_correction, 2e-6);
    const double wrong_scale = signed_sum_correction * std::sqrt(std::acos(-1.0) / 2.0);
    EXPECT_GT(std::abs(unit_correction - wrong_scale), 1e-3)
        << "fixture must detect pi/(2d) in place of sqrt(pi/2)/d";
    EXPECT_FLOAT_EQ(qjl.asymmetricCorrection(nullptr, nullptr, 0.0f), 0.0f);
}

TEST(CirculantGaussianQjlV1Test, PersistentAndScratchMemoryAreAllocatorAccountedAndLinear) {
    for (size_t dim : {size_t{7}, size_t{1024}, size_t{3072}}) {
        auto allocator = VecSimAllocator::newVecsimAllocator();
        const uint64_t baseline = allocator->getAllocationSize();
        size_t persistent_bytes = 0;
        size_t scratch_bytes = 0;
        {
            CirculantGaussianQjlV1 qjl(allocator, dim, 17);
            persistent_bytes = qjl.persistentAllocationBytes();
            EXPECT_EQ(allocator->getAllocationSize(), baseline + persistent_bytes)
                << "persistent dimension " << dim;
            EXPECT_LT(persistent_bytes, 140 * dim)
                << "persistent state must remain O(d), dimension " << dim;
            {
                auto scratch = qjl.createScratch();
                scratch_bytes = qjl.scratchAllocationBytes();
                EXPECT_EQ(scratch.storageBytes() + VecSimAllocator::getAllocationOverheadSize(),
                          scratch_bytes);
                EXPECT_EQ(allocator->getAllocationSize(),
                          baseline + persistent_bytes + scratch_bytes)
                    << "scratch dimension " << dim;
                EXPECT_LT(scratch_bytes, 70 * dim)
                    << "operation scratch must remain O(d), dimension " << dim;
            }
            EXPECT_EQ(allocator->getAllocationSize(), baseline + persistent_bytes);
        }
        EXPECT_EQ(allocator->getAllocationSize(), baseline) << "destroy dimension " << dim;
    }
}

TEST(CirculantGaussianQjlV1Test, ProjectionCallsAllocateNothingAfterScratchCreation) {
    constexpr size_t dim = 1000;
    auto allocator = VecSimAllocator::newVecsimAllocator();
    CirculantGaussianQjlV1 qjl(allocator, dim, 17);
    auto scratch = qjl.createScratch();
    const auto input = MakeInput(dim, 1);
    const auto query = MakeInput(dim, 2);
    std::vector<float> projected(dim);
    std::vector<uint8_t> packed(qjl.packedSignBytes());
    const uint64_t bytes_before = allocator->getAllocationSize();
#ifdef BUILD_TESTS
    const uint64_t calls_before = allocator->getAllocationCount();
#endif
    for (size_t iteration = 0; iteration < 32; ++iteration) {
        qjl.projectForward(input.data(), projected.data(), scratch);
        qjl.projectAdjoint(query.data(), projected.data(), scratch);
        qjl.projectQuery(query.data(), projected.data(), scratch);
        qjl.encodeSigns(input.data(), packed.data(), projected.data(), scratch);
        EXPECT_TRUE(
            std::isfinite(qjl.asymmetricCorrection(packed.data(), projected.data(), 0.75f)));
    }
    EXPECT_EQ(allocator->getAllocationSize(), bytes_before);
#ifdef BUILD_TESTS
    EXPECT_EQ(allocator->getAllocationCount(), calls_before);
#endif
}

TEST(CirculantGaussianQjlV1ModelTest, FullProfileIdentityValidationAndAllocatorEstimate) {
    constexpr size_t dim = 1024;
    constexpr uint64_t seed = 17;
    auto allocator = VecSimAllocator::newVecsimAllocator();
    auto config = TQCodecConfig::FastStructured(dim, 4, dim, seed);
    EXPECT_NO_THROW(TQFlatDetails::ValidateTQCodecConfig(config));

    auto mismatched_rotation = config;
    mismatched_rotation.rotation_backend_version =
        TQFlatDetails::TQRotationBackendVersion::DenseHaarV1;
    EXPECT_THROW(TQFlatDetails::ValidateTQCodecConfig(mismatched_rotation), std::invalid_argument);
    auto mismatched_qjl = config;
    mismatched_qjl.qjl_backend_version = TQFlatDetails::TQQjlBackendVersion::DenseGaussianV1;
    EXPECT_THROW(TQFlatDetails::ValidateTQCodecConfig(mismatched_qjl), std::invalid_argument);

    const uint64_t baseline = allocator->getAllocationSize();
    auto model = TQFlatDetails::AllocateFastStructuredTQModelState(allocator, dim, 4, dim, seed);
    const auto identity = model->modelIdentity();
    EXPECT_EQ(identity.model_transform_version,
              TQFlatDetails::TQModelTransformVersion::FastStructuredV1);
    EXPECT_EQ(identity.rotation_backend_version,
              TQFlatDetails::TQRotationBackendVersion::FastStructuredV1);
    EXPECT_EQ(identity.qjl_backend_version,
              TQFlatDetails::TQQjlBackendVersion::CirculantGaussianV1);
    EXPECT_EQ(identity.qjl_gaussian_generation_version,
              CirculantGaussianQjlV1::kGaussianGenerationVersion);
    EXPECT_EQ(identity.qjl_sign_generation_version, CirculantGaussianQjlV1::kSignGenerationVersion);
    EXPECT_EQ(identity.qjl_convolution_version, CirculantGaussianQjlV1::kConvolutionVersion);
    EXPECT_EQ(model->qjlGaussianGenerationVersion(), identity.qjl_gaussian_generation_version);
    EXPECT_EQ(model->qjlSignGenerationVersion(), identity.qjl_sign_generation_version);
    EXPECT_EQ(model->qjlConvolutionVersion(), identity.qjl_convolution_version);
    EXPECT_EQ(allocator->getAllocationSize() - baseline,
              TQFlatDetails::EstimateFastStructuredTQModelAllocationSize(dim, 4, dim, seed));
    EXPECT_LT(
        TQFlatDetails::EstimateFastStructuredTQModelAllocationSize(dim, 4, dim, seed),
        TQFlatDetails::EstimateFastStructuredRotationTQModelAllocationSize(dim, 4, dim, seed));
    model.reset();
    EXPECT_EQ(allocator->getAllocationSize(), baseline);
}

TEST(CirculantGaussianQjlV1ModelTest, ModelQueryProjectionBindsForwardKernel) {
    constexpr size_t dim = 7;
    constexpr uint64_t seed = 31;
    auto allocator = VecSimAllocator::newVecsimAllocator();
    auto model = TQFlatDetails::AllocateFastStructuredTQModelState(allocator, dim, 4, dim, seed);
    const auto query = MakeInput(dim, 71);
    std::vector<float> projected(dim);
    auto qjl_scratch = model->createQjlScratch();
    model->projectQjl(query.data(), projected.data(), qjl_scratch);

    CirculantGaussianQjlV1 direct(allocator, dim, seed + TQFlatDetails::kQjlSeedOffset);
    const auto expected_forward = DirectForward(direct, query.data());
    const auto wrong_adjoint = DirectAdjoint(direct, query.data());
    double maximum_adjoint_difference = 0.0;
    for (size_t coordinate = 0; coordinate < dim; ++coordinate) {
        EXPECT_NEAR(projected[coordinate], expected_forward[coordinate],
                    KernelTolerance(dim, query.data()));
        maximum_adjoint_difference = std::max(
            maximum_adjoint_difference,
            std::abs(static_cast<double>(projected[coordinate]) - wrong_adjoint[coordinate]));
    }
    EXPECT_GT(maximum_adjoint_difference, 0.25);
}

TEST(CirculantGaussianQjlV1ModelTest, ReusableFullPreprocessingContextAllocatesNothingPerCall) {
    constexpr size_t dim = 257;
    auto allocator = VecSimAllocator::newVecsimAllocator();
    auto model = TQFlatDetails::AllocateFastStructuredTQModelState(allocator, dim, 4, dim, 73);
    TQFlatDetails::TQPreprocessor<VecSimMetric_IP> preprocessor(allocator, model);
    auto context = preprocessor.createOperationContext();
    void *storage_blob = allocator->allocate_aligned(model->storageBlobSize(), alignof(float));
    void *query_blob = allocator->allocate_aligned(model->queryBlobSize(), alignof(float));
    ASSERT_NE(storage_blob, nullptr);
    ASSERT_NE(query_blob, nullptr);
    const uint64_t bytes_before = allocator->getAllocationSize();
#ifdef BUILD_TESTS
    const uint64_t calls_before = allocator->getAllocationCount();
#endif

    std::vector<uint8_t> first_storage(model->storageBlobSize());
    std::vector<uint8_t> first_query(model->queryBlobSize());
    for (size_t iteration = 0; iteration < 24; ++iteration) {
        const auto input = MakeInput(dim, iteration + 1, 0.25f + iteration * 0.125f);
        size_t storage_size = dim * sizeof(float);
        size_t query_size = dim * sizeof(float);
        preprocessor.preprocessForStorageWithContext(input.data(), storage_blob, storage_size,
                                                     alignof(float), context);
        preprocessor.preprocessQueryWithContext(input.data(), query_blob, query_size,
                                                alignof(float), context);
        EXPECT_EQ(storage_size, model->storageBlobSize());
        EXPECT_EQ(query_size, model->queryBlobSize());
        EXPECT_TRUE(std::isfinite(model->queryView(query_blob).norm_sq));
        if (iteration == 0) {
            std::memcpy(first_storage.data(), storage_blob, model->storageBlobSize());
            std::memcpy(first_query.data(), query_blob, model->queryBlobSize());
        }
    }
    EXPECT_EQ(allocator->getAllocationSize(), bytes_before);
#ifdef BUILD_TESTS
    EXPECT_EQ(allocator->getAllocationCount(), calls_before);
#endif

    const auto first_input = MakeInput(dim, 1, 0.25f);
    size_t storage_size = dim * sizeof(float);
    size_t query_size = dim * sizeof(float);
    preprocessor.preprocessForStorageWithContext(first_input.data(), storage_blob, storage_size,
                                                 alignof(float), context);
    preprocessor.preprocessQueryWithContext(first_input.data(), query_blob, query_size,
                                            alignof(float), context);
    EXPECT_EQ(std::memcmp(first_storage.data(), storage_blob, model->storageBlobSize()), 0);
    EXPECT_EQ(std::memcmp(first_query.data(), query_blob, model->queryBlobSize()), 0);
    EXPECT_EQ(allocator->getAllocationSize(), bytes_before);
#ifdef BUILD_TESTS
    EXPECT_EQ(allocator->getAllocationCount(), calls_before);
#endif
    TQFlatDetails::TQDistanceCalculator<VecSimMetric_IP> query_calculator(
        allocator, model, TQFlatDetails::TQStoredDistanceMode::CoarseMse);
    TQFlatDetails::TQDistanceCalculator<VecSimMetric_IP> decode_calculator(
        allocator, model, TQFlatDetails::TQStoredDistanceMode::FullDecodeReference);
    EXPECT_TRUE(
        std::isfinite(query_calculator.calcDistanceForQuery(storage_blob, query_blob, model->dim)));
    EXPECT_TRUE(
        std::isfinite(decode_calculator.calcDistance(storage_blob, storage_blob, model->dim)));
    EXPECT_EQ(allocator->getAllocationSize(), bytes_before);
    allocator->free_allocation(storage_blob);
    allocator->free_allocation(query_blob);
}

TEST(CirculantGaussianQjlV1ModelTest, InternalComponentSeamAcceptsExplicitCandidateState) {
    auto allocator = VecSimAllocator::newVecsimAllocator();
    auto model = TQFlatDetails::AllocateFastStructuredTQModelState(allocator, 7, 4, 7, 17);
    auto components = TQFlatDetails::CreateTQComponents<VecSimMetric_IP>(
        allocator, model, TQFlatDetails::TQStoredDistanceMode::CoarseMse);
    ASSERT_NE(components.indexCalculator, nullptr);
    ASSERT_NE(components.preprocessors, nullptr);
    delete components.indexCalculator;
    delete components.preprocessors;

    auto other_allocator = VecSimAllocator::newVecsimAllocator();
    EXPECT_THROW(TQFlatDetails::CreateTQComponents<VecSimMetric_IP>(
                     other_allocator, model, TQFlatDetails::TQStoredDistanceMode::CoarseMse),
                 std::invalid_argument);
    EXPECT_THROW(TQFlatDetails::CreateTQComponents<VecSimMetric_IP>(
                     allocator, std::shared_ptr<const TQFlatDetails::TQModelState>{},
                     TQFlatDetails::TQStoredDistanceMode::CoarseMse),
                 std::invalid_argument);
}

TEST(CirculantGaussianQjlV1Test, ImmutableStateSupportsConcurrentIndependentScratch) {
    constexpr size_t dim = 1000;
    auto allocator = VecSimAllocator::newVecsimAllocator();
    CirculantGaussianQjlV1 qjl(allocator, dim, 999);
    constexpr size_t thread_count = 4;
    std::array<std::vector<float>, thread_count> inputs;
    std::array<std::vector<float>, thread_count> expected;
    for (size_t thread = 0; thread < thread_count; ++thread) {
        inputs[thread] = MakeInput(dim, thread + 1);
        const auto direct = DirectForward(qjl, inputs[thread].data());
        expected[thread].assign(direct.begin(), direct.end());
    }

    std::array<std::vector<float>, thread_count> outputs;
    std::array<std::thread, thread_count> threads;
    for (size_t thread = 0; thread < thread_count; ++thread) {
        threads[thread] = std::thread([&, thread] {
            auto scratch = qjl.createScratch();
            outputs[thread].resize(dim);
            for (size_t iteration = 0; iteration < 8; ++iteration) {
                qjl.projectQuery(inputs[thread].data(), outputs[thread].data(), scratch);
            }
        });
    }
    for (auto &thread : threads) {
        thread.join();
    }
    for (size_t thread = 0; thread < thread_count; ++thread) {
        for (size_t coordinate = 0; coordinate < dim; ++coordinate) {
            EXPECT_NEAR(outputs[thread][coordinate], expected[thread][coordinate],
                        KernelTolerance(dim, inputs[thread].data()));
        }
    }
}

struct Scenario {
    std::string name;
    std::vector<float> residual_direction;
    std::vector<float> query;
    double residual_norm;
};

std::vector<Scenario> MakeScenarios(size_t dim) {
    auto gaussian = MakeGaussianInput(dim, 101);
    auto independent = MakeInput(dim, 307);
    auto alternate = MakeInput(dim, 911);
    Normalize(gaussian);
    Normalize(independent);
    Normalize(alternate);

    const double projection = Dot(gaussian.data(), independent.data(), dim);
    for (size_t coordinate = 0; coordinate < dim; ++coordinate) {
        independent[coordinate] -= static_cast<float>(projection * gaussian[coordinate]);
    }
    Normalize(independent);

    auto negative = gaussian;
    for (float &value : negative) {
        value = -value;
    }
    auto near_zero = independent;
    auto correlated = alternate;
    for (size_t coordinate = 0; coordinate < dim; ++coordinate) {
        near_zero[coordinate] += 0.002f * gaussian[coordinate];
        correlated[coordinate] = 0.7f * gaussian[coordinate] + 0.3f * alternate[coordinate];
    }
    Normalize(near_zero);
    Normalize(correlated);

    auto normalized_distribution = MakeInput(dim, 1701);
    auto normalized_query = MakeInput(dim, 1901);
    Normalize(normalized_distribution);
    Normalize(normalized_query);
    return {{"zero_residual", std::vector<float>(dim, 0.0f), gaussian, 0.0},
            {"parallel_gaussian", gaussian, gaussian, 1.0},
            {"negative_gaussian", gaussian, negative, 1.0},
            {"orthogonal", gaussian, independent, 1.0},
            {"near_zero", gaussian, near_zero, 1.0},
            {"correlated", gaussian, correlated, 1.0},
            {"normalized", normalized_distribution, normalized_query, 1.0}};
}

struct ErrorSummary {
    double bias;
    double mae;
    double rmse;
    double p95;
    double variance;
    double confidence_half_width;
};

ErrorSummary Summarize(std::vector<double> errors) {
    EXPECT_GT(errors.size(), 1);
    const double count = static_cast<double>(errors.size());
    const double bias = std::accumulate(errors.begin(), errors.end(), 0.0) / count;
    double absolute_sum = 0.0;
    double square_sum = 0.0;
    double centered_square_sum = 0.0;
    for (double error : errors) {
        absolute_sum += std::abs(error);
        square_sum += error * error;
        centered_square_sum += (error - bias) * (error - bias);
    }
    std::vector<double> absolute_errors(errors.size());
    std::transform(errors.begin(), errors.end(), absolute_errors.begin(),
                   [](double error) { return std::abs(error); });
    std::sort(absolute_errors.begin(), absolute_errors.end());
    const size_t p95_index =
        static_cast<size_t>(0.95 * static_cast<double>(absolute_errors.size() - 1));
    const double variance = centered_square_sum / (count - 1.0);
    // A conservative four-standard-error interval avoids encoding a quality gate before the
    // reviewed multi-seed baseline while still detecting a material signed-bias mutation.
    return {.bias = bias,
            .mae = absolute_sum / count,
            .rmse = std::sqrt(square_sum / count),
            .p95 = absolute_errors[p95_index],
            .variance = variance,
            .confidence_half_width = 4.0 * std::sqrt(variance / count)};
}

void RecordSummary(const std::string &prefix, const ErrorSummary &summary) {
    testing::Test::RecordProperty(prefix + "_signed_bias", summary.bias);
    testing::Test::RecordProperty(prefix + "_mae", summary.mae);
    testing::Test::RecordProperty(prefix + "_rmse", summary.rmse);
    testing::Test::RecordProperty(prefix + "_p95_absolute_error", summary.p95);
    testing::Test::RecordProperty(prefix + "_empirical_variance", summary.variance);
    testing::Test::RecordProperty(prefix + "_bias_ci_half_width", summary.confidence_half_width);
}

TEST(CirculantGaussianQjlV1Test, DenseAndCirculantStatisticalBiasAndErrorDiagnostics) {
    // These deterministic diagnostics establish unbiasedness and expose all requested error
    // statistics. They intentionally do not set the candidate's variance/p95 acceptance budget;
    // that remains a maintainer-reviewed multi-seed benchmark and rollout gate.
    constexpr size_t seed_count = 384;
    for (size_t dim : {size_t{7}, size_t{31}, size_t{64}}) {
        const auto scenarios = MakeScenarios(dim);
        std::vector<std::vector<double>> dense_errors(scenarios.size());
        std::vector<std::vector<double>> circulant_errors(scenarios.size());
        for (size_t seed_index = 0; seed_index < seed_count; ++seed_index) {
            const uint64_t seed = SplitMixFinalizer(0x6A09E667F3BCC909ULL ^ seed_index);
            auto allocator = VecSimAllocator::newVecsimAllocator();
            auto dense_config = TQCodecConfig::DenseReference(dim, 2, dim, seed);
            dense_config.qjl_seed = seed;
            DenseGaussianQjlV1State dense(allocator, dense_config);
            CirculantGaussianQjlV1 circulant(allocator, dim, seed);
            auto scratch = circulant.createScratch();
            std::vector<float> residual_projection(dim);
            std::vector<float> query_projection(dim);
            std::vector<uint8_t> packed((dim + 7) / 8);

            for (size_t scenario_index = 0; scenario_index < scenarios.size(); ++scenario_index) {
                const auto &scenario = scenarios[scenario_index];
                const double exact =
                    scenario.residual_norm *
                    Dot(scenario.residual_direction.data(), scenario.query.data(), dim);

                dense.project(scenario.residual_direction.data(), residual_projection.data(),
                              nullptr);
                packed = PackSigns(residual_projection.data(), dim);
                dense.project(scenario.query.data(), query_projection.data(), nullptr);
                dense_errors[scenario_index].push_back(
                    Correction(packed, query_projection.data(), dim, scenario.residual_norm) -
                    exact);

                circulant.encodeSigns(scenario.residual_direction.data(), packed.data(),
                                      residual_projection.data(), scratch);
                circulant.projectQuery(scenario.query.data(), query_projection.data(), scratch);
                circulant_errors[scenario_index].push_back(
                    circulant.asymmetricCorrection(packed.data(), query_projection.data(),
                                                   scenario.residual_norm) -
                    exact);
            }
        }

        for (size_t scenario_index = 0; scenario_index < scenarios.size(); ++scenario_index) {
            const auto dense = Summarize(std::move(dense_errors[scenario_index]));
            const auto circulant = Summarize(std::move(circulant_errors[scenario_index]));
            const std::string suffix = std::to_string(dim) + "_" + scenarios[scenario_index].name;
            RecordSummary("dense_" + suffix, dense);
            RecordSummary("circulant_" + suffix, circulant);
            EXPECT_TRUE(std::isfinite(dense.mae));
            EXPECT_TRUE(std::isfinite(dense.rmse));
            EXPECT_TRUE(std::isfinite(dense.p95));
            EXPECT_TRUE(std::isfinite(circulant.mae));
            EXPECT_TRUE(std::isfinite(circulant.rmse));
            EXPECT_TRUE(std::isfinite(circulant.p95));
            EXPECT_LE(std::abs(dense.bias), dense.confidence_half_width + 2e-5)
                << "dense bias dimension/scenario " << suffix;
            EXPECT_LE(std::abs(circulant.bias), circulant.confidence_half_width + 2e-5)
                << "circulant bias dimension/scenario " << suffix;
        }
    }
}

} // namespace
