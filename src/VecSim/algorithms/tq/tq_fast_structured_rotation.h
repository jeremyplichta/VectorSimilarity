/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

#include "VecSim/memory/vecsim_malloc.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace TQFlatDetails {

/**
 * The deterministic, scalar rotation backend FastStructuredV1, first selected by model profile
 * FastStructuredRotationV1.
 *
 * Each of the three stages applies independent Rademacher signs, a global permutation, and
 * normalized Walsh-Hadamard transforms over the greedy power-of-two partition with maximum block
 * size 256. The versioned generation rules are:
 *
 *  - SplitMix64 with the standard increment and finalizer below.
 *  - stage stream seed = SplitMix64Finalizer(model_seed XOR domain XOR
 *    (0x9e3779b97f4a7c15 * (stage + 1))).
 *  - signs consume one word per coordinate, using the low bit (0 => -1, 1 => +1).
 *  - permutations start as the identity and use descending Fisher-Yates. A random index in
 *    [0, i] is sampled with rejection before modulo reduction.
 *
 * These choices, including the permutation's destination-to-source representation and FWHT
 * butterfly order, are part of FastStructuredV1 and are locked by golden tests.
 */
class FastStructuredRotationV1 {
private:
    static constexpr size_t kStages = 3;
    static constexpr size_t kMaximumBlockSize = 256;
    static constexpr uint64_t kSplitMixIncrement = 0x9E3779B97F4A7C15ULL;
    static constexpr uint64_t kSignDomain = 0xD1B54A32D192ED03ULL;
    static constexpr uint64_t kPermutationDomain = 0x8CB92BA72F3D8DD7ULL;

    using SignVector = std::vector<int8_t, VecsimSTLAllocator<int8_t>>;
    using PermutationVector = std::vector<size_t, VecsimSTLAllocator<size_t>>;

    class SplitMix64 {
    public:
        explicit SplitMix64(uint64_t seed) : state(seed) {}

        uint64_t next() {
            state += kSplitMixIncrement;
            return mix(state);
        }

        uint64_t bounded(uint64_t bound) {
            assert(bound != 0);
            // (-bound) mod bound is the number of low words that would make modulo reduction
            // biased. Unsigned wraparound is intentional and defined by C++.
            const uint64_t rejection_threshold = (uint64_t{0} - bound) % bound;
            uint64_t value;
            do {
                value = next();
            } while (value < rejection_threshold);
            return value % bound;
        }

        static uint64_t mix(uint64_t value) {
            value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
            value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
            return value ^ (value >> 31);
        }

    private:
        uint64_t state;
    };

    struct StageState {
        StageState(std::shared_ptr<VecSimAllocator> allocator, size_t dim, uint64_t sign_seed,
                   uint64_t permutation_seed)
            : signs(dim, int8_t{0}, VecsimSTLAllocator<int8_t>(allocator)),
              permutation(dim, size_t{0}, VecsimSTLAllocator<size_t>(allocator)),
              inverse_permutation(dim, size_t{0}, VecsimSTLAllocator<size_t>(allocator)) {
            SplitMix64 sign_rng(sign_seed);
            for (size_t coordinate = 0; coordinate < dim; ++coordinate) {
                signs[coordinate] = (sign_rng.next() & 1U) == 0 ? int8_t{-1} : int8_t{1};
                permutation[coordinate] = coordinate;
            }

            SplitMix64 permutation_rng(permutation_seed);
            for (size_t remaining = dim; remaining > 1; --remaining) {
                const size_t last = remaining - 1;
                const size_t selected = static_cast<size_t>(permutation_rng.bounded(remaining));
                std::swap(permutation[last], permutation[selected]);
            }
            for (size_t destination = 0; destination < dim; ++destination) {
                inverse_permutation[permutation[destination]] = destination;
            }
        }

        SignVector signs;
        // P(D(x))[destination] = signs[source] * x[source], source = permutation[destination].
        PermutationVector permutation;
        PermutationVector inverse_permutation;
    };

public:
    FastStructuredRotationV1(std::shared_ptr<VecSimAllocator> allocator, size_t dim, uint64_t seed)
        : allocator(validateAllocator(std::move(allocator))), dim(validateDimension(dim)),
          seed(seed),
          stages{StageState(this->allocator, this->dim, deriveSeed(seed, kSignDomain, 0),
                            deriveSeed(seed, kPermutationDomain, 0)),
                 StageState(this->allocator, this->dim, deriveSeed(seed, kSignDomain, 1),
                            deriveSeed(seed, kPermutationDomain, 1)),
                 StageState(this->allocator, this->dim, deriveSeed(seed, kSignDomain, 2),
                            deriveSeed(seed, kPermutationDomain, 2))} {}

    size_t dimension() const { return dim; }
    uint64_t modelSeed() const { return seed; }

    /**
     * Apply R_2(R_1(R_0(input))). output and scratch must each address dim floats and must not
     * overlap. input may alias output; all other overlap is unsupported.
     */
    void forwardRotate(const float *input, float *output, float *scratch) const {
        assert(input != nullptr);
        assert(output != nullptr);
        assert(scratch != nullptr);
        assert(output != scratch);
        assert(input != scratch);

        if (input == output) {
            applyForwardStage(stages[0], input, scratch);
            applyForwardStage(stages[1], scratch, output);
            applyForwardStage(stages[2], output, scratch);
            std::memcpy(output, scratch, dim * sizeof(float));
            return;
        }

        applyForwardStage(stages[0], input, output);
        applyForwardStage(stages[1], output, scratch);
        applyForwardStage(stages[2], scratch, output);
    }

    /**
     * Apply R_0^-1(R_1^-1(R_2^-1(input))). The aliasing rules match forwardRotate.
     */
    void inverseRotate(const float *input, float *output, float *scratch) const {
        assert(input != nullptr);
        assert(output != nullptr);
        assert(scratch != nullptr);
        assert(output != scratch);
        assert(input != scratch);

        applyInverseStage(stages[2], input, output, scratch);
        applyInverseStage(stages[1], output, output, scratch);
        applyInverseStage(stages[0], output, output, scratch);
    }

    void forwardRotateInPlace(float *values, float *scratch) const {
        forwardRotate(values, values, scratch);
    }

    void inverseRotateInPlace(float *values, float *scratch) const {
        inverseRotate(values, values, scratch);
    }

    int signAt(size_t stage, size_t coordinate) const {
        assert(stage < kStages);
        assert(coordinate < dim);
        return stages[stage].signs[coordinate];
    }

    size_t permutationAt(size_t stage, size_t destination) const {
        assert(stage < kStages);
        assert(destination < dim);
        return stages[stage].permutation[destination];
    }

    size_t inversePermutationAt(size_t stage, size_t source) const {
        assert(stage < kStages);
        assert(source < dim);
        return stages[stage].inverse_permutation[source];
    }

private:
    static std::shared_ptr<VecSimAllocator>
    validateAllocator(std::shared_ptr<VecSimAllocator> allocator) {
        if (!allocator) {
            throw std::invalid_argument("FastStructuredV1 requires a VecSim allocator");
        }
        return allocator;
    }

    static size_t validateDimension(size_t dim) {
        if (dim == 0) {
            throw std::invalid_argument("FastStructuredV1 requires dimension >= 1");
        }
        if (dim > std::numeric_limits<size_t>::max() / sizeof(size_t)) {
            throw std::invalid_argument("FastStructuredV1 dimension exceeds addressable state");
        }
        if constexpr (std::numeric_limits<size_t>::digits > std::numeric_limits<uint64_t>::digits) {
            if (dim > std::numeric_limits<uint64_t>::max()) {
                throw std::invalid_argument("FastStructuredV1 dimension exceeds the PRNG range");
            }
        }
        return dim;
    }

    static uint64_t deriveSeed(uint64_t model_seed, uint64_t domain, size_t stage) {
        return SplitMix64::mix(model_seed ^ domain ^
                               (kSplitMixIncrement * static_cast<uint64_t>(stage + 1)));
    }

    static size_t nextBlockSize(size_t remaining) {
        size_t block_size = 1;
        const size_t limit = std::min(remaining, kMaximumBlockSize);
        while (block_size <= limit / 2) {
            block_size *= 2;
        }
        return block_size;
    }

    static void applyNormalizedBlockFwht(float *values, size_t block_size) {
        for (size_t half_width = 1; half_width < block_size; half_width *= 2) {
            const size_t butterfly_width = half_width * 2;
            for (size_t base = 0; base < block_size; base += butterfly_width) {
                for (size_t offset = 0; offset < half_width; ++offset) {
                    const float lhs = values[base + offset];
                    const float rhs = values[base + offset + half_width];
                    values[base + offset] = lhs + rhs;
                    values[base + offset + half_width] = lhs - rhs;
                }
            }
        }

        const float normalization = 1.0f / std::sqrt(static_cast<float>(block_size));
        for (size_t i = 0; i < block_size; ++i) {
            values[i] *= normalization;
        }
    }

    void applyBlockFwht(float *values) const {
        size_t offset = 0;
        while (offset < dim) {
            const size_t block_size = nextBlockSize(dim - offset);
            applyNormalizedBlockFwht(values + offset, block_size);
            offset += block_size;
        }
    }

    void applyForwardStage(const StageState &stage, const float *input, float *output) const {
        assert(input != output);
        for (size_t destination = 0; destination < dim; ++destination) {
            const size_t source = stage.permutation[destination];
            output[destination] = static_cast<float>(stage.signs[source]) * input[source];
        }
        applyBlockFwht(output);
    }

    void applyInverseStage(const StageState &stage, const float *input, float *output,
                           float *scratch) const {
        std::memcpy(scratch, input, dim * sizeof(float));
        applyBlockFwht(scratch);
        for (size_t source = 0; source < dim; ++source) {
            output[source] = static_cast<float>(stage.signs[source]) *
                             scratch[stage.inverse_permutation[source]];
        }
    }

    std::shared_ptr<VecSimAllocator> allocator;
    size_t dim;
    uint64_t seed;
    std::array<StageState, kStages> stages;
};

} // namespace TQFlatDetails
