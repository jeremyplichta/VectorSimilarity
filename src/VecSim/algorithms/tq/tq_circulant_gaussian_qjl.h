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
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace TQFlatDetails {

/**
 * Experimental O(d log d) Gaussian-circulant QJL backend.
 *
 * For independently generated g[j] ~ N(0, 1) and epsilon[j] in {-1, +1}, this class defines
 *
 *   S[i,j] = g[(i - j) mod d] * epsilon[j].
 *
 * Each row is marginally an IID standard-Gaussian vector, so sign(Sr) paired with Sq preserves
 * the one-bit QJL expectation. The rows share g and are correlated, however, so this backend is
 * not the paper's independent dense Gaussian matrix. It remains an internal, non-default
 * candidate until its variance, tail error, recall, persistence, and platform gates are reviewed.
 *
 * projectForward computes Sx. projectAdjoint computes S^T x for algebraic validation and future
 * callers that explicitly need the adjoint. TurboQuant's asymmetric score uses sign(Sr) with Sq,
 * so projectQuery deliberately dispatches to projectForward, not projectAdjoint.
 *
 * The portable convolution uses a radix-2 complex FFT in FP64 working precision. The forward FFT
 * has the exp(-2*pi*i*k/n) convention. The inverse has the opposite sign and divides by the FFT
 * length. A length-d circular convolution is obtained by a zero-padded length-(2d-1) linear
 * convolution followed by folding coefficient i+d into coefficient i. No 1/sqrt(d) or other JL
 * normalization is applied. Inputs and outputs are FP32. Scratch has natural Complex alignment,
 * is owned by the caller, and is the only mutable operation state.
 */
class CirculantGaussianQjlV1 {
public:
    static constexpr uint8_t kGaussianGenerationVersion = 1;
    static constexpr uint8_t kSignGenerationVersion = 1;
    static constexpr uint8_t kConvolutionVersion = 1;

    struct Complex {
        double real = 0.0;
        double imag = 0.0;
    };

private:
    using GaussianVector = std::vector<float, VecsimSTLAllocator<float>>;
    using SignVector = std::vector<int8_t, VecsimSTLAllocator<int8_t>>;
    using ComplexVector = std::vector<Complex, VecsimSTLAllocator<Complex>>;

public:
    class Scratch {
        friend class CirculantGaussianQjlV1;

    public:
        Scratch(const Scratch &) = delete;
        Scratch &operator=(const Scratch &) = delete;
        Scratch(Scratch &&) noexcept = default;
        Scratch &operator=(Scratch &&) noexcept = default;

        size_t complexCount() const { return values.size(); }
        size_t storageBytes() const { return values.size() * sizeof(Complex); }

    private:
        Scratch(const std::shared_ptr<VecSimAllocator> &allocator, size_t fft_length)
            : values(fft_length, Complex{}, VecsimSTLAllocator<Complex>(allocator)) {}

        ComplexVector values;
    };

    CirculantGaussianQjlV1(std::shared_ptr<VecSimAllocator> allocator, size_t dimension,
                           uint64_t seed)
        : allocator(validateAllocator(std::move(allocator))), dim(validateDimension(dimension)),
          seed(seed), gaussian_seed(deriveSeed(seed, kGaussianDomain)),
          sign_seed(deriveSeed(seed, kSignDomain)), linear_length(linearLength(dim)),
          fft_length(fftLength(linear_length)),
          generator(dim, 0.0f, VecsimSTLAllocator<float>(this->allocator)),
          signs(dim, int8_t{0}, VecsimSTLAllocator<int8_t>(this->allocator)),
          forward_spectrum(fft_length, Complex{}, VecsimSTLAllocator<Complex>(this->allocator)),
          adjoint_spectrum(fft_length, Complex{}, VecsimSTLAllocator<Complex>(this->allocator)) {
        initializeRandomState();
        initializeSpectra();
    }

    CirculantGaussianQjlV1(const CirculantGaussianQjlV1 &) = delete;
    CirculantGaussianQjlV1 &operator=(const CirculantGaussianQjlV1 &) = delete;

    size_t dimension() const { return dim; }
    size_t projections() const { return dim; }
    uint64_t modelSeed() const { return seed; }
    uint64_t gaussianSeed() const { return gaussian_seed; }
    uint64_t signSeed() const { return sign_seed; }
    uint8_t gaussianGenerationVersion() const { return kGaussianGenerationVersion; }
    uint8_t signGenerationVersion() const { return kSignGenerationVersion; }
    uint8_t convolutionVersion() const { return kConvolutionVersion; }
    size_t fftLength() const { return fft_length; }
    static size_t requiredFftLength(size_t dimension) {
        validateDimension(dimension);
        return fftLength(linearLength(dimension));
    }
    size_t packedSignBytes() const { return (dim + 7) / 8; }
    float gaussianAt(size_t coordinate) const {
        assert(coordinate < dim);
        return generator[coordinate];
    }
    int signAt(size_t coordinate) const {
        assert(coordinate < dim);
        return signs[coordinate];
    }

    Scratch createScratch() const { return Scratch(allocator, fft_length); }

    size_t persistentAllocationBytes() const {
        const size_t payload =
            checkedAdd(checkedBytes(dim, sizeof(float), "Circulant QJL state size overflow"),
                       checkedBytes(dim, sizeof(int8_t), "Circulant QJL state size overflow"),
                       "Circulant QJL state size overflow");
        const size_t spectrum_bytes = checkedBytes(
            checkedMultiply(fft_length, size_t{2}, "Circulant QJL spectrum size overflow"),
            sizeof(Complex), "Circulant QJL spectrum size overflow");
        return checkedAdd(checkedAdd(payload, spectrum_bytes, "Circulant QJL state size overflow"),
                          checkedMultiply(size_t{4}, VecSimAllocator::getAllocationOverheadSize(),
                                          "Circulant QJL allocation overhead overflow"),
                          "Circulant QJL state size overflow");
    }

    size_t scratchAllocationBytes() const {
        return checkedAdd(
            checkedBytes(fft_length, sizeof(Complex), "Circulant QJL scratch size overflow"),
            VecSimAllocator::getAllocationOverheadSize(), "Circulant QJL scratch size overflow");
    }

    /**
     * Compute Sx. input may exactly alias output; partial overlap is unsupported. Scratch must
     * have been created for a backend with the same convolution length.
     */
    void projectForward(const float *input, float *output, Scratch &scratch) const {
        project(input, output, scratch, forward_spectrum, true, false);
    }

    /**
     * Compute S^T x. This is not the projection used by TurboQuant query scoring.
     */
    void projectAdjoint(const float *input, float *output, Scratch &scratch) const {
        project(input, output, scratch, adjoint_spectrum, false, true);
    }

    /**
     * Project a query for the asymmetric estimator. Algorithm 2 stores sign(Sr), hence the paired
     * query vector is Sq. Keeping this named entry point prevents an accidental adjoint dispatch.
     */
    void projectQuery(const float *input, float *output, Scratch &scratch) const {
        projectForward(input, output, scratch);
    }

    /**
     * Compute Sx and pack sign(x_i) with the persisted tie rule sign(0) = +1. Bit i is one for a
     * nonnegative projection. The destination is cleared first, including unused tail bits.
     */
    void encodeSigns(const float *input, uint8_t *packed_signs, float *projected,
                     Scratch &scratch) const {
        if (packed_signs == nullptr || projected == nullptr) {
            throw std::invalid_argument("Circulant QJL sign output must not be null");
        }
        std::memset(packed_signs, 0, packedSignBytes());
        projectForward(input, projected, scratch);
        for (size_t projection = 0; projection < dim; ++projection) {
            if (projected[projection] >= 0.0f) {
                packed_signs[projection / 8] |=
                    static_cast<uint8_t>(uint8_t{1} << (projection % 8));
            }
        }
    }

    /**
     * Apply the unchanged one-bit QJL correction to a forward-projected query. A zero residual
     * returns exactly zero without consulting sign payload bytes.
     */
    float asymmetricCorrection(const uint8_t *packed_signs, const float *projected_query,
                               float residual_norm) const {
        if (residual_norm == 0.0f) {
            return 0.0f;
        }
        if (packed_signs == nullptr || projected_query == nullptr) {
            throw std::invalid_argument("Circulant QJL correction input must not be null");
        }
        if (!(residual_norm > 0.0f) || !std::isfinite(residual_norm)) {
            throw std::invalid_argument(
                "Circulant QJL residual norm must be finite and nonnegative");
        }

        double signed_sum = 0.0;
        for (size_t projection = 0; projection < dim; ++projection) {
            const bool positive = (packed_signs[projection / 8] &
                                   static_cast<uint8_t>(uint8_t{1} << (projection % 8))) != 0;
            signed_sum += (positive ? 1.0 : -1.0) * projected_query[projection];
        }
        constexpr double kSqrtPiOverTwo = 1.2533141373155002512;
        return static_cast<float>(static_cast<double>(residual_norm) * kSqrtPiOverTwo * signed_sum /
                                  static_cast<double>(dim));
    }

private:
    static constexpr uint64_t kSplitMixIncrement = 0x9E3779B97F4A7C15ULL;
    static constexpr uint64_t kGaussianDomain = 0xA24BAED4963EE407ULL;
    static constexpr uint64_t kSignDomain = 0x9FB21C651E98DF25ULL;

    class SplitMix64 {
    public:
        explicit SplitMix64(uint64_t seed) : state(seed) {}

        uint64_t next() {
            state += kSplitMixIncrement;
            return mix(state);
        }

        static uint64_t mix(uint64_t value) {
            value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
            value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
            return value ^ (value >> 31);
        }

    private:
        uint64_t state;
    };

    class GaussianRng {
    public:
        explicit GaussianRng(uint64_t seed) : words(seed) {}

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
        double uniformOpen() {
            constexpr double kInv53 = 1.0 / static_cast<double>(uint64_t{1} << 53);
            return (static_cast<double>(words.next() >> 11) + 0.5) * kInv53;
        }

        SplitMix64 words;
        double spare = 0.0;
        bool has_spare = false;
    };

    static std::shared_ptr<VecSimAllocator>
    validateAllocator(std::shared_ptr<VecSimAllocator> allocator) {
        if (!allocator) {
            throw std::invalid_argument("CirculantGaussianQjlV1 requires a VecSim allocator");
        }
        return allocator;
    }

    static size_t validateDimension(size_t dimension) {
        if (dimension < 2) {
            throw std::invalid_argument("CirculantGaussianQjlV1 requires dimension >= 2");
        }
        linearLength(dimension);
        return dimension;
    }

    static size_t checkedAdd(size_t lhs, size_t rhs, const char *what) {
        if (rhs > std::numeric_limits<size_t>::max() - lhs) {
            throw std::overflow_error(what);
        }
        return lhs + rhs;
    }

    static size_t checkedMultiply(size_t lhs, size_t rhs, const char *what) {
        if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
            throw std::overflow_error(what);
        }
        return lhs * rhs;
    }

    static size_t checkedBytes(size_t count, size_t element_size, const char *what) {
        return checkedMultiply(count, element_size, what);
    }

    static size_t linearLength(size_t dimension) {
        return checkedMultiply(dimension, size_t{2}, "Circulant QJL convolution length overflow") -
               1;
    }

    static size_t fftLength(size_t required_length) {
        size_t result = 1;
        while (result < required_length) {
            if (result > std::numeric_limits<size_t>::max() / 2) {
                throw std::overflow_error("Circulant QJL FFT length overflow");
            }
            result *= 2;
        }
        checkedBytes(result, sizeof(Complex), "Circulant QJL FFT state size overflow");
        return result;
    }

    static uint64_t deriveSeed(uint64_t model_seed, uint64_t domain) {
        return SplitMix64::mix(model_seed ^ domain);
    }

    void initializeRandomState() {
        GaussianRng gaussian_rng(gaussian_seed);
        SplitMix64 sign_rng(sign_seed);
        for (size_t coordinate = 0; coordinate < dim; ++coordinate) {
            generator[coordinate] = gaussian_rng.normal();
            signs[coordinate] = (sign_rng.next() & uint64_t{1}) == 0 ? int8_t{-1} : int8_t{1};
        }
    }

    void initializeSpectra() {
        for (size_t coordinate = 0; coordinate < dim; ++coordinate) {
            forward_spectrum[coordinate].real = generator[coordinate];
            const size_t reversed = coordinate == 0 ? 0 : dim - coordinate;
            adjoint_spectrum[coordinate].real = generator[reversed];
        }
        fft(forward_spectrum, false);
        fft(adjoint_spectrum, false);
    }

    static void fft(ComplexVector &values, bool inverse) {
        const size_t count = values.size();
        assert(count != 0 && (count & (count - 1)) == 0);

        for (size_t source = 1, destination = 0; source < count; ++source) {
            size_t bit = count >> 1;
            while ((destination & bit) != 0) {
                destination ^= bit;
                bit >>= 1;
            }
            destination ^= bit;
            if (source < destination) {
                std::swap(values[source], values[destination]);
            }
        }

        const double direction = inverse ? 2.0 : -2.0;
        for (size_t width = 2;; width *= 2) {
            const double angle = direction * std::acos(-1.0) / static_cast<double>(width);
            const Complex step{std::cos(angle), std::sin(angle)};
            const size_t half_width = width / 2;
            for (size_t base = 0; base < count; base += width) {
                Complex twiddle{1.0, 0.0};
                for (size_t offset = 0; offset < half_width; ++offset) {
                    const Complex even = values[base + offset];
                    const Complex odd = multiply(values[base + offset + half_width], twiddle);
                    values[base + offset] = {even.real + odd.real, even.imag + odd.imag};
                    values[base + offset + half_width] = {even.real - odd.real,
                                                          even.imag - odd.imag};
                    twiddle = multiply(twiddle, step);
                }
            }
            if (width == count) {
                break;
            }
        }

        if (inverse) {
            const double scale = 1.0 / static_cast<double>(count);
            for (Complex &value : values) {
                value.real *= scale;
                value.imag *= scale;
            }
        }
    }

    static Complex multiply(const Complex &lhs, const Complex &rhs) {
        return {lhs.real * rhs.real - lhs.imag * rhs.imag,
                lhs.real * rhs.imag + lhs.imag * rhs.real};
    }

    void project(const float *input, float *output, Scratch &scratch, const ComplexVector &spectrum,
                 bool apply_input_signs, bool apply_output_signs) const {
        if (input == nullptr || output == nullptr) {
            throw std::invalid_argument("Circulant QJL projection input must not be null");
        }
        if (scratch.values.size() != fft_length) {
            throw std::invalid_argument("Circulant QJL scratch has the wrong convolution length");
        }

        std::fill(scratch.values.begin(), scratch.values.end(), Complex{});
        for (size_t coordinate = 0; coordinate < dim; ++coordinate) {
            const double sign = apply_input_signs ? static_cast<double>(signs[coordinate]) : 1.0;
            scratch.values[coordinate].real = sign * static_cast<double>(input[coordinate]);
        }

        fft(scratch.values, false);
        for (size_t frequency = 0; frequency < fft_length; ++frequency) {
            scratch.values[frequency] = multiply(scratch.values[frequency], spectrum[frequency]);
        }
        fft(scratch.values, true);

        for (size_t coordinate = 0; coordinate < dim; ++coordinate) {
            double value = scratch.values[coordinate].real;
            if (coordinate + dim < linear_length) {
                value += scratch.values[coordinate + dim].real;
            }
            if (apply_output_signs) {
                value *= static_cast<double>(signs[coordinate]);
            }
            output[coordinate] = static_cast<float>(value);
        }
    }

    std::shared_ptr<VecSimAllocator> allocator;
    size_t dim;
    uint64_t seed;
    uint64_t gaussian_seed;
    uint64_t sign_seed;
    size_t linear_length;
    size_t fft_length;
    GaussianVector generator;
    SignVector signs;
    ComplexVector forward_spectrum;
    ComplexVector adjoint_spectrum;
};

} // namespace TQFlatDetails
