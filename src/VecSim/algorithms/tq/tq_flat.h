/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

#include "VecSim/algorithms/brute_force/brute_force_single.h"
#include "VecSim/algorithms/tq/tq_model.h"
#include "VecSim/algorithms/tq/tq_stored_distance.h"
#include "VecSim/spaces/computer/calculator.h"
#include "VecSim/spaces/computer/preprocessor_container.h"
#include "VecSim/utils/vec_utils.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>

namespace TQFlatDetails {

#ifdef BUILD_TESTS
inline std::atomic_size_t query_preprocessing_count{0};

inline void ResetQueryPreprocessingCount() { query_preprocessing_count.store(0); }

inline size_t GetQueryPreprocessingCount() { return query_preprocessing_count.load(); }
#endif

inline float NormalizeInPlace(float *values, size_t dim) {
    const float norm_sq = SumSquaresScalar(values, dim);
    if (norm_sq == 0.0f) {
        return 0.0f;
    }
    const float norm = std::sqrt(norm_sq);
    const float inverse_norm = 1.0f / norm;
    for (size_t i = 0; i < dim; ++i) {
        values[i] *= inverse_norm;
    }
    return norm;
}

inline MemoryUtils::unique_blob AllocateTQScratch(const TQModelState &state, size_t float_count) {
    const size_t bytes =
        CheckedBytes(float_count, sizeof(float), "TurboQuant operation scratch size overflow");
    auto scratch = state.getAllocator()->allocate_unique(bytes);
    if (!scratch) {
        throw std::bad_alloc();
    }
    return scratch;
}

/**
 * Reusable, operation-local TurboQuant preprocessing state. The generic FP32 workspace and the
 * selected QJL backend's scratch are allocated once here. Callers that retain a context can encode
 * and preprocess queries repeatedly without further allocator activity, provided destination
 * blobs are also retained. A context belongs to one immutable model and is not thread-safe; use
 * one context per concurrent worker.
 */
class TQOperationContext {
public:
    explicit TQOperationContext(std::shared_ptr<const TQModelState> state)
        : state(validateState(std::move(state))),
          generic_scratch(AllocateTQScratch(
              *this->state, CheckedMultiply(this->state->dim, size_t{6},
                                            "TurboQuant reusable operation scratch overflow"))),
          qjl_scratch(this->state->createQjlScratch()) {}

    TQOperationContext(const TQOperationContext &) = delete;
    TQOperationContext &operator=(const TQOperationContext &) = delete;
    TQOperationContext(TQOperationContext &&) noexcept = default;
    TQOperationContext &operator=(TQOperationContext &&) noexcept = default;

private:
    template <VecSimMetric Metric>
    friend class TQPreprocessor;

    void validate(const TQModelState *expected) const {
        if (state.get() != expected) {
            throw std::invalid_argument("TurboQuant operation context belongs to another model");
        }
    }

    static std::shared_ptr<const TQModelState>
    validateState(std::shared_ptr<const TQModelState> state) {
        if (!state) {
            throw std::invalid_argument("TurboQuant operation context requires model state");
        }
        return state;
    }

    float *words() const { return static_cast<float *>(generic_scratch.get()); }

    std::shared_ptr<const TQModelState> state;
    MemoryUtils::unique_blob generic_scratch;
    TQModelState::QjlScratch qjl_scratch;
};

template <VecSimMetric Metric>
class TQDistanceCalculator : public IndexCalculatorInterface<float> {
private:
    static float calcStoredFullDecodeWithContext(const void *opaque_state, const void *lhs_blob,
                                                 const void *rhs_blob, size_t dim) {
        const auto *state = static_cast<const TQModelState *>(opaque_state);
        assert(dim == state->dim);
        auto scratch = AllocateTQScratch(
            *state, CheckedMultiply(dim, size_t{6}, "TurboQuant decode scratch overflow"));
        auto *words = static_cast<float *>(scratch.get());
        float *lhs = words;
        float *rhs = lhs + dim;
        float *lhs_rotated = rhs + dim;
        float *lhs_transform_scratch = lhs_rotated + dim;
        float *rhs_rotated = lhs_transform_scratch + dim;
        float *rhs_transform_scratch = rhs_rotated + dim;
        auto qjl_scratch = state->createQjlScratch();
        state->decode(state->storageView(lhs_blob), lhs, lhs_rotated, lhs_transform_scratch,
                      qjl_scratch);
        state->decode(state->storageView(rhs_blob), rhs, rhs_rotated, rhs_transform_scratch,
                      qjl_scratch);

        if constexpr (Metric == VecSimMetric_Cosine) {
            NormalizeInPlace(lhs, dim);
            NormalizeInPlace(rhs, dim);
        }
        if constexpr (Metric == VecSimMetric_L2) {
            float distance = 0.0f;
            for (size_t i = 0; i < dim; ++i) {
                const float difference = lhs[i] - rhs[i];
                distance += difference * difference;
            }
            return distance;
        }
        return 1.0f - DotProductScalar(lhs, rhs, dim);
    }

    static float calcStoredCoarseMseWithContext(const void *opaque_state, const void *lhs_blob,
                                                const void *rhs_blob, size_t dim) {
        const auto *state = static_cast<const TQModelState *>(opaque_state);
        assert(dim == state->dim);
        const auto lhs = state->storageView(lhs_blob);
        const auto rhs = state->storageView(rhs_blob);

        // Both supported metrics define the distance to a zero source as one. Read alpha before
        // touching the packed indices so the zero-vector behavior matches the reference path.
        if (lhs.source_scale == 0.0f || rhs.source_scale == 0.0f) {
            return 1.0f;
        }

        const auto accumulators = AccumulateCoarseMse(*state, lhs, rhs);
        if constexpr (Metric == VecSimMetric_Cosine) {
            if (accumulators.lhs_norm_sq == 0.0f || accumulators.rhs_norm_sq == 0.0f) {
                RecordZeroCoarseNorm();
                return 1.0f;
            }
            const float inverse_norm =
                1.0f / std::sqrt(accumulators.lhs_norm_sq * accumulators.rhs_norm_sq);
            return 1.0f - accumulators.dot * inverse_norm;
        }
        if constexpr (Metric == VecSimMetric_IP) {
            return 1.0f - lhs.source_scale * rhs.source_scale * accumulators.dot;
        }
        // CoarseMse is an HNSW experiment for the publicly supported TQ IP/COSINE metrics only.
        assert(false && "CoarseMse does not define an L2 stored distance");
        return 0.0f;
    }

    static float calcQueryWithContext(const void *opaque_state, const void *storage_blob,
                                      const void *query_blob, size_t dim) {
        const auto *state = static_cast<const TQModelState *>(opaque_state);
        assert(dim == state->dim);
        const auto storage = state->storageView(storage_blob);
        const auto query = state->queryView(query_blob);
        const float estimate = state->estimateInnerProduct(storage, query);
        if constexpr (Metric == VecSimMetric_L2) {
            const float storage_norm_sq = storage.source_scale * storage.source_scale;
            return std::max(query.norm_sq + storage_norm_sq - 2.0f * estimate, 0.0f);
        }
        return 1.0f - estimate;
    }

public:
    TQDistanceCalculator(
        std::shared_ptr<VecSimAllocator> allocator, std::shared_ptr<const TQModelState> state,
        TQStoredDistanceMode stored_distance_mode = TQStoredDistanceMode::FullDecodeReference)
        : IndexCalculatorInterface<float>(allocator), state(std::move(state)),
          stored_distance_mode(stored_distance_mode) {
        if constexpr (Metric == VecSimMetric_L2) {
            if (stored_distance_mode == TQStoredDistanceMode::CoarseMse) {
                throw std::invalid_argument("CoarseMse supports TQ IP and COSINE only");
            }
        }
    }

    float calcDistance(const void *v1, const void *v2, size_t dim) const override {
        return stored_distance_mode == TQStoredDistanceMode::CoarseMse
                   ? calcStoredCoarseMseWithContext(state.get(), v1, v2, dim)
                   : calcStoredFullDecodeWithContext(state.get(), v1, v2, dim);
    }

    float calcDistanceForQuery(const void *candidate_vector, const void *query_vector,
                               size_t dim) const override {
        return calcQueryWithContext(state.get(), candidate_vector, query_vector, dim);
    }

    DistanceDispatch<float> getDistanceDispatch(DistanceMode mode) const override {
        if (mode == DistanceMode::StoredToQuery) {
            return DistanceDispatch<float>::stateful(state.get(), calcQueryWithContext);
        }
        // HNSW snapshots this callback during construction, so the hot stored-to-stored path does
        // not branch on the experimental construction mode.
        return stored_distance_mode == TQStoredDistanceMode::CoarseMse
                   ? DistanceDispatch<float>::stateful(state.get(), calcStoredCoarseMseWithContext)
                   : DistanceDispatch<float>::stateful(state.get(),
                                                       calcStoredFullDecodeWithContext);
    }

    TQStoredDistanceMode getStoredDistanceMode() const { return stored_distance_mode; }

private:
    std::shared_ptr<const TQModelState> state;
    TQStoredDistanceMode stored_distance_mode;
};

template <VecSimMetric Metric>
class TQPreprocessor : public PreprocessorInterface {
public:
    TQPreprocessor(std::shared_ptr<VecSimAllocator> allocator,
                   std::shared_ptr<const TQModelState> state)
        : PreprocessorInterface(allocator), state(std::move(state)) {}

    TQOperationContext createOperationContext() const { return TQOperationContext(state); }

    void preprocess(const void *original_blob, void *&storage_blob, void *&query_blob,
                    size_t &storage_blob_size, size_t &query_blob_size,
                    unsigned char storage_alignment, unsigned char query_alignment) const override {
        const bool owns_storage_on_success = storage_blob == nullptr;
        const bool owns_query_on_success = query_blob == nullptr;
        auto context = createOperationContext();
        preprocessForStorageWithContext(original_blob, storage_blob, storage_blob_size,
                                        storage_alignment, context);
        try {
            preprocessQueryWithContext(original_blob, query_blob, query_blob_size, query_alignment,
                                       context);
        } catch (...) {
            if (owns_storage_on_success) {
                this->allocator->free_allocation(storage_blob);
                storage_blob = nullptr;
            }
            if (owns_query_on_success && query_blob) {
                this->allocator->free_allocation(query_blob);
                query_blob = nullptr;
            }
            throw;
        }
    }

    void preprocessForStorage(const void *original_blob, void *&storage_blob,
                              size_t &input_blob_size,
                              unsigned char storage_alignment) const override {
        auto context = createOperationContext();
        preprocessForStorageWithContext(original_blob, storage_blob, input_blob_size,
                                        storage_alignment, context);
    }

    void preprocessForStorageWithContext(const void *original_blob, void *&storage_blob,
                                         size_t &input_blob_size, unsigned char storage_alignment,
                                         TQOperationContext &context) const {
        context.validate(state.get());
        if (!storage_blob) {
            storage_blob =
                this->allocator->allocate_aligned(state->storageBlobSize(), storage_alignment);
            if (!storage_blob) {
                throw std::bad_alloc();
            }
        }
        std::memset(storage_blob, 0, state->storageBlobSize());

        const auto *input = static_cast<const float *>(original_blob);
        auto *words = context.words();
        float *unit = words;
        float *reconstructed = unit + state->dim;
        float *residual = reconstructed + state->dim;
        float *rotated = residual + state->dim;
        float *reconstructed_rotated = rotated + state->dim;
        float *transform_scratch = reconstructed_rotated + state->dim;
        std::copy(input, input + state->dim, unit);
        const float original_norm = NormalizeInPlace(unit, state->dim);
        if (original_norm == 0.0f) {
            state->writeMetadata(storage_blob, 0.0f, 0.0f);
            input_blob_size = state->storageBlobSize();
            return;
        }

        auto *indices = static_cast<uint8_t *>(storage_blob);
        auto *signs = indices + state->packedIndexBytes();
        state->encodeMse(unit, indices, reconstructed, rotated, reconstructed_rotated,
                         transform_scratch);
        for (size_t i = 0; i < state->dim; ++i) {
            residual[i] = unit[i] - reconstructed[i];
        }

        const float residual_norm = std::sqrt(SumSquaresScalar(residual, state->dim));
        if (residual_norm > 0.0f) {
            const float inverse_residual_norm = 1.0f / residual_norm;
            for (size_t i = 0; i < state->dim; ++i) {
                residual[i] *= inverse_residual_norm;
            }
            state->packResidualSigns(residual, signs, rotated, context.qjl_scratch);
        } else {
            // A nonzero source with a zero residual still persists sign(0) = +1 for every used
            // projection bit. A zero source returned before this point and leaves its blob zero.
            state->packResidualSigns(residual, signs, rotated, context.qjl_scratch);
        }

        const float source_scale = Metric == VecSimMetric_Cosine ? 1.0f : original_norm;
        state->writeMetadata(storage_blob, source_scale, residual_norm);
        input_blob_size = state->storageBlobSize();
    }

    void preprocessQuery(const void *original_blob, void *&query_blob, size_t &input_blob_size,
                         unsigned char alignment) const override {
        auto context = createOperationContext();
        preprocessQueryWithContext(original_blob, query_blob, input_blob_size, alignment, context);
    }

    void preprocessQueryWithContext(const void *original_blob, void *&query_blob,
                                    size_t &input_blob_size, unsigned char alignment,
                                    TQOperationContext &context) const {
#ifdef BUILD_TESTS
        query_preprocessing_count.fetch_add(1);
#endif
        context.validate(state.get());
        if (!query_blob) {
            query_blob = this->allocator->allocate_aligned(state->queryBlobSize(), alignment);
            if (!query_blob) {
                throw std::bad_alloc();
            }
        }

        const auto *input = static_cast<const float *>(original_blob);
        auto *query = context.words();
        auto *backend_scratch = query + state->dim;
        std::copy(input, input + state->dim, query);
        if constexpr (Metric == VecSimMetric_Cosine) {
            NormalizeInPlace(query, state->dim);
        }

        auto *words = static_cast<float *>(query_blob);
        auto *rotated = words;
        auto *projected = rotated + state->dim;
        auto *norm_sq = projected + state->projections;
        state->applyRotation(query, rotated, backend_scratch);
        state->projectQjl(query, projected, context.qjl_scratch);
        *norm_sq = SumSquaresScalar(query, state->dim);
        input_blob_size = state->queryBlobSize();
    }

    void preprocessStorageInPlace(void *original_blob, size_t input_blob_size) const override {
        assert(original_blob);
        assert(input_blob_size >= state->storageBlobSize());
        auto encoded = this->allocator->allocate_unique(state->storageBlobSize());
        if (!encoded) {
            throw std::bad_alloc();
        }
        void *encoded_blob = encoded.get();
        size_t encoded_size = input_blob_size;
        preprocessForStorage(original_blob, encoded_blob, encoded_size, 0);
        std::memcpy(original_blob, encoded.get(), state->storageBlobSize());
    }

private:
    std::shared_ptr<const TQModelState> state;
};

template <VecSimMetric Metric>
inline size_t GetStorageDataSize(const TQFlatParams *params) {
    ValidateTQCodecConfig(TQCodecConfig::DenseReference(
        params->dim, params->bits, params->projections, params->seed, params->useRotation));
    return CheckedAdd(CheckedAdd(PackedBytes(params->dim, params->bits - 1),
                                 PackedBytes(params->dim, 1),
                                 "TurboQuant payload byte size overflow"),
                      2 * sizeof(float), "TurboQuant payload byte size overflow");
}

template <VecSimMetric Metric>
inline IndexComponents<float, float> CreateTQComponents(
    std::shared_ptr<VecSimAllocator> allocator, std::shared_ptr<const TQModelState> state,
    TQStoredDistanceMode stored_distance_mode = TQStoredDistanceMode::FullDecodeReference) {
    if (!allocator || !state) {
        throw std::invalid_argument("TurboQuant components require allocator and model state");
    }
    if (state->getAllocator() != allocator) {
        throw std::invalid_argument(
            "TurboQuant components and model state must share one VecSim allocator");
    }
    auto index_calculator = std::unique_ptr<TQDistanceCalculator<Metric>>(
        new (allocator) TQDistanceCalculator<Metric>(allocator, state, stored_distance_mode));
    auto preprocessors = std::unique_ptr<MultiPreprocessorsContainer<float, 1>>(
        new (allocator) MultiPreprocessorsContainer<float, 1>(allocator, alignof(float)));
    auto tq_preprocessor = std::unique_ptr<TQPreprocessor<Metric>>(
        new (allocator) TQPreprocessor<Metric>(allocator, state));
    const int rc = preprocessors->addPreprocessor(tq_preprocessor.get());
    if (rc == -1) {
        throw std::runtime_error("TQ preprocessor registration failed");
    }
    tq_preprocessor.release();
    return {index_calculator.release(), preprocessors.release()};
}

template <VecSimMetric Metric>
inline IndexComponents<float, float> CreateTQComponents(
    std::shared_ptr<VecSimAllocator> allocator, TQCodecConfig config,
    TQStoredDistanceMode stored_distance_mode = TQStoredDistanceMode::FullDecodeReference) {
    auto state = AllocateTQModelState(allocator, std::move(config));
    return CreateTQComponents<Metric>(std::move(allocator), std::move(state), stored_distance_mode);
}

template <VecSimMetric Metric>
inline IndexComponents<float, float> CreateTQComponents(
    std::shared_ptr<VecSimAllocator> allocator, const TQFlatParams *params,
    TQStoredDistanceMode stored_distance_mode = TQStoredDistanceMode::FullDecodeReference) {
    auto state =
        AllocateDenseReferenceTQModelState(allocator, params->dim, params->bits,
                                           params->projections, params->seed, params->useRotation);
    return CreateTQComponents<Metric>(std::move(allocator), std::move(state), stored_distance_mode);
}

template <VecSimMetric Metric>
inline IndexComponents<float, float>
CreateTQHNSWComponents(std::shared_ptr<VecSimAllocator> allocator, const TQFlatParams *params) {
    return CreateTQComponents<Metric>(std::move(allocator), params,
                                      TQStoredDistanceMode::FullDecodeReference);
}

template <VecSimMetric Metric>
inline IndexComponents<float, float>
CreateTQHNSWComponents(std::shared_ptr<VecSimAllocator> allocator, const TQFlatParams *params,
                       TQStoredDistanceMode stored_distance_mode) {
    return CreateTQComponents<Metric>(std::move(allocator), params, stored_distance_mode);
}

template <VecSimMetric Metric>
inline IndexComponents<float, float>
CreateTQHNSWComponents(std::shared_ptr<VecSimAllocator> allocator, TQCodecConfig config,
                       TQStoredDistanceMode stored_distance_mode) {
    return CreateTQComponents<Metric>(std::move(allocator), std::move(config),
                                      stored_distance_mode);
}

template <VecSimMetric Metric>
inline IndexComponents<float, float>
CreateTQHNSWComponents(std::shared_ptr<VecSimAllocator> allocator,
                       std::shared_ptr<const TQModelState> state,
                       TQStoredDistanceMode stored_distance_mode) {
    return CreateTQComponents<Metric>(std::move(allocator), std::move(state), stored_distance_mode);
}

class TQFlatIndex : public BruteForceIndex_Single<float, float> {
public:
    TQFlatIndex(const BFParams *params, const AbstractIndexInitParams &abstract_init_params,
                const IndexComponents<float, float> &components)
        : BruteForceIndex_Single<float, float>(params, abstract_init_params, components) {}

    int addVector(const void *vector_data, labelType label) override {
        const auto existing_id = this->labelToIdLookup.find(label);
        if (existing_id != this->labelToIdLookup.end()) {
            const auto processed_blob = this->preprocessForStorage(vector_data);
            this->vectors->updateElement(existing_id->second, processed_blob.get());
            return 0;
        }
        this->appendVector(vector_data, label);
        return 1;
    }

    double getDistanceFrom_Unsafe(labelType label, const void *vector_data) const override {
        const auto optional_id = this->labelToIdLookup.find(label);
        if (optional_id == this->labelToIdLookup.end()) {
            return INVALID_SCORE;
        }
        const auto processed_query = this->preprocessQuery(vector_data);
        return this->calcDistanceForQuery(this->getDataByInternalId(optional_id->second),
                                          processed_query.get());
    }

    VecSimIndexDebugInfo debugInfo() const override {
        VecSimIndexDebugInfo info = BruteForceIndex_Single<float, float>::debugInfo();
        info.commonInfo.basicInfo.algo = VecSimAlgo_TQ;
        return info;
    }

    VecSimIndexBasicInfo basicInfo() const override {
        VecSimIndexBasicInfo info = this->getBasicInfo();
        info.algo = VecSimAlgo_TQ;
        info.isTiered = false;
        return info;
    }
};

} // namespace TQFlatDetails
