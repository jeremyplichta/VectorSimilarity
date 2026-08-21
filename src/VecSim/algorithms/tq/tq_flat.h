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
#include "VecSim/spaces/computer/calculator.h"
#include "VecSim/spaces/computer/preprocessor_container.h"
#include "VecSim/utils/vec_utils.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace TQFlatDetails {

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

template <VecSimMetric Metric>
class TQDistanceCalculator : public IndexCalculatorInterface<float> {
private:
    static float calcStoredWithContext(const void *opaque_state, const void *lhs_blob,
                                       const void *rhs_blob, size_t dim) {
        const auto *state = static_cast<const TQModelState *>(opaque_state);
        assert(dim == state->dim);
        std::vector<float> lhs(dim);
        std::vector<float> rhs(dim);
        state->decode(state->storageView(lhs_blob), lhs.data());
        state->decode(state->storageView(rhs_blob), rhs.data());

        if constexpr (Metric == VecSimMetric_Cosine) {
            NormalizeInPlace(lhs.data(), dim);
            NormalizeInPlace(rhs.data(), dim);
        }
        if constexpr (Metric == VecSimMetric_L2) {
            float distance = 0.0f;
            for (size_t i = 0; i < dim; ++i) {
                const float difference = lhs[i] - rhs[i];
                distance += difference * difference;
            }
            return distance;
        }
        return 1.0f - DotProductScalar(lhs.data(), rhs.data(), dim);
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
    TQDistanceCalculator(std::shared_ptr<VecSimAllocator> allocator,
                         std::shared_ptr<TQModelState> state)
        : IndexCalculatorInterface<float>(allocator), state(std::move(state)) {}

    float calcDistance(const void *v1, const void *v2, size_t dim) const override {
        return calcStoredWithContext(state.get(), v1, v2, dim);
    }

    float calcDistanceForQuery(const void *candidate_vector, const void *query_vector,
                               size_t dim) const override {
        return calcQueryWithContext(state.get(), candidate_vector, query_vector, dim);
    }

    DistanceDispatch<float> getDistanceDispatch(DistanceMode mode) const override {
        return mode == DistanceMode::StoredToStored
                   ? DistanceDispatch<float>::stateful(state.get(), calcStoredWithContext)
                   : DistanceDispatch<float>::stateful(state.get(), calcQueryWithContext);
    }

private:
    std::shared_ptr<TQModelState> state;
};

template <VecSimMetric Metric>
class TQPreprocessor : public PreprocessorInterface {
public:
    TQPreprocessor(std::shared_ptr<VecSimAllocator> allocator, std::shared_ptr<TQModelState> state)
        : PreprocessorInterface(allocator), state(std::move(state)) {}

    void preprocess(const void *original_blob, void *&storage_blob, void *&query_blob,
                    size_t &storage_blob_size, size_t &query_blob_size,
                    unsigned char storage_alignment, unsigned char query_alignment) const override {
        preprocessForStorage(original_blob, storage_blob, storage_blob_size, storage_alignment);
        preprocessQuery(original_blob, query_blob, query_blob_size, query_alignment);
    }

    void preprocessForStorage(const void *original_blob, void *&storage_blob,
                              size_t &input_blob_size,
                              unsigned char storage_alignment) const override {
        if (!storage_blob) {
            storage_blob =
                this->allocator->allocate_aligned(state->storageBlobSize(), storage_alignment);
        }
        std::memset(storage_blob, 0, state->storageBlobSize());

        const auto *input = static_cast<const float *>(original_blob);
        std::vector<float> unit(input, input + state->dim);
        const float original_norm = NormalizeInPlace(unit.data(), state->dim);
        if (original_norm == 0.0f) {
            state->writeMetadata(storage_blob, 0.0f, 0.0f);
            input_blob_size = state->storageBlobSize();
            return;
        }

        auto *indices = static_cast<uint8_t *>(storage_blob);
        auto *signs = indices + state->packedIndexBytes();
        std::vector<float> reconstructed(state->dim);
        std::vector<float> residual(state->dim);
        state->encodeMse(unit.data(), indices, reconstructed.data());
        for (size_t i = 0; i < state->dim; ++i) {
            residual[i] = unit[i] - reconstructed[i];
        }

        const float residual_norm = std::sqrt(SumSquaresScalar(residual.data(), state->dim));
        if (residual_norm > 0.0f) {
            const float inverse_residual_norm = 1.0f / residual_norm;
            for (float &value : residual) {
                value *= inverse_residual_norm;
            }
            state->packResidualSigns(residual.data(), signs);
        } else {
            // sign(0) is +1. The signs are ignored because gamma is zero, but keeping the
            // canonical mathematical value makes byte-level tests unambiguous.
            std::memset(signs, 0xFF, state->packedQjlBytes());
        }

        const float source_scale = Metric == VecSimMetric_Cosine ? 1.0f : original_norm;
        state->writeMetadata(storage_blob, source_scale, residual_norm);
        input_blob_size = state->storageBlobSize();
    }

    void preprocessQuery(const void *original_blob, void *&query_blob, size_t &input_blob_size,
                         unsigned char alignment) const override {
        if (!query_blob) {
            query_blob = this->allocator->allocate_aligned(state->queryBlobSize(), alignment);
        }

        const auto *input = static_cast<const float *>(original_blob);
        std::vector<float> query(input, input + state->dim);
        if constexpr (Metric == VecSimMetric_Cosine) {
            NormalizeInPlace(query.data(), state->dim);
        }

        auto *words = static_cast<float *>(query_blob);
        auto *rotated = words;
        auto *projected = rotated + state->dim;
        auto *norm_sq = projected + state->projections;
        state->applyRotation(query.data(), rotated);
        state->projectQjl(query.data(), projected);
        *norm_sq = SumSquaresScalar(query.data(), state->dim);
        input_blob_size = state->queryBlobSize();
    }

    void preprocessStorageInPlace(void *original_blob, size_t input_blob_size) const override {
        assert(original_blob);
        assert(input_blob_size >= state->storageBlobSize());
        std::vector<uint8_t> encoded(state->storageBlobSize());
        void *encoded_blob = encoded.data();
        size_t encoded_size = input_blob_size;
        preprocessForStorage(original_blob, encoded_blob, encoded_size, 0);
        std::memcpy(original_blob, encoded.data(), state->storageBlobSize());
    }

private:
    std::shared_ptr<TQModelState> state;
};

template <VecSimMetric Metric>
inline size_t GetStorageDataSize(const TQFlatParams *params) {
    MseBits(params->bits);
    if (params->dim < 2) {
        throw std::invalid_argument("TurboQuant requires dimension >= 2");
    }
    if (params->projections != params->dim) {
        throw std::invalid_argument("Paper-faithful TurboQuant requires projections == dim");
    }
    return PackedBytes(params->dim, params->bits - 1) + PackedBytes(params->dim, 1) +
           2 * sizeof(float);
}

template <VecSimMetric Metric>
inline IndexComponents<float, float> CreateTQComponents(std::shared_ptr<VecSimAllocator> allocator,
                                                        const TQFlatParams *params) {
    auto state = std::make_shared<TQModelState>(params->dim, params->bits, params->projections,
                                                params->seed, params->useRotation);
    auto *index_calculator = new (allocator) TQDistanceCalculator<Metric>(allocator, state);
    auto *preprocessors =
        new (allocator) MultiPreprocessorsContainer<float, 1>(allocator, alignof(float));
    auto *tq_preprocessor = new (allocator) TQPreprocessor<Metric>(allocator, state);
    const int rc = preprocessors->addPreprocessor(tq_preprocessor);
    UNUSED(rc);
    assert(rc != -1 && "TQ preprocessor was not added correctly");
    return {index_calculator, preprocessors};
}

template <VecSimMetric Metric>
inline IndexComponents<float, float>
CreateTQHNSWComponents(std::shared_ptr<VecSimAllocator> allocator, const TQFlatParams *params) {
    return CreateTQComponents<Metric>(std::move(allocator), params);
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
