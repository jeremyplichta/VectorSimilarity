/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

#include "VecSim/algorithms/hnsw/hnsw_single.h"
#include "VecSim/algorithms/hnsw/hnsw_multi.h"

namespace TQHNSWDetails {

template <typename IndexType>
class TQHNSWAdhocBfCtx final : public VecSimAdhocBfCtx {
public:
    TQHNSWAdhocBfCtx(const IndexType &index, const void *raw_query)
        : VecSimAdhocBfCtx(index.getAllocator()), index(index),
          processed_query(index.preprocessQuery(raw_query, true)) {}

    double getDistanceFrom(labelType label) const override {
        return index.getDistanceFromPreprocessedQuery(label, processed_query.get());
    }

    // TQ-HNSW does not retain raw vectors. As with other RAM contexts, this method scores every
    // label through the same approximate TQ estimator; it is not exact FP32 reranking.
    void getExactDistances(const labelType *labels, double *distances_out,
                           size_t count) const override {
        for (size_t i = 0; i < count; ++i) {
            distances_out[i] = getDistanceFrom(labels[i]);
        }
    }

private:
    const IndexType &index;
    MemoryUtils::unique_blob processed_query;
};

template <typename DataType, typename DistType>
class TQHNSWIndex_Single : public HNSWIndex_Single<DataType, DistType> {
public:
    TQHNSWIndex_Single(const HNSWParams *params, const AbstractIndexInitParams &abstractInitParams,
                       const IndexComponents<DataType, DistType> &components,
                       size_t random_seed = 100)
        : HNSWIndex_Single<DataType, DistType>(params, abstractInitParams, components,
                                               random_seed) {}

    double getDistanceFrom_Unsafe(labelType label, const void *raw_query) const override {
        if (!this->hasLabel(label)) {
            return INVALID_SCORE;
        }
        const auto processed_query = this->preprocessQuery(raw_query);
        return getDistanceFromPreprocessedQuery(label, processed_query.get());
    }

    VecSimAdhocBfCtx *newAdhocBfCtx(const void *raw_query) const override {
        return new (this->getAllocator()) TQHNSWAdhocBfCtx<TQHNSWIndex_Single>(*this, raw_query);
    }

    double getDistanceFromPreprocessedQuery(labelType label, const void *processed_query) const {
        return HNSWIndex_Single<DataType, DistType>::getDistanceFromInternal(label,
                                                                             processed_query);
    }

    VecSimIndexDebugInfo debugInfo() const override {
        VecSimIndexDebugInfo info = HNSWIndex_Single<DataType, DistType>::debugInfo();
        info.commonInfo.basicInfo.algo = VecSimAlgo_TQ_HNSW;
        return info;
    }

    VecSimIndexBasicInfo basicInfo() const override {
        VecSimIndexBasicInfo info = this->getBasicInfo();
        info.algo = VecSimAlgo_TQ_HNSW;
        info.isTiered = false;
        return info;
    }

#ifdef BUILD_TESTS
    void saveIndex(const std::string &location) override {
        UNUSED(location);
        throw std::runtime_error("TQ-HNSW serialization is not supported yet");
    }
#endif
};

template <typename DataType, typename DistType>
class TQHNSWIndex_Multi : public HNSWIndex_Multi<DataType, DistType> {
public:
    TQHNSWIndex_Multi(const HNSWParams *params, const AbstractIndexInitParams &abstractInitParams,
                      const IndexComponents<DataType, DistType> &components,
                      size_t random_seed = 100)
        : HNSWIndex_Multi<DataType, DistType>(params, abstractInitParams, components, random_seed) {
    }

    double getDistanceFrom_Unsafe(labelType label, const void *raw_query) const override {
        if (!this->hasLabel(label)) {
            return INVALID_SCORE;
        }
        const auto processed_query = this->preprocessQuery(raw_query);
        return getDistanceFromPreprocessedQuery(label, processed_query.get());
    }

    VecSimAdhocBfCtx *newAdhocBfCtx(const void *raw_query) const override {
        return new (this->getAllocator()) TQHNSWAdhocBfCtx<TQHNSWIndex_Multi>(*this, raw_query);
    }

    double getDistanceFromPreprocessedQuery(labelType label, const void *processed_query) const {
        return HNSWIndex_Multi<DataType, DistType>::getDistanceFromInternal(label, processed_query);
    }

    VecSimIndexDebugInfo debugInfo() const override {
        VecSimIndexDebugInfo info = HNSWIndex_Multi<DataType, DistType>::debugInfo();
        info.commonInfo.basicInfo.algo = VecSimAlgo_TQ_HNSW;
        return info;
    }

    VecSimIndexBasicInfo basicInfo() const override {
        VecSimIndexBasicInfo info = this->getBasicInfo();
        info.algo = VecSimAlgo_TQ_HNSW;
        info.isTiered = false;
        return info;
    }

#ifdef BUILD_TESTS
    void saveIndex(const std::string &location) override {
        UNUSED(location);
        throw std::runtime_error("TQ-HNSW serialization is not supported yet");
    }
#endif
};

} // namespace TQHNSWDetails
