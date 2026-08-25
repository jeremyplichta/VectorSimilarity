/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include "VecSim/index_factories/tq_factory.h"

#include "VecSim/algorithms/tq/tq_flat.h"

namespace TQFactory {
namespace {

void ValidateParams(const TQFlatParams &params) {
    TQFlatDetails::ValidatePublicTQParams(params);
    if (params.multi) {
        throw std::invalid_argument("TQ-FLAT currently supports single-value indexes only");
    }
}

void AddEstimate(size_t &estimate, size_t bytes) {
    estimate =
        TQFlatDetails::CheckedAdd(estimate, bytes, "TurboQuant initial-size estimate overflow");
}

AbstractIndexInitParams NewAbstractInitParams(const TQFlatParams *params, void *logCtx,
                                              std::shared_ptr<VecSimAllocator> allocator,
                                              size_t stored_data_size) {
    return {.allocator = allocator,
            .dim = params->dim,
            .vecType = params->type,
            .storedDataSize = stored_data_size,
            .metric = params->metric,
            .blockSize = params->blockSize,
            .multi = false,
            .isDisk = false,
            .logCtx = logCtx,
            .inputBlobSize =
                TQFlatDetails::CheckedBytes(params->dim, VecSimType_sizeof(params->type),
                                            "TurboQuant input byte size overflow")};
}

template <VecSimMetric Metric>
VecSimIndex *NewIndexImpl(const VecSimParams *params) {
    const auto &tq_params = params->algoParams.tqFlatParams;
    auto allocator = VecSimAllocator::newVecsimAllocator();
    auto components = TQFlatDetails::CreateTQComponents<Metric>(allocator, &tq_params);
    auto stored_data_size = TQFlatDetails::GetStorageDataSize<Metric>(&tq_params);
    auto abstract_init_params =
        NewAbstractInitParams(&tq_params, params->logCtx, allocator, stored_data_size);
    BFParams bf_params = {.type = tq_params.type,
                          .dim = tq_params.dim,
                          .metric = tq_params.metric,
                          .multi = false,
                          .initialCapacity = tq_params.initialCapacity,
                          .blockSize = tq_params.blockSize};
    return new (allocator) TQFlatDetails::TQFlatIndex(&bf_params, abstract_init_params, components);
}

template <VecSimMetric Metric>
size_t EstimateInitialSizeImpl(const TQFlatParams *params) {
    size_t allocations_overhead = VecSimAllocator::getAllocationOverheadSize();
    size_t est = TQFlatDetails::CheckedAdd(sizeof(VecSimAllocator), allocations_overhead,
                                           "TurboQuant initial-size estimate overflow");
    AddEstimate(est, sizeof(TQFlatDetails::TQFlatIndex));
    AddEstimate(est, sizeof(DataBlocksContainer) + allocations_overhead);
    AddEstimate(est, allocations_overhead + sizeof(TQFlatDetails::TQDistanceCalculator<Metric>));
    AddEstimate(est, allocations_overhead + sizeof(MultiPreprocessorsContainer<float, 1>));
    AddEstimate(est, allocations_overhead + sizeof(TQFlatDetails::TQPreprocessor<Metric>));
    AddEstimate(est, TQFlatDetails::EstimateTQModelAllocationSize(
                         TQFlatDetails::TQCodecConfigFromPublicParams(*params)));
    return est;
}

template <VecSimMetric Metric>
size_t EstimateElementSizeImpl(const TQFlatParams *params) {
    size_t estimate = TQFlatDetails::GetStorageDataSize<Metric>(params);
    AddEstimate(estimate, sizeof(labelType));
    AddEstimate(estimate, sizeof(void *));
    return estimate;
}

} // namespace

VecSimIndex *NewIndex(const VecSimParams *params) {
    const auto &tq_params = params->algoParams.tqFlatParams;
    ValidateParams(tq_params);

    switch (tq_params.metric) {
    case VecSimMetric_L2:
        break;
    case VecSimMetric_IP:
        return NewIndexImpl<VecSimMetric_IP>(params);
    case VecSimMetric_Cosine:
        return NewIndexImpl<VecSimMetric_Cosine>(params);
    }
    return nullptr;
}

size_t EstimateInitialSize(const TQFlatParams *params) {
    ValidateParams(*params);
    switch (params->metric) {
    case VecSimMetric_L2:
        break;
    case VecSimMetric_IP:
        return EstimateInitialSizeImpl<VecSimMetric_IP>(params);
    case VecSimMetric_Cosine:
        return EstimateInitialSizeImpl<VecSimMetric_Cosine>(params);
    }
    return 0;
}

size_t EstimateElementSize(const TQFlatParams *params) {
    ValidateParams(*params);
    switch (params->metric) {
    case VecSimMetric_L2:
        break;
    case VecSimMetric_IP:
        return EstimateElementSizeImpl<VecSimMetric_IP>(params);
    case VecSimMetric_Cosine:
        return EstimateElementSizeImpl<VecSimMetric_Cosine>(params);
    }
    return 0;
}

} // namespace TQFactory
