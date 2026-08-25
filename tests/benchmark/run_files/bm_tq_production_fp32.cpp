/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

// Native decision-record benchmark for the TurboQuant production matrix.  This intentionally
// has a small standalone runner instead of sharing the dataset-backed VecSim benchmark fixtures:
// its inputs, metadata, unsupported-backend records, and smoke assertions are part of its public
// reproducibility contract.

#include "VecSim/algorithms/tq/tq_flat.h"
#include "VecSim/algorithms/tq/tq_hnsw.h"
#include "VecSim/algorithms/tq/tq_stored_distance.h"
#include "VecSim/vec_sim.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#include <sys/utsname.h>

#ifndef TQ_BENCH_GIT_COMMIT
#define TQ_BENCH_GIT_COMMIT "unknown"
#endif
#ifndef TQ_BENCH_GIT_DIRTY
#define TQ_BENCH_GIT_DIRTY "unknown"
#endif
#ifndef TQ_BENCH_BUILD_MODE
#define TQ_BENCH_BUILD_MODE "unknown"
#endif

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::array<size_t, 6> kProductionDimensions = {768, 1000, 1024, 1536, 3000, 3072};
constexpr std::array<size_t, 3> kBitWidths = {2, 4, 8};
constexpr std::array<VecSimMetric, 2> kMetrics = {VecSimMetric_IP, VecSimMetric_Cosine};
constexpr size_t kSeed = 0x5EED5EED;
constexpr size_t kM = 16;
constexpr size_t kEfConstruction = 64;
constexpr size_t kEfRuntime = 32;
constexpr size_t kTopK = 10;

enum class Distribution { Gaussian, UnitGaussian, LogUniformNormIp, Correlated };
enum class RunMode { Smoke, Full, List };

struct Config {
    RunMode mode = RunMode::Smoke;
    size_t corpus_size = 32;
    size_t query_count = 8;
    size_t repetitions = 3;
    std::optional<size_t> dim_filter;
    std::optional<size_t> bits_filter;
    std::optional<VecSimMetric> metric_filter;
    std::optional<Distribution> distribution_filter;
    std::optional<std::string> implementation_filter;
    std::optional<size_t> corpus_size_override;
    std::optional<size_t> query_count_override;
    std::optional<size_t> repetitions_override;
};

struct Metadata {
    std::string git_commit = TQ_BENCH_GIT_COMMIT;
    std::string git_dirty = TQ_BENCH_GIT_DIRTY;
    std::string build_mode = TQ_BENCH_BUILD_MODE;
    std::string compiler;
    std::string cpu_model;
    std::string architecture;
    std::string simd;
    std::string assertions;
};

struct Result {
    std::string implementation;
    std::string run_profile;
    std::string operation;
    std::string status = "ok";
    std::string note;
    std::string transform_version = "not_applicable";
    std::string qjl_version = "not_applicable";
    std::string model_profile_version = "not_applicable";
    std::string stored_distance_version = "not_applicable";
    size_t dim = 0;
    size_t bits = 0;
    VecSimMetric metric = VecSimMetric_IP;
    Distribution distribution = Distribution::Gaussian;
    size_t corpus_size = 0;
    size_t query_count = 0;
    double median_ns = 0.0;
    double p95_ns = 0.0;
    double checksum = 0.0;
    size_t sample_count = 0;
    double recall_at_10 = std::numeric_limits<double>::quiet_NaN();
    double signed_bias = std::numeric_limits<double>::quiet_NaN();
    double mae = std::numeric_limits<double>::quiet_NaN();
    double rmse = std::numeric_limits<double>::quiet_NaN();
    double p95_absolute_error = std::numeric_limits<double>::quiet_NaN();
    size_t payload_bytes = 0;
    size_t metadata_bytes = 0;
    size_t model_bytes = 0;
    size_t graph_and_container_bytes = 0;
    size_t allocator_total_bytes = 0;
    std::string peak_memory_status = "not_supported_by_current_allocator_api";
    std::string graph_integrity_status = "not_checked";
    size_t graph_connections_to_repair = 0;
    size_t graph_bidirectional_connections = 0;
    size_t graph_unidirectional_connections = 0;
    double graph_average_directed_degree = std::numeric_limits<double>::quiet_NaN();
};

std::string MetricName(VecSimMetric metric) { return metric == VecSimMetric_IP ? "IP" : "COSINE"; }

std::string DistributionName(Distribution distribution) {
    switch (distribution) {
    case Distribution::Gaussian:
        return "iid_standard_gaussian";
    case Distribution::UnitGaussian:
        return "unit_normalized_gaussian";
    case Distribution::LogUniformNormIp:
        return "log_uniform_norm_ip";
    case Distribution::Correlated:
        return "correlated_decaying_coordinate_variance";
    }
    return "unknown";
}

std::string CpuModel() {
#if defined(__APPLE__)
    size_t size = 0;
    if (sysctlbyname("machdep.cpu.brand_string", nullptr, &size, nullptr, 0) == 0 && size != 0) {
        std::string result(size, '\0');
        if (sysctlbyname("machdep.cpu.brand_string", result.data(), &size, nullptr, 0) == 0) {
            result.resize(size > 0 ? size - 1 : 0);
            return result;
        }
    }
#endif
#if defined(__linux__)
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        const size_t separator = line.find(':');
        if (separator != std::string::npos &&
            (line.rfind("model name", 0) == 0 || line.rfind("Hardware", 0) == 0)) {
            const size_t first = line.find_first_not_of(" \t", separator + 1);
            return first == std::string::npos ? "unavailable" : line.substr(first);
        }
    }
#endif
    return "unavailable";
}

Metadata GetMetadata() {
    struct utsname uts {};
    uname(&uts);
    Metadata metadata;
    metadata.cpu_model = CpuModel();
    metadata.architecture = uts.machine;
#if defined(__clang__)
    metadata.compiler = std::string("clang-") + __clang_version__;
#elif defined(__GNUC__)
    metadata.compiler = "gcc-" + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__);
#else
    metadata.compiler = "unknown";
#endif
    metadata.simd = TQFlatDetails::HasPaperTqSimd() ? "paper_tq_simd" : "scalar";
#ifdef NDEBUG
    metadata.assertions = "disabled";
#else
    metadata.assertions = "enabled";
#endif
    return metadata;
}

class DeterministicUniformRng {
public:
    explicit DeterministicUniformRng(uint64_t seed) : state(seed) {}

    double uniformOpen() {
        constexpr double kInv53 = 1.0 / static_cast<double>(uint64_t{1} << 53);
        return (static_cast<double>(next() >> 11) + 0.5) * kInv53;
    }

private:
    uint64_t next() {
        uint64_t value = (state += 0x9E3779B97F4A7C15ULL);
        value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
        value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
        return value ^ (value >> 31);
    }

    uint64_t state;
};

std::vector<float> MakeVector(size_t dim, Distribution distribution,
                              TQFlatDetails::DeterministicGaussianRng &normal,
                              DeterministicUniformRng &uniform) {
    std::vector<float> result(dim);
    constexpr float kCorrelation = 0.85f;
    constexpr float kInnovationScale = 0.5267827f;
    float previous = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float value = normal.normal();
        if (distribution == Distribution::Correlated) {
            // AR(1) makes neighboring coordinates genuinely correlated; the deterministic
            // envelope then supplies the requested decaying coordinate variance.
            const float correlated =
                i == 0 ? value : kCorrelation * previous + kInnovationScale * value;
            previous = correlated;
            value = correlated * std::pow(0.996f, static_cast<float>(i));
        }
        result[i] = value;
    }
    const float norm_sq = TQFlatDetails::SumSquaresScalar(result.data(), dim);
    if (distribution == Distribution::UnitGaussian && norm_sq > 0.0f) {
        const float inverse_norm = 1.0f / std::sqrt(norm_sq);
        for (float &value : result) {
            value *= inverse_norm;
        }
    } else if (distribution == Distribution::LogUniformNormIp && norm_sq > 0.0f) {
        const float target_norm =
            std::pow(10.0f, static_cast<float>(-3.0 + 6.0 * uniform.uniformOpen()));
        const float scale = target_norm / std::sqrt(norm_sq);
        for (float &value : result) {
            value *= scale;
        }
    }
    return result;
}

std::vector<std::vector<float>> MakeDataset(size_t count, size_t dim, Distribution distribution,
                                            uint64_t seed) {
    TQFlatDetails::DeterministicGaussianRng normal(seed);
    DeterministicUniformRng uniform(seed ^ 0x9E3779B97F4A7C15ULL);
    std::vector<std::vector<float>> vectors;
    vectors.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        vectors.push_back(MakeVector(dim, distribution, normal, uniform));
    }
    return vectors;
}

std::vector<std::vector<float>> MakeCorpusDataset(const Config &config, size_t dim,
                                                  Distribution distribution) {
    return MakeDataset(config.corpus_size, dim, distribution, kSeed + dim);
}

std::vector<std::vector<float>> MakeQueryDataset(const Config &config, size_t dim,
                                                 Distribution distribution) {
    return MakeDataset(config.query_count, dim, distribution, kSeed + 0x200000 + dim);
}

TQFlatParams TqFlatParams(size_t dim, size_t bits, VecSimMetric metric, size_t capacity) {
    return {.type = VecSimType_FLOAT32,
            .dim = dim,
            .metric = metric,
            .multi = false,
            .initialCapacity = capacity,
            .blockSize = std::max<size_t>(capacity, 1),
            .bits = bits,
            .projections = dim,
            .seed = kSeed,
            .useRotation = true};
}

TQHNSWParams TqHnswParams(size_t dim, size_t bits, VecSimMetric metric, size_t capacity) {
    return {.type = VecSimType_FLOAT32,
            .dim = dim,
            .metric = metric,
            .multi = false,
            .initialCapacity = capacity,
            .blockSize = std::max<size_t>(capacity, 1),
            .bits = bits,
            .projections = dim,
            .seed = kSeed,
            .useRotation = true,
            .M = kM,
            .efConstruction = kEfConstruction,
            .efRuntime = kEfRuntime,
            .epsilon = 0.0};
}

TQFlatDetails::TQFlatIndex *NewTqFlat(size_t dim, size_t bits, VecSimMetric metric,
                                      size_t capacity) {
    VecSimParams params = {
        .algo = VecSimAlgo_TQ,
        .algoParams = {.tqFlatParams = TqFlatParams(dim, bits, metric, capacity)},
        .logCtx = nullptr};
    return static_cast<TQFlatDetails::TQFlatIndex *>(VecSimIndex_New(&params));
}

size_t TqStorageBlobSize(size_t dim, size_t bits) {
    return TQFlatDetails::PackedBytes(dim, bits - 1) + TQFlatDetails::PackedBytes(dim, 1) +
           2 * sizeof(float);
}

template <VecSimMetric Metric>
TQFlatDetails::TQFlatIndex *NewTqFlatWithConfig(size_t dim, size_t bits, size_t capacity,
                                                TQFlatDetails::TQCodecConfig config,
                                                TQFlatDetails::TQStoredDistanceMode mode) {
    auto allocator = VecSimAllocator::newVecsimAllocator();
    auto components = TQFlatDetails::CreateTQComponents<Metric>(allocator, std::move(config), mode);
    const BFParams bf_params = {.type = VecSimType_FLOAT32,
                                .dim = dim,
                                .metric = Metric,
                                .multi = false,
                                .initialCapacity = capacity,
                                .blockSize = std::max<size_t>(capacity, 1)};
    const AbstractIndexInitParams init = {
        .allocator = allocator,
        .dim = dim,
        .vecType = VecSimType_FLOAT32,
        .storedDataSize = TqStorageBlobSize(dim, bits),
        .metric = Metric,
        .blockSize = bf_params.blockSize,
        .multi = false,
        .isDisk = false,
        .logCtx = nullptr,
        .inputBlobSize = dim * sizeof(float),
    };
    return new (allocator) TQFlatDetails::TQFlatIndex(&bf_params, init, components);
}

using TQTypedHNSWIndex = TQHNSWDetails::TQHNSWIndex_Single<float, float>;

template <VecSimMetric Metric>
TQTypedHNSWIndex *NewTqHnswWithStoredDistanceMode(size_t dim, size_t bits, size_t capacity,
                                                  TQFlatDetails::TQStoredDistanceMode mode) {
    const TQHNSWParams params = TqHnswParams(dim, bits, Metric, capacity);
    const TQFlatParams flat_params = TqFlatParams(dim, bits, Metric, capacity);
    auto allocator = VecSimAllocator::newVecsimAllocator();
    auto components = TQFlatDetails::CreateTQHNSWComponents<Metric>(allocator, &flat_params, mode);
    const AbstractIndexInitParams init = {
        .allocator = allocator,
        .dim = dim,
        .vecType = VecSimType_FLOAT32,
        .storedDataSize = TQFlatDetails::GetStorageDataSize<Metric>(&flat_params),
        .metric = Metric,
        .blockSize = params.blockSize,
        .multi = false,
        .isDisk = false,
        .logCtx = nullptr,
        .inputBlobSize = dim * sizeof(float),
    };
    const HNSWParams hnsw_params = {.type = params.type,
                                    .dim = params.dim,
                                    .metric = params.metric,
                                    .multi = false,
                                    .initialCapacity = params.initialCapacity,
                                    .blockSize = params.blockSize,
                                    .M = params.M,
                                    .efConstruction = params.efConstruction,
                                    .efRuntime = params.efRuntime,
                                    .epsilon = params.epsilon};
    return new (allocator)
        TQHNSWDetails::TQHNSWIndex_Single<float, float>(&hnsw_params, init, components, kSeed);
}

template <VecSimMetric Metric>
TQTypedHNSWIndex *NewTqHnswWithConfig(size_t dim, size_t bits, size_t capacity,
                                      TQFlatDetails::TQCodecConfig config,
                                      TQFlatDetails::TQStoredDistanceMode mode) {
    const TQHNSWParams params = TqHnswParams(dim, bits, Metric, capacity);
    auto allocator = VecSimAllocator::newVecsimAllocator();
    auto components =
        TQFlatDetails::CreateTQHNSWComponents<Metric>(allocator, std::move(config), mode);
    const AbstractIndexInitParams init = {
        .allocator = allocator,
        .dim = dim,
        .vecType = VecSimType_FLOAT32,
        .storedDataSize = TqStorageBlobSize(dim, bits),
        .metric = Metric,
        .blockSize = params.blockSize,
        .multi = false,
        .isDisk = false,
        .logCtx = nullptr,
        .inputBlobSize = dim * sizeof(float),
    };
    const HNSWParams hnsw_params = {.type = params.type,
                                    .dim = params.dim,
                                    .metric = params.metric,
                                    .multi = false,
                                    .initialCapacity = params.initialCapacity,
                                    .blockSize = params.blockSize,
                                    .M = params.M,
                                    .efConstruction = params.efConstruction,
                                    .efRuntime = params.efRuntime,
                                    .epsilon = params.epsilon};
    return new (allocator)
        TQHNSWDetails::TQHNSWIndex_Single<float, float>(&hnsw_params, init, components, kSeed);
}

TQTypedHNSWIndex *NewTqHnsw(size_t dim, size_t bits, VecSimMetric metric, size_t capacity,
                            TQFlatDetails::TQStoredDistanceMode mode) {
    if (metric == VecSimMetric_IP) {
        return NewTqHnswWithStoredDistanceMode<VecSimMetric_IP>(dim, bits, capacity, mode);
    }
    return NewTqHnswWithStoredDistanceMode<VecSimMetric_Cosine>(dim, bits, capacity, mode);
}

VecSimIndex *NewFp32Flat(size_t dim, VecSimMetric metric, size_t capacity) {
    BFParams params = {.type = VecSimType_FLOAT32,
                       .dim = dim,
                       .metric = metric,
                       .multi = false,
                       .initialCapacity = capacity,
                       .blockSize = std::max<size_t>(capacity, 1)};
    VecSimParams index_params = {
        .algo = VecSimAlgo_BF, .algoParams = {.bfParams = params}, .logCtx = nullptr};
    return VecSimIndex_New(&index_params);
}

void Populate(VecSimIndex *index, const std::vector<std::vector<float>> &vectors) {
    for (size_t i = 0; i < vectors.size(); ++i) {
        if (VecSimIndex_AddVector(index, vectors[i].data(), i) != 1) {
            throw std::runtime_error("failed to add benchmark vector");
        }
    }
}

double Percentile(std::vector<double> samples, double fraction) {
    if (samples.empty()) {
        return 0.0;
    }
    std::sort(samples.begin(), samples.end());
    const size_t position =
        std::min(samples.size() - 1, static_cast<size_t>(std::ceil(fraction * samples.size())) - 1);
    return samples[position];
}

template <typename Function>
std::pair<double, double> Timed(size_t repetitions, Function &&function) {
    std::vector<double> samples;
    samples.reserve(repetitions);
    function(); // warmup; creation/allocation is intentionally measured only by its own operation.
    for (size_t repetition = 0; repetition < repetitions; ++repetition) {
        const auto begin = Clock::now();
        function();
        const auto end = Clock::now();
        samples.push_back(std::chrono::duration<double, std::nano>(end - begin).count());
    }
    return {Percentile(samples, 0.50), Percentile(samples, 0.95)};
}

struct TimingSummary {
    double median_ns;
    double p95_ns;
    size_t sample_count;
};

template <typename Items, typename Prepare, typename Function>
TimingSummary TimedEachPrepared(size_t repetitions, const Items &items, Prepare &&prepare,
                                Function &&function) {
    if (items.empty()) {
        throw std::invalid_argument("benchmark timing requires at least one item");
    }
    for (const auto &item : items) {
        prepare(item);
        function(item); // Warm every query represented by the recorded distribution.
    }
    if (repetitions > std::numeric_limits<size_t>::max() / items.size()) {
        throw std::overflow_error("benchmark timing sample count overflow");
    }
    const size_t sample_count = repetitions * items.size();
    std::vector<double> samples;
    samples.reserve(sample_count);
    for (size_t repetition = 0; repetition < repetitions; ++repetition) {
        for (const auto &item : items) {
            prepare(item);
            const auto begin = Clock::now();
            function(item);
            const auto end = Clock::now();
            samples.push_back(std::chrono::duration<double, std::nano>(end - begin).count());
        }
    }
    return {.median_ns = Percentile(samples, 0.50),
            .p95_ns = Percentile(samples, 0.95),
            .sample_count = sample_count};
}

template <typename Items, typename Function>
TimingSummary TimedEach(size_t repetitions, const Items &items, Function &&function) {
    return TimedEachPrepared(
        repetitions, items, [](const auto &) {}, std::forward<Function>(function));
}

void ApplyTiming(Result &result, const TimingSummary &timing) {
    result.median_ns = timing.median_ns;
    result.p95_ns = timing.p95_ns;
    result.sample_count = timing.sample_count;
}

template <typename Setup, typename Function, typename Teardown>
std::pair<double, double> TimedPrepared(size_t repetitions, Setup &&setup, Function &&function,
                                        Teardown &&teardown) {
    auto warmup = setup();
    function(warmup);
    teardown(warmup);
    std::vector<double> samples;
    samples.reserve(repetitions);
    for (size_t repetition = 0; repetition < repetitions; ++repetition) {
        auto state = setup();
        const auto begin = Clock::now();
        function(state);
        const auto end = Clock::now();
        samples.push_back(std::chrono::duration<double, std::nano>(end - begin).count());
        teardown(state);
    }
    return {Percentile(samples, 0.50), Percentile(samples, 0.95)};
}

std::vector<size_t> Labels(VecSimQueryReply *reply) {
    std::vector<size_t> labels;
    VecSimQueryReply_Iterator *iterator = VecSimQueryReply_GetIterator(reply);
    while (VecSimQueryReply_IteratorHasNext(iterator)) {
        labels.push_back(
            static_cast<size_t>(VecSimQueryResult_GetId(VecSimQueryReply_IteratorNext(iterator))));
    }
    VecSimQueryReply_IteratorFree(iterator);
    return labels;
}

double ExactDistance(const std::vector<float> &query, const std::vector<float> &candidate,
                     VecSimMetric metric) {
    const double dot =
        TQFlatDetails::DotProductScalar(query.data(), candidate.data(), query.size());
    if (metric == VecSimMetric_IP) {
        return 1.0 - dot;
    }
    const double query_norm_sq = TQFlatDetails::SumSquaresScalar(query.data(), query.size());
    const double candidate_norm_sq =
        TQFlatDetails::SumSquaresScalar(candidate.data(), candidate.size());
    if (query_norm_sq == 0.0 || candidate_norm_sq == 0.0) {
        return 1.0;
    }
    return 1.0 - dot / std::sqrt(query_norm_sq * candidate_norm_sq);
}

double RecallAt10(VecSimIndex *index, const std::vector<std::vector<float>> &queries,
                  const std::vector<std::vector<float>> &corpus, VecSimMetric metric) {
    size_t matches = 0;
    for (const auto &query : queries) {
        std::vector<std::pair<double, size_t>> exact;
        exact.reserve(corpus.size());
        for (size_t i = 0; i < corpus.size(); ++i) {
            exact.emplace_back(ExactDistance(query, corpus[i], metric), i);
        }
        std::partial_sort(exact.begin(), exact.begin() + std::min(kTopK, exact.size()),
                          exact.end());
        std::set<size_t> expected;
        for (size_t i = 0; i < std::min(kTopK, exact.size()); ++i) {
            expected.insert(exact[i].second);
        }
        HNSWRuntimeParams runtime = {.efRuntime = kEfRuntime};
        VecSimQueryParams params = {.hnswRuntimeParams = runtime};
        VecSimQueryReply *reply =
            VecSimIndex_TopKQuery(index, query.data(), kTopK, &params, BY_SCORE);
        for (size_t label : Labels(reply)) {
            matches += expected.contains(label) ? 1 : 0;
        }
        VecSimQueryReply_Free(reply);
    }
    return queries.empty() ? 0.0 : static_cast<double>(matches) / (queries.size() * kTopK);
}

struct ErrorStats {
    double signed_bias;
    double mae;
    double rmse;
    double p95_absolute_error;
};

Result BaseResult(std::string implementation, std::string operation, size_t dim, size_t bits,
                  VecSimMetric metric, Distribution distribution, const Config &config);
std::string RunProfile(const Config &config);

ErrorStats CompareWithExactFp32(TQFlatDetails::TQFlatIndex *index,
                                const std::vector<std::vector<float>> &queries,
                                const std::vector<std::vector<float>> &corpus,
                                VecSimMetric metric) {
#ifdef BUILD_TESTS
    const size_t preprocessing_before = TQFlatDetails::GetQueryPreprocessingCount();
#endif
    std::vector<double> absolute_errors;
    absolute_errors.reserve(queries.size() * corpus.size());
    double signed_sum = 0.0;
    double absolute_sum = 0.0;
    double squared_sum = 0.0;
    for (const auto &query : queries) {
        // This benchmark populates monotonically increasing labels into an empty single-value
        // flat index, so label and internal id are identical. Preprocess once, then reuse the
        // cached stored-to-query dispatch for the complete candidate loop.
        const auto processed_query = index->preprocessQuery(query.data(), true);
        for (size_t label = 0; label < corpus.size(); ++label) {
            const double approximate = index->calcDistanceForQuery(
                index->getDataByInternalId(label), processed_query.get());
            // Candidate calculators return distance.  exact - approximate distance is the
            // corresponding approximate - exact similarity estimator bias.
            const double error = ExactDistance(query, corpus[label], metric) - approximate;
            signed_sum += error;
            absolute_sum += std::abs(error);
            squared_sum += error * error;
            absolute_errors.push_back(std::abs(error));
        }
    }
#ifdef BUILD_TESTS
    if (TQFlatDetails::GetQueryPreprocessingCount() - preprocessing_before != queries.size()) {
        throw std::runtime_error("dense quality benchmark did not preprocess each query once");
    }
#endif
    const double count = static_cast<double>(absolute_errors.size());
    return {.signed_bias = signed_sum / count,
            .mae = absolute_sum / count,
            .rmse = std::sqrt(squared_sum / count),
            .p95_absolute_error = Percentile(absolute_errors, 0.95)};
}

template <VecSimMetric Metric>
ErrorStats
CompareDirectModelWithExact(const std::shared_ptr<VecSimAllocator> &allocator,
                            const std::shared_ptr<const TQFlatDetails::TQModelState> &state,
                            const std::vector<std::vector<float>> &queries,
                            const std::vector<std::vector<float>> &corpus) {
    TQFlatDetails::TQPreprocessor<Metric> preprocessor(allocator, state);
    TQFlatDetails::TQDistanceCalculator<Metric> calculator(
        allocator, state, TQFlatDetails::TQStoredDistanceMode::CoarseMse);
    std::vector<void *> encoded;
    encoded.reserve(corpus.size());
    for (const auto &vector : corpus) {
        void *blob = nullptr;
        size_t blob_size = 0;
        preprocessor.preprocessForStorage(vector.data(), blob, blob_size, alignof(float));
        encoded.push_back(blob);
    }
    std::vector<double> absolute_errors;
    absolute_errors.reserve(queries.size() * corpus.size());
    double signed_sum = 0.0;
    double absolute_sum = 0.0;
    double squared_sum = 0.0;
    for (const auto &query : queries) {
        void *query_blob = nullptr;
        size_t query_blob_size = 0;
        preprocessor.preprocessQuery(query.data(), query_blob, query_blob_size, alignof(float));
        for (size_t label = 0; label < corpus.size(); ++label) {
            const double approximate =
                calculator.calcDistanceForQuery(encoded[label], query_blob, state->dim);
            const double error = ExactDistance(query, corpus[label], Metric) - approximate;
            signed_sum += error;
            absolute_sum += std::abs(error);
            squared_sum += error * error;
            absolute_errors.push_back(std::abs(error));
        }
        allocator->free_allocation(query_blob);
    }
    for (void *blob : encoded) {
        allocator->free_allocation(blob);
    }
    const double count = static_cast<double>(absolute_errors.size());
    return {.signed_bias = signed_sum / count,
            .mae = absolute_sum / count,
            .rmse = std::sqrt(squared_sum / count),
            .p95_absolute_error = Percentile(absolute_errors, 0.95)};
}

template <VecSimMetric Metric>
Result MeasureStoredToStored(const Config &config, std::string implementation, size_t dim,
                             size_t bits, Distribution distribution,
                             const std::vector<std::vector<float>> &corpus,
                             TQFlatDetails::TQStoredDistanceMode mode) {
    Result result = BaseResult(std::move(implementation), "stored_to_stored_construction_score",
                               dim, bits, Metric, distribution, config);
    auto allocator = VecSimAllocator::newVecsimAllocator();
    const size_t allocator_baseline = allocator->getAllocationSize();
    auto state =
        TQFlatDetails::AllocateDenseReferenceTQModelState(allocator, dim, bits, dim, kSeed, true);
    result.model_bytes = allocator->getAllocationSize() - allocator_baseline;
    TQFlatDetails::TQPreprocessor<Metric> preprocessor(allocator, state);
    TQFlatDetails::TQDistanceCalculator<Metric> calculator(allocator, state, mode);
    std::vector<void *> encoded;
    encoded.reserve(corpus.size());
    for (const auto &vector : corpus) {
        void *blob = nullptr;
        size_t blob_size = 0;
        preprocessor.preprocessForStorage(vector.data(), blob, blob_size, alignof(float));
        encoded.push_back(blob);
    }
#ifdef BUILD_TESTS
    const uint64_t allocations_before = allocator->getAllocationCount();
#endif
    std::tie(result.median_ns, result.p95_ns) = Timed(config.repetitions, [&] {
        for (size_t i = 1; i < encoded.size(); ++i) {
            result.checksum += calculator.calcDistance(encoded[i - 1], encoded[i], dim);
        }
    });
    result.sample_count = config.repetitions;
#ifdef BUILD_TESTS
    const uint64_t allocations_after = allocator->getAllocationCount();
    const uint64_t allocation_delta = allocations_after - allocations_before;
    result.note = "vecsim_allocations_in_hot_path=" + std::to_string(allocation_delta);
    if (mode == TQFlatDetails::TQStoredDistanceMode::CoarseMse && allocation_delta != 0) {
        throw std::runtime_error("CoarseMse stored scoring allocated in the hot path");
    }
#else
    result.note = "allocator call counter unavailable in this build";
#endif
    result.allocator_total_bytes = allocator->getAllocationSize() - allocator_baseline;
    for (void *blob : encoded) {
        allocator->free_allocation(blob);
    }
    return result;
}

template <VecSimMetric Metric>
Result MeasureQueryPreprocess(const Config &config, std::string implementation, size_t dim,
                              size_t bits, Distribution distribution,
                              const std::vector<std::vector<float>> &queries) {
    Result result = BaseResult(std::move(implementation), "query_preprocess", dim, bits, Metric,
                               distribution, config);
    auto allocator = VecSimAllocator::newVecsimAllocator();
    const size_t allocator_baseline = allocator->getAllocationSize();
    auto state =
        TQFlatDetails::AllocateDenseReferenceTQModelState(allocator, dim, bits, dim, kSeed, true);
    result.model_bytes = allocator->getAllocationSize() - allocator_baseline;
    TQFlatDetails::TQPreprocessor<Metric> preprocessor(allocator, state);
    ApplyTiming(result, TimedEach(config.repetitions, queries, [&](const auto &query) {
                    void *blob = nullptr;
                    size_t blob_size = 0;
                    preprocessor.preprocessQuery(query.data(), blob, blob_size, alignof(float));
                    result.checksum += static_cast<double>(blob_size);
                    allocator->free_allocation(blob);
                }));
    result.allocator_total_bytes = allocator->getAllocationSize() - allocator_baseline;
    result.note = "allocation is part of query-context creation, not stored scoring";
    return result;
}

template <VecSimMetric Metric>
Result MeasureStoredToQuery(const Config &config, std::string implementation, size_t dim,
                            size_t bits, Distribution distribution,
                            const std::vector<std::vector<float>> &corpus,
                            const std::vector<std::vector<float>> &queries) {
    Result result = BaseResult(std::move(implementation), "stored_to_query_score", dim, bits,
                               Metric, distribution, config);
    auto allocator = VecSimAllocator::newVecsimAllocator();
    const size_t allocator_baseline = allocator->getAllocationSize();
    auto state =
        TQFlatDetails::AllocateDenseReferenceTQModelState(allocator, dim, bits, dim, kSeed, true);
    result.model_bytes = allocator->getAllocationSize() - allocator_baseline;
    TQFlatDetails::TQPreprocessor<Metric> preprocessor(allocator, state);
    TQFlatDetails::TQDistanceCalculator<Metric> calculator(allocator, state);
    std::vector<void *> encoded;
    encoded.reserve(corpus.size());
    for (const auto &vector : corpus) {
        void *blob = nullptr;
        size_t blob_size = 0;
        preprocessor.preprocessForStorage(vector.data(), blob, blob_size, alignof(float));
        encoded.push_back(blob);
    }
    auto query_context = preprocessor.createOperationContext();
    auto query_blob_owner =
        allocator->allocate_aligned_unique(state->queryBlobSize(), alignof(float));
    if (!query_blob_owner) {
        throw std::bad_alloc();
    }
    void *query_blob = query_blob_owner.get();
    size_t query_blob_size = state->queryBlobSize();
#ifdef BUILD_TESTS
    const uint64_t allocations_before = allocator->getAllocationCount();
#endif
    ApplyTiming(result, TimedEachPrepared(
                            config.repetitions, queries,
                            [&](const auto &query) {
                                preprocessor.preprocessQueryWithContext(
                                    query.data(), query_blob, query_blob_size, alignof(float),
                                    query_context);
                            },
                            [&](const auto &) {
                                for (void *blob : encoded) {
                                    result.checksum +=
                                        calculator.calcDistanceForQuery(blob, query_blob, dim);
                                }
                            }));
#ifdef BUILD_TESTS
    const uint64_t allocations_after = allocator->getAllocationCount();
    const uint64_t allocation_delta = allocations_after - allocations_before;
    result.note = "vecsim_allocations_in_hot_path=" + std::to_string(allocation_delta);
    if (allocation_delta != 0) {
        throw std::runtime_error("stored-to-query scoring allocated in the hot path");
    }
#else
    result.note = "allocator call counter unavailable in this build";
#endif
    result.allocator_total_bytes = allocator->getAllocationSize() - allocator_baseline;
    for (void *blob : encoded) {
        allocator->free_allocation(blob);
    }
    return result;
}

template <VecSimMetric Metric, typename StateFactory>
Result MeasureVectorEncodeWithModel(const Config &config, std::string implementation, size_t dim,
                                    size_t bits, Distribution distribution,
                                    const std::vector<std::vector<float>> &corpus,
                                    StateFactory &&state_factory, std::string note) {
    Result result = BaseResult(std::move(implementation), "vector_encode", dim, bits, Metric,
                               distribution, config);
    auto allocator = VecSimAllocator::newVecsimAllocator();
    const size_t allocator_baseline = allocator->getAllocationSize();
    auto state = state_factory(allocator);
    result.model_bytes = allocator->getAllocationSize() - allocator_baseline;
    TQFlatDetails::TQPreprocessor<Metric> preprocessor(allocator, state);
    std::vector<void *> encoded(corpus.size(), nullptr);
    for (void *&blob : encoded) {
        blob = allocator->allocate_aligned(state->storageBlobSize(), alignof(float));
        if (!blob) {
            throw std::bad_alloc();
        }
    }
    std::tie(result.median_ns, result.p95_ns) = Timed(config.repetitions, [&] {
        for (size_t i = 0; i < corpus.size(); ++i) {
            size_t blob_size = state->storageBlobSize();
            preprocessor.preprocessForStorage(corpus[i].data(), encoded[i], blob_size,
                                              alignof(float));
            result.checksum += static_cast<const uint8_t *>(encoded[i])[0];
        }
    });
    result.sample_count = config.repetitions;
    result.allocator_total_bytes = allocator->getAllocationSize() - allocator_baseline;
    result.note = std::move(note);
    for (void *blob : encoded) {
        allocator->free_allocation(blob);
    }
    return result;
}

template <VecSimMetric Metric>
Result MeasureVectorEncode(const Config &config, std::string implementation, size_t dim,
                           size_t bits, Distribution distribution,
                           const std::vector<std::vector<float>> &corpus) {
    return MeasureVectorEncodeWithModel<Metric>(
        config, std::move(implementation), dim, bits, distribution, corpus,
        [=](const std::shared_ptr<VecSimAllocator> &allocator) {
            return TQFlatDetails::AllocateDenseReferenceTQModelState(allocator, dim, bits, dim,
                                                                     kSeed, true);
        },
        "direct codec preprocessing into preallocated storage buffers");
}

Result BaseResult(std::string implementation, std::string operation, size_t dim, size_t bits,
                  VecSimMetric metric, Distribution distribution, const Config &config) {
    Result result;
    result.implementation = std::move(implementation);
    result.run_profile = RunProfile(config);
    result.operation = std::move(operation);
    result.dim = dim;
    result.bits = bits;
    result.metric = metric;
    result.distribution = distribution;
    result.corpus_size = config.corpus_size;
    result.query_count = config.query_count;
    result.payload_bytes =
        TQFlatDetails::PackedBytes(dim, bits - 1) + TQFlatDetails::PackedBytes(dim, 1);
    result.metadata_bytes = 2 * sizeof(float);
    return result;
}

std::string JsonEscape(std::string_view value) {
    std::string escaped;
    for (char character : value) {
        if (character == '\\' || character == '\"') {
            escaped.push_back('\\');
        }
        escaped.push_back(character);
    }
    return escaped;
}

std::string TransformVersion(TQFlatDetails::TQRotationBackendVersion version) {
    switch (version) {
    case TQFlatDetails::TQRotationBackendVersion::DenseHaarV1:
        return "DenseHaarV1";
    case TQFlatDetails::TQRotationBackendVersion::FastStructuredV1:
        return "FastStructuredRotationV1";
    }
    throw std::invalid_argument("unknown TurboQuant rotation backend identity");
}

std::string QjlVersion(TQFlatDetails::TQQjlBackendVersion version) {
    switch (version) {
    case TQFlatDetails::TQQjlBackendVersion::DenseGaussianV1:
        return "DenseGaussianQjlV1";
    case TQFlatDetails::TQQjlBackendVersion::CirculantGaussianV1:
        return "CirculantGaussianQjlV1";
    }
    throw std::invalid_argument("unknown TurboQuant QJL backend identity");
}

std::string ModelProfileVersion(TQFlatDetails::TQModelTransformVersion version) {
    switch (version) {
    case TQFlatDetails::TQModelTransformVersion::DenseReferenceV1:
        return "DenseReferenceV1";
    case TQFlatDetails::TQModelTransformVersion::FastStructuredRotationV1:
        return "FastStructuredRotationV1";
    case TQFlatDetails::TQModelTransformVersion::FastStructuredV1:
        return "FastStructuredV1";
    }
    throw std::invalid_argument("unknown TurboQuant model profile identity");
}

std::string StoredDistanceVersion(TQFlatDetails::TQStoredDistanceMode mode) {
    switch (mode) {
    case TQFlatDetails::TQStoredDistanceMode::FullDecodeReference:
        return "FullDecodeReferenceV1";
    case TQFlatDetails::TQStoredDistanceMode::CoarseMse:
        return "CoarseMseV1";
    }
    throw std::invalid_argument("unknown TurboQuant construction score identity");
}

void ApplyTqIdentity(Result &result, const TQFlatDetails::TQCodecConfig &config,
                     TQFlatDetails::TQStoredDistanceMode stored_distance_mode) {
    TQFlatDetails::ValidateTQCodecConfig(config);
    result.transform_version = TransformVersion(config.rotation_backend_version);
    result.qjl_version = QjlVersion(config.qjl_backend_version);
    result.model_profile_version = ModelProfileVersion(config.model_transform_version);
    result.stored_distance_version = StoredDistanceVersion(stored_distance_mode);
}

std::string RunProfile(const Config &config) {
    if (config.mode == RunMode::Smoke) {
        return "tiny_smoke";
    }
    if (config.mode == RunMode::Full &&
        (config.corpus_size_override || config.query_count_override ||
         config.repetitions_override)) {
        return "bounded_production_dimension_diagnostic";
    }
    return config.mode == RunMode::Full ? "production_gate" : "matrix_manifest";
}

void PrintFiniteOrNull(double value) {
    if (std::isfinite(value)) {
        std::cout << value;
    } else {
        std::cout << "null";
    }
}

double TimedWorkItems(const Result &result) {
    if (result.operation == "vector_encode" ||
        result.operation == "hnsw_insert_and_graph_construction") {
        return static_cast<double>(result.corpus_size);
    }
    if (result.operation == "stored_to_query_score" ||
        result.operation == "stored_to_query_score_exact_baseline") {
        return static_cast<double>(result.corpus_size);
    }
    if (result.operation == "stored_to_stored_construction_score") {
        return static_cast<double>(result.corpus_size - 1);
    }
    if (result.operation == "flat_scan" || result.operation == "flat_scan_exact_baseline") {
        return static_cast<double>(result.corpus_size);
    }
    if (result.operation == "hnsw_query_and_exact_neighbor_recall" ||
        result.operation == "query_preprocess") {
        return 1.0;
    }
    if (result.operation == "hnsw_delete_and_repair") {
        return static_cast<double>((result.corpus_size + 3) / 4);
    }
    return 0.0;
}

void PrintJson(const Result &result, const Metadata &metadata) {
    const bool exact_fp32 = result.implementation == "FP32Exact";
    const bool identity_not_applicable = result.transform_version == "not_applicable" ||
                                         result.qjl_version == "not_applicable" ||
                                         result.model_profile_version == "not_applicable" ||
                                         result.stored_distance_version == "not_applicable";
    if (metadata.git_commit.empty() || metadata.git_dirty.empty() || metadata.compiler.empty() ||
        metadata.build_mode.empty() || metadata.cpu_model.empty() ||
        metadata.architecture.empty() || metadata.simd.empty() || metadata.assertions.empty() ||
        result.implementation.empty() || result.run_profile.empty() || result.operation.empty() ||
        result.transform_version.empty() || result.qjl_version.empty() ||
        result.model_profile_version.empty() || result.stored_distance_version.empty() ||
        result.dim == 0 || result.corpus_size == 0 || result.query_count == 0 ||
        (!exact_fp32 && identity_not_applicable)) {
        throw std::runtime_error("benchmark result is missing required reproducibility metadata");
    }
    std::cout << std::setprecision(12) << "{\"git_commit\":\"" << JsonEscape(metadata.git_commit)
              << "\",\"git_dirty\":\"" << JsonEscape(metadata.git_dirty) << "\",\"compiler\":\""
              << JsonEscape(metadata.compiler) << "\",\"build_mode\":\""
              << JsonEscape(metadata.build_mode) << "\",\"cpu_model\":\""
              << JsonEscape(metadata.cpu_model) << "\",\"architecture\":\""
              << JsonEscape(metadata.architecture) << "\",\"simd\":\"" << JsonEscape(metadata.simd)
              << "\",\"assertions\":\"" << JsonEscape(metadata.assertions)
              << "\",\"implementation\":\"" << JsonEscape(result.implementation)
              << "\",\"run_profile\":\"" << JsonEscape(result.run_profile) << "\",\"operation\":\""
              << JsonEscape(result.operation) << "\",\"status\":\"" << JsonEscape(result.status)
              << "\",\"note\":\"" << JsonEscape(result.note) << "\",\"transform_version\":\""
              << JsonEscape(result.transform_version) << "\",\"model_profile_version\":\""
              << JsonEscape(result.model_profile_version) << "\",\"qjl_version\":\""
              << JsonEscape(result.qjl_version) << "\",\"stored_distance_version\":\""
              << JsonEscape(result.stored_distance_version) << "\",\"dim\":" << result.dim
              << ",\"metric\":\"" << MetricName(result.metric) << "\",\"bits\":" << result.bits
              << ",\"distribution\":\"" << DistributionName(result.distribution)
              << "\",\"seed\":" << kSeed << ",\"corpus_size\":" << result.corpus_size
              << ",\"query_count\":" << result.query_count << ",\"M\":" << kM
              << ",\"efConstruction\":" << kEfConstruction << ",\"efRuntime\":" << kEfRuntime
              << ",\"sample_count\":" << result.sample_count << ",\"median_ns\":";
    PrintFiniteOrNull(result.median_ns);
    std::cout << ",\"p95_ns\":";
    PrintFiniteOrNull(result.p95_ns);
    std::cout << ",\"total_seconds\":";
    PrintFiniteOrNull(result.median_ns / 1e9);
    if (result.operation == "hnsw_query_and_exact_neighbor_recall") {
        std::cout << ",\"query_median_us\":";
        PrintFiniteOrNull(result.median_ns / 1e3);
        std::cout << ",\"query_p95_us\":";
        PrintFiniteOrNull(result.p95_ns / 1e3);
    }
    std::cout << ",\"median_ms\":";
    PrintFiniteOrNull(result.median_ns / 1e6);
    const double work_items = TimedWorkItems(result);
    std::cout << ",\"median_ns_per_work_item\":";
    PrintFiniteOrNull(work_items == 0.0 ? std::numeric_limits<double>::quiet_NaN()
                                        : result.median_ns / work_items);
    std::cout << ",\"work_items_per_second\":";
    PrintFiniteOrNull(result.median_ns == 0.0 || work_items == 0.0
                          ? std::numeric_limits<double>::quiet_NaN()
                          : work_items * 1e9 / result.median_ns);
    if (result.operation == "vector_encode" || result.operation == "query_preprocess" ||
        result.operation == "flat_scan" || result.operation == "flat_scan_exact_baseline" ||
        result.operation == "hnsw_insert_and_graph_construction") {
        std::cout << ",\"median_ns_per_vector\":";
        PrintFiniteOrNull(work_items == 0.0 ? std::numeric_limits<double>::quiet_NaN()
                                            : result.median_ns / work_items);
        std::cout << ",\"vectors_per_second\":";
        PrintFiniteOrNull(result.median_ns == 0.0 || work_items == 0.0
                              ? std::numeric_limits<double>::quiet_NaN()
                              : work_items * 1e9 / result.median_ns);
    }
    if (result.operation == "stored_to_query_score" ||
        result.operation == "stored_to_query_score_exact_baseline" ||
        result.operation == "stored_to_stored_construction_score") {
        std::cout << ",\"median_ns_per_score\":";
        PrintFiniteOrNull(work_items == 0.0 ? std::numeric_limits<double>::quiet_NaN()
                                            : result.median_ns / work_items);
        std::cout << ",\"scores_per_second\":";
        PrintFiniteOrNull(result.median_ns == 0.0 || work_items == 0.0
                              ? std::numeric_limits<double>::quiet_NaN()
                              : work_items * 1e9 / result.median_ns);
    }
    if (result.operation == "hnsw_query_and_exact_neighbor_recall") {
        std::cout << ",\"query_qps\":";
        PrintFiniteOrNull(result.median_ns == 0.0 ? std::numeric_limits<double>::quiet_NaN()
                                                  : 1e9 / result.median_ns);
    }
    std::cout << ",\"checksum\":";
    PrintFiniteOrNull(result.checksum);
    std::cout << ",\"recall_at_10\":";
    PrintFiniteOrNull(result.recall_at_10);
    std::cout << ",\"signed_estimator_bias\":";
    PrintFiniteOrNull(result.signed_bias);
    std::cout << ",\"mae\":";
    PrintFiniteOrNull(result.mae);
    std::cout << ",\"rmse\":";
    PrintFiniteOrNull(result.rmse);
    std::cout << ",\"p95_absolute_error\":";
    PrintFiniteOrNull(result.p95_absolute_error);
    std::cout << ",\"encoded_payload_bytes\":" << result.payload_bytes
              << ",\"metadata_bytes\":" << result.metadata_bytes
              << ",\"model_bytes\":" << result.model_bytes
              << ",\"graph_and_container_bytes\":" << result.graph_and_container_bytes
              << ",\"allocator_total_bytes\":" << result.allocator_total_bytes
              << ",\"allocator_bytes_per_vector\":"
              << static_cast<double>(result.allocator_total_bytes) / result.corpus_size
              << ",\"amortized_model_bytes_per_vector\":"
              << static_cast<double>(result.model_bytes) / result.corpus_size
              << ",\"amortized_graph_and_container_bytes_per_vector\":"
              << static_cast<double>(result.graph_and_container_bytes) / result.corpus_size
              << ",\"peak_memory_status\":\"" << JsonEscape(result.peak_memory_status)
              << "\",\"graph_integrity_status\":\"" << JsonEscape(result.graph_integrity_status)
              << "\",\"graph_connections_to_repair\":" << result.graph_connections_to_repair
              << ",\"graph_bidirectional_connections\":" << result.graph_bidirectional_connections
              << ",\"graph_unidirectional_connections\":" << result.graph_unidirectional_connections
              << ",\"graph_average_directed_degree\":";
    PrintFiniteOrNull(result.graph_average_directed_degree);
    std::cout << ",\"payload_bits_per_dimension\":"
              << (result.dim == 0 ? 0.0 : 8.0 * result.payload_bytes / result.dim)
              << ",\"stored_bits_per_dimension_including_metadata\":"
              << (result.dim == 0
                      ? 0.0
                      : 8.0 * (result.payload_bytes + result.metadata_bytes) / result.dim)
              << "}" << std::endl;
}

void VerifyPayloadAndChecksum() {
    for (size_t bits : kBitWidths) {
        const auto params = TqFlatParams(1000, bits, VecSimMetric_IP, 2);
        const size_t expected = TQFlatDetails::PackedBytes(params.dim, bits - 1) +
                                TQFlatDetails::PackedBytes(params.dim, 1) + 2 * sizeof(float);
        if (TQFlatDetails::GetStorageDataSize<VecSimMetric_IP>(&params) != expected) {
            throw std::runtime_error("TQ payload layout disagrees with benchmark accounting");
        }
    }
    auto index = NewTqFlat(8, 4, VecSimMetric_IP, 2);
    std::array<float, 8> lhs{};
    std::array<float, 8> rhs{};
    lhs[0] = 1.0f;
    rhs[1] = 1.0f;
    Populate(index, {std::vector<float>(lhs.begin(), lhs.end()),
                     std::vector<float>(rhs.begin(), rhs.end())});
    const double first = VecSimIndex_GetDistanceFrom_Unsafe(index, 0, lhs.data());
    const double second = VecSimIndex_GetDistanceFrom_Unsafe(index, 1, lhs.data());
    VecSimIndex_Free(index);
    if (first == second) {
        throw std::runtime_error("TQ benchmark checksum would not observe changed scores");
    }

#ifdef BUILD_TESTS
    auto allocator = VecSimAllocator::newVecsimAllocator();
    auto state =
        TQFlatDetails::AllocateDenseReferenceTQModelState(allocator, 10, 4, 10, kSeed, false);
    TQFlatDetails::TQPreprocessor<VecSimMetric_IP> preprocessor(allocator, state);
    std::array<float, 10> zero_vector{};
    void *zero_storage = nullptr;
    size_t zero_storage_size = 0;
    preprocessor.preprocessForStorage(zero_vector.data(), zero_storage, zero_storage_size,
                                      alignof(float));
    const auto zero_view = state->storageView(zero_storage);
    if (zero_view.source_scale != 0.0f || zero_view.residual_norm != 0.0f) {
        throw std::runtime_error("zero vector does not retain canonical TQ metadata");
    }
    allocator->free_allocation(zero_storage);
    std::array<float, 10> zero_residual{};
    std::array<uint8_t, 2> signs = {0xFF, 0xFF};
    state->packResidualSigns(zero_residual.data(), signs.data());
    if (signs[0] != 0xFF || signs[1] != 0x03) {
        throw std::runtime_error("zero-residual sign payload is not canonical");
    }
#endif
}

void VerifyTqIdentityMetadata() {
    const auto verify = [](const TQFlatDetails::TQCodecConfig &config,
                           TQFlatDetails::TQStoredDistanceMode stored_distance_mode,
                           std::string_view transform, std::string_view qjl,
                           std::string_view model_profile, std::string_view construction) {
        Result result;
        ApplyTqIdentity(result, config, stored_distance_mode);
        if (result.transform_version != transform || result.qjl_version != qjl ||
            result.model_profile_version != model_profile ||
            result.stored_distance_version != construction) {
            throw std::runtime_error("TurboQuant benchmark identity metadata mismatch");
        }
    };
    verify(TQFlatDetails::TQCodecConfig::DenseReference(8, 4, 8, kSeed, true),
           TQFlatDetails::TQStoredDistanceMode::FullDecodeReference, "DenseHaarV1",
           "DenseGaussianQjlV1", "DenseReferenceV1", "FullDecodeReferenceV1");
    verify(TQFlatDetails::TQCodecConfig::FastStructuredRotation(8, 4, 8, kSeed),
           TQFlatDetails::TQStoredDistanceMode::CoarseMse, "FastStructuredRotationV1",
           "DenseGaussianQjlV1", "FastStructuredRotationV1", "CoarseMseV1");
    verify(TQFlatDetails::TQCodecConfig::FastStructured(8, 4, 8, kSeed),
           TQFlatDetails::TQStoredDistanceMode::CoarseMse, "FastStructuredRotationV1",
           "CirculantGaussianQjlV1", "FastStructuredV1", "CoarseMseV1");
}

void ApplyGraphIntegrity(Result &result, const TQTypedHNSWIndex &index) {
    const HNSWIndexMetaData integrity = index.checkIntegrity();
    result.graph_integrity_status = integrity.valid_state ? "valid" : "invalid";
    if (!integrity.valid_state) {
        result.note = result.note.empty() ? "HNSW integrity validation failed"
                                          : result.note + "; HNSW integrity validation failed";
        return;
    }
    result.graph_connections_to_repair = integrity.connections_to_repair;
    result.graph_bidirectional_connections = integrity.double_connections;
    result.graph_unidirectional_connections = integrity.unidirectional_connections;
    const size_t directed_connections =
        integrity.double_connections + integrity.unidirectional_connections;
    result.graph_average_directed_degree =
        index.indexSize() == 0 ? 0.0
                               : static_cast<double>(directed_connections) / index.indexSize();
}

void CopyGraphIntegrity(const Result &source, Result &destination) {
    destination.graph_integrity_status = source.graph_integrity_status;
    destination.graph_connections_to_repair = source.graph_connections_to_repair;
    destination.graph_bidirectional_connections = source.graph_bidirectional_connections;
    destination.graph_unidirectional_connections = source.graph_unidirectional_connections;
    destination.graph_average_directed_degree = source.graph_average_directed_degree;
}

size_t GraphAndContainerBytes(const Result &result) {
    const size_t vector_bytes = result.corpus_size * (result.payload_bytes + result.metadata_bytes);
    if (result.allocator_total_bytes <= result.model_bytes + vector_bytes) {
        return 0;
    }
    return result.allocator_total_bytes - result.model_bytes - vector_bytes;
}

void RunFp32ExactBaseline(const Config &config, const Metadata &metadata, size_t dim,
                          VecSimMetric metric, Distribution distribution) {
    const auto corpus = MakeCorpusDataset(config, dim, distribution);
    const auto queries = MakeQueryDataset(config, dim, distribution);
    auto index = NewFp32Flat(dim, metric, config.corpus_size);
    Populate(index, corpus);
    Result score = BaseResult("FP32Exact", "stored_to_query_score_exact_baseline", dim, 32, metric,
                              distribution, config);
    score.payload_bytes = dim * sizeof(float);
    score.metadata_bytes = 0;
    ApplyTiming(score, TimedEach(config.repetitions, queries, [&](const auto &query) {
                    for (const auto &candidate : corpus) {
                        score.checksum += ExactDistance(query, candidate, metric);
                    }
                }));
    score.allocator_total_bytes = VecSimIndex_StatsInfo(index).memory;
    PrintJson(score, metadata);
    Result scan =
        BaseResult("FP32Exact", "flat_scan_exact_baseline", dim, 32, metric, distribution, config);
    scan.payload_bytes = dim * sizeof(float);
    scan.metadata_bytes = 0;
    ApplyTiming(scan, TimedEach(config.repetitions, queries, [&](const auto &query) {
                    auto reply =
                        VecSimIndex_TopKQuery(index, query.data(), kTopK, nullptr, BY_SCORE);
                    for (size_t label : Labels(reply)) {
                        scan.checksum += static_cast<double>(label);
                    }
                    VecSimQueryReply_Free(reply);
                }));
    scan.recall_at_10 = 1.0;
    scan.allocator_total_bytes = VecSimIndex_StatsInfo(index).memory;
    PrintJson(scan, metadata);
    VecSimIndex_Free(index);
}

template <VecSimMetric Metric>
void RunExplicitStructuredForMetric(const Config &config, const Metadata &metadata, size_t dim,
                                    size_t bits, Distribution distribution,
                                    TQFlatDetails::TQCodecConfig codec_config,
                                    std::string implementation, std::string transform_note,
                                    const std::vector<std::vector<float>> &corpus,
                                    const std::vector<std::vector<float>> &queries) {
    std::vector<Result> results;
    Result create =
        BaseResult(implementation, "model_create", dim, bits, Metric, distribution, config);
    std::tie(create.median_ns, create.p95_ns) = Timed(1, [&] {
        auto allocator = VecSimAllocator::newVecsimAllocator();
        const size_t baseline = allocator->getAllocationSize();
        [[maybe_unused]] auto state = TQFlatDetails::AllocateTQModelState(allocator, codec_config);
        create.model_bytes += allocator->getAllocationSize() - baseline;
    });
    create.sample_count = 1;
    create.model_bytes /= 2; // Timed(1) runs a warmup plus exactly one recorded sample.
    create.allocator_total_bytes = create.model_bytes;
    results.push_back(create);

    Result transform = BaseResult(implementation, "fast_structured_rotation_transform", dim, bits,
                                  Metric, distribution, config);
    auto transform_allocator = VecSimAllocator::newVecsimAllocator();
    const size_t transform_baseline = transform_allocator->getAllocationSize();
    auto transform_state = TQFlatDetails::AllocateTQModelState(transform_allocator, codec_config);
    transform.model_bytes = transform_allocator->getAllocationSize() - transform_baseline;
    std::vector<float> rotated(dim);
    std::vector<float> scratch(dim);
    std::tie(transform.median_ns, transform.p95_ns) = Timed(config.repetitions, [&] {
        transform_state->applyRotation(corpus.front().data(), rotated.data(), scratch.data());
        transform.checksum += rotated.front();
    });
    transform.sample_count = config.repetitions;
    transform.allocator_total_bytes = transform_allocator->getAllocationSize() - transform_baseline;
    transform.note = std::move(transform_note);
    results.push_back(transform);

    const auto fast_state_factory = [=](const std::shared_ptr<VecSimAllocator> &allocator) {
        return TQFlatDetails::AllocateTQModelState(allocator, codec_config);
    };
    results.push_back(MeasureVectorEncodeWithModel<Metric>(
        config, implementation, dim, bits, distribution, corpus, fast_state_factory,
        "direct FastStructuredRotationV1 codec preprocessing into preallocated storage buffers"));

    Result query_preprocess =
        BaseResult(implementation, "query_preprocess", dim, bits, Metric, distribution, config);
    auto query_allocator = VecSimAllocator::newVecsimAllocator();
    const size_t query_baseline = query_allocator->getAllocationSize();
    auto query_state = fast_state_factory(query_allocator);
    query_preprocess.model_bytes = query_allocator->getAllocationSize() - query_baseline;
    TQFlatDetails::TQPreprocessor<Metric> query_preprocessor(query_allocator, query_state);
    ApplyTiming(query_preprocess, TimedEach(config.repetitions, queries, [&](const auto &query) {
                    void *blob = nullptr;
                    size_t blob_size = 0;
                    query_preprocessor.preprocessQuery(query.data(), blob, blob_size,
                                                       alignof(float));
                    query_preprocess.checksum += blob_size;
                    query_allocator->free_allocation(blob);
                }));
    query_preprocess.allocator_total_bytes = query_allocator->getAllocationSize() - query_baseline;
    query_preprocess.note = "direct FastStructuredRotationV1 query preprocessing";
    results.push_back(query_preprocess);

    Result score = BaseResult(implementation, "stored_to_query_score", dim, bits, Metric,
                              distribution, config);
    auto score_allocator = VecSimAllocator::newVecsimAllocator();
    const size_t score_baseline = score_allocator->getAllocationSize();
    auto score_state = fast_state_factory(score_allocator);
    score.model_bytes = score_allocator->getAllocationSize() - score_baseline;
    TQFlatDetails::TQPreprocessor<Metric> score_preprocessor(score_allocator, score_state);
    TQFlatDetails::TQDistanceCalculator<Metric> score_calculator(
        score_allocator, score_state, TQFlatDetails::TQStoredDistanceMode::CoarseMse);
    std::vector<void *> encoded;
    encoded.reserve(corpus.size());
    for (const auto &vector : corpus) {
        void *blob = nullptr;
        size_t blob_size = 0;
        score_preprocessor.preprocessForStorage(vector.data(), blob, blob_size, alignof(float));
        encoded.push_back(blob);
    }
    auto query_context = score_preprocessor.createOperationContext();
    auto query_blob_owner =
        score_allocator->allocate_aligned_unique(score_state->queryBlobSize(), alignof(float));
    if (!query_blob_owner) {
        throw std::bad_alloc();
    }
    void *query_blob = query_blob_owner.get();
    size_t query_blob_size = score_state->queryBlobSize();
#ifdef BUILD_TESTS
    const uint64_t allocations_before = score_allocator->getAllocationCount();
#endif
    ApplyTiming(score, TimedEachPrepared(
                           config.repetitions, queries,
                           [&](const auto &query) {
                               score_preprocessor.preprocessQueryWithContext(
                                   query.data(), query_blob, query_blob_size, alignof(float),
                                   query_context);
                           },
                           [&](const auto &) {
                               for (void *blob : encoded) {
                                   score.checksum +=
                                       score_calculator.calcDistanceForQuery(blob, query_blob, dim);
                               }
                           }));
#ifdef BUILD_TESTS
    const uint64_t allocation_delta = score_allocator->getAllocationCount() - allocations_before;
    score.note = "vecsim_allocations_in_hot_path=" + std::to_string(allocation_delta);
    if (allocation_delta != 0) {
        throw std::runtime_error("fast stored-to-query scoring allocated in the hot path");
    }
#endif
    score.allocator_total_bytes = score_allocator->getAllocationSize() - score_baseline;
    for (void *blob : encoded) {
        score_allocator->free_allocation(blob);
    }
    results.push_back(score);

    Result stored_score = BaseResult(implementation, "stored_to_stored_construction_score", dim,
                                     bits, Metric, distribution, config);
    auto construction_allocator = VecSimAllocator::newVecsimAllocator();
    const size_t construction_baseline = construction_allocator->getAllocationSize();
    auto construction_state = fast_state_factory(construction_allocator);
    stored_score.model_bytes = construction_allocator->getAllocationSize() - construction_baseline;
    TQFlatDetails::TQPreprocessor<Metric> construction_preprocessor(construction_allocator,
                                                                    construction_state);
    TQFlatDetails::TQDistanceCalculator<Metric> construction_calculator(
        construction_allocator, construction_state, TQFlatDetails::TQStoredDistanceMode::CoarseMse);
    std::vector<void *> construction_encoded;
    construction_encoded.reserve(corpus.size());
    for (const auto &vector : corpus) {
        void *blob = nullptr;
        size_t blob_size = 0;
        construction_preprocessor.preprocessForStorage(vector.data(), blob, blob_size,
                                                       alignof(float));
        construction_encoded.push_back(blob);
    }
#ifdef BUILD_TESTS
    const uint64_t construction_allocations_before = construction_allocator->getAllocationCount();
#endif
    std::tie(stored_score.median_ns, stored_score.p95_ns) = Timed(config.repetitions, [&] {
        for (size_t i = 1; i < construction_encoded.size(); ++i) {
            stored_score.checksum += construction_calculator.calcDistance(
                construction_encoded[i - 1], construction_encoded[i], dim);
        }
    });
    stored_score.sample_count = config.repetitions;
#ifdef BUILD_TESTS
    const uint64_t construction_allocation_delta =
        construction_allocator->getAllocationCount() - construction_allocations_before;
    stored_score.note =
        "vecsim_allocations_in_hot_path=" + std::to_string(construction_allocation_delta);
    if (construction_allocation_delta != 0) {
        throw std::runtime_error("fast CoarseMse construction scoring allocated in the hot path");
    }
#endif
    stored_score.allocator_total_bytes =
        construction_allocator->getAllocationSize() - construction_baseline;
    for (void *blob : construction_encoded) {
        construction_allocator->free_allocation(blob);
    }
    results.push_back(stored_score);

    Result quality = BaseResult(implementation, "quality_against_exact_fp32", dim, bits, Metric,
                                distribution, config);
    auto quality_allocator = VecSimAllocator::newVecsimAllocator();
    const size_t quality_baseline = quality_allocator->getAllocationSize();
    auto quality_state = fast_state_factory(quality_allocator);
    quality.model_bytes = quality_allocator->getAllocationSize() - quality_baseline;
    const ErrorStats errors =
        CompareDirectModelWithExact<Metric>(quality_allocator, quality_state, queries, corpus);
    quality.signed_bias = errors.signed_bias;
    quality.mae = errors.mae;
    quality.rmse = errors.rmse;
    quality.p95_absolute_error = errors.p95_absolute_error;
    quality.sample_count = queries.size() * corpus.size();
    quality.allocator_total_bytes = quality_allocator->getAllocationSize() - quality_baseline;
    quality.note = "direct CoarseMse quality versus exact FP32";
    results.push_back(quality);

    auto flat = NewTqFlatWithConfig<Metric>(dim, bits, config.corpus_size, codec_config,
                                            TQFlatDetails::TQStoredDistanceMode::CoarseMse);
    Populate(flat, corpus);
    Result scan = BaseResult(implementation, "flat_scan", dim, bits, Metric, distribution, config);
    ApplyTiming(scan, TimedEach(config.repetitions, queries, [&](const auto &query) {
                    auto reply =
                        VecSimIndex_TopKQuery(flat, query.data(), kTopK, nullptr, BY_SCORE);
                    for (size_t label : Labels(reply)) {
                        scan.checksum += static_cast<double>(label);
                    }
                    VecSimQueryReply_Free(reply);
                }));
    scan.allocator_total_bytes = VecSimIndex_StatsInfo(flat).memory;
    scan.model_bytes = create.model_bytes;
    results.push_back(scan);

    Result insert = BaseResult(implementation, "hnsw_insert_and_graph_construction", dim, bits,
                               Metric, distribution, config);
    std::tie(insert.median_ns, insert.p95_ns) = TimedPrepared(
        config.repetitions,
        [&] {
            return NewTqHnswWithConfig<Metric>(dim, bits, config.corpus_size, codec_config,
                                               TQFlatDetails::TQStoredDistanceMode::CoarseMse);
        },
        [&](VecSimIndex *index) { Populate(index, corpus); },
        [](VecSimIndex *index) { VecSimIndex_Free(index); });
    insert.sample_count = config.repetitions;
    insert.model_bytes = create.model_bytes;

    auto hnsw = NewTqHnswWithConfig<Metric>(dim, bits, config.corpus_size, codec_config,
                                            TQFlatDetails::TQStoredDistanceMode::CoarseMse);
    Populate(hnsw, corpus);
    insert.allocator_total_bytes = VecSimIndex_StatsInfo(hnsw).memory;
    insert.graph_and_container_bytes = GraphAndContainerBytes(insert);
    ApplyGraphIntegrity(insert, *hnsw);
    results.push_back(insert);

    Result query = BaseResult(implementation, "hnsw_query_and_exact_neighbor_recall", dim, bits,
                              Metric, distribution, config);
    HNSWRuntimeParams runtime = {.efRuntime = kEfRuntime};
    VecSimQueryParams query_params = {.hnswRuntimeParams = runtime};
    ApplyTiming(query, TimedEach(config.repetitions, queries, [&](const auto &query_vector) {
                    auto reply = VecSimIndex_TopKQuery(hnsw, query_vector.data(), kTopK,
                                                       &query_params, BY_SCORE);
                    for (size_t label : Labels(reply)) {
                        query.checksum += static_cast<double>(label);
                    }
                    VecSimQueryReply_Free(reply);
                }));
    query.recall_at_10 = RecallAt10(hnsw, queries, corpus, Metric);
    query.allocator_total_bytes = VecSimIndex_StatsInfo(hnsw).memory;
    query.model_bytes = create.model_bytes;
    query.graph_and_container_bytes = insert.graph_and_container_bytes;
    CopyGraphIntegrity(insert, query);
    results.push_back(query);

    Result deletion = BaseResult(implementation, "hnsw_delete_and_repair", dim, bits, Metric,
                                 distribution, config);
    const auto delete_vectors = [&](VecSimIndex *index) {
        for (size_t i = 0; i < corpus.size(); i += 4) {
            deletion.checksum += VecSimIndex_DeleteVector(index, i);
        }
    };
    auto deletion_warmup =
        NewTqHnswWithConfig<Metric>(dim, bits, config.corpus_size, codec_config,
                                    TQFlatDetails::TQStoredDistanceMode::CoarseMse);
    Populate(deletion_warmup, corpus);
    delete_vectors(deletion_warmup);
    VecSimIndex_Free(deletion_warmup);
    std::vector<double> deletion_samples;
    deletion_samples.reserve(config.repetitions);
    for (size_t repetition = 0; repetition < config.repetitions; ++repetition) {
        auto prepared = NewTqHnswWithConfig<Metric>(dim, bits, config.corpus_size, codec_config,
                                                    TQFlatDetails::TQStoredDistanceMode::CoarseMse);
        Populate(prepared, corpus);
        const auto begin = Clock::now();
        delete_vectors(prepared);
        const auto end = Clock::now();
        deletion_samples.push_back(std::chrono::duration<double, std::nano>(end - begin).count());
        deletion.allocator_total_bytes = VecSimIndex_StatsInfo(prepared).memory;
        VecSimIndex_Free(prepared);
    }
    deletion.median_ns = Percentile(deletion_samples, 0.50);
    deletion.p95_ns = Percentile(deletion_samples, 0.95);
    deletion.sample_count = config.repetitions;
    deletion.model_bytes = create.model_bytes;
    deletion.graph_and_container_bytes = insert.graph_and_container_bytes;
    auto deletion_integrity =
        NewTqHnswWithConfig<Metric>(dim, bits, config.corpus_size, codec_config,
                                    TQFlatDetails::TQStoredDistanceMode::CoarseMse);
    Populate(deletion_integrity, corpus);
    for (size_t i = 0; i < corpus.size(); i += 4) {
        VecSimIndex_DeleteVector(deletion_integrity, i);
    }
    ApplyGraphIntegrity(deletion, *deletion_integrity);
    VecSimIndex_Free(deletion_integrity);
    results.push_back(deletion);

    Result memory = BaseResult(implementation, "peak_and_steady_state_memory", dim, bits, Metric,
                               distribution, config);
    memory.allocator_total_bytes = VecSimIndex_StatsInfo(hnsw).memory;
    memory.model_bytes = create.model_bytes;
    memory.graph_and_container_bytes = GraphAndContainerBytes(memory);
    CopyGraphIntegrity(insert, memory);
    memory.checksum = static_cast<double>(memory.allocator_total_bytes);
    results.push_back(memory);

    VecSimIndex_Free(hnsw);
    VecSimIndex_Free(flat);

    for (Result &result : results) {
        ApplyTqIdentity(result, codec_config, TQFlatDetails::TQStoredDistanceMode::CoarseMse);
        PrintJson(result, metadata);
    }
}

void RunExplicitStructured(const Config &config, const Metadata &metadata, size_t dim, size_t bits,
                           VecSimMetric metric, Distribution distribution,
                           std::string implementation, TQFlatDetails::TQCodecConfig codec_config,
                           std::string transform_note) {
    const auto corpus = MakeCorpusDataset(config, dim, distribution);
    const auto queries = MakeQueryDataset(config, dim, distribution);
    if (metric == VecSimMetric_IP) {
        RunExplicitStructuredForMetric<VecSimMetric_IP>(config, metadata, dim, bits, distribution,
                                                        codec_config, std::move(implementation),
                                                        std::move(transform_note), corpus, queries);
    } else {
        RunExplicitStructuredForMetric<VecSimMetric_Cosine>(
            config, metadata, dim, bits, distribution, codec_config, std::move(implementation),
            std::move(transform_note), corpus, queries);
    }
}

void RunDenseReference(const Config &config, const Metadata &metadata, size_t dim, size_t bits,
                       VecSimMetric metric, Distribution distribution, std::string implementation,
                       bool coarse_mse) {
    const auto stored_distance_mode =
        coarse_mse ? TQFlatDetails::TQStoredDistanceMode::CoarseMse
                   : TQFlatDetails::TQStoredDistanceMode::FullDecodeReference;
    const auto codec_config =
        TQFlatDetails::TQCodecConfig::DenseReference(dim, bits, dim, kSeed, true);
    const auto corpus = MakeCorpusDataset(config, dim, distribution);
    const auto queries = MakeQueryDataset(config, dim, distribution);
    std::vector<Result> results;

    Result create =
        BaseResult(implementation, "model_create", dim, bits, metric, distribution, config);
    std::tie(create.median_ns, create.p95_ns) = Timed(1, [&] {
        auto allocator = VecSimAllocator::newVecsimAllocator();
        const size_t allocator_baseline = allocator->getAllocationSize();
        [[maybe_unused]] auto state = TQFlatDetails::AllocateDenseReferenceTQModelState(
            allocator, dim, bits, dim, kSeed, true);
        create.model_bytes += allocator->getAllocationSize() - allocator_baseline;
    });
    create.sample_count = 1;
    create.model_bytes /= 2; // Timed(1) performs one warmup and one measured construction.
    create.allocator_total_bytes = create.model_bytes;
    results.push_back(create);

    Result encode = metric == VecSimMetric_IP
                        ? MeasureVectorEncode<VecSimMetric_IP>(config, implementation, dim, bits,
                                                               distribution, corpus)
                        : MeasureVectorEncode<VecSimMetric_Cosine>(config, implementation, dim,
                                                                   bits, distribution, corpus);
    results.push_back(encode);

    auto flat = NewTqFlat(dim, bits, metric, config.corpus_size);
    Populate(flat, corpus);

    Result score = metric == VecSimMetric_IP
                       ? MeasureStoredToQuery<VecSimMetric_IP>(config, implementation, dim, bits,
                                                               distribution, corpus, queries)
                       : MeasureStoredToQuery<VecSimMetric_Cosine>(
                             config, implementation, dim, bits, distribution, corpus, queries);
    const ErrorStats quality = CompareWithExactFp32(flat, queries, corpus, metric);
    score.signed_bias = quality.signed_bias;
    score.mae = quality.mae;
    score.rmse = quality.rmse;
    score.p95_absolute_error = quality.p95_absolute_error;
    results.push_back(score);

    Result query_preprocess = metric == VecSimMetric_IP
                                  ? MeasureQueryPreprocess<VecSimMetric_IP>(
                                        config, implementation, dim, bits, distribution, queries)
                                  : MeasureQueryPreprocess<VecSimMetric_Cosine>(
                                        config, implementation, dim, bits, distribution, queries);
    results.push_back(query_preprocess);

    Result scan = BaseResult(implementation, "flat_scan", dim, bits, metric, distribution, config);
    ApplyTiming(scan, TimedEach(config.repetitions, queries, [&](const auto &query) {
                    auto reply =
                        VecSimIndex_TopKQuery(flat, query.data(), kTopK, nullptr, BY_SCORE);
                    for (size_t label : Labels(reply)) {
                        scan.checksum += static_cast<double>(label);
                    }
                    VecSimQueryReply_Free(reply);
                }));
    scan.allocator_total_bytes = VecSimIndex_StatsInfo(flat).memory;
    scan.model_bytes = create.model_bytes;
    results.push_back(scan);

    Result stored_score =
        metric == VecSimMetric_IP
            ? MeasureStoredToStored<VecSimMetric_IP>(config, implementation, dim, bits,
                                                     distribution, corpus, stored_distance_mode)
            : MeasureStoredToStored<VecSimMetric_Cosine>(
                  config, implementation, dim, bits, distribution, corpus, stored_distance_mode);
    results.push_back(stored_score);

    Result insert = BaseResult(implementation, "hnsw_insert_and_graph_construction", dim, bits,
                               metric, distribution, config);
    std::tie(insert.median_ns, insert.p95_ns) = TimedPrepared(
        config.repetitions,
        [&] {
            // HNSW/model construction is deliberately outside this timer: model_create is a
            // separate operation, while this record is populated-graph construction only.
            return NewTqHnsw(dim, bits, metric, config.corpus_size, stored_distance_mode);
        },
        [&](VecSimIndex *build_index) { Populate(build_index, corpus); },
        [](VecSimIndex *build_index) { VecSimIndex_Free(build_index); });
    insert.sample_count = config.repetitions;
    insert.model_bytes = create.model_bytes;

    auto hnsw = NewTqHnsw(dim, bits, metric, config.corpus_size, stored_distance_mode);
    Populate(hnsw, corpus);
    insert.allocator_total_bytes = VecSimIndex_StatsInfo(hnsw).memory;
    insert.graph_and_container_bytes = GraphAndContainerBytes(insert);
    ApplyGraphIntegrity(insert, *hnsw);
    results.push_back(insert);

    Result query = BaseResult(implementation, "hnsw_query_and_exact_neighbor_recall", dim, bits,
                              metric, distribution, config);
    HNSWRuntimeParams runtime = {.efRuntime = kEfRuntime};
    VecSimQueryParams query_params = {.hnswRuntimeParams = runtime};
    ApplyTiming(query, TimedEach(config.repetitions, queries, [&](const auto &query_vector) {
                    auto reply = VecSimIndex_TopKQuery(hnsw, query_vector.data(), kTopK,
                                                       &query_params, BY_SCORE);
                    for (size_t label : Labels(reply)) {
                        query.checksum += static_cast<double>(label);
                    }
                    VecSimQueryReply_Free(reply);
                }));
    query.recall_at_10 = RecallAt10(hnsw, queries, corpus, metric);
    query.allocator_total_bytes = VecSimIndex_StatsInfo(hnsw).memory;
    query.model_bytes = create.model_bytes;
    query.graph_and_container_bytes = insert.graph_and_container_bytes;
    CopyGraphIntegrity(insert, query);
    results.push_back(query);

    Result deletion = BaseResult(implementation, "hnsw_delete_and_repair", dim, bits, metric,
                                 distribution, config);
    const auto delete_vectors = [&](VecSimIndex *prepared) {
        for (size_t i = 0; i < corpus.size(); i += 4) {
            deletion.checksum += VecSimIndex_DeleteVector(prepared, i);
        }
    };
    auto deletion_warmup = NewTqHnsw(dim, bits, metric, config.corpus_size, stored_distance_mode);
    Populate(deletion_warmup, corpus);
    delete_vectors(deletion_warmup);
    VecSimIndex_Free(deletion_warmup);
    std::vector<double> deletion_samples;
    deletion_samples.reserve(config.repetitions);
    for (size_t repetition = 0; repetition < config.repetitions; ++repetition) {
        auto prepared = NewTqHnsw(dim, bits, metric, config.corpus_size, stored_distance_mode);
        Populate(prepared, corpus);
        const auto begin = Clock::now();
        delete_vectors(prepared);
        const auto end = Clock::now();
        deletion_samples.push_back(std::chrono::duration<double, std::nano>(end - begin).count());
        // Memory observation and destruction are purposefully outside the deletion timer.
        deletion.allocator_total_bytes = VecSimIndex_StatsInfo(prepared).memory;
        VecSimIndex_Free(prepared);
    }
    deletion.median_ns = Percentile(deletion_samples, 0.50);
    deletion.p95_ns = Percentile(deletion_samples, 0.95);
    deletion.sample_count = config.repetitions;
    deletion.model_bytes = create.model_bytes;
    deletion.graph_and_container_bytes = insert.graph_and_container_bytes;
    auto deletion_integrity =
        NewTqHnsw(dim, bits, metric, config.corpus_size, stored_distance_mode);
    Populate(deletion_integrity, corpus);
    for (size_t i = 0; i < corpus.size(); i += 4) {
        VecSimIndex_DeleteVector(deletion_integrity, i);
    }
    ApplyGraphIntegrity(deletion, *deletion_integrity);
    VecSimIndex_Free(deletion_integrity);
    results.push_back(deletion);

    Result memory = BaseResult(implementation, "peak_and_steady_state_memory", dim, bits, metric,
                               distribution, config);
    memory.allocator_total_bytes = VecSimIndex_StatsInfo(hnsw).memory;
    memory.model_bytes = create.model_bytes;
    memory.graph_and_container_bytes = GraphAndContainerBytes(memory);
    CopyGraphIntegrity(insert, memory);
    memory.checksum = static_cast<double>(memory.allocator_total_bytes);
    results.push_back(memory);

    VecSimIndex_Free(hnsw);
    VecSimIndex_Free(flat);
    for (Result &result : results) {
        ApplyTqIdentity(result, codec_config, stored_distance_mode);
        PrintJson(result, metadata);
    }
}

Config ParseArgs(int argc, char **argv) {
    Config config;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--tq-mode=smoke") {
            config.mode = RunMode::Smoke;
        } else if (arg == "--tq-mode=full") {
            config.mode = RunMode::Full;
            config.corpus_size = 5000;
            config.query_count = 1000;
            config.repetitions = 9;
        } else if (arg == "--tq-mode=list") {
            config.mode = RunMode::List;
        } else if (arg.starts_with("--tq-dim=")) {
            config.dim_filter =
                std::stoull(std::string(arg.substr(std::string_view("--tq-dim=").size())));
        } else if (arg.starts_with("--tq-bits=")) {
            config.bits_filter =
                std::stoull(std::string(arg.substr(std::string_view("--tq-bits=").size())));
        } else if (arg == "--tq-metric=ip") {
            config.metric_filter = VecSimMetric_IP;
        } else if (arg == "--tq-metric=cosine") {
            config.metric_filter = VecSimMetric_Cosine;
        } else if (arg == "--tq-distribution=gaussian") {
            config.distribution_filter = Distribution::Gaussian;
        } else if (arg == "--tq-distribution=unit") {
            config.distribution_filter = Distribution::UnitGaussian;
        } else if (arg == "--tq-distribution=log_uniform") {
            config.distribution_filter = Distribution::LogUniformNormIp;
        } else if (arg == "--tq-distribution=correlated") {
            config.distribution_filter = Distribution::Correlated;
        } else if (arg.starts_with("--tq-implementation=")) {
            config.implementation_filter =
                std::string(arg.substr(std::string_view("--tq-implementation=").size()));
        } else if (arg.starts_with("--tq-corpus-size=")) {
            config.corpus_size_override =
                std::stoull(std::string(arg.substr(std::string_view("--tq-corpus-size=").size())));
        } else if (arg.starts_with("--tq-query-count=")) {
            config.query_count_override =
                std::stoull(std::string(arg.substr(std::string_view("--tq-query-count=").size())));
        } else if (arg.starts_with("--tq-repetitions=")) {
            config.repetitions_override =
                std::stoull(std::string(arg.substr(std::string_view("--tq-repetitions=").size())));
        } else {
            throw std::invalid_argument(
                "usage: bm_tq_production_fp32 --tq-mode=smoke|full|list "
                "[--tq-dim=N] [--tq-bits=N] [--tq-metric=ip|cosine] "
                "[--tq-distribution=gaussian|unit|log_uniform|correlated] "
                "[--tq-implementation=fp32|dense-full|dense-coarse|fast-dense|fast-circulant] "
                "[--tq-corpus-size=N] [--tq-query-count=N] [--tq-repetitions=N]");
        }
    }
    if (config.corpus_size_override) {
        config.corpus_size = *config.corpus_size_override;
    }
    if (config.query_count_override) {
        config.query_count = *config.query_count_override;
    }
    if (config.repetitions_override) {
        config.repetitions = *config.repetitions_override;
    }
    if (config.corpus_size == 0 || config.query_count == 0 || config.repetitions == 0) {
        throw std::invalid_argument("corpus size, query count, and repetitions must be positive");
    }
    return config;
}

std::string DistributionOption(Distribution distribution) {
    switch (distribution) {
    case Distribution::Gaussian:
        return "gaussian";
    case Distribution::UnitGaussian:
        return "unit";
    case Distribution::LogUniformNormIp:
        return "log_uniform";
    case Distribution::Correlated:
        return "correlated";
    }
    return "unknown";
}

bool MatchesFilters(const Config &config, size_t dim, size_t bits, VecSimMetric metric,
                    Distribution distribution) {
    return (!config.dim_filter || *config.dim_filter == dim) &&
           (!config.bits_filter || *config.bits_filter == bits) &&
           (!config.metric_filter || *config.metric_filter == metric) &&
           (!config.distribution_filter || *config.distribution_filter == distribution);
}

bool RunsImplementation(const Config &config, std::string_view name) {
    return !config.implementation_filter || *config.implementation_filter == name;
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Config config = ParseArgs(argc, argv);
        const Metadata metadata = GetMetadata();
        std::cout << "# run_profile: " << RunProfile(config) << std::endl;
        std::cout << "# Full production command (run sequentially, capture with tee): " << argv[0]
                  << " --tq-mode=full" << std::endl;
        if (config.mode == RunMode::List) {
            for (size_t dim : kProductionDimensions) {
                for (size_t bits : kBitWidths) {
                    for (VecSimMetric metric : kMetrics) {
                        for (Distribution distribution :
                             {Distribution::Gaussian, Distribution::UnitGaussian,
                              Distribution::LogUniformNormIp, Distribution::Correlated}) {
                            std::cout
                                << argv[0] << " --tq-mode=full"
                                << " --tq-dim=" << dim << " --tq-bits=" << bits
                                << " --tq-metric=" << (metric == VecSimMetric_IP ? "ip" : "cosine")
                                << " --tq-distribution=" << DistributionOption(distribution)
                                << std::endl;
                        }
                    }
                }
            }
            return 0;
        }
        VerifyPayloadAndChecksum();
        VerifyTqIdentityMetadata();
        const std::vector<size_t> dimensions =
            config.mode == RunMode::Smoke
                ? std::vector<size_t>{8}
                : std::vector<size_t>(kProductionDimensions.begin(), kProductionDimensions.end());
        for (size_t dim : dimensions) {
            for (size_t bits : kBitWidths) {
                for (VecSimMetric metric : kMetrics) {
                    for (Distribution distribution :
                         {Distribution::Gaussian, Distribution::UnitGaussian,
                          Distribution::LogUniformNormIp, Distribution::Correlated}) {
                        if (!MatchesFilters(config, dim, bits, metric, distribution)) {
                            continue;
                        }
                        if ((bits == kBitWidths.front() || config.bits_filter) &&
                            RunsImplementation(config, "fp32")) {
                            RunFp32ExactBaseline(config, metadata, dim, metric, distribution);
                        }
                        if (RunsImplementation(config, "dense-full")) {
                            RunDenseReference(config, metadata, dim, bits, metric, distribution,
                                              "DenseReference + FullDecodeReference", false);
                        }
                        if (RunsImplementation(config, "dense-coarse")) {
                            RunDenseReference(config, metadata, dim, bits, metric, distribution,
                                              "DenseReference + CoarseMse", true);
                        }
                        if (RunsImplementation(config, "fast-dense")) {
                            RunExplicitStructured(
                                config, metadata, dim, bits, metric, distribution,
                                "FastStructuredRotationV1 + DenseGaussianQjl + CoarseMse",
                                TQFlatDetails::TQCodecConfig::FastStructuredRotation(dim, bits, dim,
                                                                                     kSeed),
                                "scalar FastStructuredRotationV1 forward transform");
                        }
                        if (RunsImplementation(config, "fast-circulant")) {
                            RunExplicitStructured(
                                config, metadata, dim, bits, metric, distribution,
                                "FastStructuredRotationV1 + CirculantGaussianQjlV1 + CoarseMse",
                                TQFlatDetails::TQCodecConfig::FastStructured(dim, bits, dim, kSeed),
                                "scalar FastStructuredV1 forward transform");
                        }
                    }
                }
            }
        }
        return 0;
    } catch (const std::exception &exception) {
        std::cerr << "TurboQuant production benchmark failed: " << exception.what() << std::endl;
        return 1;
    }
}
