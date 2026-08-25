/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

// Run from the VectorSimilarity root:
// c++ -std=c++20 -O0 -I tests/unit tests/unit/generate_tq_paper_golden.cpp \
//   tests/unit/tq_paper_reference.cpp -o /tmp/generate_tq_paper_golden && \
//   /tmp/generate_tq_paper_golden

#include "tq_paper_reference.h"

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

std::vector<float> diagnosticVector(size_t dim, size_t salt, float scale) {
    std::vector<float> values(dim);
    for (size_t i = 0; i < dim; ++i) {
        values[i] = scale * (0.73f * std::sin(static_cast<float>((i + 1) * (salt + 3)) * 0.317f) +
                             0.21f * std::cos(static_cast<float>((i + 2) * (salt + 7)) * 0.193f));
    }
    return values;
}

void dump(size_t dim, size_t bits, size_t seed, tq_paper_reference::Metric metric) {
    constexpr uint64_t kQjlSeedOffset = 0xCAFEBABE00000001ULL;
    const auto input = diagnosticVector(dim, bits + seed, 2.75f);
    const auto query = diagnosticVector(dim, bits * 3 + seed, 1.6f);
    tq_paper_reference::ReferenceDenseModel model(dim, bits, seed, seed + kQjlSeedOffset);
    const auto encoded = model.encode(input, metric);
    const auto prepared = model.preprocessQuery(query, metric);
    const auto decoded = model.decode(encoded);
    const auto &codebook = model.codebook();

    std::cout << "fixture " << dim << " " << bits << " " << seed << "\nbytes";
    for (uint8_t value : encoded.bytes) {
        std::cout << " " << static_cast<unsigned>(value);
    }
    std::cout << "\nalpha_gamma " << std::setprecision(17) << encoded.alpha << " " << encoded.gamma
              << "\nrotated";
    for (double value : prepared.rotated) {
        std::cout << " " << value;
    }
    std::cout << "\nqjl";
    for (double value : prepared.qjl_projection) {
        std::cout << " " << value;
    }
    std::cout << "\nscore " << model.estimateIp(encoded, prepared) << "\ndecoded";
    for (double value : decoded) {
        std::cout << " " << value;
    }
    std::cout << "\ncodebook";
    for (double value : codebook.centroids) {
        std::cout << " " << value;
    }
    std::cout << "\nboundaries";
    for (double value : codebook.boundaries) {
        std::cout << " " << value;
    }
    std::cout << "\n";
}

} // namespace

int main() {
    dump(3, 8, 17, tq_paper_reference::Metric::InnerProduct);
    dump(8, 4, 31, tq_paper_reference::Metric::Cosine);
    dump(15, 2, 7, tq_paper_reference::Metric::InnerProduct);
    dump(31, 2, 17, tq_paper_reference::Metric::InnerProduct);
}
