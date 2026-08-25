message("# VectorSimilarity_Benchmark root: " ${root})
message("# VectorSimilarity_Benchmark binroot: " ${binroot})

project(VectorSimilarity_Benchmark)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
include_directories(../)
include_directories(./)

enable_testing()

exec_program(${root}/tests/benchmark/benchmarks.sh OUTPUT_VARIABLE BENCHMARKS)
string(REPLACE "\n" ";" BENCHMARKS ${BENCHMARKS})

foreach(benchmark IN ITEMS ${BENCHMARKS})
	if(${benchmark} MATCHES "spaces*")
		continue()
	endif()
	# NOTE: mock_thread_pool.cpp should appear *before* the benchmark files, so we can ensure that the thread pool
	# globals are initialized before we use them in the benchmark classes (as globals initialization is done by order).
	add_executable(bm_${benchmark} ../utils/mock_thread_pool.cpp bm_vecsim_general.cpp run_files/bm_${benchmark}.cpp)
	target_link_libraries(bm_${benchmark} VectorSimilarity benchmark::benchmark gtest)
endforeach()

execute_process(COMMAND git -C ${root} rev-parse --short HEAD
                OUTPUT_VARIABLE TQ_BENCH_GIT_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND git -C ${root} status --porcelain
                OUTPUT_VARIABLE TQ_BENCH_GIT_STATUS OUTPUT_STRIP_TRAILING_WHITESPACE)
if(TQ_BENCH_GIT_STATUS)
  set(TQ_BENCH_GIT_DIRTY "dirty")
else()
  set(TQ_BENCH_GIT_DIRTY "clean")
endif()
if(CMAKE_BUILD_TYPE)
  set(TQ_BENCH_BUILD_MODE "${CMAKE_BUILD_TYPE}")
else()
  set(TQ_BENCH_BUILD_MODE "multi-config")
endif()

# The TurboQuant production benchmark owns its runner so it can emit the reproducibility
# metadata and explicit `not_implemented` records required by the production matrix.  It is
# intentionally a CTest smoke target; the full 6-dimension matrix is invoked manually.
if("tq_production_fp32" IN_LIST BENCHMARKS)
	target_compile_definitions(bm_tq_production_fp32 PRIVATE
		TQ_BENCH_GIT_COMMIT="${TQ_BENCH_GIT_COMMIT}"
		TQ_BENCH_GIT_DIRTY="${TQ_BENCH_GIT_DIRTY}"
		TQ_BENCH_BUILD_MODE="${TQ_BENCH_BUILD_MODE}")
	add_test(NAME bm_tq_production_smoke COMMAND bm_tq_production_fp32 --tq-mode=smoke)
	set_tests_properties(bm_tq_production_smoke PROPERTIES LABELS "benchmark;turboquant;smoke")
endif()

include(GoogleTest)

if (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "(aarch64)|(arm64)|(ARM64)|(armv.*)")
	include(${root}/cmake/aarch64InstructionFlags.cmake)
	add_compile_definitions(BENCHMARK_ARCH=arm64)
else()
	include(${root}/cmake/x86_64InstructionFlags.cmake)
	add_compile_definitions(BENCHMARK_ARCH=x86_64)
endif()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Spaces benchmarks								                                          #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

set(DATA_TYPE fp32 fp64 bf16 fp16 int8 uint8 sq8_fp32 sq8_fp16 sq8_sq8)
foreach(data_type IN LISTS DATA_TYPE)
	add_executable(bm_spaces_${data_type} spaces_benchmarks/bm_spaces_${data_type}.cpp)
	target_link_libraries(bm_spaces_${data_type} VectorSimilarity benchmark::benchmark)
endforeach()
