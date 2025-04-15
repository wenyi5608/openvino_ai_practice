// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <format>
#include "openvino/genai/llm_pipeline.hpp"
#include "utils.h"
#include "config.h" // path for prompts
#include <windows.h>
#include <stdlib.h>
#include <psapi.h>
#pragma comment(lib,"psapi.lib") //PrintMemoryInfo
#include <stdio.h>
#include "processthreadsapi.h"
#include <openvino/openvino.hpp>


#ifdef WIN32
// To ensure correct resolution of symbols, add Psapi.lib to TARGETLIBS
// and compile with -DPSAPI_VERSION=1
static void DebugMemoryInfo(const char* header) {
    PROCESS_MEMORY_COUNTERS_EX2 pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        // The following printout corresponds to the value of Resource Memory, respectively
        printf("%s:\tCommit \t\t\t=  0x%08X- %u (MB)\n", header, pmc.PrivateUsage, pmc.PrivateUsage / (1024 * 1024));
        printf("%s:\tWorkingSetSize\t\t\t=  0x%08X- %u (MB)\n",
               header,
               pmc.WorkingSetSize,
               pmc.WorkingSetSize / (1024 * 1024));
        printf("%s:\tPrivateWorkingSetSize\t\t\t=  0x%08X- %u (MB)\n",
               header,
               pmc.PrivateWorkingSetSize,
               pmc.PrivateWorkingSetSize / (1024 * 1024));
    }
}
#endif  //  WIN32

enum class TestMode {
    invalid = 0,
    infer_with_memory = 1,
    infer_with_performance = 2,
};

TestMode parse_args(const std::string& mode) {
    if (mode == "infer_with_memory") {
        return TestMode::infer_with_memory;
    } else if (mode == "infer_with_performance") {
        return TestMode::infer_with_performance;
    } else {
        throw std::runtime_error("Invalid test mode.\n");
    }
    return TestMode::invalid;
}
int main(int argc, char* argv[]) try {
    if (3 != argc) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] +
                                 " <MODEL_DIR> <TEST_MODE> ");
    }

    std::filesystem::path models_path = argv[1];

    std::filesystem::path prompts_path{CURRENT_SOURCE_DIR};
    prompts_path /= "sentences.txt";
    std::vector<std::string> sentences = read_file_lines(prompts_path);


    TestMode test_mode = parse_args(std::string(argv[2]));
    std::cout << "test mode:" << (int)test_mode << std::endl;

    std::string device = "GPU";  // CPU can be used as well

    using namespace ov::genai;
    std::cout << ov::get_openvino_version() << std::endl;
 

    ov::AnyMap mp;

    mp = {{"ATTENTION_BACKEND", "SDPA"},
          ov::device::properties(device, ov::cache_dir(std::format("{}_cache", device)))};

    LLMPipeline pipe(models_path, device, mp);

    // only used in memory test , test the memory usage after the first inference
    auto streamer = [](std::string subword) {
#ifdef WIN32
       DebugMemoryInfo("First token ");
#endif
        return ov::genai::StreamingStatus::STOP;
    };

    // input length, output length, first time, other time
    std::vector<std::tuple<size_t, size_t, float, float>> perf_records;
    ov::genai::GenerationConfig config;
    if (test_mode == TestMode::infer_with_memory) {
        config.max_new_tokens = 1;  // streamer may inpact the performance test, only infer first token for the memory test
    } else {
        config.max_new_tokens = 200; // perfromance test
    }

    for (int i = 0 ; i<sentences.size(); ++i ) {
        ov::genai::DecodedResults res;
        const auto& prompt = sentences[i];
        if (test_mode == TestMode::infer_with_memory) {
            res = pipe.generate(prompt, config, streamer);
        } else {
            res = pipe.generate(prompt, config);
        }

        ov::genai::PerfMetrics metrics = res.perf_metrics;

        size_t input_tokens_len = metrics.get_num_input_tokens();
        size_t num_generated_tokens = metrics.get_num_generated_tokens();

        if (!i) {
            std::cout << "Compile LLM model took " << metrics.get_load_time() << " ms" << std::endl;
        }
            
        perf_records.emplace_back(input_tokens_len,
                                  num_generated_tokens,
                                  metrics.get_ttft().mean,
                                  metrics.get_tpot().mean);
    }

    if (test_mode == TestMode::infer_with_performance) {
        std::cout << "input id, input token len, out token len, first token time, average time" << std::endl;
        size_t index = 0;
        for (auto i : perf_records) {
            std::cout << index << ", " << std::get<0>(i) << ", " << std::get<1>(i) << ", " << std::get<2>(i) << ", "
                      << std::get<3>(i) << std::endl;
            index++;
        }
    }

    perf_records.clear();

} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
