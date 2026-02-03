// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <openvino/genai/visual_language/pipeline.hpp>

#include "load_image.hpp"

#ifdef _WIN32
#    include <fcntl.h>
#    include <io.h>
#    include <stdlib.h>
#    include <windows.h>
#    include <psapi.h>

#    include <codecvt>
#    pragma comment(lib, "psapi.lib")  // PrintMemoryInfo
#    include <stdio.h>

#    include "processthreadsapi.h"
#endif

#ifdef WIN32
// To ensure correct resolution of symbols, add Psapi.lib to TARGETLIBS
// and compile with -DPSAPI_VERSION=1
static void DebugMemoryInfo(const char* header) {
    PROCESS_MEMORY_COUNTERS_EX2 pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        // The following printout corresponds to the value of Resource Memory, respectively
        printf("%s Commit \t\t\t=  0x%08X- %u (MB)\n", header, pmc.PrivateUsage, pmc.PrivateUsage / (1024 * 1024));
        printf("%s WorkingSetSize\t\t\t=  0x%08X- %u (MB)\n",
               header,
               pmc.WorkingSetSize,
               pmc.WorkingSetSize / (1024 * 1024));
        printf("%s PrivateWorkingSetSize\t\t\t=  0x%08X- %u (MB)\n",
               header,
               pmc.PrivateWorkingSetSize,
               pmc.PrivateWorkingSetSize / (1024 * 1024));
    }
}
#endif  //  WIN32

struct GenaiArgs {
    std::string vlm_model_path = "";
    std::string image_path = "";
    std::string device = "GPU";
    std::string test_mode = "memory";
};

static void usage(const std::string& prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "\n"
              << "options:\n"
              << "  -h, --help              show this help message and exit\n"
              << "  -m, --model PATH        vlm model path \n"
              << "  -img, --image PATH      image path \n"
              << "  -d, --device            Device (default: GPU)\n"
              << "  --test_mode             test mode (default: memory)\n";
}

static GenaiArgs parse_args(const std::vector<std::string>& argv) {
    GenaiArgs args;

    for (size_t i = 1; i < argv.size(); i++) {
        const std::string& arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            exit(EXIT_SUCCESS);
        } else if (arg == "-m" || arg == "--model") {
            args.vlm_model_path = argv[++i];
        } else if (arg == "-img" || arg == "--image") {
            args.image_path = argv[++i];
        } else if (arg == "-d" || arg == "--device") {
            args.device = argv[++i];
        } else if (arg == "--test_mode") {
            args.test_mode = argv[++i];
        }
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            usage(argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    return args;
}

static GenaiArgs parse_args(int argc, char** argv) {
    std::vector<std::string> argv_vec;
    argv_vec.reserve(argc);

#ifdef _WIN32
    LPWSTR* wargs = CommandLineToArgvW(GetCommandLineW(), &argc);

    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    for (int i = 0; i < argc; i++) {
        argv_vec.emplace_back(converter.to_bytes(wargs[i]));
    }

    LocalFree(wargs);
#else
    for (int i = 0; i < argc; i++) {
        argv_vec.emplace_back(argv[i]);
    }
#endif

    return parse_args(argv_vec);
}

ov::genai::StreamingStatus print_subword(std::string&& subword) {
    std::cout << subword << std::flush;
    return ov::genai::StreamingStatus::RUNNING;
}

enum class TestMode {
    invalid = 0,
    performance = 1,
    memory = 2
};

TestMode parse_args(const std::string& mode) {
    if (mode == "performance") {
        return TestMode::performance;
    } else if (mode == "memory") {
        return TestMode::memory;
    } else {
        throw std::runtime_error("Invalid test mode.\n");
    }
    return TestMode::invalid;
}

namespace fs = std::filesystem;

int main(int argc, char* argv[]) try {

    GenaiArgs genai_args = parse_args(argc, argv);

    std::filesystem::path models_path = genai_args.vlm_model_path;
    std::filesystem::path image_path = genai_args.image_path;

    if (image_path.empty() || !fs::exists(image_path)) {
        throw std::runtime_error{"Path to images is empty or does not exist."};
    }

    TestMode test_mode = parse_args(std::string(genai_args.test_mode));
    std::cout << "test mode:" << (int)test_mode << std::endl;

    std::cout << ov::get_openvino_version() << std::endl;

    std::string device = genai_args.device;
    ov::AnyMap enable_compile_cache;
    if (device == "GPU") {
        // Cache compiled models on disk for GPU to save time on the
        // next run. It's not beneficial for CPU.
        //enable_compile_cache.insert({ov::cache_dir("vlm_cache")});
    }
    ov::genai::VLMPipeline pipe(models_path, device, enable_compile_cache);

    ov::genai::GenerationConfig generation_config;

    if (test_mode == TestMode::memory) {
        generation_config.max_new_tokens = 1;  // streamer may inpact the performance test, only infer first token for the memory test
    } else {
        generation_config.max_new_tokens = 200;  // perfromance test
    }

    auto streamer = [](std::string subword) {
#ifdef WIN32
        DebugMemoryInfo("First token ");
#endif
        return ov::genai::StreamingStatus::STOP;
    };

    // input length, output length, first time, other time
    std::vector<std::tuple<size_t, size_t, float, float>> perf_records;

    std::string prompt = "Describe this image.";
    ov::genai::VLMDecodedResults vlm_res;

    size_t img_idx = 0;

    if (fs::is_directory(image_path)) {
        std::set<fs::path> sorted_images{fs::directory_iterator(image_path), fs::directory_iterator()};
        for (const fs::path& dir_entry : sorted_images) {
            std::vector<ov::Tensor> rgbs = {utils::load_image(dir_entry)};

           if (test_mode == TestMode::memory ) {
                vlm_res = pipe.generate(prompt,
                                        ov::genai::images(rgbs),
                                        ov::genai::generation_config(generation_config),
                                        ov::genai::streamer(streamer));
           } else {
               vlm_res = pipe.generate(prompt, ov::genai::images(rgbs), ov::genai::generation_config(generation_config));
           }
            
            ov::genai::PerfMetrics metrics = vlm_res.perf_metrics;
            size_t input_tokens_len = metrics.get_num_input_tokens();
            size_t num_generated_tokens = metrics.get_num_generated_tokens();

            std::string output = vlm_res.texts[0];
            std::cout << "------------------output-----------------" << std::endl;
            std::cout << output << std::endl;

            if (!img_idx) {
                std::cout << "Compile LLM model took " << metrics.get_load_time() << " ms" << std::endl;
            }
            
            perf_records.emplace_back(input_tokens_len,
                                      num_generated_tokens,
                                      metrics.get_ttft().mean,
                                      metrics.get_tpot().mean);

            img_idx++;

        }
    } else {
        std::vector<ov::Tensor> rgbs = {utils::load_image(image_path)};

        vlm_res = pipe.generate(prompt, ov::genai::images(rgbs), ov::genai::generation_config(generation_config)); //,
                                //ov::genai::streamer(print_subword));

        ov::genai::PerfMetrics metrics = vlm_res.perf_metrics;
        size_t input_tokens_len = metrics.get_num_input_tokens();
        size_t num_generated_tokens = metrics.get_num_generated_tokens();

        std::string output = vlm_res.texts[0];
        std::cout << "------------------output-----------------" << std::endl;
        std::cout << output << std::endl;

        std::cout << "Compile VLM model took " << metrics.get_load_time() << " ms" << std::endl;

        perf_records.emplace_back(input_tokens_len,
                                  num_generated_tokens,
                                  metrics.get_ttft().mean,
                                  metrics.get_tpot().mean);

    }

    size_t index = 0;
    for (auto i : perf_records) {
        std::cout << index << ", " << std::get<0>(i) << ", " << std::get<1>(i) << ", " << std::get<2>(i) << ", "
                  << std::get<3>(i) << std::endl;
        index++;
    }
    
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {
    }
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {
    }
    return EXIT_FAILURE;
}

