
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <openvino/genai/visual_language/pipeline.hpp>

#include "utils.h"
#include "config.h" // path for prompts

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
    std::string lora_path = "adapter_model.safetensors";
    float lora_alpha = 0.5;
    std::string device = "GPU";
    std::string test_mode = "memory";
};

static void usage(const std::string& prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "\n"
              << "options:\n"
              << "  -h, --help              show this help message and exit\n"
              << "  -m, --model PATH        vlm model path \n"
              << "  -lora_adapter PATH      lora adapter model file (default: adapter_model.safetensors)\n"
              << "  -lora_alpha N           lora_alpha (default: 0.5)\n"
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
        } else if (arg == "-lora_adapter") {
            args.lora_path = argv[++i];
        } else if (arg == "-lora_alpha") {
            args.lora_alpha = std::stof(argv[++i]);
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
    memory = 2,
    lora_performance = 3,
    lora_memory = 4
};

TestMode parse_args(const std::string& mode) {
    if (mode == "performance") {
        return TestMode::performance;
    } else if (mode == "memory") {
        return TestMode::memory;
    } else if (mode == "lora_performance") {
        return TestMode::lora_performance;
    } else if (mode == "lora_memory") {
        return TestMode::lora_memory;
    } else {
        throw std::runtime_error("Invalid test mode.\n");
    }
    return TestMode::invalid;
}

namespace fs = std::filesystem;

int main(int argc, char* argv[]) try {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    _setmode(_fileno(stdin), _O_WTEXT);
#endif

    GenaiArgs genai_args = parse_args(argc, argv);

    std::filesystem::path models_path = genai_args.vlm_model_path;
    std::filesystem::path adapter_path = genai_args.lora_path;

    TestMode test_mode = parse_args(std::string(genai_args.test_mode));
    std::cout << "test mode:" << (int)test_mode << std::endl;

    std::cout << ov::get_openvino_version() << std::endl;

    std::filesystem::path prompts_path{CURRENT_SOURCE_DIR};
    prompts_path /= "sentences.txt";
    std::vector<std::string> sentences = read_file_lines(prompts_path);

    std::string device = genai_args.device;

    ov::genai::Adapter adapter;
    if (!(test_mode == TestMode::memory) && !(test_mode == TestMode::performance)) {
        adapter = ov::genai::Adapter(adapter_path);
    }

    ov::AnyMap mp;
    if (test_mode == TestMode::memory || test_mode == TestMode::performance) {
        mp = {{"ATTENTION_BACKEND", "PA"},
              ov::device::properties(device, ov::cache_dir(std::format("{}_cache", device)))};

    } else {
        mp = {{"ATTENTION_BACKEND", "PA"},
              ov::device::properties(device, ov::cache_dir(std::format("{}_cache", device))),
              ov::genai::adapters(adapter)};
    }

    ov::genai::VLMPipeline pipe(models_path, device, mp);

    ov::genai::GenerationConfig generation_config;

    // input length, output length, first time, other time
    std::vector<std::tuple<size_t, size_t, float, float>> perf_records;

    if (test_mode == TestMode::memory || test_mode == TestMode::lora_memory) {
        generation_config.max_new_tokens = 1;  // streamer may inpact the performance test, only infer first token for the memory test
    } else {
        generation_config.max_new_tokens = 200;  // perfromance test
    }

    if (test_mode == TestMode::lora_memory || test_mode == TestMode::lora_performance) {
        generation_config.adapters = ov::genai::AdapterConfig{adapter, genai_args.lora_alpha};
    }

    auto streamer = [](std::string subword) {
#ifdef WIN32
        DebugMemoryInfo("First token ");
#endif
        return ov::genai::StreamingStatus::STOP;
    };

    for (int i = 0; i < sentences.size(); ++i) {
        ov::genai::VLMDecodedResults res;
        const auto& prompt = sentences[i];

        if (test_mode == TestMode::memory || test_mode == TestMode::lora_memory) {
            res = pipe.generate(prompt,
                                    ov::genai::generation_config(generation_config),
                                    ov::genai::streamer(streamer));

        } else {
            res = pipe.generate(prompt, ov::genai::generation_config(generation_config));;
        }

        ov::genai::PerfMetrics metrics = res.perf_metrics;

        //std::string output = res.texts[0];
        //std::cout << "------------------output-----------------" << std::endl;
        //std::cout << output << std::endl;

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

    if (test_mode == TestMode::performance || test_mode == TestMode::lora_performance) {
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

