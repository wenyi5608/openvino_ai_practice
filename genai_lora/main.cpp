// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <filesystem>
#include <format>
#include "openvino/genai/llm_pipeline.hpp"
#include "utils.h"
#include "config.h" // path for prompts

#ifdef _WIN32
#include <codecvt>
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#include <stdlib.h>
#include <psapi.h>
#pragma comment(lib,"psapi.lib") //PrintMemoryInfo
#include <stdio.h>
#include "processthreadsapi.h"
#endif

#include <openvino/openvino.hpp>


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
    std::string llm_model_path = "";
    std::string lora_path = "adapter_model.safetensors";
    std::string attention_backend = "";
    std::string device = "GPU";
    float lora_alpha = 0.5;
    bool enable_lora = false;
    int output_fixed_len = 0;
    std::string test_mode = "no_lora_memory";     
};

static void usage(const std::string& prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "\n"
              << "options:\n"
              << "  -h, --help              show this help message and exit\n"
              << "  -m, --model PATH        llm model path \n"
              << "  -lora_adapter PATH      lora adapter model file (default: adapter_model.safetensors)\n"
              << "  -lora_alpha N           lora_alpha (default: 0.5)\n"
              << "  -d, --device            Device (default: GPU)\n"
              << "  --attention_backend     genai attention backend(SDPA or PA) (default: PA)\n"
              << "  --test_mode             test mode (default: no_lora_memory)\n";
        //<< "  --output_fixed_len N    set output fixed lenth (default: 0, output lenth is determined by the model)\n";
}

static GenaiArgs parse_args(const std::vector<std::string>& argv) {
    GenaiArgs args;

    for (size_t i = 1; i < argv.size(); i++) {
        const std::string& arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            exit(EXIT_SUCCESS);
        } else if (arg == "-m" || arg == "--model") {
            args.llm_model_path = argv[++i];
        } else if (arg == "-lora_adapter") {
            args.lora_path = argv[++i];
        } else if (arg == "-lora_alpha") {
            args.lora_alpha = std::stof(argv[++i]);
        } else if (arg == "-attention_backend") {
            args.attention_backend = argv[++i];
        } else if (arg == "-d" || arg == "--device") {
            args.device = argv[++i];
        } else if (arg == "--test_mode") {
            args.test_mode = argv[++i];
        } /*else if (arg == "--output_fixed_len") {
            args.output_fixed_len = std::stoi(argv[++i]);
        }*/ else {
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

std::pair<ov::Tensor, ov::Tensor> tokenize(ov::InferRequest& tokenizer, std::string&& prompt) {
    constexpr size_t BATCH_SIZE = 1;
    tokenizer.set_input_tensor(ov::Tensor{ov::element::string, {BATCH_SIZE}, &prompt});
    tokenizer.infer();
    return {tokenizer.get_tensor("input_ids"), tokenizer.get_tensor("attention_mask")};
}

std::string detokenize(ov::InferRequest& detokenizer, std::vector<int64_t>& tokens) {
    constexpr size_t BATCH_SIZE = 1;
    detokenizer.set_input_tensor(ov::Tensor{ov::element::i64, {BATCH_SIZE, tokens.size()}, tokens.data()});
    detokenizer.infer();
    return detokenizer.get_output_tensor().data<std::string>()[0];
}

// The following reasons require TextStreamer to keep a cache of previous tokens:
// detokenizer removes starting ' '. For example detokenize(tokenize(" a")) == "a",
// but detokenize(tokenize("prefix a")) == "prefix a"
// 1 printable token may consist of 2 token ids: detokenize(incomplete_token_idx) == "�"
struct TextStreamer {
    ov::InferRequest detokenizer;
    std::vector<int64_t> token_cache;
    size_t print_len = 0;

    void put(int64_t token) {
        token_cache.push_back(token);
        std::string text = detokenize(detokenizer, token_cache);
        if (!text.empty() && '\n' == text.back()) {
            // Flush the cache after the new line symbol
            std::cout << std::string_view{text.data() + print_len, text.size() - print_len};
            token_cache.clear();
            print_len = 0;
            return;
        }
        if (text.size() >= 3 && text.compare(text.size() - 3, 3, "�") == 0) {
            // Don't print incomplete text
            return;
        }
        std::cout << std::string_view{text.data() + print_len, text.size() - print_len} << std::flush;
        print_len = text.size();
    }

    void end() {
        std::string text = detokenize(detokenizer, token_cache);
        std::cout << std::string_view{text.data() + print_len, text.size() - print_len} << '\n';
        token_cache.clear();
        print_len = 0;
    }
};


enum class TestMode {
    invalid = 0,
    no_lora_performance = 1,
    no_lora_memory = 2,
    empty_lora_performance = 3,
    empty_lora_memory = 4,
    infer_with_lora_performance = 5,
    infer_with_lora_memory = 6,
};

TestMode parse_args(const std::string& mode) {
    if (mode == "no_lora_performance") {
        return TestMode::no_lora_performance;
    } else if (mode == "no_lora_memory") {
        return TestMode::no_lora_memory;
    } else if (mode == "empty_lora_performance") {
        return TestMode::empty_lora_performance;
    } else if (mode == "empty_lora_memory") {
        return TestMode::empty_lora_memory;
    } else if (mode == "infer_with_lora_performance") {
        return TestMode::infer_with_lora_performance;
    } else if (mode == "infer_with_lora_memory") {
        return TestMode::infer_with_lora_memory;
    } else {
        throw std::runtime_error("Invalid test mode.\n");
    }
    return TestMode::invalid;
}

int main(int argc, char* argv[]) try {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    _setmode(_fileno(stdin), _O_WTEXT);
#endif

    GenaiArgs genai_args = parse_args(argc, argv);

    std::filesystem::path models_path = genai_args.llm_model_path;
    std::filesystem::path adapter_path = genai_args.lora_path;

    TestMode test_mode = parse_args(std::string(genai_args.test_mode));
    std::cout << "test mode:" << (int)test_mode << std::endl;

    std::string device = genai_args.device;  // CPU can be used as well

    std::filesystem::path prompts_path{CURRENT_SOURCE_DIR};
    prompts_path /= "sentences.txt";
    std::vector<std::string> sentences = read_file_lines(prompts_path);

    ov::Core core;

#ifdef _WIN32
    core.add_extension("openvino_tokenizers.dll");
#elif defined(__linux__)
    core.add_extension("libopenvino_tokenizers.so");
#endif

    std::filesystem::path prompts_path_test{CURRENT_SOURCE_DIR};
    prompts_path_test /= "A_Bid_for_Fortune.txt";

    std::string str;
    std::ifstream file(prompts_path_test);
    std::getline(file, str, '\0');

    std::string token_model_path = genai_args.llm_model_path;
    std::string token_model = token_model_path + "/openvino_tokenizer.xml";
    std::string detoken_model = token_model_path + "/openvino_detokenizer.xml";

    ov::InferRequest tokenizer = core.compile_model(token_model, "CPU").create_infer_request();
    ov::InferRequest detokenizer = core.compile_model(detoken_model, "CPU").create_infer_request();
    TextStreamer text_streamer{detokenizer};

    auto input_ids = tokenizer.get_tensor("input_ids");

    tokenize(tokenizer, str.c_str());

    std::cout << "input_ids length " << input_ids.get_size() << std::endl;

    using namespace ov::genai;
    std::cout << ov::get_openvino_version() << std::endl;
 
    auto start_time = std::chrono::steady_clock::now();
    Adapter adapter;

    if (!(test_mode == TestMode::no_lora_memory) && !(test_mode == TestMode::no_lora_performance)) {
        adapter = Adapter(adapter_path);
    }
 
    auto stop_time = std::chrono::steady_clock::now();

    size_t load_time = PerfMetrics::get_microsec(stop_time - start_time);
    std::cout << "lora load time " << load_time / 1000 << " ms" << std::endl;

    ov::AnyMap mp; //{"ATTENTION_BACKEND", "SDPA"},
    if (test_mode == TestMode::no_lora_memory || test_mode == TestMode::no_lora_performance) {
        mp = {
              ov::device::properties(device, ov::cache_dir(std::format("{}_cache", device)))};
              //mp[ov::cache_mode.name()] = ov::CacheMode::OPTIMIZE_SIZE;
    } else {
        mp = {
              ov::device::properties(device, ov::cache_dir(std::format("{}_cache", device))),
              adapters(adapter)};
    }

    LLMPipeline pipe(models_path, device, mp);

    int idx = 0;
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

    if (test_mode == TestMode::no_lora_memory 
        || test_mode == TestMode::empty_lora_memory
        || test_mode == TestMode::infer_with_lora_memory) {
        config.max_new_tokens = 1;  // streamer may inpact the performance test, only infer first token for the memory test
    } else {
        config.max_new_tokens = 200; // perfromance test
    }

    if (test_mode == TestMode::empty_lora_memory
        || test_mode == TestMode::empty_lora_performance) {
        config.adapters = ov::genai::AdapterConfig{};
    } else if (test_mode == TestMode::infer_with_lora_memory
        || test_mode == TestMode::infer_with_lora_performance) {
        config.adapters = ov::genai::AdapterConfig{adapter, 0.5};
    }

    size_t input_ids_len[] = { 1000, 3000, 8000, 10000, 15000, 20000, 25000, 30000};

    for (size_t i = 0; i < 6; i++) {
        ov::genai::EncodedResults enc_res;
        size_t input_id_len = std::min(input_ids_len[i], input_ids.get_size());

        ov::Tensor encode_inputs_id = ov::Tensor(ov::element::i64, {1, input_id_len}, &input_ids.data<int64_t>()[i*10]);

        if (test_mode == TestMode::no_lora_memory
            || test_mode == TestMode::empty_lora_memory
            || test_mode == TestMode::infer_with_lora_memory) {
            enc_res = pipe.generate(encode_inputs_id, config, streamer);
        } else {
            enc_res = pipe.generate(encode_inputs_id, config);
        }

        ov::genai::PerfMetrics metrics = enc_res.perf_metrics;

        size_t input_tokens_len = metrics.get_num_input_tokens();
        size_t num_generated_tokens = metrics.get_num_generated_tokens();

        if (!i) {
            std::cout << "Compile LLM model took " << metrics.get_load_time() << " ms" << std::endl;
        }

        if (test_mode == TestMode::infer_with_lora_performance
            || test_mode == TestMode::empty_lora_performance
            || test_mode == TestMode::no_lora_performance) {

            for (size_t idx = 0; idx < enc_res.tokens[0].size(); ++idx) {
                text_streamer.put(enc_res.tokens[0][idx]);
             }

            text_streamer.end();
        }

        perf_records.emplace_back(input_tokens_len,
                                  num_generated_tokens,
                                  metrics.get_ttft().mean,
                                  metrics.get_tpot().mean);

    }

    if (test_mode == TestMode::no_lora_performance
        || test_mode == TestMode::empty_lora_performance
        || test_mode == TestMode::infer_with_lora_performance) {
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
