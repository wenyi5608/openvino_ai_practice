# GenAI VLMPipeline

This is an example that shows the performance and memory testing of a OpenVINO.GenAI VLM pipeline.

## Build and Run
Windows

Download and Install VS2022, Cmake:

VS2022: Install latest [Visual Studio 2022 Community](https://visualstudio.microsoft.com/zh-hans/downloads/) and Install C and C++ support in Visual Studio.
Cmake: If Cmake not installed in the terminal Command Prompt, please download and install Cmake or use the terminal Developer Command Prompt for VS 2022 instead.

openvino_genai can be download from https://storage.openvinotoolkit.org/repositories/openvino_genai/packages

```
<OpenVINO_GenAI_DIR>\setupvars.bat
cd genai_vlm
mkdir build
cmake -S . -B build && cmake --build build --config Release
.\build\Release\genai_llm.exe -m \\path\\to\\Qwen3-VL-4B -img \\path\\to\\image -d GPU  --test_mode memory
.\build\Release\genai_vlm.exe -m \\path\\to\\Qwen3-VL-4B -img \\path\\to\\image -d GPU  --test_mode performance
.\build\Release\genai_vlm.exe -m \\path\\to\\Qwen3-VL-4B -img \\path\\to\\image -lora_adapter \\path\\to\\adapter_model.safetensors -d GPU --test_mode lora_memory
.\build\Release\genai_vlm.exe -m \\path\\to\\Qwen3-VL-4B -img \\path\\to\\image -lora_adapter \\path\\to\\adapter_model.safetensors -d GPU --test_mode lora_performance

```
