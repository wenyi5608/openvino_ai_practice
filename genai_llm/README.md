# GenAI LLMPipeline

This is an example that shows the performance and memory testing of a OpenVINO.GenAI LLM pipeline.

## Build and Run

openvino_genai can be download from https://storage.openvinotoolkit.org/repositories/openvino_genai/packages

```
<OpenVINO_GenAI_DIR>\setupvars.bat
cd genai_llm
mkdir build
cmake -S . -B build && cmake --build build --config Release
.\build\Release\genai_llm.exe  "C:\\Users\\yourname\\Qwen2.5-1.5B" "infer_with_memory"
.\build\Release\genai_llm.exe  "C:\\Users\\yourname\\Qwen2.5-1.5B" "infer_with_performance"
```
