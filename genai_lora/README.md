# Utilizing LoRa with GenAI LLMPipeline

This is an example that shows the performance and memory testing of a OpenVINO.GenAI LLM pipeline integrated with LoRa.

## Build and Run


```
<OpenVINO_GenAI_DIR>\setupvars.bat
cd genai_lora
mkdir build
cmake -S . -B build && cmake --build build --config Release
.\build\Release\genai_llm.exe -m \path\to\ov_llm_model -lora_adapter \path\to\adapter_model.safetensors -d GPU --test_mode  "infer_with_lora_memory"
```
