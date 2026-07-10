# FastAPI Local Server serving GGUF fine-tuned NLU Model
# Install requirements: pip install fastapi uvicorn llama-cpp-python pydantic

import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama

app = FastAPI(title="Mimo NLU - Vistral/PhoGPT GGUF Local API")

# Initialize GGUF model path
MODEL_PATH = "./vismimo_gguf-unsloth.Q4_K_M.gguf"
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=4,      # Number of CPU threads
    n_gpu_layers=35   # Number of layers to offload to GPU (set to 0 for CPU-only execution)
)

class QueryRequest(BaseModel):
    text: str

@app.post("/v1/nlu")
def run_nlu(request: QueryRequest):
    sys_instruction = (
        "<s>[INST] <<SYS>>\n"
        "Bạn là Mimo, trợ lý tài chính cá nhân thân thiện và thông thái của hệ thống spending-diary. "
        "Hãy phân tích ý định người dùng và trả về một cấu trúc JSON hợp lệ có dạng: "
        '{"intent": "...", "action_type": "...", "slots": {...}, "emotion": "...", "response": "..."}.\n'
        "<</SYS>>\n\n"
    )
    
    prompt = f"{sys_instruction}{request.text} [/INST]"
    
    # Generate response from local GGUF model
    response = llm(
        prompt,
        max_tokens=256,
        stop=["</s>"],
        temperature=0.1   # Keep temperature low for deterministic JSON schema output
    )
    
    output_text = response["choices"][0]["text"].strip()
    try:
        parsed_json = json.loads(output_text)
        return parsed_json
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail=f"Model output could not be parsed as valid JSON. Raw output: {output_text}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)
