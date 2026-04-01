import torch
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "/root/autodl-tmp/Qwen3-ASR-0.6B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    max_inference_batch_size=32, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
    max_new_tokens=256, # Maximum number of tokens to generate. Set a larger value for long audio input.
)

def recognize(audio_path: str, output_path: str) -> str:
    results = model.transcribe(audio_path, language="Chinese")  
    with open(output_path, "w") as writer:
        writer.write(results[0].text)