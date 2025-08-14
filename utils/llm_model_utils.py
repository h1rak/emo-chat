import os
import transformers
from utils.steering_layer import SteeringLayer
from dotenv import load_dotenv
from transformers import BitsAndBytesConfig
import torch

load_dotenv()

def load_llm_model(device):
    MODEL_WEIGHTS_FOLDER = os.getenv("MODEL_WEIGHTS_FOLDER")
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
    )

    # 元のコードを以下に置き換え
    llm_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_WEIGHTS_FOLDER,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map={"": 0},   # .to(DEVICE)の代わりに使用
        torch_dtype=torch.float16
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_WEIGHTS_FOLDER)

    return llm_model,tokenizer

def add_steering_layers(llm_model,insertion_layers):
    for layer in insertion_layers:
        llm_model.model.layers[layer].mlp = SteeringLayer(llm_model.model.layers[layer].mlp)

    return llm_model

def load_llm_model_with_insertions(device,insertion_layers):
    llm_model,tokenizer = load_llm_model(device)
    llm_model = add_steering_layers(llm_model,insertion_layers)

    return llm_model,tokenizer