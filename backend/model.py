from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# DEFAULT: small code model for local dev (fast, low RAM)
# DEFAULT_MODEL = "Salesforce/codegen-350M-mono"  
DEFAULT_MODEL = "bigcode/starcoderbase-1b"

_tokenizer = None
_model = None
_MODEL_NAME = DEFAULT_MODEL

def load_llm(model_name=None):
    global _tokenizer, _model, _MODEL_NAME
    if model_name:
        _MODEL_NAME = model_name
    if _tokenizer is None or _model is None:
        print(f"Loading LLM: {_MODEL_NAME} (this may take a while)...")
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(_MODEL_NAME, device_map="auto")
        # _model = AutoModelForCausalLM.from_pretrained(_MODEL_NAME, device_map="auto",load_in_4bit=True,torch_dtype=torch.float16,trust_remote_code=True)
        _model.eval()
    return _tokenizer, _model

def llm_summarize(code: str, max_new_tokens: int = 150) -> str:
    tokenizer, model = load_llm()
    prompt = f"\"\"\"\n{code}\n\"\"\"\n# Explain in simple terms what this code does:\n"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.2, do_sample=False)
    summary = tokenizer.decode(out[0], skip_special_tokens=True)
    # Trim to the part after the prompt
    return summary.replace(prompt, "").strip()