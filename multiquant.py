import argparse
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

# Function for quantizing using AWQ
def quantize_awq(model_path):
    try:
        from awq import AutoAWQForCausalLM
        import torchvision  # Check for torchvision
    except ImportError:
        print("AWQ quantization requires the `autoawq` and `torchvision` packages. Please install them via pip.")
        return
    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    }
    
    # Load model and tokenizer
    model = AutoAWQForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, device_map="auto", use_cache=False)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Quantize the model
    model.quantize(tokenizer, quant_config=quant_config)

    # Save quantized model
    SAVE_DIR = model_path.split("/")[-1] + "-AWQ"
    model.save_quantized(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

    print(f'Model is quantized and saved at "{SAVE_DIR}"')

# Function for quantizing using HQQ
def quantize_hqq(model_path):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, HqqConfig
    except ImportError:
        print("HQQ quantization requires the `hqq` packages. Please install them via pip.")
        return
    
    quant_config = HqqConfig(nbits=4, group_size=128, axis=1)
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                             torch_dtype=torch.float16,
                                             device_map="cuda:0",
                                             quantization_config=quant_config,
                                             low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    SAVE_DIR = model_path.split("/")[-1] + "-HQQ-4bit"
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

    print(f'Model is quantized and saved at "{SAVE_DIR}"')


# Function for quantizing using GPTQ
def quantize_gptq(model_path):
    try:
        from transformers import AutoModelForCausalLM, GPTQConfig, AutoTokenizer
        import optimum  # Check for optimum
        import auto_gptq
    except ImportError:
        print("GPTQ quantization requires the `optimum` and `auto-gptq` packages. Please install it via pip.")
        return
    from transformers import AutoModelForCausalLM, GPTQConfig, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    gptq_config = GPTQConfig(
        bits=4,
        dataset="wikitext2",
        group_size=128,
        desc_act=True,
        use_cuda_fp16=True,
        tokenizer=tokenizer
    )

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", quantization_config=gptq_config, attn_implementation="sdpa")
    model.config.quantization_config.dataset = None

    SAVE_DIR = model_path.split("/")[-1] + "-GPTQ-128G"
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

    print(f'Model is quantized and saved at "{SAVE_DIR}"')


# Function for quantizing using FP8
def quantize_fp8(model_path):
    try:
        from llmcompressor.modifiers.quantization import QuantizationModifier
        from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot
    except ImportError:
        print("FP8 quantization requires the `llmcompressor` package. Please install it via pip.")
        return
        
    # Load model and tokenizer
    model = SparseAutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Configure the quantization algorithm and scheme
    recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])
    
    # Apply quantization
    oneshot(model=model, recipe=recipe)
    
    # Save quantized model
    SAVE_DIR = model_path.split("/")[1] + "-FP8-Dynamic"
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

    print(f'Model is quantized and saved at "{SAVE_DIR}"')


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run model quantization based on the selected method.")
    
    # Argument for model path
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to the model to be quantized.")
    
    # Option to select which scripts to run (comma-separated)
    parser.add_argument("-r", "--run", type=str, required=True, help="Comma-separated list of scripts to run (e.g., 'awq,gptq,fp8').")

    # Parse arguments
    args = parser.parse_args()

    # Parse which scripts to run
    scripts_to_run = [script.strip() for script in args.run.split(',')]

    # Run scripts based on the flags
    if 'awq' in scripts_to_run:
        quantize_awq(args.model_path)
    
    if 'gptq' in scripts_to_run:
        quantize_gptq(args.model_path)
    
    if 'fp8' in scripts_to_run:
        quantize_fp8(args.model_path)

    if 'hqq' in scripts_to_run:
        quantize_hqq(args.model_path)


if __name__ == "__main__":
    main()