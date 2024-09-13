# QuantSrcipts
Repository of low user effort quantization scripts for multiple formats.

For example if you wanted to quantize Qwen1.5-0.5B to GPTQ, AWQ, and FP8 in one command, can do this.
```
python3 multiquant.py -r "gptq,awq,fp8" -m "Qwen/Qwen1.5-0.5B"
```
