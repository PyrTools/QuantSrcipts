# QuantSrcipts
Repository of low user effort quantization scripts for multiple formats.

Depenencies
ALL ``pip install llmcompressor transformers optimum autoawq auto-gptq torchvision hqq``

GPTQ ``pip install transformers optimum auto-gptq``

AWQ ``pip install transformers autoawq torchvision``

FP8 ``pip install llmcompressor transformers``

HQQ ``pip install hqq transformers``

For example if you wanted to quantize Qwen1.5-0.5B to GPTQ, AWQ, HQQ, and FP8 in one command, can do this.
```
python3 multiquant.py -r "gptq,awq,fp8,hqq" -m "Qwen/Qwen1.5-0.5B"
```
