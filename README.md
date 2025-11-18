## InferLlama

A high-performance C/C++ inference engine for LLaMA-based models, optimized for CPU execution.

Current model weights is from here - (https://huggingface.co/unsloth/llama-2-7b-chat)

1. Convert the weights into .bin file.

```bash
python3 convert_to_bin.py
```

### The binary format structure

![](artifacts/bin.png)


### Roadmap

- [ ] Add a makefile for builds
- [ ] Implement SIMD instructions for faster computation
- [ ] Add quantization algorithms with performance benchmarking
- [ ] Support GPU operations via CUDA C++