dInfer is an efficient and extensible inference framework for dLLMs. It modularizes inference into four components:
model, diffusion iteration manager, decoding strategy and KV-cache management, and provides well-designed APIs for
flexible combinations of algorithms in each component.

<p align="center">
  <img src="https://code.alipay.com/937b4b84-1ea5-40aa-a37f-bbf907733e79" alt="dInfer v0.1 architecture" width="600">
  <br>
  <b>Figure</b>: Overall Architecture of dInfer
</p>

dInfer supports multiple dLLM variants, including LLaDA and LLaDA-MoE. It introduces a soft diffusion iteration algorithm
for smoother denoising, hierarchical and credit decoding for enhanced parallel decoding, and a vicinity refresh strategy
for KV-cache management to mitigate cache staleness.
Beyond algorithmic improvements, it integrates several system-level optimizations. It supports both tensor parallelism
(TP) and expert parallelism (EP) to maximize GPU utilization even at batch size 1. It leverages PyTorch compilation and
NVIDIA CUDA Graphs for efficient kernel execution, and introduces a loop unrolling mechanism to eliminate CUDA stream
bubbles across diffusion iterations.

