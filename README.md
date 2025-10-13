# dInfer: An Efficient Inference Framework for Diffusion Language Models

dInfer is an efficient and extensible inference framework for dLLMs. It modularizes inference into four components:
model, diffusion iteration manager, decoding strategy and KV-cache management, and provides well-designed APIs for
flexible combinations of algorithms in each component.

<p align="center">
  <img src="https://raw.githubusercontent.com/inclusionAI/dInfer/refs/heads/add_readme/assets/Framework2.png" alt="dInfer v0.1 architecture" width="600">
  <br>
  <b>Figure</b>: Overall Architecture of dInfer
</p>

dInfer supports multiple dLLM variants, including LLaDA and LLaDA-MoE. It introduces multiple algorithms in each of
the components to improve the decoding quality and inference speed. This includes a soft diffusion iteration algorithm
for smoother denoising, hierarchical and credit decoding for enhanced parallel decoding, and a vicinity refresh strategy
for KV-cache management to mitigate cache staleness.
Beyond algorithmic improvements, it integrates several system-level optimizations. It supports both tensor parallelism
(TP) and expert parallelism (EP) to maximize GPU utilization even at batch size 1. It leverages PyTorch compilation and
NVIDIA CUDA Graphs for efficient kernel execution, and introduces a loop unrolling mechanism to eliminate CUDA stream
bubbles across diffusion iterations.

## Benchmark results

<p align="center">
  <img src="https://raw.githubusercontent.com/inclusionAI/dInfer/refs/heads/add_readme/assets/dinfer_tps.png" alt="dInfer v0.1 speedup" width="600">
  <br>
  <b>Figure</b>: Benchmark results
</p>

On HumanEval, dInfer achieves over 1,100 TPS at batch size 1, and averages more than 800 TPS across six benchmarks on
a single node with $8\times$ H800 GPUs. Compared to Fast-dLLM, dInfer delivers more than a $10\times$ speedup while
maintaining accuracy; on LLaDA-MoE it provides a $2-3\times$ speedup over QWen2.5-3B on vLLM with comparable quality.

## Get started

Please follow the instruction below to install dInfer.

```
git clone https://github.com/inclusionAI/dInfer.git
cd dInfer
pip install .
```

## Cite

```
@article{dinfer,
    title={dInfer: An Efficient Inference Framework for Diffusion Language Models},
    author={Yuxin Ma, Lun Du, Lanning Wei, Kun Chen, Qian Xu, Kangyu Wang, Guofeng Feng, Guoshan Lu, Lin Liu, Xiaojing Qi, Xinyuan Zhang, Zhen Tao, Haibo Feng, Ziyun Jiang, Ying Xu, Zenan Huang, Yihong Zhuang, Haokai Xu, Jiaqi Hu, Zhenzhong Lan, Junbo Zhao, Jianguo Li, Da Zheng},
    year={2025},
    journal={arXiv preprint arXiv:2510.08666}
}
```
