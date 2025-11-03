import torch
import numpy as np
from torch._C import dtype
import torch.nn.functional as F
import os
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM
import torch.distributed as dist
import time
import tqdm
from vllm.config import CompilationConfig, ParallelConfig
from vllm.config import VllmConfig, set_current_vllm_config, get_current_vllm_config
from vllm.forward_context import set_forward_context
import json
from multiprocessing import Process
from pathlib import Path

from dinfer.model import BailingMoeV2ForCausalLM, LLaDA2MoeModelLM
from dinfer import BlockIteratorFactory, KVCacheFactory
from dinfer import ThresholdParallelDecoder, BlockDiffusionLLMAttnmask, BlockDiffusionLLM
import difflib

#model_path = '/mnt/dllm/luxiaocheng/moe-mini-v2-e256-1009-fp8-ml4-grouprouter-20T-mdmcpt-block-diffusion-bl32-4k-noshift-100B'
#model_path = '/data/dulun/models/dllm-mini-final/dllm-mini/block-diffusion-sft-2k-v2-full-bd/lladamoe-mini-sft-ep5-final-251017'
model_path = '/mnt/infra/dulun.dl/models/dllm-mini/block-diffusion-sft-2k-v2-full-bd/LLaDA2-mini-preview-ep4-v0'
dataset_path = '/ossfs/workspace/dumped_prompts'
dataset='openai_humaneval'

FILE_PATH = Path(__file__).resolve()
sample_path = FILE_PATH.with_name(f"{FILE_PATH.stem}_sample.json")

def run_bd(use_kvcache, threshold=0.9, gpu=0):
  with open(sample_path, "r") as f:
    samples = json.load(f)
  torch.cuda.set_device(gpu)
  device = torch.device(gpu)
  # setup EP
  parallel_config = ParallelConfig(enable_expert_parallel = True)
  with set_current_vllm_config(VllmConfig(parallel_config = parallel_config)):
    vllm_config = get_current_vllm_config()

    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = LLaDA2MoeModelLM(config=model_config).eval()

    model.load_weights(model_path, torch_dtype=torch.float32)
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    decoder = ThresholdParallelDecoder(temperature=0, threshold=threshold, mask_id=156895, eos_id=156892) 
    
    print('Test BD model without kvcache')
    ans = []
    for sample in samples:
      prompt = [sample['question']]
      print("prompt: ")
      print(prompt)
      prompt[0] = '<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>'+prompt[0]+'<|role_end|><role>ASSISTANT</role>' 
      
      input_ids = tokenizer(prompt)['input_ids']
      input_ids = torch.tensor(input_ids).to(device)
      print(input_ids.shape)

      if not use_kvcache:
        dllm = BlockDiffusionLLMAttnmask(model, decoder, BlockIteratorFactory(use_block_diffusion=True), early_stop=True)
      else:
        dllm = BlockDiffusionLLM(model, decoder, BlockIteratorFactory(use_block_diffusion=True),  cache_factory=KVCacheFactory('prefix',is_bd_model=True), early_stop=True)
      out = dllm.generate(input_ids, gen_length=256, block_length=32)
      new_ans = tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
      
      assert(new_ans == sample['answer'])
      ans.append(new_ans)
    
    return ans

def test_bd():
  from vllm import distributed
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12388'
  distributed.init_distributed_environment(1, 0, 'env://', 0, 'nccl')
  distributed.initialize_model_parallel(1, backend='nccl')

  run_bd(use_kvcache=False, gpu=0)
  run_bd(use_kvcache=True, gpu=1)

  


if __name__ == '__main__':
    
  from vllm import distributed
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12388'
  distributed.init_distributed_environment(1, 0, 'env://', 0, 'nccl')
  distributed.initialize_model_parallel(1, backend='nccl')

  print("Testing Block Diffusion Model without kvcache...")
  ans = run_bd(use_kvcache=False, gpu=0)
  print(ans)

  print("Testing Block Diffusion Model with kvcache...")
  ans = run_bd(use_kvcache=True, gpu=2)
  print(ans)

