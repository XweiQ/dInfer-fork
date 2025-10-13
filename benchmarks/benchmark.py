import json
import torch
import numpy as np
import torch.nn.functional as F
import os
from transformers import AutoTokenizer, AutoModel

import torch.distributed as dist

import time
import tqdm

from dinfer.model import LLaDAModelLM
from dinfer import BlockIteratorFactory, KVCacheFactory
from dinfer import ThresholdParallelDecoder, BlockWiseDiffusionLLM, SlidingWindowDiffusionLLM

def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12346'
    print(f'rank={rank}, world size={world_size}')
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def benchmark_gen(rank, model, tokenizer, prompt, gen_len, block_len, threshold, cache,
                  num_test_iter=1, have_warmup=True, sliding=False, prefix_look=0, after_look=0,
                  warmup_steps=1):
    device = model.device
    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    print('prompt len:', input_ids.shape[1], ', total len:', input_ids.shape[1] + gen_len)

    decoder = ThresholdParallelDecoder(0, threshold=threshold)
    if sliding:
        cf_type = cache if cache in ('prefix', 'dual') else 'dual'
        dllm = SlidingWindowDiffusionLLM(
            model, decoder, BlockIteratorFactory(),
            KVCacheFactory(cf_type),
            prefix_look=prefix_look, after_look=after_look, warmup_steps=warmup_steps
        )
    else:
        if cache == 'prefix':
            dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(True), cache_factory=KVCacheFactory('prefix'), early_stop=True)
        elif cache == 'dual':
            dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(True), cache_factory=KVCacheFactory('dual'), early_stop=True)
        else:
            dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(True), early_stop=True)

    if have_warmup:
        for _ in range(2):
            dllm.generate(input_ids, gen_length=gen_len, block_length=block_len)

    dist.barrier()
    prev_forwards = dllm.num_forwards
    prev_cache_updates = dllm.cache_updates
    start = time.time()
    num_tokens = 0
    for i in tqdm.trange(num_test_iter):
        out = dllm.generate(input_ids, gen_length=gen_len, block_length=block_len)
        num_tokens += out.shape[1] - input_ids.shape[1]
    stop = time.time()
    dist.barrier()
    total_forward = dllm.num_forwards - prev_forwards
    total_cache_updates = dllm.cache_updates - prev_cache_updates
    tps = num_tokens/(stop-start)
    if rank==0:
        print(f'Iter: {i}, Forward: {total_forward}, cache updates: {total_cache_updates}, Time: {stop-start}, FPS: {total_forward/(stop-start)}, TPS: {num_tokens/(stop-start)}')
        print(tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=False))
    return tps

@ torch.no_grad()
def main(world_size, rank, gpu_id, args):
    print('started', world_size, rank, gpu_id, args)
    torch.cuda.set_device(gpu_id)
    device = torch.device(gpu_id)
    setup_distributed(rank, world_size)

    model = LLaDAModelLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, init_device='cuda:'+str(gpu_id)).eval()
    if world_size>1:
        model.tensor_parallel(rank, world_size)
    model = model.to(torch.bfloat16)
    model = model.to(device)
    if not args.sliding:
        model = torch.compile(model, mode='reduce-overhead', fullgraph=True)



    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = model.to(device)

    if args.input_data is None:
        prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she can run 6 kilometers per hour. How many kilometers can she run in 8 hours? "
        m = [{"role": "user", "content": prompt}, ]
        prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        benchmark_gen(rank, model, tokenizer, prompt, args.gen_len, args.block_length, args.threshold, args.cache,
                num_test_iter=args.num_test_iter, have_warmup=True, sliding=args.sliding,
                prefix_look=args.prefix_look, after_look=args.after_look,
                warmup_steps=args.warmup_steps)
    else:
        with open(args.input_data, 'r') as f:
            data = json.load(f)
        res_list = []
        for prompt in data:
            tps = benchmark_gen(rank, model, tokenizer, prompt, args.gen_len, args.block_length, args.threshold, args.cache,
                    num_test_iter=args.num_test_iter, have_warmup=False, sliding=args.sliding,
                    prefix_look=args.prefix_look, after_look=args.after_look,
                    warmup_steps=args.warmup_steps)
            res_list.append(tps)
        import statistics
        print(statistics.mean(res_list))

    dist.destroy_process_group()
    
from multiprocessing import Process
import argparse

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='/mnt/dllm/model_hub/LLaDA-1.5/')
    parser.add_argument('--input_data', type=str, default=None)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_test_iter', type=int, default=2)
    parser.add_argument('--gen_len', type=int, default=1024)
    parser.add_argument('--block_length', type=int, default=64)
    parser.add_argument('--threshold', type=float, default=0.9)
    parser.add_argument('--exp_name', type=str, default='exp')
    parser.add_argument('--cache', type=str, default='')
    parser.add_argument('--tp', action='store_true')
    parser.add_argument('--sliding', action='store_true')
    parser.add_argument('--prefix_look', type=int, default=0)
    parser.add_argument('--after_look', type=int, default=0)
    parser.add_argument('--warmup_steps', type=int, default=1)
    args = parser.parse_args()
    procs = []
    print(args)

    gpus = [int(gpu) for gpu in args.gpu.split(',')]
    
    if len(gpus) == 1:
        main(1, 0, gpus[0], args)
    else:
        for i, gpu in enumerate(gpus):
            p = Process(target=main, args=(len(gpus), i, gpu, args))
            p.daemon = True
            procs.append(p)
            p.start()
        for p in procs:
            p.join()
