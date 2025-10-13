import torch
import numpy as np
import torch.nn.functional as F
import os
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch.distributed as dist
import time
import tqdm
from vllm.config import CompilationConfig, ParallelConfig
from vllm.config import VllmConfig, set_current_vllm_config, get_current_vllm_config
from vllm.forward_context import set_forward_context
import json

from dinfer.model import FusedOlmoeForCausalLM, LLaDAModelLM
from dinfer import BlockIteratorFactory, KVCacheFactory
from dinfer import ThresholdParallelDecoder,CreditThresholdParallelDecoder, HierarchyDecoder, BlockWiseDiffusionLLM, BlockWiseDiffusionLLMCont, SlidingWindowDiffusionLLM, SlidingWindowDiffusionLLMCont

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12345'
    print(f'rank={rank}, world size={world_size}')
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

bucket_size = 8
used_buckets = []

def get_bucket_length(length):
    #bucket_length = bucket_size*((length+bucket_size-1)//bucket_size)
    bucket_length = bucket_size*(length//bucket_size)
    if bucket_length not in used_buckets:
        used_buckets.append(bucket_length)
    return bucket_length

def load_inputs(dataset, tokenizer):
    with open(dataset, 'r') as f:
        data = json.load(f)
    prompts = []
    questions = []
    ids = []
    all_input_ids = []
    if "judge_details" in data.keys():
        details_data = data['judge_details']
    else:
        details_data = data['details']
    for id, judge_detail in enumerate(details_data):
        ids.append(id)
        prompt = judge_detail['prompt']
        prompts.append(prompt)
        questions.append(prompt)
        prompt = '<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>'+prompt+'<|role_end|><role>ASSISTANT</role>'   

        input_ids = tokenizer(prompt)['input_ids']
        input_ids = torch.tensor(input_ids).unsqueeze(0)
        all_input_ids.append(input_ids)
    return all_input_ids, prompts, questions, ids

def cal_bucket_len(args, all_input_ids):
    max_prompt_length = 0
    gen_len = args.gen_len
    padded_gen_lens = []

    for i in range(len(all_input_ids)):
        input_ids = all_input_ids[i]
        if input_ids.shape[1] > max_prompt_length:
            max_prompt_length = input_ids.shape[1]
        padded_length = get_bucket_length(input_ids.shape[1]+gen_len)
        padded_gen_lens.append(padded_length - input_ids.shape[1])
    return padded_gen_lens

def warmup_cudagraph(rank, device, dllm, args):
    if rank==0:
        print('warmup')
        print(used_buckets)
        iterator = tqdm.tqdm(used_buckets)
    else:
        iterator = used_buckets
    offset = 0
    for i in iterator:   
        offset = (offset + 1) % bucket_size
        input_ids = torch.randint(0, 140000, (1, i - args.gen_len+offset), dtype=torch.long, device=device)
        dllm.generate(input_ids, gen_length=args.gen_len-offset, block_length=args.block_length)

@ torch.no_grad()
def main(world_size, rank, gpu_id, args):
    print('started', world_size, rank, gpu_id, args)
    torch.cuda.set_device(gpu_id)
    device = torch.device(gpu_id)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    all_input_ids, prompts, questions, ids = load_inputs(args.dataset, tokenizer)
    padded_gen_lens = cal_bucket_len(args, all_input_ids)

    block_length=args.block_length
    dataset_name = args.dataset.split('/')[-1]
    os.makedirs(args.output_dir, exist_ok=True)

    from vllm import distributed
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(45601+args.port_offset)
    distributed.init_distributed_environment(world_size, rank, 'env://', rank, 'nccl')
    distributed.initialize_model_parallel(args.tp_size, backend='nccl')
    print("[Loading model]")
    # setup EP
    parallel_config = ParallelConfig(enable_expert_parallel = True)
    with set_current_vllm_config(VllmConfig(parallel_config = parallel_config)):
        vllm_config = get_current_vllm_config()
        print("EP Enabled:", vllm_config.parallel_config.enable_expert_parallel)

        model_config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
        model = FusedOlmoeForCausalLM(config=model_config).eval()
        model.load_weights(args.model_name, torch_dtype=torch.bfloat16)
        if args.tp_size>1 and args.use_tp:
            print('enabling tp')
            model.tensor_parallel(args.tp_size)
        model.forward = torch.compile(model.forward, mode='reduce-overhead', fullgraph=False, dynamic=True)
        model = model.to(device)

        if args.parallel_decoding == 'threshold':
            if args.use_credit:
                decoder = CreditThresholdParallelDecoder(temperature=0, threshold=args.threshold, mask_id=156895, eos_id=156892)
            else:
                decoder = ThresholdParallelDecoder(temperature=0, threshold=args.threshold, mask_id=156895, eos_id=156892)

        else:
            decoder = HierarchyDecoder(temperature=0, threshold=args.threshold, low_threshold=args.low_threshold, mask_id=156895, eos_id=156892)
        use_sw = args.prefix_look > 0 or args.after_look > 0 or args.warmup_times > 0
        if args.cache == 'prefix' or args.cache == 'dual':
            cache_factory=KVCacheFactory(args.cache)
        else:
            cache_factory=None

        if args.cont_weight>0:
            if use_sw:
                dllm = SlidingWindowDiffusionLLMCont(model, decoder, BlockIteratorFactory(start_block_align=True), cache_factory=cache_factory, early_stop=True, 
                    cont_weight=args.cont_weight, prefix_look=args.prefix_look, after_look=args.after_look, warmup_steps=args.warmup_times)
            else:
                dllm = BlockWiseDiffusionLLMCont(model, decoder, BlockIteratorFactory(start_block_align=True), cache_factory=cache_factory, early_stop=True, cont_weight=args.cont_weight)
        else:
            if use_sw:
                dllm = SlidingWindowDiffusionLLM(model, decoder, BlockIteratorFactory(start_block_align=True), cache_factory=cache_factory, early_stop=True, 
                    prefix_look=args.prefix_look, after_look=args.after_look, warmup_steps=args.warmup_times)
            else:
                dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(start_block_align=True), cache_factory=cache_factory, early_stop=True)
            
        warmup_cudagraph(rank, device, dllm, args)

        for wi in range(1):
            outputs = []
            total_forward = 0
            if rank==0:
                iterator = tqdm.trange(len(all_input_ids))
            else:
                iterator = range(len(all_input_ids))
            start = time.time()
            tpfs = []
            tpss = []
            fpss = []
            total_token = 0
            token_numbers = []
            for i in iterator:   
                input_ids = all_input_ids[i].to(device)
                padded_gen_len = padded_gen_lens[i]
                inner_start = time.time()
                prev_forwards = dllm.num_forwards
                out = dllm.generate(input_ids, gen_length=padded_gen_len, block_length=block_length)
                nfe = dllm.num_forwards - prev_forwards
                inner_stop = time.time()
                sample_time = inner_stop - inner_start
                outputs.append(out)
                total_forward += nfe
                token_number = out.shape[1] - input_ids.shape[1]
                token_numbers.append(token_number)
                tpf = token_number/nfe
                tps = token_number/sample_time
                fps = nfe/sample_time
                if rank == 0:
                    print(f'iter={i}, fps={fps}, nfe={nfe}')
                tpfs.append(tpf)
                tpss.append(tps)
                fpss.append(fps)
                total_token += token_number

            total_token = total_token

            stop = time.time()
        if rank==0:
            answers = []
            for i in tqdm.trange(len(outputs)):
                out = outputs[i]
                answer = (tokenizer.decode(out[0, all_input_ids[i].shape[1]:], skip_special_tokens=True))
                answers.append(answer)
            print(f'Forward: {total_forward}, Time: {stop-start}, FPS: {total_forward/(stop-start)}({np.mean(fpss)}), TPS: {total_token/(stop-start)}({np.mean(tpss)}), TPF: {total_token/total_forward}({np.mean(tpfs)})')
            filename = args.output_dir+'/'+'_'.join([str(item) for item in [args.exp_name, dataset_name, args.config, args.parallel_decoding, args.threshold, args.prefix_look]])+'.jsonl'
            with open (filename, 'w') as f:
                for i in range(len(answers)):
                    question = questions[i]
                    prompt = prompts[i]
                    answer = answers[i]
                    id = ids[i]
                    json.dump({'id':id, 'question':question, 'prompt':prompt, 'answer': answer, 'generated_length': token_numbers[i], 'tpf':tpfs[i], 'tps':tpss[i], 'fps':fpss[i], }, f, indent=4)
                    f.write('\n')
            with open('results.txt', 'a+') as f:
                print(args.exp_name, args.config, args.parallel_decoding, args.threshold, args.prefix_look, total_forward, stop-start, total_token / len(all_input_ids), total_forward/(stop-start), total_token/(stop-start), total_token/total_forward, sum(padded_gen_lens)/total_forward, np.mean(fpss), np.mean(tpss), np.mean(tpfs), args.dataset, file=f)

    
from multiprocessing import Process
import argparse

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--gpu', type=str, default='0,1,2,3')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gen_len', type=int, default=1024)
    parser.add_argument('--prefix_look', type=int, default=0)
    parser.add_argument('--after_look', type=int, default=0)
    parser.add_argument('--block_length', type=int, default=64)
    parser.add_argument('--threshold', type=float, default=0.9)
    parser.add_argument('--warmup_times', type=int, default=0)
    parser.add_argument('--low_threshold', type=float, default=0.3)
    parser.add_argument('--cont_weight', type=float, default=0)
    parser.add_argument('--parallel_decoding', type=str, default='hierarchy_faster')
    parser.add_argument('--use_credit', action='store_true')
    parser.add_argument('--exp_name', type=str, default='exp')
    parser.add_argument('--cache', type=str, default='')
    parser.add_argument('--use_tp', action='store_true')
    parser.add_argument('--output_dir', type=str, default='/ossfs/workspace/detailed_results_0917')
    parser.add_argument('--config', type=int, default=0)
    args = parser.parse_args()

    if args.config == 1:
        args.cache = ''
        args.parallel_decoding = 'threshold'
        args.prefix_look = 0
        args.after_look = 0
        args.threshold = 0.95
        args.warmup_times = 0
    elif args.config == 2:
        args.cache = 'dual'
        args.parallel_decoding = 'threshold'
        args.prefix_look = 0
        args.after_look = 0
        args.threshold = 0.95
        args.warmup_times = 0
    elif args.config == 3:
        args.cache = 'dual'
        args.parallel_decoding = 'threshold'
        args.prefix_look = 16
        args.after_look = 16
        args.threshold = 0.95
        args.warmup_times = 4
    elif args.config == 4:
        args.cache = ''
        args.parallel_decoding = 'threshold'
        args.prefix_look = 0
        args.after_look = 0
        args.threshold = 0.8
        args.warmup_times = 0
    elif args.config == 5:
        args.cache = ''
        args.parallel_decoding = 'hierarchy_faster'
        args.prefix_look = 0
        args.after_look = 0
        args.threshold = 0.8
        args.low_threshold = 0.5
        args.warmup_times = 0
    elif args.config == 6:
        args.cache = 'dual'
        args.parallel_decoding = 'hierarchy_faster'
        args.prefix_look = 16
        args.after_look = 16
        args.threshold = 0.8
        args.low_threshold = 0.5
        args.warmup_times = 4
    elif args.config == 9:
        args.cache = 'dual'
        args.parallel_decoding = 'threshold'
        args.prefix_look = 16
        args.after_look = 16
        args.threshold = 0.9
        args.low_threshold = 0.7
        args.warmup_times = 4

    elif args.config == 10:
        args.cache = 'dual'
        args.parallel_decoding = 'threshold'
        args.prefix_look = 16
        args.after_look = 16
        args.threshold = 0.85
        args.warmup_times = 4
    elif args.config == 11:
        args.cache = 'dual'
        args.parallel_decoding = 'threshold'
        args.prefix_look = 16
        args.after_look = 16
        args.threshold = 0.8
        args.low_threshold = 0.75
        args.warmup_times = 4

    elif args.config == 12:
        args.cache = 'dual'
        args.parallel_decoding = 'threshold'
        args.prefix_look = 16
        args.after_look = 16
        args.threshold = 0.85
        args.low_threshold = 0.5
        args.warmup_times = 4
        
    elif args.config == 13:
        args.cache = 'dual'
        args.parallel_decoding = 'threshold'
        args.prefix_look = 16
        args.after_look = 16
        args.threshold = 0.8
        args.warmup_times = 4

    elif args.config == 14:
        args.cache = 'dual'
        args.parallel_decoding = 'hierarchy_faster'
        args.prefix_look = 16
        args.after_look = 16
        args.threshold = 0.9
        args.low_threshold = 0.7
        args.warmup_times = 4

    elif args.config == 15:
        args.cache = 'dual'
        args.parallel_decoding = 'hierarchy_faster'
        args.prefix_look = 16
        args.after_look = 16
        args.threshold = 0.85
        args.low_threshold = 0.75
        args.warmup_times = 4
    procs = []
    print(args)

    gpus = [int(gpu) for gpu in args.gpu.split(',')]
    args.tp_size = len(gpus)
    args.use_tp = args.tp_size > 1
    args.port_offset = gpus[0]
    
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
