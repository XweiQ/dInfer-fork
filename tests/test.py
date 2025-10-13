import os
import logging
from multiprocessing import Process

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModel, AutoConfig
from vllm.config import CompilationConfig, ParallelConfig
from vllm.config import VllmConfig, set_current_vllm_config, get_current_vllm_config

from dinfer.model import FusedOlmoeForCausalLM, LLaDAModelLM
from dinfer import BlockWiseDiffusionLLM, SlidingWindowDiffusionLLM, BlockWiseDiffusionLLMWithSP
from dinfer import ThresholdParallelDecoder, HierarchyDecoder

from dinfer.model.modeling_llada_fastdllm import LLaDAModelLM as LLaDAModelLM_fastdllm
from dinfer.decoding.generate_fastdllm import generate, generate_with_prefix_cache, generate_with_dual_cache
from dinfer.decoding.generate_dist import generate as generate_sp
from dinfer.decoding.generate_hierarchy import generate_hierarchy
from dinfer.decoding.utils import TokenArray, DistAlignedTokenArray, BlockIterator, BlockIteratorFactory, KVCacheFactory, gather_sequence_block, BlockLoc
from dinfer.decoding.generate_merge import generate_merge
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

#model_path = "/mnt/dllm/model_hub/LLaDA-1.5/"
model_path = "/data/myx/llm/vllm/model/LLaDA-1_5/"
#moe_model_path = '/mnt/dllm/fengling/moe/workdir/7bA1b_anneal_15t_0827_500B_further_8k_enneal_train_4k_ep3_v7_1e-5/step45567_converted_hf_fusemoe'
moe_model_path = '/data/dulun/models/llada-moe-sft/llada-moe-sft-model/7bA1b_anneal_19t_500B_further_8k_anneal_train_4k_ep3_v8p5/step45567_converted_hf_fusemoe/'

def test_block_iterator():
    prompt = torch.tensor([1, 2, 3, 4, 5, 6, 7]).view(1, 7)
    x = TokenArray(prompt, gen_length=10, mask_id=17, eos_id=18, device='cpu')
    it = BlockIterator(x, block_length=5)
    num_iters = 0
    for block_id, (block_loc, block) in enumerate(it):
        num_iters += 1
        assert block_loc.start == block_id * 5 + prompt.shape[1]
        assert block_loc.end == min((block_id + 1) * 5 + prompt.shape[1], prompt.shape[1] + 10)
    assert num_iters == 2

def test_token_array():
    prompt = torch.tensor([1, 2, 3, 4, 5, 6, 7]).view(1, 7)
    arr = TokenArray(prompt, gen_length=20, mask_id=17, eos_id=18, device='cpu')
    assert arr.total_length == prompt.shape[1] + 20
    assert torch.all(arr[0:5] == prompt[:, 0:5])
    arr[8:10] = torch.tensor([9, 10]).view(1, 2)

    arr = DistAlignedTokenArray(prompt, gen_length=20, mask_id=17, eos_id=18, device='cpu', rank=0, world_size=4)
    assert arr.total_length == prompt.shape[1] + 20 + 1
    assert torch.all(arr[0:5] == prompt[:, 0:5])
    arr[8:10] = torch.tensor([9, 10]).view(1, 2)

class SimulateBlockIterator:
    """ This class simulates the block iterator in SlidingWindowDiffusionLLM.
    """
    def __init__(self, x, block_length, mask_id):
        self.x = x
        self.iter = 0
        self.block_length = block_length
        self.mask_id = mask_id

    def __iter__(self):
        self.iter = 0
        return self

    def move_next(self):
        current_block_start = self.x.prompt.shape[1] + self.iter * self.block_length
        current_block_end = current_block_start + self.block_length
        current_block_end = min(current_block_end, self.x.total_length)
        # If all tokens have been decoded, move to the next block.
        if torch.all(self.x[current_block_start:current_block_end] != self.mask_id):
            self.iter += 1

    def __next__(self):
        self.move_next()
        current_block_start = self.x.prompt.shape[1] + self.iter * self.block_length
        if current_block_start >= self.x.total_length:
            raise StopIteration
        current_block_end = current_block_start + self.block_length
        current_block_end = min(current_block_end, self.x.total_length)
        return BlockLoc(current_block_start, current_block_end), self.x[current_block_start:current_block_end]

class SimulateBlockIteratorFactory:
    def create(self, x, block_length):
        return SimulateBlockIterator(x, block_length, 126336)


def test_moe_diffusion():
    torch.cuda.set_device(0)
    device = torch.device(0)

    decoder = ThresholdParallelDecoder(0, threshold=0.9, mask_id=156895, eos_id=156892, use_float64=True)
    h_decoder = HierarchyDecoder(0, threshold=0.9, mask_id=156895, eos_id=156892, low_threshold=0.4)
    tokenizer = AutoTokenizer.from_pretrained(moe_model_path, trust_remote_code=True)
    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she can run 6 kilometers per hour. How many kilometers can she run in 8 hours? "
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(prompt)['input_ids']
    batch_size = 1
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0).repeat(batch_size, 1)

    from vllm import distributed
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12346'
    distributed.init_distributed_environment(1, 0, 'env://', 0, 'nccl')
    distributed.initialize_model_parallel(1, backend='nccl')
    print("[Loading model]")
    # setup EP
    parallel_config = ParallelConfig(enable_expert_parallel = True)
    with set_current_vllm_config(VllmConfig(parallel_config = parallel_config)):
        model_config = AutoConfig.from_pretrained(moe_model_path, trust_remote_code=True)
        model = FusedOlmoeForCausalLM(config=model_config).eval()
        model.load_weights(moe_model_path, torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(moe_model_path, trust_remote_code=True)
        model = model.to(device)

        # Test generation without cache.
        print('Test block-wise diffusion MOE-LLM without KV-cache')
        dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(), early_stop=True)
        res = dllm.generate(input_ids, gen_length=128, block_length=32)
        res1, nfe = generate(model, input_ids, gen_length=128, block_length=32, threshold=0.9, mask_id=156895, eos_id=156892)
        res2, nfe = generate_merge(model, input_ids, None, gen_length=128, block_length=32, threshold=0.9, mask_id=156895, eos_id=156892, parallel_decoding='threshold', early_stop=False,)
        res1 = res1[res1 != 156892]
        res2 = res2[res2 != 156892]
        assert res.shape[1] == len(res1)
        assert res.shape[1] == len(res2)
        assert torch.all(res == res1)
        assert torch.all(res == res2)

        # Test generation with dual cache
        print('Test block-wise diffusion MOE-LLM with dual KV-cache')
        dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(), early_stop=True, cache_factory=KVCacheFactory('dual'))
        res = dllm.generate(input_ids, gen_length=256, block_length=32)
        res1, nfe = generate_with_dual_cache(model, input_ids, gen_length=256, block_length=32, threshold=0.9, mask_id=156895, eos_id=156892)
        res1 = res1[res1 != 156892]
        assert res.shape[1] == len(res1)
        assert torch.all(res == res1)

        # Test generation without cache.
        print('Test block-wise hierarchical diffusion MOE-LLM without KV-cache')
        dllm = BlockWiseDiffusionLLM(model, h_decoder, BlockIteratorFactory(), early_stop=True)
        res = dllm.generate(input_ids, gen_length=128, block_length=32)
        res1, nfe = generate_hierarchy(model, input_ids, gen_length=128, block_length=32, threshold=0.9, mask_id=156895, eos_id=156892,decoding='hierarchy_fast_v2',
                                        low_threshold=0.4, remask_threshold=0.4)
        res1 = res1[res1 != 156892]
        assert res.shape[1] == len(res1)
        assert torch.all(res == res1)

def test_diffusion():
    torch.cuda.set_device(0)
    device = torch.device(0)
    config = AutoConfig.from_pretrained(model_path)
    config.flash_attention = True
    model = LLaDAModelLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, config=config).eval()
    model = model.to(device)
    fastdllm_model = LLaDAModelLM_fastdllm.from_pretrained(model_path, torch_dtype=torch.bfloat16, config=config).eval()
    fastdllm_model = fastdllm_model.to(device)
    decoder = ThresholdParallelDecoder(0, threshold=0.9, use_float64=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she can run 6 kilometers per hour. How many kilometers can she run in 8 hours? "
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(prompt)['input_ids']
    batch_size = 1
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0).repeat(batch_size, 1)

    print('Test sliding-window diffusion LLM with dual KV-cache')
    dllm = SlidingWindowDiffusionLLM(model, decoder, SimulateBlockIteratorFactory(), KVCacheFactory('dual'))
    res = dllm.generate(input_ids, gen_length=128, block_length=32)
    res1, nfe = generate_with_dual_cache(fastdllm_model, input_ids, gen_length=128, block_length=32, threshold=0.9)
    res1 = res1[res1 != 126081]
    assert res.shape[1] == len(res1)
    assert torch.all(res == res1)

    # Test generation without cache.
    print('Test block-wise diffusion LLM without KV-cache')
    dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(), early_stop=True)
    res = dllm.generate(input_ids, gen_length=128, block_length=32)
    res1, nfe = generate(fastdllm_model, input_ids, gen_length=128, block_length=32, threshold=0.9)
    res1 = res1[res1 != 126081]
    assert res.shape[1] == len(res1)
    assert torch.all(res == res1)

    # Test generation with prefix cache
    print('Test block-wise diffusion LLM with prefix KV-cache')
    dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(), early_stop=True, cache_factory=KVCacheFactory('prefix'))
    res = dllm.generate(input_ids, gen_length=128, block_length=32)
    res1, nfe = generate_with_prefix_cache(fastdllm_model, input_ids, gen_length=128, block_length=32, threshold=0.9)
    res1 = res1[res1 != 126081]
    assert res.shape[1] == len(res1)
    assert torch.all(res == res1)

    # Test generation with dual cache
    print('Test block-wise diffusion LLM with dual KV-cache')
    dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(), cache_factory=KVCacheFactory('dual'), early_stop=True)
    res = dllm.generate(input_ids, gen_length=128, block_length=32)
    res1, nfe = generate_with_dual_cache(fastdllm_model, input_ids, gen_length=128, block_length=32, threshold=0.9)
    res1 = res1[res1 != 126081]
    assert res.shape[1] == len(res1)
    assert torch.all(res == res1)

def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12347'
    print(f'rank={rank}, world size={world_size}')
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def test_worker(rank, world_size, gpu):
    setup_distributed(rank, world_size)
    torch.cuda.set_device(gpu)
    device = torch.device(gpu)
    full_data = torch.arange(100).view(4, 25).to(device)

    # Partition size is smaller than block size.
    block_size = 6
    part_size = 4
    first_part_start = 1
    last_part_end = first_part_start + part_size * world_size
    assert last_part_end <= full_data.shape[1]
    partial_start = first_part_start + part_size * rank
    partial_end = partial_start + part_size
    part_data = full_data[:, partial_start:partial_end]
    # The accessed block must be covered by all parts.
    for block_start in range(first_part_start, last_part_end - block_size):
        block_end = block_start + block_size
        block_data = gather_sequence_block(part_data, partial_start, partial_end, block_start, block_end, rank, world_size)
        assert torch.all(block_data == full_data[:, block_start:block_end])

    # Partition size is larger than block size.
    block_size = 4
    part_size = 6
    first_part_start = 1
    last_part_end = first_part_start + part_size * world_size
    assert last_part_end <= full_data.shape[1]
    partial_start = first_part_start + part_size * rank
    partial_end = partial_start + part_size
    part_data = full_data[:, partial_start:partial_end]
    # The accessed block must be covered by all parts.
    for block_start in range(first_part_start, last_part_end - block_size):
        block_end = block_start + block_size
        block_data = gather_sequence_block(part_data, partial_start, partial_end, block_start, block_end, rank, world_size)
        assert torch.all(block_data == full_data[:, block_start:block_end])

    # Partition size is equal to block size.
    block_size = 4
    part_size = 4
    first_part_start = 1
    last_part_end = first_part_start + part_size * world_size
    assert last_part_end <= full_data.shape[1]
    partial_start = first_part_start + part_size * rank
    partial_end = partial_start + part_size
    part_data = full_data[:, partial_start:partial_end]
    # The accessed block must be covered by all parts.
    for block_start in range(first_part_start, last_part_end - block_size):
        block_end = block_start + block_size
        block_data = gather_sequence_block(part_data, partial_start, partial_end, block_start, block_end, rank, world_size)
        assert torch.all(block_data == full_data[:, block_start:block_end])

    dist.destroy_process_group()

def test_dist():
    num_gpus = 4
    procs = []
    for i, gpu in enumerate(range(num_gpus)):
        p = Process(target=test_worker, args=(i, num_gpus, i))
        procs.append(p)
        p.start()
    for p in procs:
        p.join()

def test_diffusion_worker(rank, world_size, gpu):
    setup_distributed(rank, world_size)
    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    config = AutoConfig.from_pretrained(model_path)
    config.flash_attention = True
    model = LLaDAModelLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, config=config).eval()
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she can run 6 kilometers per hour. How many kilometers can she run in 8 hours? "
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(prompt)['input_ids']
    batch_size = 1
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0).repeat(batch_size, 1)

    # Test generation without cache.
    print('Test diffusion LLM without KV-cache')
    decoder = ThresholdParallelDecoder(0, threshold=0.9, use_float64=True)
    dllm = BlockWiseDiffusionLLMWithSP(rank, world_size, model, decoder, BlockIteratorFactory())
    res = dllm.generate(input_ids, gen_length=128, block_length=32)
    res1, nfe = generate_sp(model, input_ids, rank=rank, world_size=world_size, gen_length=128, block_length=32, threshold=0.9)
    res1 = res1[res1 != 126081]
    assert res.shape[1] == len(res1)
    assert torch.all(res == res1)

    dist.destroy_process_group()

def test_diffusion_sp():
    num_gpus = 4
    procs = []
    for i, gpu in enumerate(range(num_gpus)):
        p = Process(target=test_diffusion_worker, args=(i, num_gpus, i))
        procs.append(p)
        p.start()
    for p in procs:
        p.join()

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    logging.basicConfig(level=logging.INFO)
    test_moe_diffusion()
    test_diffusion()

    test_token_array()
    test_block_iterator()

    test_dist()
    test_diffusion_sp()
