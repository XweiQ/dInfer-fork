import torch
import numpy as np
import logging

from .utils import TokenArray, DistAlignedTokenArray, gather_sequence_block
from .utils import calculate_op_num

logger = logging.getLogger(__name__)

class DiffusionLLM:
    """ Diffusion LLM inference

    Parameters
    ----------
    model : Torch.Module
        The LLM model
    decoder : ParallelDecoder
        The decoder that decodes the tokens from the logits computed by the Transformer model
    iterator_facotry : IteratorFactory
        The factory class that generates the iterator on the input token array.
    cache_factory : KVCacheFactory (optional)
        The KV-cache factory that generates a kv-cache for LLM.
    """

    @ torch.no_grad()
    def generate(self, prompt, gen_length=128, block_length=128):
        ''' Generate tokens with diffusion iterations.

        Parameters:
        ----------
        prompt: Torch.Tensor
            A tensor of shape (1, L) that contains the input prompt.
        gen_length: int
            Generated answer length.
        block_length: int
            Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.

        Returns
        -------
        Torch.Tensor: A tensor of shape (1, L') that contains the prompt tokens and the generated tokens.
            EOS and any tokens after EOS have been removed.
        '''

class BlockWiseDiffusionLLM:
    """ This diffusion LLM inference generates tokens block by block.

    The decoding algorithm break the generation sequence into blocks.
    It runs diffusion iterations on the first block and decodes all tokens
    in the block before moving to the next block.
    This is a classifical dLLM decoding algorithm.
    """
    def __init__(self, model, decoder, iterator_factory, early_stop=True, cache_factory=None, maximum_unroll=4, expected_tpf=8):
        self.model = model
        self.cache_factory = cache_factory
        self.decoder = decoder
        self.iterator_factory = iterator_factory
        self.num_forwards = 0
        self.cache_updates = 0
        self.early_stop = early_stop
        self.maximum_unroll = maximum_unroll
        self.expected_tpf = expected_tpf

    @ torch.no_grad()
    def generate(self, prompt, gen_length=128, block_length=128):
        ''' Generate tokens with diffusion iterations block by block.
        '''
        x = TokenArray(prompt, gen_length, self.decoder.mask_id, self.decoder.eos_id, self.model.device)
        it = self.iterator_factory.create(x, block_length)

        iter_no = 0
        kv_cache = self.cache_factory.create() if self.cache_factory is not None else None
        for block_id, (block_loc, block) in enumerate(it):
            self.decoder.block_init(block, block_id)

            while (block == self.decoder.mask_id).sum() > 0:
                unroll_k = max(min((block == self.decoder.mask_id).sum()//self.expected_tpf, self.maximum_unroll), 1)
                for unroll_i in range(unroll_k):
                    # Update KV-cache
                    if kv_cache is not None and kv_cache.require_update(iter_no, block_loc.start, block_loc.end):
                        output = self.model(x.data, use_cache=True)
                        self.num_forwards += 1
                        # use the generated output to decode.
                        self.decoder.decode(output.logits[:, block_loc.start:block_loc.end], block_loc.start, block_loc.end, x)
                        # update KV-cache
                        kv_cache.update(output.past_key_values)
                        past_key_values, replace_position = kv_cache.get_key_values(block_loc.start, block_loc.end)
                        self.cache_updates += 1

                    if kv_cache is None:
                        logits = self.model(x.data).logits[:, block_loc.start:block_loc.end]
                    elif kv_cache.cache_type == 'prefix':
                        logits = self.model(x[block_loc.start:], past_key_values=past_key_values, use_cache=True,
                                            replace_position=replace_position).logits
                        block_length = block_loc.end - block_loc.start
                        logits = logits[:, :block_length]
                    else:
                        # cache position is the position between current_block_start and current_block_end
                        logits = self.model(block, past_key_values=past_key_values, use_cache=True,
                                            replace_position=replace_position).logits
                    self.num_forwards += 1
                    self.decoder.decode(logits, block_loc.start, block_loc.end, x)
                    iter_no += 1

            if self.early_stop and torch.any(x[block_loc.start:block_loc.end] == self.decoder.eos_id):
                # Find the first location of EOS and set all tokens after the location to EOS.
                # Here we assume that don't perform remasking.
                # TODO(zhengda) here we assume the batch size is 1.
                x[block_loc.end:] = self.decoder.eos_id
                break

        logger.info(f'The number of diffusion iterations: {self.num_forwards}')
        return x.get_generated_tokens()


class BlockWiseDiffusionLLMCont(BlockWiseDiffusionLLM):
    """ This diffusion LLM inference generates tokens block by block.

    The decoding algorithm break the generation sequence into blocks.
    It runs diffusion iterations on the first block and decodes all tokens
    in the block before moving to the next block.
    This is a classifical dLLM decoding algorithm.
    """
    def __init__(self, model, decoder, iterator_factory, early_stop=True, cache_factory=None, maximum_unroll=4, expected_tpf=8,
                cont_weight=0.3, cont_weight_init=0.15, cont_weight_growth=0.02, threshold_decay=0.02):
        self.model = model
        self.cache_factory = cache_factory
        self.decoder = decoder
        self.iterator_factory = iterator_factory
        self.num_forwards = 0
        self.cache_updates = 0
        self.early_stop = early_stop
        self.cont_weight = cont_weight
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            self.h2e = self.model.module.h2e
        else:
            self.h2e = self.model.h2e
        self.cont_weight_init = cont_weight_init
        self.cont_weight_growth = cont_weight_growth
        self.threshold_decay = threshold_decay
        self.maximum_unroll = maximum_unroll
        self.expected_tpf = expected_tpf

    @ torch.no_grad()
    def generate(self, prompt, gen_length=128, block_length=128):
        ''' Generate tokens with diffusion iterations block by block.
        '''
        x = TokenArray(prompt, gen_length, self.decoder.mask_id, self.decoder.eos_id, self.model.device)
        it = self.iterator_factory.create(x, block_length)
        inputs_embeds = self.h2e(x.data)
        iter_no = 0
        kv_cache = self.cache_factory.create() if self.cache_factory is not None else None
        for block_id, (block_loc, block) in enumerate(it):
            self.decoder.block_init(block, block_id)

            while (block == self.decoder.mask_id).sum() > 0:
                unroll_k = max(min((block == self.decoder.mask_id).sum()//self.expected_tpf, self.maximum_unroll), 1)
                for unroll_i in range(unroll_k):

                    iter_cont_weight = min(self.cont_weight_init+self.cont_weight_growth*iter_no, self.cont_weight)
                    iter_threshold = max(1-iter_no*self.threshold_decay, self.decoder.threshold)
                    # Update KV-cache
                    if kv_cache is not None and kv_cache.require_update(iter_no, block_loc.start, block_loc.end):
                        output = self.model(inputs_embeds=inputs_embeds, use_cache=True)
                        self.num_forwards += 1
                        # use the generated output to decode.
                        self.decoder.decode(output.logits[:, block_loc.start:block_loc.end], block_loc.start, block_loc.end, x, iter_threshold)
                        # update KV-cache
                        mask_index = (x.data == self.decoder.mask_id)
                        inputs_embeds = self.h2e(x.data, mask_index, output.logits, iter_cont_weight)
                        kv_cache.update(output.past_key_values)
                        past_key_values, replace_position = kv_cache.get_key_values(block_loc.start, block_loc.end)
                        self.cache_updates += 1
                        iter_no += 1

                    iter_cont_weight = min(self.cont_weight_init+self.cont_weight_growth*iter_no, self.cont_weight)
                    iter_threshold = max(1-iter_no*self.threshold_decay, self.decoder.threshold)
                    if kv_cache is None:
                        logits = self.model(inputs_embeds=inputs_embeds).logits
                        self.decoder.decode(logits[:, block_loc.start:block_loc.end], block_loc.start, block_loc.end, x, iter_threshold)
                        mask_index = (x.data == self.decoder.mask_id)
                        inputs_embeds = self.h2e(x.data, mask_index, logits, iter_cont_weight)
                    elif kv_cache.cache_type == 'prefix':
                        logits = self.model(inputs_embeds=inputs_embeds[:, block_loc.start:], past_key_values=past_key_values, use_cache=True,
                                            replace_position=replace_position).logits
                        block_length = block_loc.end - block_loc.start
                        self.decoder.decode(logits[:, :block_length], block_loc.start, block_loc.end, x, iter_threshold)
                        mask_index = (x.data[:, block_loc.start:] == self.decoder.mask_id)
                        inputs_embeds[:, block_loc.start:] = self.h2e(x.data[:, block_loc.start:], mask_index, logits, iter_cont_weight)
                    else:
                        # cache position is the position between current_block_start and current_block_end
                        logits = self.model(inputs_embeds=inputs_embeds[:, block_loc.start:block_loc.end], past_key_values=past_key_values, use_cache=True,
                                            replace_position=replace_position).logits
                        self.decoder.decode(logits, block_loc.start, block_loc.end, x, iter_threshold)
                        mask_index = (x.data[:, block_loc.start:block_loc.end] == self.decoder.mask_id)
                        inputs_embeds[:, block_loc.start:block_loc.end] = self.h2e(x.data[:, block_loc.start:block_loc.end], mask_index, logits, iter_cont_weight)
                    self.num_forwards += 1

                    iter_no += 1

            if self.early_stop and torch.any(x[block_loc.start:block_loc.end] == self.decoder.eos_id):
                # Find the first location of EOS and set all tokens after the location to EOS.
                # Here we assume that don't perform remasking.
                # TODO(zhengda) here we assume the batch size is 1.
                x[block_loc.end:] = self.decoder.eos_id
                break

        logger.info(f'The number of diffusion iterations: {self.num_forwards}')
        return x.get_generated_tokens()

class SlidingWindowDiffusionLLM(DiffusionLLM):
    """ This diffusion LLM inference generates tokens in a sliding window manner.

    The decoding algorithm defines a window to decode tokens in each diffusion iteration.
    After each iteration, the decoding window may slide forward to cover more masked tokens.
    """
    def __init__(self, model, decoder, iterator_factory, cache_factory, maximum_unroll=4, expected_tpf=8,
                 prefix_look=0, after_look=0, warmup_steps=0, early_stop=True):
        self.model = model
        self.cache_factory = cache_factory
        self.decoder = decoder
        self.iterator_factory = iterator_factory
        self.num_forwards = 0
        self.cache_updates = 0
        self.prefix_look = int(prefix_look)
        self.after_look = int(after_look)
        self.warmup_steps = int(warmup_steps)
        self.early_stop = early_stop
        self.maximum_unroll = maximum_unroll
        self.expected_tpf = expected_tpf
        assert cache_factory is not None, "This class requires a KV-cache."

    @ torch.no_grad()
    def generate(self, prompt, gen_length=128, block_length=128):
        ''' Generate tokens with diffusion iterations block by block.
        '''
        x = TokenArray(prompt, gen_length, self.decoder.mask_id, self.decoder.eos_id, self.model.device)
        it = self.iterator_factory.create(x, block_length)

        kv_cache = self.cache_factory.create()
        prompt_len = x.prompt.shape[1]
        total_len = x.total_length

        for block_idx, (block_loc, block) in enumerate(it):
            block_start, block_end = block_loc.start, block_loc.end
            left_start = max(0, block_start - self.prefix_look)
            right_end = min(total_len, block_end + self.after_look)

            iter_in_block = 0
            while (x[block_start:block_end] == self.decoder.mask_id).sum() > 0:
                unroll_k = max(min((block == self.decoder.mask_id).sum()//self.expected_tpf, self.maximum_unroll), 1)
                for unroll_i in range(unroll_k):
                    if block_idx == 0 and iter_in_block < self.warmup_steps:
                        out_full = self.model(x.data)
                        self.num_forwards += 1
                        self.decoder.decode(out_full.logits[:, block_start:block_end], block_start, block_end, x)
                        iter_in_block += 1
                        continue

                    if kv_cache.past_key_values is None or (kv_cache.require_update(iter_in_block, block_start, block_end) and block_idx > 0):
                        out_full = self.model(x.data, use_cache=True)
                        self.num_forwards += 1
                        self.decoder.decode(out_full.logits[:, block_start:block_end], block_start, block_end, x)
                        kv_cache.update(out_full.past_key_values)
                        self.cache_updates += 1
                        iter_in_block += 1

                    window_input = x.data[:, left_start:right_end]
                    past_key_values, replace_position = kv_cache.get_key_values(left_start, right_end)
                    out_step = self.model(
                        window_input,
                        past_key_values=past_key_values,
                        use_cache=True,
                        replace_position=replace_position
                    )
                    self.num_forwards += 1
                    offset = block_start - left_start
                    logits_block = out_step.logits[:, offset:offset + (block_end - block_start)]
                    self.decoder.decode(logits_block, block_start, block_end, x)

                    iter_in_block += 1

            if self.early_stop and torch.any(x[block_start:block_end] == self.decoder.eos_id):
                x[block_end:] = self.decoder.eos_id
                break

        logger.info(f'The number of diffusion iterations with kv-cache: {self.num_forwards}')
        return x.get_generated_tokens()


class SlidingWindowDiffusionLLMCont(DiffusionLLM):
    """ This diffusion LLM inference generates tokens in a sliding window manner.

    The decoding algorithm defines a window to decode tokens in each diffusion iteration.
    After each iteration, the decoding window may slide forward to cover more masked tokens.
    """
    def __init__(self, model, decoder, iterator_factory, cache_factory, maximum_unroll=4, expected_tpf=8,
                 prefix_look=0, after_look=0, warmup_steps=0, early_stop=True, cont_weight=0.3, 
                 cont_weight_init=0.15, cont_weight_growth=0.02, threshold_decay=0.02):
        self.model = model
        self.cache_factory = cache_factory
        self.decoder = decoder
        self.iterator_factory = iterator_factory
        self.num_forwards = 0
        self.cache_updates = 0
        self.prefix_look = int(prefix_look)
        self.after_look = int(after_look)
        self.warmup_steps = int(warmup_steps)
        self.early_stop = early_stop
        self.cont_weight = cont_weight
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            self.h2e = self.model.module.h2e
        else:
            self.h2e = self.model.h2e
        self.cont_weight_init = cont_weight_init
        self.cont_weight_growth = cont_weight_growth
        self.threshold_decay = threshold_decay
        self.maximum_unroll = maximum_unroll
        self.expected_tpf = expected_tpf
        assert cache_factory is not None, "This class requires a KV-cache."

    @ torch.no_grad()
    def generate(self, prompt, gen_length=128, block_length=128):
        ''' Generate tokens with diffusion iterations block by block.
        '''
        x = TokenArray(prompt, gen_length, self.decoder.mask_id, self.decoder.eos_id, self.model.device)
        it = self.iterator_factory.create(x, block_length)

        kv_cache = self.cache_factory.create()
        prompt_len = x.prompt.shape[1]
        total_len = x.total_length
        inputs_embeds = self.h2e(x.data)
        iter_no = 0

        for block_idx, (block_loc, block) in enumerate(it):
            block_start, block_end = block_loc.start, block_loc.end
            left_start = max(0, block_start - self.prefix_look)
            right_end = min(total_len, block_end + self.after_look)

            iter_in_block = 0
            while (x[block_start:block_end] == self.decoder.mask_id).sum() > 0:
                unroll_k = max(min((block == self.decoder.mask_id).sum()//self.expected_tpf, self.maximum_unroll), 1)
                for unroll_i in range(unroll_k):
                    iter_cont_weight = min(self.cont_weight_init+self.cont_weight_growth*iter_no, self.cont_weight)
                    iter_threshold = max(1-iter_no*self.threshold_decay, self.decoder.threshold)
                    if block_idx == 0 and iter_in_block < self.warmup_steps:
                        out_full = self.model(inputs_embeds=inputs_embeds)
                        self.num_forwards += 1
                        self.decoder.decode(out_full.logits[:, block_start:block_end], block_start, block_end, x, iter_threshold)
                        mask_index = (x.data == self.decoder.mask_id)
                        inputs_embeds = self.h2e(x.data, mask_index, out_full.logits, iter_cont_weight)
                        iter_in_block += 1
                        iter_no += 1
                        continue

                    if kv_cache.past_key_values is None or (kv_cache.require_update(iter_no, block_start, block_end) and block_idx > 0):
                        out_full = self.model(inputs_embeds=inputs_embeds, use_cache=True)
                        self.num_forwards += 1
                        self.decoder.decode(out_full.logits[:, block_start:block_end], block_start, block_end, x, iter_threshold)
                        mask_index = (x.data == self.decoder.mask_id)
                        inputs_embeds = self.h2e(x.data, mask_index, out_full.logits, iter_cont_weight)
                        kv_cache.update(out_full.past_key_values)
                        self.cache_updates += 1
                        iter_in_block+=1
                        iter_no += 1
                        continue

                    iter_cont_weight = min(self.cont_weight_init+self.cont_weight_growth*iter_no, self.cont_weight)
                    iter_threshold = max(1-iter_no*self.threshold_decay, self.decoder.threshold)
                    past_key_values, replace_position = kv_cache.get_key_values(left_start, right_end)
                    out_step = self.model(
                        inputs_embeds=inputs_embeds[:, left_start:right_end],
                        past_key_values=past_key_values,
                        use_cache=True,
                        replace_position=replace_position
                    )

                    self.num_forwards += 1
                    iter_no += 1
                    offset = block_start - left_start
                    logits_block = out_step.logits[:, offset:offset + (block_end - block_start)]
                    self.decoder.decode(logits_block, block_start, block_end, x, iter_threshold)
                    mask_index = (x.data[:, left_start:right_end] == self.decoder.mask_id)
                    inputs_embeds[:, left_start:right_end] = self.h2e(x.data[:, left_start:right_end], mask_index, out_step.logits, iter_cont_weight)

                    iter_in_block += 1

            if self.early_stop and torch.any(x[block_start:block_end] == self.decoder.eos_id):
                x[block_end:] = self.decoder.eos_id
                break

        logger.info(f'The number of diffusion iterations with kv-cache: {self.num_forwards}')
        return x.get_generated_tokens()

class BlockWiseDiffusionLLMWithSP(DiffusionLLM):
    """ Diffusion LLM inference with sequence parallel.

    This class performs diffusion LLM inference with sequence parallel.

    Parameters
    ----------
    rank : int
        The rank of the process
    world_size : int
        The number of processes to perform diffusion LLM inference with sequence parallel.
    model : Torch.Module
        The diffusion LLM model
    decoder : ParallelDecoder
        The decoder that decodes the tokens from the logits computed by the Transformer model
    iterator_facotry : IteratorFactory
        The factory class that generates the iterator on the input token array.
    """
    def __init__(self, rank, world_size, model, decoder, iterator_factory):
        self.model = model
        self.decoder = decoder
        self.iterator_factory = iterator_factory
        self.rank = rank
        self.world_size = world_size
        self.num_forwards = 0

    @ torch.no_grad()
    def generate(self, prompt, gen_length=128, block_length=128):
        '''
        Args:
            prompt: A tensor of shape (1, L).
            gen_length: Generated answer length.
            block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        '''
        op_num = 0
        x = DistAlignedTokenArray(prompt, gen_length, self.decoder.mask_id, self.decoder.eos_id, self.model.device, self.rank, self.world_size)
        it = self.iterator_factory.create(x, block_length)

        for block_id, (block_loc, block) in enumerate(it):
            self.decoder.block_init(block, block_id)
            while (block == self.decoder.mask_id).sum()>0:
                part = x.total_length // self.world_size
                # TODO(zhengda) How does the model collect KV from other processes.
                partial_logits = self.model(x[(self.rank * part):((self.rank + 1) * part)].clone()).logits
                op_num += calculate_op_num(x[self.rank*part:(self.rank+1)*part])

                logits = gather_sequence_block(partial_logits, self.rank * part, (self.rank + 1) * part, block_loc.start, block_loc.end,
                        self.rank, self.world_size)
                self.decoder.decode(logits, block_loc.start, block_loc.end, x)
                self.num_forwards += 1
        return x.get_generated_tokens()
