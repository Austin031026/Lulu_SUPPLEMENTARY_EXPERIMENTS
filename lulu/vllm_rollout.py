"""One resident vLLM engine per Student GPU, isolated from Student DDP."""
from __future__ import annotations
import contextlib
import ctypes
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import time
import traceback


def snapshot_path(output_dir, round_index):
    path = (Path(output_dir)/'checkpoints/latest').resolve(strict=True)
    state = json.loads((path/'lulu_state.json').read_text())
    if state['completed_rounds'] != round_index:
        raise RuntimeError('vLLM rollout checkpoint differs from the frozen Student round')
    return str(path)


def isolate_vllm_allocator_environment():
    """Keep Student allocator tuning out of the spawned vLLM service."""
    return os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)


def worker(settings, gpu, rank, conn):
    engine = None
    # Die with the Student parent even if it is killed during a failed DDP round.
    parent = os.getppid()
    ctypes.CDLL(None).prctl(1, signal.SIGTERM)
    if os.getppid() != parent:
        return
    def stop(signum, frame):
        raise SystemExit(f'Rollout service received signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    try:
        # The rollout engine uses vLLM's CuMem-backed sleep pool.  PyTorch
        # allocator tuning inherited from the Student process can either be
        # incompatible with that pool (``expandable_segments``) or distort
        # vLLM's startup memory profile enough to leave no KV-cache blocks
        # (for example ``max_split_size_mb``).  The rollout service is a
        # spawned process, so removing the setting here leaves the Student
        # allocator unchanged while giving vLLM its required default allocator.
        isolate_vllm_allocator_environment()
        for key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT'):
            os.environ.pop(key, None)
        os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), VLLM_WORKER_MULTIPROC_METHOD='spawn',
                          TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1')
        cache = Path(settings['output_dir'])/'.engine_cache'/f'rollout-{rank}'
        for key, name in [('VLLM_CACHE_ROOT','vllm'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('TRITON_CACHE_DIR','triton')]:
            os.environ[key] = str(cache/name)
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        engine = LLM(model=settings['model'], dtype=settings['dtype'], tensor_parallel_size=1,
            max_model_len=settings['max_sequence_tokens'],
            worker_extension_cls='lulu.weight_sync.FullWeightWorkerExtension',
            enable_sleep_mode=settings.get('rollout_vllm_sleep', False),
            compilation_config={'cudagraph_capture_sizes':[1,2,4,8,16,32]},
            gpu_memory_utilization=settings['rollout_vllm_memory'],
            max_num_seqs=settings['rollout_vllm_max_seqs'], max_num_batched_tokens=4096,
            enable_chunked_prefill=True, enable_lora=bool(settings['lora_rank']), max_loras=1, max_cpu_loras=1,
            max_lora_rank=max(16, settings['lora_rank']), swap_space=0,
            seed=settings['seed']+rank, generation_config='vllm')
        conn.send({'op':'ready'})
        loaded_round = None
        sleeping = False
        while True:
            message = conn.recv()
            if message['op'] == 'stop':
                break
            if message['op'] == 'sleep':
                if not settings.get('rollout_vllm_sleep', False):
                    raise ValueError('Sleep was not enabled for this rollout engine')
                if not sleeping:
                    engine.sleep(level=1)
                    sleeping = True
                conn.send({'op': 'sleeping', 'loaded_round': loaded_round})
                continue
            if message['op'] != 'generate':
                raise ValueError('Unknown rollout command')
            if sleeping:
                engine.wake_up()
                sleeping = False
            version = message['round']
            adapter = snapshot_path(settings['output_dir'], version)
            request = None
            sync_result = None
            if settings['lora_rank']:
                request = LoRARequest(f'round-{version}', version+1, adapter)
            elif loaded_round != version:
                sync_result = engine.collective_rpc('refresh_student_weights', args=(adapter,))
                engine.reset_prefix_cache()
                loaded_round = version
            greedy = message.get('greedy', False)
            params = [SamplingParams(temperature=0. if greedy else settings['temperature'],
                top_p=1. if greedy else settings['top_p'],
                top_k=-1 if greedy else (settings['rollout_top_k'] or -1),
                max_tokens=settings['max_new_tokens'],
                seed=settings['seed'] if greedy else settings['seed']+100003*version+int(index),
                repetition_penalty=1.0)
                for index in message['indices']]
            started = time.monotonic()
            generated = engine.generate([{'prompt_token_ids':ids} for ids in message['prompts']],
                                         params, lora_request=request, use_tqdm=False)
            conn.send({'op':'generated','round':version,'checkpoint':adapter,
                'responses':[list(result.outputs[0].token_ids) for result in generated],
                'finish_reasons':[result.outputs[0].finish_reason for result in generated], 'weight_sync':sync_result,
                'seconds':time.monotonic()-started})
    except BaseException as exc:
        with contextlib.suppress(OSError, EOFError):
            conn.send({'op':'error','error':str(exc),'traceback':traceback.format_exc()})
        raise
    finally:
        if engine is not None:
            core = getattr(engine.llm_engine, 'engine_core', None)
            if core is not None:
                core.shutdown()
        conn.close()


class RolloutClient:
    def __init__(self, args, gpu, rank):
        ctx = mp.get_context('spawn')
        self.connection, child = ctx.Pipe()
        self.timeout = args.worker_timeout
        self.process = ctx.Process(target=worker, args=(vars(args).copy(), gpu, rank, child))
        self.process.start()
        child.close()
        try:
            self.receive('ready')
        except BaseException:
            self.close()
            raise

    def receive(self, expected):
        if not self.connection.poll(self.timeout):
            raise TimeoutError('vLLM rollout service timed out')
        message = self.connection.recv()
        if message['op'] != expected:
            raise RuntimeError(f'vLLM rollout failed: {message}')
        return message

    def generate(self, records, round_index, *, greedy=False):
        self.connection.send({'op':'generate','round':round_index, 'greedy':greedy,
            'indices':[r['index'] for r in records],
            'prompts':[r['causal_prompt_ids'] for r in records]})
        result = self.receive('generated')
        if result['round'] != round_index or len(result['responses']) != len(records):
            raise RuntimeError('Mismatched vLLM rollout response')
        self.last_result = result
        for record in records:
            record['rollout_checkpoint'] = result['checkpoint']
        return result['responses'], result['finish_reasons']

    def sleep(self):
        self.connection.send({'op': 'sleep'})
        return self.receive('sleeping')

    def close(self):
        with contextlib.suppress(OSError, EOFError):
            self.connection.send({'op':'stop'})
        self.process.join(timeout=15)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=15)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=5)
        self.connection.close()
