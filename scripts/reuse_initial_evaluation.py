#!/usr/bin/env python3
"""Reuse frozen Base answers only when dev selected identical initial weights."""
from __future__ import annotations
import contextlib,hashlib,json,shutil
from pathlib import Path


def verify_identical_weights(initial, base):
    import torch
    from safetensors import safe_open
    def files(path):
        index=path/'model.safetensors.index.json'
        if index.exists():return {key:path/name for key,name in json.loads(index.read_text())['weight_map'].items()}
        single=path/'model.safetensors'
        with safe_open(single,framework='pt',device='cpu') as handle:return {key:single for key in handle.keys()}
    initial,base=Path(initial),Path(base)
    left,right=files(initial),files(base)
    def config(path):return json.loads((path/'config.json').read_text()) if (path/'config.json').exists() else {}
    ca,cb=config(initial),config(base)
    for key in ('model_type','architectures','hidden_act','rope_theta','rope_scaling','rms_norm_eps','num_attention_heads','num_key_value_heads','tie_word_embeddings','bos_token_id','eos_token_id','pad_token_id'):
        if ca.get(key)!=cb.get(key):raise ValueError(f'Initial/Base model config differs: {key}')
    alias_a='lm_head.weight' not in left and ca.get('tie_word_embeddings',False)
    alias_b='lm_head.weight' not in right and cb.get('tie_word_embeddings',False)
    if alias_a:left['lm_head.weight']=left['model.embed_tokens.weight']
    if alias_b:right['lm_head.weight']=right['model.embed_tokens.weight']
    if left.keys()!=right.keys():raise ValueError('Initial/Base tensor names differ')
    count=0
    with contextlib.ExitStack() as stack:
        handles={p:stack.enter_context(safe_open(p,framework='pt',device='cpu')) for p in set(left.values())|set(right.values())}
        for key in sorted(left):
            a=handles[left[key]].get_tensor('model.embed_tokens.weight' if alias_a and key=='lm_head.weight' else key)
            b=handles[right[key]].get_tensor('model.embed_tokens.weight' if alias_b and key=='lm_head.weight' else key)
            if a.shape!=b.shape:raise ValueError(f'Initial/Base tensor shape differs: {key}')
            # The saved trainable FP32 copy must equal the original BF16 values,
            # even before the BF16 inference cast. Check in bounded chunks.
            a=a.reshape(-1);b=b.reshape(-1)
            for start in range(0,a.numel(),1_000_000):
                if not torch.equal(a[start:start+1_000_000].float(),b[start:start+1_000_000].float()):
                    raise ValueError(f'Initial/Base tensor values differ: {key}')
            count+=a.numel()
    return {'identical':True,'tensors':len(left),'tensor_elements_checked_including_tied_aliases':count,'scope':'Exact tensor values in FP32, no tolerance; before inference dtype cast'}


def reuse(root, plan, effective):
    import evaluate_lulu as evaluation
    from huggingface_hub import snapshot_download
    root=Path(root);selected=json.loads((root/'train/validation/selected.json').read_text())
    if selected['round']!=0:return False
    source=Path(plan['reference_evaluation']['path']);previous=json.loads((source/'eval_plan.json').read_text())
    command=effective['eval_command'];current=evaluation.build_plan(evaluation.argument_parser().parse_args(command[next(i for i,x in enumerate(command) if Path(x).name=='evaluate_lulu.py')+1:]))
    for key in ('sampling','thinking','max_response_tokens','max_prompt_tokens','max_examples','dtype','decoding','split','backend','vllm','benchmarks','devices'):
        if current[key]!=previous[key]:raise ValueError(f'Cached Base protocol mismatch: {key}')
    if current['models']!=[{'name':'selected','model':selected['checkpoint']}]:raise ValueError('Unexpected selected model')
    for name,digest in plan['reference_evaluation']['shard_sha256'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen Base shard changed')
    base=Path(snapshot_download(plan['student'],cache_dir=plan['hf_cache'],local_files_only=True))
    identity=verify_identical_weights(Path(selected['checkpoint']),base)
    from transformers import AutoTokenizer
    a=AutoTokenizer.from_pretrained(selected['checkpoint'],local_files_only=True)
    b=AutoTokenizer.from_pretrained(base,local_files_only=True)
    if (json.loads(a.backend_tokenizer.to_str())!=json.loads(b.backend_tokenizer.to_str())
            or a.chat_template!=b.chat_template or a.all_special_ids!=b.all_special_ids):
        raise ValueError('Initial/Base tokenizer or chat template differs')
    identity['tokenizer_and_template_identical']=True
    provenance={'kind':'reuse_identical_initial_model','source':str(source),'selected_round':0,'generated_new_responses':0,'identity':identity,'reason':'Dev did not select a trained checkpoint; avoid regenerating the identical Base model.'}
    destination=root/'evaluation'
    if destination.exists():raise ValueError('Refuse to replace existing evaluation')
    stage=root/'.initial_evaluation_cache'
    if stage.exists():raise ValueError('Unexpected cache staging directory')
    stage.mkdir()
    try:
        for benchmark in current['benchmarks']:
            name=benchmark['name'];folder=stage/'selected'/name;folder.mkdir(parents=True)
            for p in sorted((source/'base'/name).glob('shard-*.jsonl')):shutil.copy2(p,folder/p.name)
        current['cache_reuse']=provenance
        (stage/'eval_plan.json').write_text(json.dumps(current,indent=2)+'\n')
        (stage/'reuse_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
        # Publish only a complete scored cache, so the controller never sees a
        # partial evaluation. Preserve raw response files byte-for-byte.
        staged_plan=dict(current,output_dir=str(stage))
        result=evaluation.finish_suite(staged_plan)
        summary=json.loads((stage/'summary.json').read_text());summary['cache_reuse']=provenance
        (stage/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        stage.replace(destination)
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True)
        raise
    return True
