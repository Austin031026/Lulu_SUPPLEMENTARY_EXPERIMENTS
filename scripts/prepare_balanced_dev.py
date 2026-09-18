"""Freeze a disjoint DAPO dev set; CPU-only exact-overlap and tokenizer audit."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lulu.data import load_prepared_jsonl, build_prompt_views


def freeze_dev(source, pool, benchmarks, output, tokenizer, size=256, seed=42):
    import unicodedata
    import pyarrow.parquet as pq
    source,pool,benchmarks,output=map(Path,(source,pool,benchmarks,output))
    pool_meta=json.loads((pool/'manifest.json').read_text())
    rules=pool_meta['exclusion_audit']['audit']
    def canonical(messages):
        text='\n'.join(m['content'] for m in messages if m['role']=='user')
        for prefix in rules['removable_prefixes']:
            if text.startswith(prefix):text=text[len(prefix):];break
        if text.endswith(rules['removable_suffix']):text=text[:-len(rules['removable_suffix'])]
        return ' '.join(unicodedata.normalize('NFC',text).split())
    blocked={canonical(r['messages']) for r in load_prepared_jsonl(pool/'train.jsonl')}
    hashes={str(source.resolve()):hashlib.sha256(source.read_bytes()).hexdigest(),
            str((pool/'train.jsonl').resolve()):hashlib.sha256((pool/'train.jsonl').read_bytes()).hexdigest()}
    for entry in json.loads(benchmarks.read_text())['benchmarks'].values():
        p=Path(entry['full']);p=p if p.is_absolute() else benchmarks.parent/p
        blocked.update(canonical(r['prompt']) for r in pq.read_table(p).to_pylist())
        hashes[str(p.resolve())]=hashlib.sha256(p.read_bytes()).hexdigest()
    candidates=load_prepared_jsonl(source);eligible=[r for r in candidates if canonical(r['messages']) not in blocked]
    selected=sorted(eligible,key=lambda r:hashlib.sha256(f"lulu-dev:{seed}:{r['id']}".encode()).digest())[:size]
    if len(selected)!=size or len({canonical(r['messages']) for r in selected})!=size:
        raise ValueError('Not enough unique disjoint dev questions; do not silently change validation size')
    lengths=[]
    for r in selected:
        ids=build_prompt_views(tokenizer,r['messages'],r['gold_answer'],enable_thinking=True)['causal_prompt_ids']
        lengths.append(len(ids))
    if max(lengths)>4096:raise ValueError('Dev prompt exceeds configured budget')
    raw=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected).encode()
    meta={'size':size,'seed':seed,'source_questions':len(candidates),'excluded_exact_overlap':len(candidates)-len(eligible),
          'overlap_with_training_pool':0,'overlap_with_benchmarks':0,'overlap_scope':rules['normalizer_policy'],
          'source_hashes':hashes,'sha256':hashlib.sha256(raw).hexdigest(),'max_prompt_tokens':max(lengths),
          'selection':'seeded SHA256 order of existing held-out dev; no outcome-based selection',
          'question_ids':[r['id'] for r in selected]}
    if output.exists():
        if (output/'dev.jsonl').read_bytes()!=raw or json.loads((output/'manifest.json').read_text())!=meta:
            raise FileExistsError('Frozen validation directory has different content')
    else:
        output.mkdir(parents=True);(output/'dev.jsonl').write_bytes(raw);(output/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    return meta


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--pool',required=True);p.add_argument('--benchmarks',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--hf-cache',required=True);a=p.parse_args()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B',cache_dir=a.hf_cache,local_files_only=True)
    print(json.dumps(freeze_dev(a.source,a.pool,a.benchmarks,a.output_dir,tok),indent=2))
if __name__=='__main__':main()
