"""Resident greedy dev evaluation and dev-only checkpoint selection."""
from __future__ import annotations
import json,time
from pathlib import Path
from lulu.data import build_prompt_views,load_prepared_jsonl
from lulu import math_parser


def validate_student(student,tok,a,rank,world,rollout_client,round_index):
    from lulu.training import atomic_json
    rows=load_prepared_jsonl(a.validation_data)[:a.validation_max_examples]
    records=[]
    for index,row in enumerate(rows):
        if index%world!=rank:continue
        prompt=build_prompt_views(tok,row['messages'],row['gold_answer'],enable_thinking=True)['causal_prompt_ids']
        records.append({'index':index,'source_id':row['id'],'causal_prompt_ids':prompt,'gold_answer':row['gold_answer']})
    started=time.monotonic()
    student.eval()
    student.gradient_checkpointing_disable()
    responses,finish=rollout_client.generate(records,round_index,greedy=True)
    result=[]
    for record,ids,reason in zip(records,responses,finish):
        text=tok.decode(ids,skip_special_tokens=True)
        prediction=math_parser.extract_answer(text,'math_dapo')
        correct=bool(math_parser.math_equal(prediction,record['gold_answer']))
        result.append({'index':record['index'],'source_id':record['source_id'],'gold_answer':record['gold_answer'],
                       'prediction':prediction,'correct':correct,'response_tokens':len(ids),
                       'hit_cap':reason=='length','response':text,'checkpoint':record['rollout_checkpoint']})
    path=Path(a.output_dir)/'validation'/f'round_{round_index:06d}'
    path.mkdir(parents=True,exist_ok=True)
    temp=path/f'shard-{rank:03d}.jsonl.tmp'
    temp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result));temp.replace(path/f'shard-{rank:03d}.jsonl')
    return {'op':'validated','round':round_index,'rank':rank,'examples':len(result),
            'seconds':time.monotonic()-started}


def merge_validation(root,round_index,world,expected):
    from lulu.training import atomic_json
    root=Path(root);directory=root/'validation'/f'round_{round_index:06d}'
    rows=[json.loads(line) for rank in range(world) for line in (directory/f'shard-{rank:03d}.jsonl').read_text().splitlines() if line]
    if len(rows)!=expected or {r['index'] for r in rows}!=set(range(expected)) or len({r['source_id'] for r in rows})!=expected:
        raise RuntimeError('Dev coverage is incomplete or duplicated')
    checkpoint=root/'checkpoints'/f'round_{round_index:06d}'
    if any(Path(r['checkpoint']).resolve()!=checkpoint.resolve() for r in rows):
        raise RuntimeError('Dev evaluated stale model weights')
    metrics={'round':round_index,'checkpoint':str(checkpoint),'examples':len(rows),
             'correct':sum(r['correct'] for r in rows),'accuracy':sum(r['correct'] for r in rows)/len(rows),
             'hit_cap_fraction':sum(r['hit_cap'] for r in rows)/len(rows),
             'mean_response_tokens':sum(r['response_tokens'] for r in rows)/len(rows),
             'decoding':'greedy','temperature':0.,'selection_rule':'maximum dev accuracy; ties prefer earlier round'}
    atomic_json(directory/'summary.json',metrics)
    history_path=root/'validation/history.json'
    history=json.loads(history_path.read_text()) if history_path.exists() else []
    history=[x for x in history if x['round']!=round_index]+[metrics];history.sort(key=lambda x:x['round'])
    atomic_json(history_path,history)
    # Include round0: no benchmark evaluation can promote a trained checkpoint
    # when the held-out dev does not prefer it to the unchanged initial Student.
    best=max(history,key=lambda x:(x['accuracy'],-x['round']))
    atomic_json(root/'validation/selected.json',best)
    return metrics


def validation_round(a,workers,students,index):
    from lulu.training import atomic_json
    root=Path(a.output_dir);summary=root/'validation'/f'round_{index:06d}'/'summary.json'
    if summary.exists():return json.loads(summary.read_text())
    atomic_json(root/'phase_progress.json',{'round':index,'phase':'greedy_dev_validation','examples':a.validation_max_examples,'updated_unix':time.time()})
    for conn in students:conn.send({'op':'validate','round':index})
    for rank,conn in enumerate(students):
        response=workers.expect(conn,'validated')
        if response['round']!=index or response['rank']!=rank:raise RuntimeError('Invalid validation completion')
    return merge_validation(root,index,len(students),a.validation_max_examples)
