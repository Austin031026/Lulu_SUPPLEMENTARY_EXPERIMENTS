"""Explicit small GPU test: resident full Student plus vLLM checkpoint refresh."""
from pathlib import Path
import argparse
import copy
import json
import torch
from lulu import training as tr
from lulu.checkpoints import CheckpointManager
from lulu.vllm_rollout import RolloutClient


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);p.add_argument('--gpu',default='0')
    x=p.parse_args();root=Path(x.output_dir).resolve()
    a=tr.parser().parse_args(['--train-data','unused','--output-dir',str(root),'--lora-rank','0',
        '--master-weights-fp32','--method','ren_resolved','--rollout-backend','vllm',
        '--rollout-vllm-memory','.22','--max-new-tokens','8','--temperature','.6','--top-p','.95','--rollout-top-k','20'])
    tr.validate_args(a)
    student=tr.load_student(a,trainable=True);tok=tr.load_tokenizer(a.model)
    assert all(p.requires_grad and p.dtype==torch.float32 for p in student.parameters())
    manager=CheckpointManager(root,index_unit='rounds')
    if manager.latest_path() is None:
        manager.save(student,tok,None,0,{'completed_rounds':0,'completed_updates':0})
    elif manager.latest_metadata()['completed_rounds'] != 0:
        raise ValueError('Smoke directory already progressed; choose a new output')
    records=[{'index':0,'causal_prompt_ids':tok.apply_chat_template([{'role':'user','content':'What is 2+3?'}],
        tokenize=True,add_generation_prompt=True,enable_thinking=True)}]
    client=RolloutClient(a,x.gpu,0)
    try:
        results=[]
        for version in (0,1):
            if version:
                with torch.no_grad():student.model.norm.weight.add_(.125)
                manager.save(student,tok,None,1,{'completed_rounds':1,'completed_updates':1},final=True)
            response,finish=client.generate(copy.deepcopy(records),version)
            expected=student.model.norm.weight[:8].to(torch.bfloat16).float().tolist()
            actual=client.last_result['weight_sync'][0]
            assert actual['norm_probe']==expected,(actual,expected)
            assert len(response)==1 and 0<len(response[0])<=8
            results.append({'round':version,'weight_sync':actual,'tokens':response[0],'finish':finish})
        tr.atomic_json(root/'smoke_result.json',{'status':'passed','all_parameters_trainable':True,
            'parameter_count':sum(p.numel() for p in student.parameters()),'results':results})
    finally:client.close()


if __name__=='__main__':main()
