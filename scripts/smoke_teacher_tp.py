"""Two-GPU Teacher long-prefix smoke test; no generation or Student training."""
import argparse
from pathlib import Path
from lulu import training as tr
from lulu.persistent import Workers, _port
from lulu.teacher_service import teacher_worker


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);p.add_argument('--teacher-model',required=True)
    x=p.parse_args();a=tr.parser().parse_args(['--train-data','unused','--output-dir',x.output_dir,
        '--teacher-model',x.teacher_model,'--method','ren_resolved','--lora-rank','0',
        '--master-weights-fp32','--worker-timeout','240'])
    workers=Workers(240);port=_port()
    try:
        conn=workers.launch(teacher_worker,(a,0,2,['6','7'],'127.0.0.1',port))
        workers.launch(teacher_worker,(a,1,2,['6','7'],'127.0.0.1',port),connected=False)
        workers.expect(conn,'ready')
        results=[]
        for i,length in enumerate([32,8192]):
            conn.send({'op':'score','round':0,'request_id':i,'records':[
                {'causal_prompt_ids':[151644,872,198,17,10,18,30,151645,198,151644,77091,198],
                 'response_ids':[16]*length,'positions':[0,length//2,length-1]}]})
            result=workers.expect(conn,'scored')
            hidden=result['records'][0]['teacher_hidden']
            assert hidden.shape[0]==3 and hidden.isfinite().all()
            results.append({'length':length,'seconds':result['seconds'],'hidden_shape':list(hidden.shape)})
            print(results[-1],flush=True)
        tr.atomic_json(Path(x.output_dir)/'teacher_tp_result.json',{'status':'passed','results':results})
    finally:workers.close()


if __name__=='__main__':main()
