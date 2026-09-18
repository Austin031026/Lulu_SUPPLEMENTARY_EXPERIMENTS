"""CPU-only finalization after every frozen continuation job completes."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path

def write(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);a=p.parse_args();out=Path(a.output_dir).resolve()
    state=out/'analysis_progress.json';start=time.time();write(state,{'status':'waiting_for_continuations'})
    while True:
        progress=json.loads((out/'live_progress.json').read_text())
        if progress['status']=='complete':break
        if progress['status']=='failed' and not progress['running']:
            write(state,{'status':'blocked_by_worker_failure','failures':progress['failures']});return
        time.sleep(15)
    write(state,{'status':'analyzing','total_jobs':progress['total_jobs']})
    try:
        for script in ['analyze_semantic_utility.py','audit_utility_parser.py','report_semantic_utility.py']:
            subprocess.run([sys.executable,str(Path(__file__).with_name(script)),'--output-dir',str(out)],check=True)
        write(state,{'status':'complete','report':str(out/'REPORT.md'),'finished_utc':time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),'elapsed_seconds_including_wait':time.time()-start})
    except Exception as exc:
        write(state,{'status':'error','error':str(exc)});raise
if __name__=='__main__':main()
