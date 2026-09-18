"""CPU-only diagnosis of predeclared full-response/final-only score differences."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lulu import benchmark_parser as bp
from lulu import math_parser as mp

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);a=p.parse_args();out=Path(a.output_dir)
    diffs=[]
    for file in sorted((out/'continuations').glob('task_*.jsonl')):
        for line in file.read_text().splitlines():
            r=json.loads(line);v=r['verification']
            if v['legacy_success']==v['strict_final_success']:continue
            thought,final=r['response_text'].rsplit('</think>',1)
            old_box='boxed' in thought and 'boxed' not in final
            thought_prediction=bp.extract_answer(thought,'math500')
            conversions=[];original=mp.convert_word_number
            def capture(value):
                result=original(value);conversions.append({'input_length':len(value),'changed':result!=value,'output':result});return result
            mp.convert_word_number=capture
            try:replayed=bp.extract_answer(r['response_text'],'math500')
            finally:mp.convert_word_number=original
            assert replayed==v['legacy_prediction']
            phrase_only='he answer is' in thought and 'he answer is' not in final and 'boxed' not in r['response_text']
            diffs.append(dict(job_id=r['job_id'],uid=r['uid'],state_id=r['state_id'],arm=r['arm'],group=r['group'],
                gold=r['gold_answer'],verification=v,boxed_only_in_reasoning=old_box,
                legacy_equals_reasoning_prediction=str(thought_prediction)==v['legacy_prediction'],
                answer_phrase_only_in_reasoning=phrase_only,word_number_conversion=conversions,
                final_tail=final[-500:]))
    result={'disagreements':len(diffs),'legacy_wrong_final_correct':sum(not r['verification']['legacy_success'] and r['verification']['strict_final_success'] for r in diffs),
        'boxed_only_in_reasoning':sum(r['boxed_only_in_reasoning'] for r in diffs),
        'legacy_equals_reasoning_prediction':sum(r['legacy_equals_reasoning_prediction'] for r in diffs),
        'answer_phrase_only_in_reasoning':sum(r['answer_phrase_only_in_reasoning'] for r in diffs),
        'word_number_conversion_changes_to_legacy_prediction':sum(any(c['changed'] and c['output']==r['verification']['legacy_prediction'] for c in r['word_number_conversion']) for r in diffs),
        'note':'Frozen primary scores preserved; final-only metric was declared before generation. This count is specific to conditional continuation samples and is not a benchmark accuracy correction.',
        'examples':diffs}
    (out/'parser_disagreement_audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False));print({k:v for k,v in result.items() if k!='examples'})
if __name__=='__main__':main()
