#!/usr/bin/env python3
"""Rescore saved generations on CPU, preserving the original evaluation."""
import argparse
import hashlib
import json
from pathlib import Path

import evaluate_lulu as evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--benchmarks', default='olympiadbench')
    args = parser.parse_args()
    source, output = Path(args.source).resolve(), Path(args.output_dir).resolve()
    if not (source/'summary.json').is_file():
        raise ValueError('Source must be a completed evaluation')
    if output.exists():
        raise FileExistsError('Rescoring needs a fresh output directory')
    plan = json.loads((source/'eval_plan.json').read_text())
    chosen = set(args.benchmarks.split(','))
    if not chosen <= {b['name'] for b in plan['benchmarks']}:
        raise ValueError('Unknown benchmark to rescore')
    scorer_path = evaluation.benchmark_parser_path(None)
    scorer = evaluation.load_parser(scorer_path)
    # Validate complete coverage before writing the derived evaluation.
    for model in plan['models']:
        for benchmark in plan['benchmarks']:
            evaluation.load_complete_rows(source/model['name']/benchmark['name'],
                len(plan['devices']), benchmark['expected_examples'])
    plan.update(output_dir=str(output), backend=plan.get('backend', 'hf'), parser_path=str(scorer_path))
    output.mkdir()
    audit = {'source': str(source), 'generation_reused': True, 'gpu_jobs_launched': False,
        'rescored_benchmarks': sorted(chosen), 'changes': [],
        'parser_sha256': hashlib.sha256(scorer_path.read_bytes()).hexdigest(),
        'math_parser_sha256': hashlib.sha256((scorer_path.parent/'math_parser.py').read_bytes()).hexdigest(),
        'source_summary_sha256': hashlib.sha256((source/'summary.json').read_bytes()).hexdigest(),
        'rules': {name: (
            'Accept standalone choice labels only, including one-letter LaTeX wrappers; '
            'do not interpret the initial letter of words as a choice.'
            if name in {'mmlu_pro', 'gpqa_diamond'} else
            'Use the bundled math parser; OlympiadBench uses the final boxed answer.'
        ) for name in sorted(chosen)}}
    for model in plan['models']:
        for benchmark in plan['benchmarks']:
            directory = output/model['name']/benchmark['name']
            directory.mkdir(parents=True)
            for path in sorted((source/model['name']/benchmark['name']).glob('shard-*.jsonl')):
                with (directory/path.name).open('w') as target:
                    for line in path.read_text().split('\n'):
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        if benchmark['name'] in chosen:
                            prediction = scorer.extract_answer(row['response'], row['data_source'])
                            reward = float(scorer.math_equal(prediction, row['ground_truth']))
                            if (prediction, reward) != (row['prediction'], row['reward']):
                                audit['changes'].append({'model': model['name'], 'benchmark': benchmark['name'],
                                    'prompt_index': row['prompt_index'], 'old_prediction': row['prediction'],
                                    'new_prediction': prediction, 'old_reward': row['reward'], 'new_reward': reward})
                            row.update(prediction=prediction, reward=reward)
                        target.write(json.dumps(row, ensure_ascii=False)+'\n')
    (output/'eval_plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    (output/'rescore_audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False)+'\n')
    evaluation.finish_suite(plan)


if __name__ == '__main__':
    main()
