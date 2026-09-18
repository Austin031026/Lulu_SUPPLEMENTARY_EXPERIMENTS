"""Prefix records must remain compatible with the common evaluation summary."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace


def test_prefix_report_keeps_summary_fields_and_original_rows(monkeypatch):
    scripts=Path(__file__).resolve().parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location('recipe_report',scripts/'summarize_balanced_recipe.py')
    report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)
    monkeypatch.setattr(report,'TOKENIZER',SimpleNamespace(decode=lambda ids,**kw: '</think>\nAnswer: 42'),raising=False)
    monkeypatch.setattr(report,'parser',SimpleNamespace(extract_answer=lambda text,ds:'42',math_equal=lambda pred,gold:pred==gold))
    row=dict(prompt_index=0,data_source='math500',prompt_sha256='hash',generation_seed=42,ground_truth='42',
             prompt_tokens=117,response_token_ids=[1]*9000,hit_cap=False,finish_reason='stop')
    model,benchmark,rows=report.prefix_rows(('base','math500',[row]))
    value=report.summarize_rows(rows)
    assert value['mean_prompt_tokens']==117 and value['mean_response_tokens']==8192
    assert value['hit_cap_fraction']==1 and value['accuracy']==1
    assert len(row['response_token_ids'])==9000 and row['finish_reason']=='stop'
