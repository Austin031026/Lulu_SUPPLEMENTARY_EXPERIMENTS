import subprocess,sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
SCRIPTS=[
    'run_rq1_baselines.py','run_rq2_allocation.py','run_rq2_teacher_sweep.py','run_rq2_policy_drift.py','run_rq3_scaling.py',
    'analyze_rq2_geometry.py','analyze_inference_budget.py','summarize_rq1_baselines.py','summarize_rq2_allocation.py',
    'summarize_rq2_teacher_sweep.py','summarize_rq3_scaling.py',
]

@pytest.mark.parametrize('script',SCRIPTS)
def test_rq_script_help_is_cpu_safe(script):
    proc=subprocess.run([sys.executable,str(ROOT/'scripts'/script),'--help'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    assert proc.returncode==0,proc.stderr
