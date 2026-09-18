"""CPU tests for reusable Lulu evaluation, including a real saved PEFT student."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from lulu import benchmark_parser
from lulu import math_parser
from lulu.evaluation_runner import load_parser
from lulu.paths import DEFAULT_BENCHMARK_PARSER


def test_bundled_math_parser_is_available():
    assert callable(math_parser.extract_answer)
    assert callable(math_parser.math_equal)


def test_benchmark_parser_uses_bundled_math_parser():
    assert benchmark_parser._math_parser() is math_parser


def test_bundled_math_parser_scores_math_and_general_reasoning():
    import sympy
    assert math_parser.latex2sympy(r"\frac{1}{2}") == sympy.Rational(1, 2)
    for source in ("math500", "aime25", "olympiadbench"):
        answer = benchmark_parser.extract_answer(r"The result is \boxed{\frac{1}{2}}.", source)
        assert benchmark_parser.math_equal(answer, "0.5")
        assert not benchmark_parser.math_equal(answer, "3")
    answer = benchmark_parser.extract_answer("The answer is (C)", "mmlu_pro")
    assert benchmark_parser.math_equal(answer, "__CHOICE__C")
    assert not benchmark_parser.math_equal(answer, "__CHOICE__A")


def test_bundled_benchmark_parser_can_be_loaded_dynamically():
    parser = load_parser(DEFAULT_BENCHMARK_PARSER)
    assert callable(parser.extract_answer)
    assert callable(parser.math_equal)


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_lulu.py"
spec = importlib.util.spec_from_file_location("lulu_evaluation_tests", SCRIPT)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class EvaluationPlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "data.parquet").touch()
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps({"benchmarks": {
            name: {"full": "data.parquet", "probe": "data.parquet", "full_examples": 10,
                   "probe_examples": 4, "scorer": "choice" if name in ("mmlu_pro", "gpqa_diamond") else "math"}
            for name in evaluation.DEFAULT_BENCHMARKS
        }}))

    def args(self, *extra):
        return evaluation.argument_parser().parse_args([
            "--data-manifest", str(self.manifest), "--gpus", "2,5",
            "--output-dir", str(self.root / "out"), *extra])

    def test_default_suite_includes_general_reasoning_and_relative_paths(self):
        plan = evaluation.build_plan(self.args())
        self.assertEqual(plan["models"], [{"name": "base", "model": "Qwen/Qwen3-1.7B"}])
        self.assertEqual({b["name"] for b in plan["benchmarks"]}, set(evaluation.DEFAULT_BENCHMARKS))
        self.assertIn("mmlu_pro", [b["name"] for b in plan["benchmarks"]])
        self.assertIn("gpqa_diamond", [b["name"] for b in plan["benchmarks"]])
        self.assertTrue(all(b["path"] == str(self.root / "data.parquet") for b in plan["benchmarks"]))
        self.assertEqual(plan["backend"], "vllm")
        self.assertFalse((self.root / "out").exists())

    def test_thinking_defaults_to_recommended_sampling_with_explicit_greedy_compatibility(self):
        plan = evaluation.build_plan(self.args())
        self.assertEqual(plan["decoding"], "qwen-thinking")
        self.assertEqual(plan["sampling"], {"temperature": .6, "top_p": .95, "top_k": 20, "seed": 42})
        greedy = evaluation.build_plan(self.args("--decoding", "greedy"))
        self.assertEqual(greedy["sampling"]["temperature"], 0)
        plain = evaluation.build_plan(self.args("--no-thinking"))
        self.assertEqual(plain["decoding"], "greedy")

    def test_independent_vllm_engines_use_separate_compiler_caches(self):
        plan = evaluation.build_plan(self.args())
        first = evaluation.worker_environment(plan, 0)
        second = evaluation.worker_environment(plan, 1)
        self.assertEqual(first['CUDA_VISIBLE_DEVICES'], '2')
        self.assertEqual(second['CUDA_VISIBLE_DEVICES'], '5')
        for key in ('VLLM_CACHE_ROOT', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR'):
            self.assertNotEqual(first[key], second[key])
            self.assertTrue(Path(first[key]).is_dir())
            self.assertTrue(Path(second[key]).is_dir())

    def test_default_shared_framework_and_output_are_workspace_relative(self):
        plan = evaluation.build_plan(self.args())
        self.assertEqual(Path(plan["parser_path"]), SCRIPT.parents[1] / "lulu" / "benchmark_parser.py")
        self.assertEqual(evaluation.argument_parser().parse_args([]).output_dir,
                         str(evaluation.DEFAULT_OUTPUT_ROOT / "evaluation"))

    def test_local_parser_override_controls_plan(self):
        parser = self.root / "benchmark_parser.py"
        parser.write_text("# dry-run fixture\n")
        with patch.dict(os.environ, {"LULU_BENCHMARK_PARSER": str(parser)}):
            from_env = evaluation.build_plan(self.args())
            self.assertEqual(from_env["parser_path"], str(parser))
            explicit_parser = self.root / "explicit_parser.py"
            explicit_parser.write_text("# explicit dry-run fixture\n")
            explicit = evaluation.build_plan(self.args("--parser-path", str(explicit_parser)))
            self.assertEqual(explicit["parser_path"], str(explicit_parser))          

    def test_launcher_preserves_caller_relative_paths_from_arbitrary_directory(self):
        (self.root / "student").mkdir()
        env = dict(os.environ)
        for key in ("DATA_MANIFEST", "EVAL_DATA", "BENCHMARKS", "CHECKPOINT", "LCB_REPO",
                    "OUTPUT_DIR", "LULU_OUTPUT_ROOT"):
            env.pop(key, None)
        env.update(PYTHON_BIN=sys.executable, PYTHON="/missing/python", GPUS="cpu",
                   DATA_MANIFEST="manifest.json", CHECKPOINT="student", OUTPUT_DIR="eval-output",
                   DRY_RUN="1", INCLUDE_BASE="0", BATCH_SIZE="16", 
                   PYTHONPATH="",)
        completed = subprocess.run(["bash", str(SCRIPT.parents[1] / "runs" / "eval_lulu.sh")],
                                   cwd=self.root, env=env, check=True, text=True, capture_output=True)
        plan = json.loads(completed.stdout)
        self.assertEqual(plan["output_dir"], str(self.root / "eval-output"))
        self.assertEqual(plan["models"], [{"name": "lulu", "model": str(self.root / "student")}])
        self.assertEqual(plan["batch_size"], 16)
        self.assertEqual(plan["max_examples"], 199)
        self.assertTrue(all(b["path"] == str(self.root / "data.parquet") for b in plan["benchmarks"]))
        self.assertFalse((self.root / "eval-output").exists())

    def test_default_cap_is_per_benchmark_and_keeps_small_datasets_complete(self):
        metadata = json.loads(self.manifest.read_text())
        counts = [500, 30, 512, 1400, 198]
        for name, count in zip(evaluation.DEFAULT_BENCHMARKS, counts):
            metadata["benchmarks"][name]["full_examples"] = count
        self.manifest.write_text(json.dumps(metadata))
        plan = evaluation.build_plan(self.args("--include-base", "--checkpoint", "final=/tmp/student"))
        self.assertEqual(plan["max_examples"], 199)
        self.assertEqual([b["expected_examples"] for b in plan["benchmarks"]], [199, 30, 199, 199, 198])
        full = evaluation.build_plan(self.args("--max-examples", "0"))
        self.assertEqual([b["expected_examples"] for b in full["benchmarks"]], counts)

    def test_named_checkpoints_base_control_and_generation_flags(self):
        plan = evaluation.build_plan(self.args("--checkpoint", "ren=/tmp/adapter",
                    "--checkpoint", "opd=/tmp/opd", "--include-base", "--no-thinking", "--max-examples", "3"))
        self.assertEqual([m["name"] for m in plan["models"]], ["base", "ren", "opd"])
        self.assertFalse(plan["thinking"])
        self.assertTrue(all(b["expected_examples"] == 3 for b in plan["benchmarks"]))

    def test_checkpoint_names_cannot_escape_output_directory_or_collide(self):
        for value in ("../escape=x", "base=x"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                evaluation.build_plan(self.args("--checkpoint", value, "--include-base"))

    def test_direct_benchmark_overrides_manifest_count(self):
        plan = evaluation.build_plan(self.args("--benchmark", f"mmlu_pro={self.root / 'data.parquet'}"))
        self.assertEqual(len(plan["benchmarks"]), 1)
        self.assertIsNone(plan["benchmarks"][0]["expected_examples"])

    def test_selected_missing_benchmark_fails_before_model_load(self):
        with self.assertRaisesRegex(ValueError, "missing from manifest"):
            evaluation.build_plan(self.args("--benchmarks", "missing"))

    def test_device_detection_preserves_scheduler_visibility(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3,GPU-example"}):
            self.assertEqual(evaluation.resolve_devices("auto"), ["3", "GPU-example"])
        for value in ("", "-1", "0,0", "cpu,0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                evaluation.resolve_devices(value)

    def test_lcb_requires_official_scorer_and_complete_split(self):
        data = json.loads(self.manifest.read_text())
        data["benchmarks"]["livecodebench"] = {
            "full": "data.parquet", "scorer": "livecodebench",
            "livecodebench": {"release_version": "v6"}}
        self.manifest.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "official evaluator"):
            evaluation.build_plan(self.args("--benchmarks", "all"))
        with self.assertRaisesRegex(ValueError, "complete split"):
            evaluation.build_plan(self.args("--benchmarks", "all", "--lcb-repo", str(self.root), "--max-examples", "1"))


class EvaluationSummaryTests(unittest.TestCase):
    @staticmethod
    def row(index, reward):
        return {"prompt_index": index, "reward": reward, "response_tokens": 10 + index,
                "prompt_tokens": 5, "hit_cap": index == 1}

    def test_scores_and_paired_rescue_degradation_use_all_matching_examples(self):
        base = [self.row(0, 0.0), self.row(1, 1.0), self.row(2, 0.0)]
        candidate = [self.row(0, 1.0), self.row(1, 0.0), self.row(2, 1.0)]
        summary = evaluation.summarize_rows(candidate)
        self.assertAlmostEqual(summary["accuracy"], 2 / 3)
        self.assertEqual(summary["mean_response_tokens"], 11)
        self.assertAlmostEqual(summary["hit_cap_fraction"], 1 / 3)
        compared = evaluation.paired_comparison(base, candidate)
        self.assertEqual(compared["rescues"], 2)
        self.assertEqual(compared["degradations"], 1)
        self.assertAlmostEqual(compared["accuracy_delta"], 1 / 3)
        with self.assertRaisesRegex(ValueError, "indices must match"):
            evaluation.paired_comparison(base, candidate[:-1])

    def test_unscored_rows_are_not_treated_as_incorrect(self):
        summary = evaluation.summarize_rows([self.row(0, None)])
        self.assertEqual(summary["scored_examples"], 0)
        self.assertIsNone(summary["accuracy"])

    def test_livecodebench_helpers_use_lulu_scripts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scripts = root / "scripts"
            scripts.mkdir()
            (scripts / "export_livecodebench_custom.py").touch()
            (scripts / "inject_livecodebench_rewards.py").touch()

            plan = {
                "lcb_python": "python",
                "lcb_repo": str(root / "official"),
                "lcb_processes": 2,
            }
            benchmark = {"path": str(root / "data.parquet"),
                         "livecodebench": {"release_version": "v6"}}

            with patch.object(evaluation, "PROJECT_SCRIPTS", scripts), \
                 patch.object(evaluation.subprocess, "run") as run:
                scored = evaluation.score_livecodebench(plan, benchmark, root / "raw")
            self.assertEqual(scored, root / "raw/scored")
            self.assertEqual(run.call_args_list[0].args[0][1],
                            str(scripts / "export_livecodebench_custom.py"))
            self.assertEqual(run.call_args_list[1].kwargs["cwd"], str(root / "official"))
            self.assertEqual(run.call_args_list[2].args[0][1],
                            str(scripts / "inject_livecodebench_rewards.py"))

    def test_jsonl_response_unicode_separators_do_not_split_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = self.row(0, 1)
            row['response'] = "physics\u2028formula\u2029answer\u0085done"
            (root / 'shard-000.jsonl').write_text(json.dumps(row, ensure_ascii=False) + "\n")
            self.assertEqual(evaluation.load_complete_rows(root, 1, 1), [row])

    def test_merge_requires_complete_exact_disjoint_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "shard-000.jsonl").write_text(json.dumps(self.row(0, 1)) + "\n")
            with self.assertRaisesRegex(ValueError, "incomplete or stale"):
                evaluation.load_complete_rows(root, 2, 2)
            (root / "shard-001.jsonl").write_text(json.dumps(self.row(1, 0)) + "\n")
            self.assertEqual(len(evaluation.load_complete_rows(root, 2, 2)), 2)
            with self.assertRaisesRegex(ValueError, "coverage"):
                evaluation.load_complete_rows(root, 2, 3)
            (root / "shard-001.jsonl").write_text(json.dumps(self.row(0, 0)) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate or incorrectly"):
                evaluation.load_complete_rows(root, 2, 2)


@unittest.skipUnless(importlib.util.find_spec("transformers") and importlib.util.find_spec("peft"),
                     "model integration needs the project Python environment")
class SavedStudentIntegrationTests(unittest.TestCase):
    def test_hf_and_lora_checkpoint_loading_and_batched_general_reasoning(self):
        import torch
        import pyarrow as pa
        import pyarrow.parquet as pq
        from peft import LoraConfig, PeftModel, get_peft_model
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            adapter = root / "adapter"
            vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3,
                     "user": 4, "assistant": 5, "question": 6, "long": 7,
                     "<think>": 8, "A": 9, "B": 10, "answer": 11, "is": 12}
            backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
            backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="<bos>",
                        eos_token="<eos>", pad_token="<pad>", unk_token="<unk>")
            tokenizer.chat_template = ("{% for m in messages %}{{ m['role'] + ' ' + m['content'] + ' ' }}{% endfor %}"
                                       "assistant {% if enable_thinking %}<think>{% endif %}")
            model = LlamaForCausalLM(LlamaConfig(vocab_size=len(vocab), hidden_size=16,
                    intermediate_size=24, num_hidden_layers=1, num_attention_heads=2,
                    num_key_value_heads=2, max_position_embeddings=64,
                    eos_token_id=1, bos_token_id=2, pad_token_id=0))
            model.save_pretrained(base)
            tokenizer.save_pretrained(base)
            trained = get_peft_model(AutoModelForCausalLM.from_pretrained(base),
                                    LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"], task_type="CAUSAL_LM"))
            with torch.no_grad():
                for name, value in trained.named_parameters():
                    if 'lora_B' in name:
                        value.fill_(0.02)
            trained.eval()
            probe = torch.tensor([[2, 4, 6]])
            with torch.inference_mode():
                expected_logits = trained(probe).logits
            trained.save_pretrained(adapter)
            tokenizer.save_pretrained(adapter)
            loaded, wrapped = evaluation.load_model_assets(str(adapter), dtype=torch.float32,
                                                           device="cpu", thinking=False)
            self.assertNotIsInstance(loaded, PeftModel)
            self.assertFalse(any("lora_" in name for name, _ in loaded.named_parameters()))
            self.assertFalse(loaded.training)
            with torch.inference_mode():
                torch.testing.assert_close(loaded(probe).logits, expected_logits, atol=1e-5, rtol=1e-5)
            self.assertNotIn("<think>", wrapped.apply_chat_template(
                [{"role": "user", "content": "question"}], tokenize=False, add_generation_prompt=True))
            self.assertEqual(loaded.generation_config.repetition_penalty, 1.0)
            plain, thinking_tokenizer = evaluation.load_model_assets(str(base), dtype=torch.float32,
                                                                     device="cpu", thinking=True)
            self.assertNotIsInstance(plain, PeftModel)
            self.assertIn("<think>", thinking_tokenizer.apply_chat_template(
                [{"role": "user", "content": "question"}], tokenize=False, add_generation_prompt=True))
            data = root / "mmlu.parquet"
            pq.write_table(pa.Table.from_pylist([
                {"prompt": [{"role": "user", "content": text}], "data_source": "mmlu_pro",
                 "reward_model": {"ground_truth": "__CHOICE__A"}}
                for text in ("question", "long long question", "long question")]), data)
            args = evaluation.argument_parser().parse_args([
                "--benchmark", f"mmlu_pro={data}", "--checkpoint", f"ren={adapter}",
                "--gpus", "cpu", "--batch-size", "2", "--max-response-tokens", "2",
                "--max-prompt-tokens", "32", "--dtype", "float32", "--no-thinking",
                "--output-dir", str(root / "out")])
            plan = evaluation.build_plan(args)
            runner = evaluation.make_runner(plan, "cpu")
            try:
                # Exactly-at-limit inputs remain intact, including when a
                # shorter neighbor is padded to the same width.
                runner.max_prompt_tokens = 5
                for sid in range(2):
                    runner.evaluate_shard(model_id=str(adapter), input_parquet=str(data),
                        output=str(root / "out" / f"shard-{sid:03d}.jsonl"), shard_id=sid, num_shards=2)
                rows = evaluation.load_complete_rows(root / "out", 2, 3)
                self.assertEqual([r["prompt_index"] for r in rows], [0, 1, 2])
                self.assertEqual({r["data_source"] for r in rows}, {"mmlu_pro"})
                self.assertTrue(all(r["reward"] in (0.0, 1.0) for r in rows))
                self.assertTrue(all(r["response_tokens"] <= 2 for r in rows))
                self.assertGreater(rows[1]["prompt_tokens"], rows[0]["prompt_tokens"])
                self.assertEqual(rows[1]["prompt_tokens"], 5)
                self.assertEqual(evaluation.summarize_rows(rows)["scored_examples"], 3)
                runner.max_prompt_tokens = 4
                with patch.object(runner.model, "generate") as generate:
                    with self.assertRaisesRegex(
                        ValueError, r"prompt_index=1 has 5 tokens.*--max-prompt-tokens=4"
                    ):
                        runner.evaluate_shard(
                            model_id=str(adapter), input_parquet=str(data),
                            output=str(root / "overlong.jsonl"), shard_id=0, num_shards=1,
                        )
                    generate.assert_not_called()
            finally:
                runner.close()
            # Exercise the real process launcher and final baseline comparison,
            # using a CPU worker and local tiny models only.
            plan["output_dir"] = str(root / "suite")
            plan["models"].insert(0, {"name": "base", "model": str(base)})
            # Start both orchestrator and worker from an unrelated directory,
            # with no inherited project PYTHONPATH and only relative user paths.
            completed = subprocess.run([
                sys.executable, str(SCRIPT), "--benchmark", "mmlu_pro=mmlu.parquet",
                "--checkpoint", "ren=adapter", "--include-base", "--model", "base",
                "--gpus", "cpu", "--batch-size", "2", "--max-response-tokens", "2",
                "--max-prompt-tokens", "32", "--dtype", "float32", "--no-thinking",
                "--output-dir", "suite"], cwd=root, env={**os.environ, "PYTHONPATH": ""},
                text=True, capture_output=True, timeout=120)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr +
                             "\n" + "\n".join(p.read_text() for p in (root / "suite").glob("worker-*.log")))
            worker_plan = json.loads((root / "suite/eval_plan.json").read_text())
            self.assertEqual(worker_plan["models"][1]["model"], str(adapter))
            report = json.loads((root / "suite/summary.json").read_text())
            self.assertEqual(set(report["models"]), {"base", "ren"})
            metrics = report["models"]["ren"]["benchmarks"]["mmlu_pro"]
            self.assertEqual(metrics["examples"], 3)
            self.assertEqual(metrics["vs_base"]["paired_examples"], 3)
            with self.assertRaisesRegex(ValueError, "must be empty"):
                evaluation.run_suite(plan)


class OlympiadFinalAnswerTests(unittest.TestCase):
    def test_repeated_and_revised_boxes_use_final_conclusion(self):
        from lulu import benchmark_parser as parser
        for text, expected in [
            (r"\boxed{27}</think>Final answer: \boxed{27}", "27"),
            (r"Try \boxed{12}. Reconsider. </think>Final: \boxed{27}", "27"),
            (r"Intermediate \boxed{2}. Final: \boxed{(2,4)}", "(2,4)"),
            (r"Final \boxed{\frac{47}{300}}", r"\frac{47}{300}"),
        ]:
            with self.subTest(text=text):
                prediction = parser.extract_answer(text, 'olympiadbench')
                self.assertTrue(parser.math_equal(prediction, expected), prediction)
        self.assertFalse(parser.math_equal(parser.extract_answer(r'Final \boxed{12}', 'olympiadbench'), '27'))


if __name__ == "__main__":
    unittest.main()


def test_choice_labels_are_complete_and_accept_latex_wrappers():
    cases = [
        ('The answer is approximately 20 GeV.', '?'),
        ('The answer is based on this assumption.', '?'),
        ('Final Answer: Given the reasoning above.', '?'),
        (r'The answer is approximately 20 GeV. Final: \boxed{D}', 'D'),
        (r'The correct answer is clearly H. Final: \boxed{H}', 'H'),
        (r'The answer is $\boxed{\text{F}}$.', 'F'),
        (r'Final: \boxed{\mathrm{C}}', 'C'),
        (r'Try \boxed{\text{B}}, then final: \boxed{\text{D}}', 'D'),
        (r'The answer is (c).', 'C'),
        (r'The answer is \boxed{\text{Clearly}}', '?'),
    ]
    for source in ('mmlu_pro', 'gpqa_diamond'):
        for text, expected in cases:
            assert benchmark_parser.extract_answer(text, source) == '__CHOICE__' + expected


def test_global_queue_reuses_free_gpu_without_changing_logical_shard(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    data=tmp_path/'data.parquet'
    pq.write_table(pa.Table.from_pylist([{'x':i} for i in range(3)]),data)
    args=evaluation.argument_parser().parse_args([
        '--benchmark',f'mmlu_pro={data}','--gpus','2,5',
        '--checkpoint','a=/tmp/eval-queue-a','--checkpoint','b=/tmp/eval-queue-b',
        '--output-dir',str(tmp_path/'out'),'--backend','vllm'])
    plan=evaluation.build_plan(args)
    calls=[]
    class FakeProcess:
        def __init__(self, command, **kwargs):
            self.remaining=20 if len(calls)==1 else 1
            calls.append((command,kwargs))
        def poll(self):
            self.remaining-=1
            return None if self.remaining>0 else 0
        def wait(self, **kwargs):return 0
        def terminate(self):raise AssertionError('No successful worker should be terminated')
    monkeypatch.setattr(evaluation.subprocess,'Popen',FakeProcess)
    monkeypatch.setattr(evaluation.time,'sleep',lambda _:None)
    monkeypatch.setattr(evaluation,'finish_suite',lambda _:None)
    evaluation.run_suite(plan)
    assert len(calls)==4
    logical=[int(cmd[cmd.index('--shard-id')+1]) for cmd,kw in calls]
    physical=[kw['env']['CUDA_VISIBLE_DEVICES'] for cmd,kw in calls]
    assert logical==[0,1,0,1]
    assert physical==['2','5','2','2']
    # The slow a/shard1 is still running on GPU5 while b/shard1 uses free GPU2.
    assert 'worker-plan-b' in calls[3][0][calls[3][0].index('--worker-plan')+1]
    assert calls[0][1]['env']['VLLM_CACHE_ROOT']==calls[3][1]['env']['VLLM_CACHE_ROOT']
    assert calls[0][1]['env']['VLLM_CACHE_ROOT']!=calls[1][1]['env']['VLLM_CACHE_ROOT']
    dispatched=[json.loads(x) for x in (tmp_path/'out/dispatch.jsonl').read_text().splitlines()]
    assert [(r['gpu'],r['shard_id']) for r in dispatched]==list(zip(physical,logical))


def test_queue_failure_terminates_only_its_running_worker_group(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pytest
    import signal
    data=tmp_path/'data.parquet';pq.write_table(pa.Table.from_pylist([{'x':0}]),data)
    args=evaluation.argument_parser().parse_args([
        '--benchmark',f'mmlu_pro={data}','--gpus','2,5','--checkpoint','a=/tmp/eval-queue-a',
        '--output-dir',str(tmp_path/'out'),'--backend','vllm'])
    calls=[];killed=[]
    class FakeProcess:
        def __init__(self, command, **kwargs):
            self.pid=10000+len(calls);self.code=3 if not calls else None
            calls.append((self,kwargs))
        def poll(self):return self.code
        def wait(self, **kwargs):return self.code
    def killpg(pid,sig):
        killed.append((pid,sig))
        for proc,kw in calls:
            if proc.pid==pid:proc.code=-sig
    monkeypatch.setattr(evaluation.subprocess,'Popen',FakeProcess)
    monkeypatch.setattr(evaluation.os,'killpg',killpg)
    monkeypatch.setattr(evaluation.time,'sleep',lambda _:None)
    monkeypatch.setattr(evaluation,'finish_suite',lambda _:pytest.fail('Failed evaluation cannot be summarized'))
    with pytest.raises(RuntimeError,match='failed: 3'):
        evaluation.run_suite(evaluation.build_plan(args))
    assert all(kw['start_new_session'] for proc,kw in calls)
    assert killed==[(10001,signal.SIGTERM)]


def test_formatted_choice_fallback_keeps_final_boundary_and_rejects_formulas():
    from lulu import thinking_final_parser as parser
    cases = [
        ("draft </think> The answer is **B. -0.7**.", "B"),
        (r"draft </think> \boxed{\text{The answer is } G}", "G"),
        (r"draft </think> \boxed{\text{D. 16.9\%}}", "D"),
        (r"draft </think> \boxed{\text{B. } \$61.48}", "B"),
        (r"draft </think> \boxed{\text{C}_6\text{H}_{12}\text{O}_2}", "?"),
        (r"draft </think> \boxed{C6H12O}", "?"),
        ("draft </think> The answer is approximately 3", "?"),
    ]
    for text, expected in cases:
        assert parser.extract_answer(text, "gpqa_diamond") == "__CHOICE__" + expected
    assert parser.extract_answer("The answer is **B. -0.7**", "gpqa_diamond") == parser.MISSING_FINAL
