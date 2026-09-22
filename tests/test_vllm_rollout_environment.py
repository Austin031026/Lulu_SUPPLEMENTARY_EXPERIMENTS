import os

from lulu.vllm_rollout import isolate_vllm_allocator_environment


def test_vllm_worker_does_not_inherit_student_allocator_tuning(monkeypatch):
    value = 'max_split_size_mb:128,garbage_collection_threshold:0.80'
    monkeypatch.setenv('PYTORCH_CUDA_ALLOC_CONF', value)

    assert isolate_vllm_allocator_environment() == value
    assert 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ


def test_vllm_worker_allocator_isolation_is_idempotent(monkeypatch):
    monkeypatch.delenv('PYTORCH_CUDA_ALLOC_CONF', raising=False)

    assert isolate_vllm_allocator_environment() is None
    assert 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ
