import importlib.util,json
from pathlib import Path
import pytest
import torch
from safetensors.torch import save_file
spec=importlib.util.spec_from_file_location('initial_cache',Path(__file__).resolve().parents[1]/'scripts/reuse_initial_evaluation.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_exact_weight_values_required(tmp_path):
    a,b=tmp_path/'a',tmp_path/'b';a.mkdir();b.mkdir()
    tensor=torch.arange(12,dtype=torch.bfloat16).reshape(3,4)
    save_file({'w':tensor.float()},str(a/'model.safetensors'));save_file({'w':tensor},str(b/'model.safetensors'))
    assert module.verify_identical_weights(a,b)['tensor_elements_checked_including_tied_aliases']==12
    changed=tensor.float();changed[0,0]=0.0001
    save_file({'w':changed},str(a/'model.safetensors'))
    with pytest.raises(ValueError,match='values differ'):module.verify_identical_weights(a,b)


def test_tied_alias_is_checked_against_both_base_tensors(tmp_path):
    a,b=tmp_path/'a',tmp_path/'b';a.mkdir();b.mkdir()
    for path in (a,b):(path/'config.json').write_text(json.dumps({'tie_word_embeddings':True}))
    tensor=torch.arange(12,dtype=torch.bfloat16).reshape(3,4)
    save_file({'model.embed_tokens.weight':tensor.float()},str(a/'model.safetensors'))
    save_file({'model.embed_tokens.weight':tensor,'lm_head.weight':tensor.clone()},str(b/'model.safetensors'))
    assert module.verify_identical_weights(a,b)['identical']
    wrong=tensor.clone();wrong[0,0]=1
    save_file({'model.embed_tokens.weight':tensor,'lm_head.weight':wrong},str(b/'model.safetensors'))
    with pytest.raises(ValueError,match='values differ'):module.verify_identical_weights(a,b)


def test_architecture_difference_prevents_cache_reuse(tmp_path):
    a,b=tmp_path/'a',tmp_path/'b';a.mkdir();b.mkdir()
    for path in (a,b):save_file({'w':torch.ones(2)},str(path/'model.safetensors'))
    (a/'config.json').write_text(json.dumps({'rope_theta':10000}))
    (b/'config.json').write_text(json.dumps({'rope_theta':1000000}))
    with pytest.raises(ValueError,match='config differs'):module.verify_identical_weights(a,b)
