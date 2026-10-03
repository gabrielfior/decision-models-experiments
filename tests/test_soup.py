"""CPU tests for scripts/soup.py (LoRA checkpoint soup) and the pure helpers of scripts/fit_temps.py."""
import json

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from scripts import fit_temps, soup

ADAPTER_CFG = {"peft_type": "LORA", "r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
               "target_modules": ["q_proj", "v_proj"], "task_type": None}


def _make_run(root, name, seed, adapter_cfg=ADAPTER_CFG, temps=None):
    g = torch.Generator().manual_seed(seed)
    run = root / name
    (run / "lora").mkdir(parents=True)
    tensors = {
        "base_model.model.layers.0.q_proj.lora_A.weight": torch.randn(4, 8, generator=g),
        "base_model.model.layers.0.q_proj.lora_B.weight": torch.randn(8, 4, generator=g),
    }
    save_file(tensors, str(run / "lora" / "adapter_model.safetensors"))
    (run / "lora" / "adapter_config.json").write_text(json.dumps(adapter_cfg))
    head = {"q.weight": torch.randn(3, 5, generator=g), "k.weight": torch.randn(3, 5, generator=g)}
    torch.save(head, run / "head.pt")
    (run / "config.json").write_text(json.dumps({"lora": {"r": 16}, "head": {"name": "pointer"}, "temps": temps or {"noul": 1.0},
                                                 "torso": "Qwen/Qwen3.5-0.8B-Base", "report": {"note": name}}))
    return run, tensors, head


def test_soup_writes_the_uniform_mean_of_adapters_and_heads(tmp_path):
    a, ta, ha = _make_run(tmp_path, "a", 0)
    b, tb, hb = _make_run(tmp_path, "b", 1)
    out = soup.soup([a, b], tmp_path / "soup")
    got = load_file(str(out / "lora" / "adapter_model.safetensors"))
    assert set(got) == set(ta)
    for k in ta:
        assert torch.allclose(got[k], (ta[k] + tb[k]) / 2)
    head = torch.load(out / "head.pt")
    for k in ha:
        assert torch.allclose(head[k], (ha[k] + hb[k]) / 2)
    assert json.loads((out / "lora" / "adapter_config.json").read_text()) == ADAPTER_CFG
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["head"] == {"name": "pointer"} and cfg["temps"] == {"noul": 1.0} and cfg["torso"] == "Qwen/Qwen3.5-0.8B-Base"
    assert cfg["note"].startswith("soup of [") and str(a) in cfg["note"] and str(b) in cfg["note"]
    assert cfg["soup"] == {"runs": [str(a), str(b)], "weights": [0.5, 0.5]}


def test_soup_honours_weights_and_normalises_them(tmp_path):
    a, ta, ha = _make_run(tmp_path, "a", 2)
    b, tb, hb = _make_run(tmp_path, "b", 3)
    out = soup.soup([a, b], tmp_path / "soup", weights=[3.0, 1.0])
    got = load_file(str(out / "lora" / "adapter_model.safetensors"))
    for k in ta:
        assert torch.allclose(got[k], 0.75 * ta[k] + 0.25 * tb[k])
    head = torch.load(out / "head.pt")
    for k in ha:
        assert torch.allclose(head[k], 0.75 * ha[k] + 0.25 * hb[k])
    assert json.loads((out / "config.json").read_text())["soup"]["weights"] == [0.75, 0.25]


def test_soup_refuses_mismatched_adapter_configs(tmp_path):
    a, *_ = _make_run(tmp_path, "a", 4)
    b, *_ = _make_run(tmp_path, "b", 5, adapter_cfg={**ADAPTER_CFG, "r": 8})
    with pytest.raises(ValueError, match="r"):
        soup.soup([a, b], tmp_path / "soup")
    c, *_ = _make_run(tmp_path, "c", 6, adapter_cfg={**ADAPTER_CFG, "target_modules": ["q_proj"]})
    with pytest.raises(ValueError, match="target_modules"):
        soup.check_adapter_configs([ADAPTER_CFG, json.loads((c / "lora" / "adapter_config.json").read_text())])


def test_soup_refuses_mismatched_tensor_keys_and_bad_weights(tmp_path):
    a = {"x": torch.ones(2)}
    b = {"y": torch.ones(2)}
    with pytest.raises(ValueError, match="keys"):
        soup.average_tensors([a, b])
    with pytest.raises(ValueError, match="weights"):
        soup.average_tensors([a, a], weights=[1.0])
    with pytest.raises(ValueError, match="run"):
        soup.soup([], tmp_path / "soup")


def test_average_tensors_keeps_dtype_and_passes_integer_buffers_through():
    a = {"w": torch.tensor([1.0, 3.0], dtype=torch.bfloat16), "step": torch.tensor(7)}
    b = {"w": torch.tensor([3.0, 5.0], dtype=torch.bfloat16), "step": torch.tensor(9)}
    m = soup.average_tensors([a, b])
    assert m["w"].dtype == torch.bfloat16 and torch.equal(m["w"].float(), torch.tensor([2.0, 4.0]))
    assert torch.equal(m["step"], torch.tensor(7))      # non-float: taken from the first run


def test_merge_temps_keeps_previous_temps_and_adds_per_k_buckets():
    cfg = {"torso": "t", "temps": {"noul": 1.2, "choice": 0.9}, "report": {}}
    new = fit_temps.merge_temps(cfg, {"noul": 1.0, "choice": 1.1, "score": 0.8}, split="dev", n_rows=2000)
    assert new["temps"] == {"noul": 1.0, "choice": 1.1, "score": 0.8}
    assert new["temps_prev"] == {"noul": 1.2, "choice": 0.9}
    assert new["temps_fit"] == {"split": "dev", "n_rows": 2000, "per_k": False}
    assert new["torso"] == "t" and cfg["temps"] == {"noul": 1.2, "choice": 0.9}     # input untouched
    with_k = fit_temps.merge_temps(new, {"noul": 1.0}, per_k={"choice:4": 1.3}, split="heldout", n_rows=10)
    assert with_k["temps_prev"] == {"noul": 1.0, "choice": 1.1, "score": 0.8} and with_k["temps_per_k"] == {"choice:4": 1.3}
    assert with_k["temps_fit"]["per_k"] is True
    fresh = fit_temps.merge_temps({"torso": "t"}, {"noul": 1.0}, split="dev", n_rows=1)
    assert "temps_prev" not in fresh and fresh["temps"] == {"noul": 1.0}


def _pred(qtype, n, label, scale, spike=None):
    """A prediction whose logit spike of height `scale` sits at `spike` (default: on the label)."""
    z = np.zeros(n)
    z[label if spike is None else spike] = scale
    return {"logits": z, "label": label, "qtype": qtype, "n_options": n, "id": f"{qtype}{n}", "ms": 0.0}


def test_fit_temperature_by_k_fits_each_bucket_and_falls_back_to_the_type_temp():
    # choice/2: confident (margin 6) but wrong 1 row in 10 -> overconfident, needs T > 1
    # choice/4: always right with a tiny margin -> underconfident, needs T < 1;  score/3: too few rows -> fallback
    preds = [_pred("choice", 2, i % 2, 6.0, spike=(i % 2) if i % 10 else (i + 1) % 2) for i in range(40)]
    preds += [_pred("choice", 4, i % 4, 0.3) for i in range(40)]
    preds += [_pred("score", 3, i % 3, 1.0) for i in range(5)]
    by_type = {"choice": 1.0, "score": 2.5}
    per_k = fit_temps.fit_temperature_by_k(preds, by_type, min_rows=20)
    assert set(per_k) == {"choice:2", "choice:4", "score:3"}
    assert per_k["choice:2"] > 1.0 and per_k["choice:4"] < 1.0
    assert per_k["score:3"] == 2.5


def test_bucket_temps_select_the_bucket_or_fall_back():
    temps = {"choice": 1.0}
    assert fit_temps.temp_for({"qtype": "choice", "n_options": 4}, temps, {"choice:4": 0.5}) == 0.5
    assert fit_temps.temp_for({"qtype": "choice", "n_options": 7}, temps, {"choice:4": 0.5}) == 1.0
    assert fit_temps.temp_for({"qtype": "score", "n_options": 3}, temps, None) == 1.0
