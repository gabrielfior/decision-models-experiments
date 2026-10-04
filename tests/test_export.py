"""The exported (merged, truncated, vocab-trimmed) decider must reproduce the trained model's read-out."""
import json
import glob

import pytest
import torch

import prepare as P
from scripts import export as E

QWEN_TOK = glob.glob("/Users/gabrielfior/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/*/tokenizer.json")


def _tiny_config():
    from transformers import AutoConfig
    cfg = AutoConfig.for_model("qwen3_5_text")
    cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers, cfg.vocab_size = 32, 64, 4, 300
    cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim = 2, 1, 16
    cfg.linear_num_value_heads, cfg.linear_num_key_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim = 2, 2, 16, 16
    cfg.layer_types = ["linear_attention", "full_attention"] * 2
    cfg.max_position_embeddings = 128
    return cfg


def test_truncate_keeps_hidden_state_at_tap_exact():
    from transformers import AutoModel
    torch.manual_seed(0)
    full = AutoModel.from_config(_tiny_config()).eval()
    ids = torch.randint(0, 300, (2, 11))
    with torch.no_grad():
        ref = full(input_ids=ids, output_hidden_states=True).hidden_states[2]
        cut = E.truncate_torso(full, 2)
        got = cut(input_ids=ids, output_hidden_states=True).hidden_states[2]
    assert len(cut.layers) == 2 and cut.config.num_hidden_layers == 2 and cut.config.layer_types == ["linear_attention", "full_attention"]
    assert torch.allclose(ref, got, atol=1e-6)


def test_trim_embeddings_gathers_kept_rows_and_updates_vocab():
    from transformers import AutoModel
    torch.manual_seed(0)
    m = AutoModel.from_config(_tiny_config())
    w = m.get_input_embeddings().weight.data.clone()
    E.trim_embeddings(m, [0, 1, 2, 299])
    assert m.config.vocab_size == 4
    assert torch.equal(m.get_input_embeddings().weight.data, w[[0, 1, 2, 299]])


def test_trim_tokenizer_json_filters_merges_and_renumbers_specials():
    tj = {"model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2, "c": 3, "abc": 4}, "merges": [["a", "b"], ["ab", "c"]]},
          "added_tokens": [{"id": 5, "content": "<s>"}, {"id": 6, "content": "<e>"}]}
    new, old_ids = E.trim_tokenizer_json(tj, 3)
    assert new["model"]["vocab"] == {"a": 0, "b": 1, "ab": 2}
    assert new["model"]["merges"] == [["a", "b"]]
    assert [a["id"] for a in new["added_tokens"]] == [3, 4]
    assert old_ids == [0, 1, 2, 5, 6]


@pytest.mark.skipif(not QWEN_TOK, reason="Qwen3.5-2B tokenizer not in the local HF cache")
def test_trimmed_qwen_tokenizer_matches_original_on_common_text_and_keeps_markers(tmp_path):
    from transformers import AutoTokenizer
    src = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B-Base")
    old_ids = E.write_trimmed_tokenizer(src, tmp_path, 98304)
    new = AutoTokenizer.from_pretrained(tmp_path)
    text = "Decide whether to buy the cheaper option. The user said: 'no, not today'. Price: 12.50 EUR"
    a, b = src(text, add_special_tokens=False)["input_ids"], new(text, add_special_tokens=False)["input_ids"]
    assert all(i < 98304 for i in a) and a == b
    for m in P.QWEN_MARKERS.values():
        i_new = new.convert_tokens_to_ids(m)
        assert i_new >= 98304 and old_ids[i_new] == src.convert_tokens_to_ids(m)
    # a rare piece (id >= cut) is re-split into kept pieces and decodes back to the same string
    rare = next(t for t, i in src.get_vocab().items() if 200000 < i < 248000 and "Ġ" not in t)
    s = src.convert_tokens_to_string([rare])
    assert new.decode(new(s, add_special_tokens=False)["input_ids"]) == s
    assert all(i < 98304 for i in new(s, add_special_tokens=False)["input_ids"])
