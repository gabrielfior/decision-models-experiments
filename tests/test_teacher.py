"""CPU tests for the teacher-distillation pieces: train.kl_to_teacher and scripts/teacher.py's prompt rendering."""
import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import train
from scripts import teacher as T


# ---- train.py: KL to a teacher distribution -------------------------------------------------
def test_kl_is_zero_when_student_matches_teacher_and_positive_otherwise():
    logits = torch.tensor([[2.0, 0.0, 1.0], [0.5, 0.5, -1.0]])
    mask = torch.ones(2, 3, dtype=torch.bool)
    teacher = F.softmax(logits, dim=-1)
    assert train.kl_to_teacher(logits, teacher, mask).item() == pytest.approx(0.0, abs=1e-6)
    other = F.softmax(logits.flip(-1), dim=-1)
    assert train.kl_to_teacher(logits, other, mask).item() > 0.01


def test_kl_ignores_masked_options_and_averages_over_rows():
    # row 0 has 2 valid options; the third slot is padding with teacher mass 0 and an arbitrary logit
    logits = torch.tensor([[1.0, -1.0, 123.0], [0.0, 0.0, 0.0]])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    teacher = torch.tensor([[0.9, 0.1, 0.0], [1 / 3, 1 / 3, 1 / 3]])
    kl = train.kl_to_teacher(logits, teacher, mask)
    s0 = F.log_softmax(logits[0, :2], dim=-1)
    expect0 = float((teacher[0, :2] * (teacher[0, :2].log() - s0)).sum())
    assert kl.item() == pytest.approx(expect0 / 2, abs=1e-6)        # row 1 contributes 0 (uniform vs uniform); batchmean
    # changing the padding logit must not change the loss
    logits2 = logits.clone()
    logits2[0, 2] = -50.0
    assert train.kl_to_teacher(logits2, teacher, mask).item() == pytest.approx(kl.item(), abs=1e-6)


def test_teacher_batch_permutes_canonical_probs_into_presented_order_and_pads():
    teacher = {"r1": [0.1, 0.2, 0.7], "r2": [0.6, 0.4]}
    # presented position j shows canonical option perm[j]
    t, has = train.teacher_batch(teacher, ids=["r1", "missing", "r2"], perms=[[2, 0, 1], [0, 1], [1, 0]], K=3)
    assert t.shape == (3, 3) and has.tolist() == [True, False, True]
    assert t[0].tolist() == pytest.approx([0.7, 0.1, 0.2])
    assert t[2].tolist() == pytest.approx([0.4, 0.6, 0.0])
    assert t[1].tolist() == [0.0, 0.0, 0.0]


def test_distill_loss_mixes_ce_and_kl_per_row_and_falls_back_to_ce_without_teacher():
    head = train.head_mod.build_head(8, 1, {"name": "pointer", "width": 4, "layer": 0, "norm": False})
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, 0.5]])
    labels = torch.tensor([0, 1])
    mask = torch.ones(2, 3, dtype=torch.bool)
    teacher = torch.tensor([[0.2, 0.5, 0.3], [0.1, 0.8, 0.1]])
    has = torch.tensor([True, True])
    ce = F.cross_entropy(logits, labels)
    kl = train.kl_to_teacher(logits, teacher, mask)
    got = train.distill_loss(head, logits, labels, teacher, has, mask, alpha=0.5)
    assert got.item() == pytest.approx((0.5 * ce + 0.5 * kl).item(), abs=1e-6)
    # alpha = 0 is the plain head loss; no teacher rows is the plain head loss
    assert train.distill_loss(head, logits, labels, teacher, has, mask, alpha=0.0).item() == pytest.approx(ce.item(), abs=1e-6)
    none = torch.tensor([False, False])
    assert train.distill_loss(head, logits, labels, teacher, none, mask, alpha=0.5).item() == pytest.approx(ce.item(), abs=1e-6)
    # mixed: row 0 distilled, row 1 pure CE  ->  mean over rows of the per-row losses
    mixed = torch.tensor([True, False])
    ce_rows = F.cross_entropy(logits, labels, reduction="none")
    kl0 = train.kl_to_teacher(logits[:1], teacher[:1], mask[:1])
    expect = (0.5 * ce_rows[0] + 0.5 * kl0 + ce_rows[1]) / 2
    assert train.distill_loss(head, logits, labels, teacher, mixed, mask, alpha=0.5).item() == pytest.approx(expect.item(), abs=1e-6)


def test_teacher_batch_follows_the_shuffled_batches_of_the_training_loop():
    import prepare as P
    from tests.test_data import FakeTok, _record
    rows = P.flatten([_record("r1", "yelp"), _record("r2", "agnews")])          # choice (3), noul (2), score (3) x 2
    teacher = {r["id"]: list(np.random.default_rng(i).dirichlet(np.ones(r["n_options"]))) for i, r in enumerate(rows) if r["qtype"] != "noul"}
    batch = next(P.batches(FakeTok(), rows, len(rows), shuffle_options=True, rng=np.random.default_rng(0)))
    t, has = train.teacher_batch(teacher, batch["id"], batch["perm"], batch["opt_mask"].shape[1])
    assert has.tolist() == [r["qtype"] != "noul" for r in rows]                 # noul rows fall back to CE
    for i, (rid, perm) in enumerate(zip(batch["id"], batch["perm"])):
        if rid in teacher:
            assert t[i, : len(perm)].tolist() == pytest.approx([teacher[rid][j] for j in perm])
            assert t[i, len(perm):].abs().sum().item() == 0.0
    # the loss accepts the batch tensors as the loop passes them
    head = train.head_mod.build_head(8, 1, {"name": "pointer", "width": 4, "layer": 0, "norm": False})
    logits = torch.randn(len(rows), t.shape[1]).masked_fill(~batch["opt_mask"], -1e9)
    loss = train.distill_loss(head, logits, batch["label"], t, has, batch["opt_mask"], alpha=0.5)
    assert torch.isfinite(loss) and loss.item() > 0


def test_teacher_file_roundtrip(tmp_path):
    p = tmp_path / "t.jsonl"
    T.write_teacher(p, [("a#q", np.array([0.25, 0.75])), ("b#q", np.array([1.0, 0.0, 0.0]))])
    d = train.load_teacher(p)
    assert d == {"a#q": [0.25, 0.75], "b#q": [1.0, 0.0, 0.0]}
    assert json.loads(p.read_text().splitlines()[0]) == {"id": "a#q", "probs": [0.25, 0.75]}


# ---- scripts/teacher.py: option orders and the Mapika prompt --------------------------------
def test_perm_for_order_only_permutes_choice_questions():
    assert T.perm_for("identity", 4, "choice") == [0, 1, 2, 3]
    assert T.perm_for("reversed", 4, "choice") == [3, 2, 1, 0]
    assert T.perm_for("shift:1", 4, "choice") == [1, 2, 3, 0]
    assert T.perm_for("reversed", 3, "score") == [0, 1, 2]       # order carries meaning
    assert T.perm_for("reversed", 2, "noul") == [0, 1]


def test_mapika_option_texts_follow_render_question():
    choice = {"qtype": "choice", "options": [("billing", "money matters"), ("sales", None)]}
    assert T.mapika_option_texts(choice) == ["billing: money matters", "sales"]
    noul = {"qtype": "noul", "options": [("no", None), ("yes", None)]}
    assert T.mapika_option_texts(noul) == ["no", "yes"]
    noul_d = {"qtype": "noul", "options": [("no", "negative"), ("yes", "positive")]}
    assert T.mapika_option_texts(noul_d) == ["no: negative", "yes: positive"]
    score = {"qtype": "score", "options": [("0", "1 star"), ("1", "2 stars")]}
    assert T.mapika_option_texts(score) == ["0: 1 star", "1: 2 stars"]


def test_mapika_prompt_is_the_plain_state_first_layout():
    text = T.mapika_prompt_text("My card was charged twice.", "Which department?", ["billing", "support", "sales"])
    assert text == ("Context:\nMy card was charged twice.\n\nQuestion: Which department?\nOptions:\n"
                    "(A) billing\n(B) support\n(C) sales\nAnswer: (")


def test_mapika_isolated_level_rows_for_score_questions():
    row = {"qtype": "score", "state": "s", "instructions": "How many stars?",
           "options": [("0", "1 star"), ("1", "2 stars")], "n_options": 2}
    rows = T.mapika_isolated_rows(row)
    assert [o for _, o in rows] == [["no", "yes"], ["no", "yes"]]
    assert rows[0][0] == "How many stars?\nProposed answer: 1 star\nDoes the proposed answer fit?"
    assert rows[1][0] == "How many stars?\nProposed answer: 2 stars\nDoes the proposed answer fit?"
    assert T.combine_isolated([0.2, 0.6]) == pytest.approx([0.25, 0.75])


def test_mapika_default_temperatures_match_the_published_config():
    temps = T.mapika_temperatures(None)
    assert temps["choice"] == pytest.approx(1.11) and temps["noul"] == pytest.approx(1.56) and temps["score"] == pytest.approx(1.287)
    cfg = {"temperature": 2.0, "temperature_by_type": {"choice": 1.5}}
    temps = T.mapika_temperatures(cfg)
    assert temps == {"choice": 1.5, "noul": 2.0, "score": 2.0}


class WordTok:
    """Whitespace tokenizer in which every word, including 'A'..'Z' and 'AA'.., is one token."""
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [1 + (hash(w) % 50_000) for w in text.replace("\n", " \n ").split(" ") if w]


def test_mapika_encode_marks_the_answer_slot_and_letter_ids():
    tok = WordTok()
    row = {"qtype": "choice", "state": "hello world", "instructions": "pick one",
           "options": [("a", None), ("b", None), ("c", None)], "n_options": 3}
    enc = T.mapika_encode(tok, row, perm=[2, 0, 1])
    assert enc["slot"] == len(enc["ids"]) - 1 and enc["n_options"] == 3 and enc["perm"] == [2, 0, 1]
    assert enc["letter_ids"] == [tok.encode(L)[0] for L in "ABC"]
    # wide rendering (more than 10 options) uses one single-token label per option
    wide = {"qtype": "choice", "state": "s", "instructions": "q", "options": [(f"o{i}", None) for i in range(12)], "n_options": 12}
    enc_w = T.mapika_encode(tok, wide, perm=list(range(12)))
    assert len(enc_w["letter_ids"]) == 12 and len(set(enc_w["letter_ids"])) == 12 and enc_w["slot"] == len(enc_w["ids"]) - 1
