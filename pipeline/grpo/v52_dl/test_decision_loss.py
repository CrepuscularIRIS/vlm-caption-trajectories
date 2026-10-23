"""CPU checks for the first-action decision loss (Pro design 10-22 §7.2). They show the code matches the design, not
that the method works. Real-tokenizer checks run only where CAPTION_DL_TOKENIZER points at the training model."""
import itertools
import math
import os
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for ref in (os.environ.get("CAPTION_REF_DIR", ""),  # [scrub] server default removed
           
            str(HERE.parents[3] / "publications/vlm-caption-trajectories/code/reference")):
    if ref and os.access(Path(ref, "caption_v51.py"), os.R_OK):
        sys.path.insert(1, ref)
        break

import decision_loss as DL  # noqa: E402

LAM = 0.01


def _grad(z0: float, d: float) -> float:
    z = torch.tensor(z0, dtype=torch.float64, requires_grad=True)
    DL.pair_loss(z, torch.tensor(d, dtype=torch.float64)).backward()
    return float(z.grad)


@pytest.mark.unit
def test_per_fork_reward_then_mean():
    # success and crop count correlated: mean of Y(1-lam N) != mean(Y)(1 - lam mean(N))
    ys, ns = [1, 0, 1, 0], [1, 5, 1, 5]
    r = [y * (1 - LAM * n) for y, n in zip(ys, ns)]
    assert DL.branch_value(r) == pytest.approx(0.495)
    assert DL.branch_value(r) != pytest.approx(0.5 * (1 - LAM * 3))
    with pytest.raises(ValueError):
        DL.branch_value([])


@pytest.mark.unit
@pytest.mark.parametrize("z0", [-3.0, -0.2, 0.0, 0.7, 4.0])
@pytest.mark.parametrize("d", [-0.99, -0.24, -0.01, 0.01, 0.5, 1.0])
def test_gradient_closed_form_and_sign(z0, d):
    g = _grad(z0, d)
    assert g == pytest.approx(DL.pair_grad_z(z0, d), abs=1e-12)
    # gradient descent moves z (answer log-odds) up when D > 0, down when D < 0
    assert (g < 0) if d > 0 else (g > 0)


@pytest.mark.unit
def test_tie_is_exactly_zero():
    for z0 in (-5.0, 0.0, 3.0):
        z = torch.tensor(z0, requires_grad=True)
        loss = DL.pair_loss(z, torch.tensor(0.0))
        loss.backward()
        assert float(loss) == 0.0 and float(z.grad) == 0.0


@pytest.mark.unit
def test_small_cost_scale_is_not_renormalised():
    for z0, d in itertools.product((-1.0, 0.0, 2.0), (-0.7, 0.3, 1.0)):
        assert _grad(z0, d * 0.01) == pytest.approx(0.01 * _grad(z0, d), rel=1e-12)


@pytest.mark.unit
def test_target_is_detached():
    z = torch.tensor(0.3, requires_grad=True)
    d = torch.tensor(0.5, requires_grad=True)
    DL.pair_loss(z, d).backward()
    assert d.grad is None and z.grad is not None


@pytest.mark.unit
def test_mentor_table_at_p_half():
    # Pro §6: p = .5, lambda = .01, crop exactly once
    for (sa, sc), g in {(0, 4): 0.495, (4, 4): -0.005, (0, 0): 0.0, (4, 0): -0.5, (3, 4): 0.12}.items():
        d = DL.decision_gap([1.0] * sa + [0.0] * (4 - sa), [0.99] * sc + [0.0] * (4 - sc))
        assert _grad(0.0, d) == pytest.approx(g, abs=1e-12)


def _expected_grad(qa: float, qc: float, p: float) -> float:
    """Exact E[d l / d z] over the 25 success-count combinations, through the real loss function."""
    z0 = math.log(p / (1 - p))
    tot = 0.0
    for ba, bc in itertools.product(range(5), range(5)):
        w = math.comb(4, ba) * qa ** ba * (1 - qa) ** (4 - ba) * math.comb(4, bc) * qc ** bc * (1 - qc) ** (4 - bc)
        d = DL.decision_gap([1.0] * ba + [0.0] * (4 - ba), [0.99] * bc + [0.0] * (4 - bc))
        tot += w * _grad(z0, d)
    return tot


@pytest.mark.unit
def test_pro_finite_k_counterexamples():
    # crop slightly better, p = .01: descent RAISES the worse answer probability
    assert _expected_grad(0.50, 0.51, 0.01) == pytest.approx(-0.13153346, abs=1e-7)
    # answer slightly better, p = .9: descent LOWERS the better answer probability
    assert _expected_grad(0.50, 0.50, 0.90) == pytest.approx(0.106875, abs=1e-7)


@pytest.mark.unit
def test_select_slot_fixed_and_in_range():
    s = [DL.select_slot(42, "q1", c, 16) for c in range(50)]
    assert all(0 <= x < 16 for x in s) and len(set(s)) > 1
    assert s == [DL.select_slot(42, "q1", c, 16) for c in range(50)]


@pytest.mark.unit
def test_find_boundary():
    CLOSE, NL, LT = 151668, 198, 27
    turn = [1, 2, CLOSE, NL, LT, 9217, 36721]
    assert DL.find_boundary(turn, CLOSE, NL, LT) == 4
    assert turn[:4][-2:] == [CLOSE, NL]
    assert DL.find_boundary([1, CLOSE, 271, LT, 9217], CLOSE, NL, LT) is None   # '\n\n' variant
    assert DL.find_boundary([1, CLOSE, 366, 9217], CLOSE, NL, LT) is None       # ' <' variant
    assert DL.find_boundary([1, 2, 3], CLOSE, NL, LT) is None                   # truncated before </think>
    assert DL.find_boundary([1, CLOSE, NL], CLOSE, NL, LT) is None              # nothing after the newline


@pytest.mark.unit
def test_aux_microbatch_layout():
    pos = lambda n: torch.arange(4 * n).reshape(4, n)  # noqa: E731
    prefix = [5, 6, 7]
    rec = {"prefix": prefix, "tag_answer": [27, 9217], "tag_crop": [27, 1951, 287], "mm": {"x": 1},
           "pos_answer": pos(5), "pos_crop": pos(6), "pad_id": 0}
    mb = DL._aux_microbatch(rec, "cpu")
    assert mb["input_ids"].tolist() == [[5, 6, 7, 27, 9217, 0], [5, 6, 7, 27, 1951, 287]]
    assert mb["attention_mask"].tolist() == [[1, 1, 1, 1, 1, 0], [1] * 6]
    assert mb["responses"].tolist() == [[27, 9217, 0], [27, 1951, 287]]
    assert mb["_rmask"].tolist() == [[1, 1, 0], [1, 1, 1]]
    assert tuple(mb["position_ids"].shape) == (2, 4, 6)
    assert torch.equal(mb["position_ids"][0, :, :5], pos(5)) and mb["position_ids"][0, :, 5].sum() == 0
    # full-label log-prob is a SUM over the tag tokens (not a mean, not the shared first token only)
    lp = torch.tensor([[-0.1, -2.0, -9.0], [-0.1, -0.5, -0.01]])
    s = (lp * mb["_rmask"]).sum(-1)
    assert s.tolist() == pytest.approx([-2.1, -0.61])


@pytest.mark.unit
def test_protocol_relaxes_step1_only():
    pytest.importorskip("caption_v51")
    import caption_v52i as P
    turn = ("<think>\nStep 1 | locate\nThe sign in the upper left reads 42 [original_image].\n</think>\n"
            "<answer>B</answer>")
    p = P.parse_step(turn)
    assert P.check_step(p, 1, 0, ["original_image"]) == []
    import caption_v51 as V
    assert "behavior_requires_grounding" in V.check_step(V.parse_step(turn), 1, 0, ["original_image"])
    later = turn.replace("Step 1", "Step 2")
    assert "behavior_requires_grounding" in P.check_step(P.parse_step(later), 2, 1, ["original_image", "observation_1"])
    crop = ("<think>\nStep 1 | commit\nThe label is too small to read [original_image].\n</think>\n"
            '<grounding>{"bbox_2d": [0.1, 0.1, 0.4, 0.4], "source": "original_image"}</grounding>')
    assert P.check_step(P.parse_step(crop), 1, 0, ["original_image"]) == []
    bad_box = crop.replace("0.4, 0.4", "0.4, 1.4")
    assert "invalid_bbox" in P.check_step(P.parse_step(bad_box), 1, 0, ["original_image"])


@pytest.mark.unit
def test_real_tokenizer_tags():
    path = os.environ.get("CAPTION_DL_TOKENIZER")
    if not path:
        pytest.skip("CAPTION_DL_TOKENIZER not set")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    a = tok.encode(DL.ANSWER_TAG, add_special_tokens=False)
    c = tok.encode(DL.CROP_TAG, add_special_tokens=False)
    assert a[0] == c[0] and a != c and len(a) == 2 and len(c) == 3
    close = tok.convert_tokens_to_ids(DL.THINK_CLOSE)
    nl = tok.encode("\n", add_special_tokens=False)
    for text, tag in (("<think>\nStep 1 | commit\nx [original_image].\n</think>\n<answer>B</answer>", a),
                      ('<think>\nStep 1 | locate\nx [original_image].\n</think>\n<grounding>{"bbox_2d": [0.1, 0.1, '
                       '0.4, 0.4], "source": "original_image"}</grounding>', c)):
        ids = tok.encode(text, add_special_tokens=False)
        j = DL.find_boundary(ids, close, nl[0], a[0])
        assert j is not None and ids[j:j + len(tag)] == tag
        assert tok.decode(ids[:j]).endswith("</think>\n")
