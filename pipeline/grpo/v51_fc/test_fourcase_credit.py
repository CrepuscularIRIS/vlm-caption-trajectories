"""Unit tests for fourcase_credit.py (CPU)."""
import pytest
import torch

import fourcase_credit as FCC

LAM = 0.01


def _batch():
    # rollout 0: turn0 crop (L 0-3), receipt (3-5 masked), turn1 crop (5-7), receipt, turn2 answer (9-10)
    # rollout 1: turn0 answer only
    mask = torch.tensor([[1, 1, 1, 0, 0, 1, 1, 0, 0, 1], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
    adv = torch.tensor([[0.5] * 10, [-0.3] * 10])
    return adv, mask


@pytest.mark.unit
@pytest.mark.parametrize("vb,va,case,c", [(0.0, 1.0, "WR", 1 - LAM), (1.0, 1.0, "RR", -LAM),
                                          (0.0, 0.0, "WW", -LAM), (1.0, 0.0, "RW", -1 - LAM)])
def test_mentor_table_binary(vb, va, case, c):
    assert FCC.case_label(vb, va) == case
    assert FCC.crop_credit(vb, va, LAM) == pytest.approx(c)


@pytest.mark.unit
def test_after_label_changes_output():
    # the two-cell bug of tool_credit.py: flipping the after label must change the credit
    assert FCC.crop_credit(1.0, 1.0, LAM) != FCC.crop_credit(1.0, 0.0, LAM)
    assert FCC.crop_credit(0.0, 1.0, LAM) != FCC.crop_credit(0.0, 0.0, LAM)


@pytest.mark.unit
def test_apply_rewrites_only_crop_turns():
    adv, mask = _batch()
    v = [[0.25, 0.75, 0.75, None], [0.5, None]]
    kinds = [["grounding", "grounding", "answer"], ["answer"]]
    new, changed, st = FCC.apply_fourcase(adv, mask, v, kinds, LAM, "fourcase")
    assert new[0, 0:3].tolist() == pytest.approx([0.5 - LAM] * 3)       # 0.75 - 0.25 - lam
    assert new[0, 5:7].tolist() == pytest.approx([-LAM] * 2)            # 0.75 - 0.75 - lam
    assert new[0, 9].item() == pytest.approx(0.5)                       # answer turn keeps A
    assert new[0, 3:5].tolist() == pytest.approx([0.5, 0.5])            # masked tokens untouched
    assert torch.equal(new[1], adv[1])                                  # no-crop rollout untouched
    assert changed == 5 and st["credited"] == 2 and st["WR"] == 1 and st["RR"] == 1


@pytest.mark.unit
def test_missing_value_keeps_advantage():
    adv, mask = _batch()
    v = [[0.25, None, 0.75, None], [0.5, None]]
    kinds = [["grounding", "grounding", "answer"], ["answer"]]
    new, changed, st = FCC.apply_fourcase(adv, mask, v, kinds, LAM, "fourcase")
    assert torch.equal(new[0, 0:7], adv[0, 0:7]) and changed == 0 and st["crop_turns"] == 2


@pytest.mark.unit
def test_rloo_mode_is_identity_and_input_not_mutated():
    adv, mask = _batch()
    ref = adv.clone()
    new, changed, _ = FCC.apply_fourcase(adv, mask, [[0.0, 1.0, 1.0, None], [0.5, None]],
                                         [["grounding", "grounding", "answer"], ["answer"]], LAM, "rloo")
    assert torch.equal(new, ref) and changed == 0
    FCC.apply_fourcase(adv, mask, [[0.0, 1.0, 1.0, None], [0.5, None]],
                       [["grounding", "grounding", "answer"], ["answer"]], LAM, "fourcase")
    assert torch.equal(adv, ref)


@pytest.mark.unit
def test_bad_mode(monkeypatch):
    monkeypatch.setenv("CAPTION_FC_MODE", "x")
    with pytest.raises(ValueError):
        FCC.mode()
