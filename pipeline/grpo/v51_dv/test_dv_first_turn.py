"""CPU tests for the first-turn dV term."""
import json, os, tempfile
import torch
import dv_first_turn as D


def test_first_segment():
    assert D.first_segment([0, 0, 1, 1, 1, 0, 0, 1, 1]) == (2, 5)
    assert D.first_segment([1, 1, 0, 1]) == (0, 2)
    assert D.first_segment([0, 0, 0]) == (3, 3)


def test_sign():
    assert D.sign_of_first("grounding") == 1 and D.sign_of_first("answer") == -1
    assert D.sign_of_first("invalid") == 0 and D.sign_of_first(None) == 0


def test_delta_only_first_turn():
    mask = torch.tensor([[1, 1, 0, 0, 1, 1, 0], [1, 1, 1, 0, 0, 0, 0], [0, 1, 1, 0, 1, 0, 0]])
    d = D.first_turn_delta(mask, [0.5, 0.5, -0.25], ["grounding", "answer", "grounding"])
    assert d[0].tolist() == [0.5, 0.5, 0, 0, 0, 0, 0]          # crop first, dV>0: +dV on turn 1 only
    assert d[1].tolist() == [-0.5, -0.5, -0.5, 0, 0, 0, 0]     # direct first, dV>0: -dV
    assert d[2].tolist() == [0, -0.25, -0.25, 0, 0, 0, 0]      # crop first, dV<0: negative


def test_zero_for_invalid_and_zero_dv():
    mask = torch.tensor([[1, 1, 0], [1, 1, 0]])
    d = D.first_turn_delta(mask, [0.75, 0.0], ["invalid", "grounding"])
    assert d.abs().sum().item() == 0


def test_table_lookup_and_size_check():
    with tempfile.TemporaryDirectory() as t:
        p = os.path.join(t, "dv.json"); json.dump({"a": 0.5, "b": -0.25}, open(p, "w"))
        os.environ["CAPTION_DV_TABLE"] = p; os.environ["CAPTION_DV_EXPECT"] = "2"; D._TABLE = None; D.MISSES = 0
        assert D.lookup("a") == 0.5 and D.lookup("b") == -0.25
        assert D.lookup("val_uid") == 0.0 and D.MISSES == 1
        os.environ["CAPTION_DV_EXPECT"] = "3"; D._TABLE = None
        try:
            D.lookup("a"); raise AssertionError("size check missing")
        except ValueError:
            pass
        D._TABLE = None
