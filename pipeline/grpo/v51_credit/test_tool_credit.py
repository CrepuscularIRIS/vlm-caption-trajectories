"""CPU tests for the four-case tool credit."""
import os
import torch
import tool_credit as T


def test_case_of():
    assert T.case_of("A", "B", "B") == "WR" and T.case_of("B", "B", "B") == "RR"
    assert T.case_of("A", "C", "B") == "WW" and T.case_of("B", "A", "B") == "RW"
    assert T.case_of(None, "B", "B") is None and T.case_of("A", "B", None) is None


def test_turn_value_mentor_rules():
    for a in (1.5, -0.7, 0.0):
        assert T.turn_value("WR", a) == max(a, 0) and T.turn_value("WW", a) == max(a, 0)
        assert T.turn_value("RR", a) == min(a, 0) and T.turn_value("RW", a) == min(a, 0)
        assert T.turn_value(None, a) == a


def test_segments():
    assert T.segments([1, 1, 0, 0, 1, 0, 1, 1]) == [(0, 2), (4, 5), (6, 8)]
    assert T.segments([0, 0]) == [] and T.segments([1]) == [(0, 1)]


def test_apply_credit_changes_only_crop_turns():
    # rollout 0: two crop turns + answer turn, correct (A=+1): crop1 RR -> 0, crop2 WR -> +1, answer keeps +1
    # rollout 1: one crop turn + answer, wrong (A=-1): crop1 WW -> 0, answer keeps -1
    mask = torch.tensor([[1, 1, 0, 1, 0, 1, 1], [1, 1, 0, 1, 1, 0, 0]])
    adv = torch.tensor([[1.0] * 7, [-1.0] * 7]) * mask
    out, changed = T.apply_credit(adv, mask, [["RR", "WR"], ["WW"]])
    assert out[0].tolist() == [0, 0, 0, 1, 0, 1, 1]
    assert out[1].tolist() == [0, 0, 0, -1, -1, 0, 0]
    assert changed == 4


def test_apply_credit_misleading_and_missing():
    mask = torch.tensor([[1, 1, 0, 1], [1, 0, 1, 1]])
    adv = torch.tensor([[2.0, 2.0, 0, 2.0], [-0.5, 0, -0.5, -0.5]])
    out, _ = T.apply_credit(adv, mask, [["RW"], [None]])
    assert out[0].tolist() == [0, 0, 0, 2.0]          # misleading crop on a recovered correct rollout: no credit
    assert out[1].tolist() == [-0.5, 0, -0.5, -0.5]   # missing read: unchanged
    out2, _ = T.apply_credit(adv, mask, [[], []])
    assert torch.equal(out2, adv)


def test_parse_top_and_messages():
    top = [{"token": " C", "logprob": -0.1}, {"token": "C", "logprob": -2.0}, {"token": " (", "logprob": -3.0},
           {"token": "A", "logprob": -1.5}]
    a, c = T.parse_top(top, ["A", "B", "C"])
    assert a == "C" and 0.5 < c < 1.0
    assert T.parse_top([{"token": " (", "logprob": -0.1}], ["A"]) == (None, 0.0)
    from PIL import Image
    msgs = T.reader_messages("Q?\n(A) x\n(B) y", [Image.new("RGB", (8, 8)), Image.new("RGB", (4, 4))])
    assert msgs[-1] == {"role": "assistant", "content": "<answer>("}
    kinds = [c["type"] for c in msgs[0]["content"]]
    assert kinds == ["text", "image_url", "text", "image_url", "text"]


def test_rule_env():
    os.environ["CAPTION_CREDIT_RULE"] = "bad"
    try:
        T.rule(); raise AssertionError("rule check missing")
    except ValueError:
        pass
    finally:
        os.environ.pop("CAPTION_CREDIT_RULE")
    assert T.rule() == "mentor4"


def test_question_and_gold():
    assert T.gold_letter({"ground_truth": " b "}) == "B" and T.gold_letter("AB") is None
    assert T.question_text({"question": "Q"}) == "Q" and T.question_text("{'question': 'Q2'}") == "Q2"
    assert T.question_text(None) is None
