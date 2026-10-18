from traj_metrics import recovery, transition


def test_transition_kinds():
    assert transition([0, 0, 100, 100], [10, 10, 50, 50]) == "zoom_in"
    assert transition([10, 10, 50, 50], [0, 0, 100, 100]) == "widen"
    assert transition([0, 0, 50, 50], [60, 60, 110, 110]) == "shift"
    assert transition([0, 0, 50, 50], [2, 2, 52, 52]) == "stay"


def _rec(boxes, ok):
    turns = [{"parsed": {"kind": "grounding", "source": "original_image"}, "box_abs": b} for b in boxes[1:]]
    return {"perturb": {"asked_bbox": [0.0, 0.0, 0.5, 0.5], "shown_bbox": [0.5, 0.5, 1.0, 1.0], "ok": True},
            "image_wh": [100, 100], "boxes_abs": boxes, "ok": ok, "n_crops": len(boxes) - 1,
            "turns": turns + [{"parsed": {"kind": "answer"}}]}


def test_recovery_back_to_target_and_false_confirm():
    full = [0, 0, 100, 100]
    back = recovery(_rec([full, [50, 50, 100, 100], [0, 0, 50, 50]], 1), None)
    assert back["recrop"] == 1 and back["back_to_target"] == 1 and back["false_confirm"] == 0
    fooled = recovery(_rec([full, [50, 50, 100, 100]], 0), None)
    assert fooled["recrop"] == 0 and fooled["false_confirm"] == 1
