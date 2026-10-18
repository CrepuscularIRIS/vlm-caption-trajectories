from traj_utils import choose_perturbation, iou, rewrite_bbox, shift_box, stable_seed


def ident(b):
    return b


def test_iou_basics():
    assert iou([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0
    assert iou([0, 0, 0.5, 0.5], [0.5, 0.5, 1, 1]) == 0.0


def test_shift_stays_inside_and_keeps_size():
    b = shift_box([0.7, 0.7, 0.9, 0.9], 1, 1)
    assert all(0 <= v <= 1 for v in b)
    assert abs((b[2] - b[0]) - 0.2) < 1e-9


def test_perturbation_small_box_is_disjoint():
    p = choose_perturbation([0.4, 0.4, 0.5, 0.5], ident, 0)
    assert p["ok"] and p["iou_abs"] < 0.1


def test_perturbation_same_for_same_seed_dir():
    a = choose_perturbation([0.1, 0.2, 0.3, 0.4], ident, 3)
    b = choose_perturbation([0.1, 0.2, 0.3, 0.4], ident, 3)
    assert a == b


def test_large_box_may_fail_and_is_flagged():
    p = choose_perturbation([0.0, 0.0, 0.9, 0.9], ident, 0)
    assert p["ok"] is False


def test_seed_is_stable_and_31_bit():
    s = stable_seed(7, "q1", 2, 0)
    assert s == stable_seed(7, "q1", 2, 0) and 0 <= s < 2 ** 31
    assert s != stable_seed(7, "q1", 3, 0)


def test_rewrite_bbox_replaces_only_the_action_box():
    text = ('<think>\nStep 1 | scan\nSEE: x\nKNOW: y\nNEXT: z\n</think>\n'
            '<grounding>{"bbox_2d":[0.1,0.1,0.4,0.4],"source":"original_image"}</grounding>')
    out = rewrite_bbox(text, [0.4, 0.0, 0.7000000000000001, 0.3])
    assert out.endswith('<grounding>{"bbox_2d":[0.4,0.0,0.7,0.3],"source":"original_image"}</grounding>')
    assert out.split("<grounding>")[0] == text.split("<grounding>")[0]


def test_rewrite_bbox_without_action_returns_none():
    assert rewrite_bbox("<answer>B</answer>", [0, 0, 1, 1]) is None
