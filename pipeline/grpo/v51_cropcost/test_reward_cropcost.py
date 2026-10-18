"""CPU unit tests for the crop-cost reward: correct * (1.0 - 0.01 * crops).

Run: python test_reward_cropcost.py  (exit 0 = all pass)
"""
import sys

from reward_caption import compute_score, correct_one, score_one

DIRECT = "The chart shows 2.1 for Qtr2. <answer>A</answer>"
WRONG = "Maybe it is B. <answer>B</answer>"
TWO_ANSWERS = "<answer>A</answer> ... <answer>B</answer>"


def close(a, b):
    return abs(a - b) < 1e-9


def main() -> int:
    cases = [
        ("direct correct, 0 crops -> 1.0", score_one("caption", DIRECT, "A", {"crops": 0}), 1.0),
        ("correct with 3 crops -> 0.97", score_one("caption", DIRECT, "A", {"crops": 3}), 0.97),
        ("correct with 8 crops -> 0.92", score_one("caption", DIRECT, "A", {"crops": 8}), 0.92),
        ("wrong answer, 0 crops -> 0", score_one("caption", WRONG, "A", {"crops": 0}), 0.0),
        ("wrong answer, 8 crops -> 0 (no cost on wrong)", score_one("caption", WRONG, "A", {"crops": 8}), 0.0),
        ("no extra_info -> undiscounted 1.0", score_one("caption", DIRECT, "A"), 1.0),
        ("extra_info without crops -> 1.0", score_one("caption", DIRECT, "A", {}), 1.0),
        ("terminal!=answer path (empty text) -> 0", score_one("caption", "", "A", {"crops": 0}), 0.0),
        ("multiple <answer> -> 0", score_one("caption", TWO_ANSWERS, "A", {"crops": 0}), 0.0),
        ("non-letter gold -> 0", score_one("caption", DIRECT, "42", {"crops": 0}), 0.0),
    ]
    failed = 0
    for name, got, want in cases:
        ok = close(got, want)
        print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got}, want {want}")
        failed += 0 if ok else 1

    # compute_score must forward the per-item crops field.
    scores = compute_score([
        {"response": DIRECT, "ground_truth": "A", "crops": 3},
        {"response": DIRECT, "ground_truth": "A"},
        {"response": WRONG, "ground_truth": "A", "crops": 5},
    ])
    for got, want, name in zip(scores, [0.97, 1.0, 0.0], ["batch crops=3", "batch no crops", "batch wrong"]):
        ok = close(got, want)
        print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got}, want {want}")
        failed += 0 if ok else 1

    # correct_one stays the undiscounted terminal-correctness signal.
    ok = correct_one("caption", DIRECT, "A") == 1.0 and correct_one("caption", WRONG, "A") == 0.0
    print(f"{'PASS' if ok else 'FAIL'}  correct_one undiscounted")
    failed += 0 if ok else 1

    print(f"\n{len(cases) + 4 - failed}/{len(cases) + 4} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
