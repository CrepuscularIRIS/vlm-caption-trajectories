"""CPU tests for the CounterCredit price channel (python test_counter_credit.py; exit 0 = pass)."""
import math

import counter_credit as CC


def test_score():
    ids = list(range(100, 110))
    top = {100: math.log(0.6), 102: math.log(0.2), 999: math.log(0.2)}
    assert abs(CC.score_from_logprobs(top, ids, "A") - 0.75) < 1e-9
    assert CC.score_from_logprobs(top, ids, "B") == 0.0
    assert CC.score_from_logprobs({999: 0.0}, ids, "A") is None


def test_values_and_price():
    v = CC.crop_values(0.30, 0.90, [0.40, 0.50, 0.60])
    assert abs(v["D"] - 0.60) < 1e-9 and abs(v["E"] - 0.40) < 1e-9 and abs(v["q"] - 0.40) < 1e-9
    bad = CC.crop_values(None, 0.9, [0.1])
    assert bad == {"eligible": False}
    waste = CC.crop_values(0.95, 0.96, [0.95])
    # correct, one verified (q=.40) + one waste: cashback .5*(.40-.05)=.175, rent .05
    p = CC.trajectory_price(True, 2, [v, waste], rho=0.05, gamma=0.5, eps=0.05, cap=0.6)
    assert p["n_verified"] == 1 and abs(p["price"] - (0.175 - 0.05)) < 1e-9
    # wrong trajectory: no verification, all crops pay rent
    p = CC.trajectory_price(False, 3, [v], rho=0.05, gamma=0.5, eps=0.05, cap=0.6)
    assert p["n_verified"] == 0 and abs(p["price"] + 0.15) < 1e-9
    # cap binds
    big = [CC.crop_values(0.0, 1.0, [0.0])] * 4
    p = CC.trajectory_price(True, 4, big, rho=0.05, gamma=0.5, eps=0.05, cap=0.6)
    assert abs(p["price"] - 0.6) < 1e-9
    # direct answer: zero price
    assert CC.trajectory_price(True, 0, [])["price"] == 0.0


def test_outcome_dominance():
    # worst correct (all crops unverified) must beat best wrong (no crops): 1 - rho*n_max > 0
    worst_correct = 1.0 + CC.trajectory_price(True, CC.N_MAX, [])["price"]
    best_wrong = 0.0 + CC.trajectory_price(False, 0, [])["price"]
    assert worst_correct > best_wrong
    assert abs(CC.CAP - (1 - CC.RHO * CC.N_MAX)) < 1e-12


def test_price_advantage():
    uids = ["a", "a", "a", "b", "b"]
    prices = [0.1, -0.05, -0.05, -0.1, -0.2]
    corrects = [1, 1, 0, 0, 0]
    adv = CC.price_advantage(uids, prices, corrects)
    assert abs(sum(adv[:3])) < 1e-12 and abs(adv[0] - 0.1) < 1e-12
    assert adv[3:] == [0.0, 0.0]  # all-wrong group: no price advantage
    # all-correct group keeps the signal (this is what card B's asymmetric DS preserves)
    adv = CC.price_advantage(["c"] * 3, [0.0, -0.05, -0.10], [1, 1, 1])
    assert abs(adv[0] - 0.05) < 1e-12 and abs(adv[2] + 0.05) < 1e-12


def test_gold():
    assert CC.gold_letter({"style": "rule", "ground_truth": "c"}) == "C"
    assert CC.gold_letter("AB") is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("all passed")
