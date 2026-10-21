"""Unit tests for decision-state baselines (db_credit.py), incl. the coherence toy run through the PRODUCTION advantage
function: a 3-level crop tree where commit is possible at every state and pi(crop) varies by state, so the true
gradient is nonzero everywhere. DB and PERM must match the exact gradient within 3 Monte-Carlo SE at every state; the
crop-turn-only variant (attempt 1's rule) must show its closed-form bias -reach p(1-p)(Q_commit - J)."""
import math
import random

import torch

import db_credit as D

LAM, G, M = 0.01, 16, 2
P = {'s1': .9, 's2g': .4, 's2b': .8, 's3gg': .3, 's3gb': .7, 's3bg': .3, 's3bb': .7}   # pi(crop | s)
V = {'s1': .5, 's2g': .9, 's2b': .2, 's3gg': .95, 's3gb': .4, 's3bg': .85, 's3bb': .1}  # P(correct | commit at s)
GOOD = {'s1': .6, 's2g': .7, 's2b': .5}
Q4 = {'s3gg': .9, 's3gb': .5, 's3bg': .8, 's3bb': .15}
KIDS = {'s1': ('s2g', 's2b'), 's2g': ('s3gg', 's3gb'), 's2b': ('s3bg', 's3bb')}
ST = list(P)


def k_of(s):
    return 1 if s == 's1' else (2 if len(s) == 3 else 3)


def rollout(rng):
    s, tr = 's1', []
    while True:
        k = k_of(s)
        if rng.random() >= P[s]:
            tr.append((s, 'commit'))
            return tr, (rng.random() < V[s]) * (1 - LAM * (k - 1))
        tr.append((s, 'crop'))
        if s in KIDS:
            s = KIDS[s][0] if rng.random() < GOOD[s] else KIDS[s][1]
        else:
            tr.append(('f' + s, 'forced'))
            return tr, (rng.random() < Q4[s]) * (1 - LAM * 3)


def exact():
    def qc(s): return V[s] * (1 - LAM * (k_of(s) - 1))
    def qcrop(s):
        if s in KIDS:
            a, b = KIDS[s]
            return GOOD[s] * j(a) + (1 - GOOD[s]) * j(b)
        return Q4[s] * (1 - LAM * 3)
    def j(s): return P[s] * qcrop(s) + (1 - P[s]) * qc(s)
    reach = {'s1': 1.0}
    for s in ['s1', 's2g', 's2b']:
        a, b = KIDS[s]
        reach[a] = reach[s] * P[s] * GOOD[s]
        reach[b] = reach[s] * P[s] * (1 - GOOD[s])
    true = {s: reach[s] * P[s] * (1 - P[s]) * (qcrop(s) - qc(s)) for s in ST}
    return true, reach, qc, j('s1')


def batch(grp, rng):
    """Group -> (adv, mask, rewards, b_lists, groups, turn index map). Each turn is one token followed by a mask gap."""
    L = max(2 * len(tr) for tr, _ in grp)
    mask = torch.zeros(len(grp), L, dtype=torch.long)
    rewards = [r for _, r in grp]
    S = sum(rewards)
    adv = torch.zeros(len(grp), L)
    b_lists = []
    for i, (tr, r) in enumerate(grp):
        A = r - (S - r) / (G - 1)
        bl = []
        for t, (s, a) in enumerate(tr):
            mask[i, 2 * t] = 1
            adv[i, 2 * t] = A
            if s in P and t >= 1:
                k = k_of(s)
                bl.append(D.fork_value([int(rng.random() < V[s]) for _ in range(M)], k, LAM))
            else:
                bl.append(None)
        b_lists.append(bl)
    return adv, mask, rewards, b_lists, ["g"] * len(grp)


def simulate(n_groups=6000, seed=1):
    rng = random.Random(seed)
    arms = ("rloo", "db", "perm", "croponly")
    g = {a: {s: 0.0 for s in ST} for a in arms}
    g2 = {a: {s: 0.0 for s in ST} for a in arms}
    for _ in range(n_groups):
        grp = [rollout(rng) for _ in range(G)]
        adv, mask, rewards, b_lists, groups = batch(grp, rng)  # one item per group: PERM is within-item
        outs = {"rloo": adv, "db": D.apply_db(adv, mask, rewards, b_lists, groups, "db")[0],
                "perm": D.apply_db(adv, mask, rewards, b_lists, groups, "perm", rng)[0]}
        kinds = [["grounding" if a == "crop" else "answer" for _, a in tr] for tr, _ in grp]
        outs["croponly"] = D.apply_db(adv, mask, rewards, b_lists, groups, "croponly", None, kinds)[0]
        for i, (tr, _) in enumerate(grp):
            for t, (s, a) in enumerate(tr):
                if s not in P:
                    continue
                sc = (1 - P[s]) if a == 'crop' else -P[s]
                for arm in arms:
                    v = sc * float(outs[arm][i, 2 * t])
                    g[arm][s] += v
                    g2[arm][s] += v * v
    n = n_groups * G
    return {a: {s: g[a][s] / n for s in ST} for a in arms}, {a: {s: math.sqrt(g2[a][s] / n / n) for s in ST} for a in arms}


def test_segments_and_fork_value():
    assert D.segments([1, 1, 0, 1, 0, 1, 1]) == [(0, 2), (3, 4), (5, 7)]
    assert D.fork_value([], 2, .01) is None
    assert abs(D.fork_value([1, 0], 3, .01) - 0.98 * 0.5) < 1e-12


def test_apply_db_rewrites_followup_turns_only():
    mask = torch.tensor([[1, 1, 0, 1, 1, 0, 1]])
    adv = torch.tensor([[0.5, 0.5, 0, 0.5, 0.5, 0, 0.5]])
    out, changed = D.apply_db(adv, mask, [1.0], [[None, 0.25, 0.9]], ["g"], "db")
    assert torch.allclose(out[0], torch.tensor([0.5, 0.5, 0, 0.75, 0.75, 0, 0.1]))
    assert changed == 3
    same, c0 = D.apply_db(adv, mask, [1.0], [[None, 0.25, 0.9]], ["g"], "rloo")
    assert torch.equal(same, adv) and c0 == 0


def test_perm_draws_same_item_same_index():
    rng = random.Random(0)
    bl = [[None, 0.1], [None, 0.7], [None, 0.9], [None, None, None]]
    groups = ["g1", "g1", "g2", "g2"]
    pb = D.perm_baselines(bl, groups, [1.0, 0.0, 1.0, 0.5], rng)
    assert pb[0][1] == 0.7 and pb[1][1] == 0.1          # within item g1 only
    assert pb[2][1] == 0.5                                # g2: no other b at index 1 -> LOO mean return of reachers
    assert pb[3] == [None, None, None]


def test_toy_unbiased_db_perm_and_croponly_bias():
    true, reach, qc, J = exact()
    mean, se = simulate()
    for s in ST:
        for arm in ("db", "perm", "rloo"):
            assert abs(mean[arm][s] - true[s]) <= 3 * se[arm][s] + 1e-4, (arm, s, mean[arm][s], true[s], se[arm][s])
    for s in ST[1:]:
        pred = -reach[s] * P[s] * (1 - P[s]) * (qc(s) - J)
        meas = mean["croponly"][s] - true[s]
        assert abs(meas - pred) <= 3 * se["croponly"][s] + 1e-4, (s, meas, pred)
    assert abs(mean["croponly"]["s2g"] - true["s2g"]) > 0.02  # the shaping term is visible where it is large
