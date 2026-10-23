"""A5 (corpus ROADMAP §1, PC-F2-20): strict potential-based shaping F_t = Phi(s_{t+1}) - Phi(s_t) (gamma = 1) added to
a trajectory-level RLOO advantage. Checks the algebra numerically on random groups:
(a) Phi(terminal) = 0: the shaping sum is -Phi(s_0), identical for every sample of a prompt, so the leave-one-out
    advantage is unchanged bit for bit (up to float error);
(b) Phi(terminal) != 0 (e.g. a potential of the evidence state at the answer): it does not cancel, i.e. it is a
    terminal-state reward and changes the objective (the 'non-strict' path);
(c) per-turn return-to-go: the shaped return from turn t is R + Phi(s_T) - Phi(s_t), so with Phi(s_T) = 0 the
    shaping acts as a state baseline -Phi(s_t).
Run: python3 -m pytest -q test_pbrs_rloo.py
"""
import random


def loo(xs):
    s, n = sum(xs), len(xs)
    return [x - (s - x) / (n - 1) for x in xs]


def group(rng, terminal_zero):
    out = []
    phi0 = rng.random()  # same prompt -> same start state
    for _ in range(16):
        turns = rng.randint(1, 7)
        phis = [phi0] + [rng.random() for _ in range(turns)]
        if terminal_zero:
            phis[-1] = 0.0
        shaping = sum(phis[t + 1] - phis[t] for t in range(turns))
        out.append((rng.choice([0.0, 1.0]) * (1 - 0.01 * turns), shaping, phis))
    return out


def test_strict_pbrs_cancels_under_rloo():
    rng = random.Random(0)
    for _ in range(1000):
        g = group(rng, terminal_zero=True)
        a = loo([r for r, _, _ in g])
        b = loo([r + f for r, f, _ in g])
        assert max(abs(x - y) for x, y in zip(a, b)) < 1e-12


def test_nonzero_terminal_potential_changes_advantage():
    rng = random.Random(1)
    g = group(rng, terminal_zero=False)
    a = loo([r for r, _, _ in g])
    b = loo([r + f for r, f, _ in g])
    assert max(abs(x - y) for x, y in zip(a, b)) > 1e-3


def test_return_to_go_is_state_baseline():
    rng = random.Random(2)
    r, _, phis = group(rng, terminal_zero=True)[0]
    T = len(phis) - 1
    for t in range(T):
        shaped = r + sum(phis[k + 1] - phis[k] for k in range(t, T))
        assert abs(shaped - (r - phis[t])) < 1e-12
