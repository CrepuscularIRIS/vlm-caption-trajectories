"""CPU checks for gigpo.step_advantages (python test_gigpo.py)."""
import torch
import gigpo

uids = ["q"] * 4 + ["p"] * 2
root = (1, (), 0, 0)
a1 = (2, ((0, 0, 10, 10),), 1, 1)
anchors = [[root, a1], [root, a1], [root, (2, ((5, 5, 9, 9),), 1, 1)], [root], [root], [root]]
scores = [1.0, 0.0, 1.0, 1.0, 0.0, 0.0]
out, st = gigpo.step_advantages(uids, scores, anchors)
r = torch.tensor([1.0, 0.0, 1.0, 1.0], dtype=torch.float64)
ref = ((r - r.mean()) / (r.std() + 1e-6)).tolist()
assert [o[0] for o in out[:4]] == ref                        # root group = whole question group
assert abs(out[0][1] - 0.7071) < 1e-3 and abs(out[1][1] + 0.7071) < 1e-3  # shared step-2 anchor, R 1 vs 0
assert out[2][1] == 0.0                                       # singleton anchor -> 0
assert out[4] == [0.0] and out[5] == [0.0]                    # all-equal group -> 0 (r - mean = 0)
assert st["root_multi"] == 2 and st["nonroot_multi"] == 1 and st["nonroot_groups"] == 2
print("gigpo ok", st)
