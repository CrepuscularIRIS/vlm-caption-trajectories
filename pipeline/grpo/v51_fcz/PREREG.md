# Pre-registration: zero-credit control arm for t-fourcase-rl (written 2026-10-22 03:21:18 -0400, before any result of either arm)

Why outside the navigator: the study is at run_test 023 (t-fourcase-rl, ~8 h); the navigator accepts no new plan until
023 is interpreted. User 10-22: GPU 0/1 freed, "可以直接开始训练". This file fixes the control's design and predictions
now; after 023's interpret it is entered as a collection-only plan_test (pattern of t-rloo-control-2) that copies these
predictions verbatim.

Arm zero: identical to fc (same recipe, seed 42, data order, forks m=4 / m0=16, same rewrite condition) except crop turns
that fc rewrites get -lambda instead of v(after) - v(before) - lambda. Isolates "outcome credit removed from crop turns"
from "before/after information". Launched on GPU 0,1; code hashes in hashes_at_launch.txt.

Predictions (E1, step 160, paired item bootstrap 2,000, seed 0):
- H (before/after information matters): fc - zero accuracy CI lower bound > 0, OR fc - zero crops/q CI excludes 0 while
  fc - zero accuracy CI lower bound >= -0.03.
- Rival (only the removal of crop-turn outcome credit matters): fc - zero accuracy CI includes 0 AND fc - zero crops/q
  CI includes 0.
- Descriptive: zero vs r1 (crops, accuracy), E2, step 80, training windows.
Stop rule: one run, final; no retuning. Crash = execution failure.
