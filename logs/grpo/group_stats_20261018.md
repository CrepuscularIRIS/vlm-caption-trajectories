
## cc  (rollout steps 1-50, 600 groups: all_correct 160 (27%), mixed 412 (69%), all_wrong 28 (5%))
| group | type | rollouts | acc | mean adv | mean |adv| | mean tokens | sup. tokens | eff. tokens | share of Σ|adv|·tok |
|---|---|---|---|---|---|---|---|---|---|
| all_correct | direct | 484 | 1.00 | -0.009 | 0.025 | 51 | 24511 | 23467 | 0.0% |
| all_correct | crop | 2076 | 1.00 | +0.002 | 0.062 | 163 | 338431 | 330837 | 1.9% |
| mixed | direct | 1196 | 0.44 | -0.094 | 0.814 | 92 | 109730 | 109730 | 6.7% |
| mixed | crop | 5396 | 0.55 | +0.021 | 0.779 | 259 | 1398964 | 1398637 | 91.4% |
| all_wrong | direct | 159 | 0.00 | +0.000 | 0.000 | 106 | 16900 | 0 | 0.0% |
| all_wrong | crop | 289 | 0.00 | +0.000 | 0.000 | 244 | 70621 | 0 | 0.0% |

## cc_dsasym  (rollout steps 1-59, 708 groups: all_correct 160 (23%), mixed 508 (72%), all_wrong 40 (6%))
| group | type | rollouts | acc | mean adv | mean |adv| | mean tokens | sup. tokens | eff. tokens | share of Σ|adv|·tok |
|---|---|---|---|---|---|---|---|---|---|
| all_correct | direct | 697 | 1.00 | -0.005 | 0.024 | 54 | 37439 | 32782 | 0.1% |
| all_correct | crop | 1863 | 1.00 | +0.002 | 0.062 | 166 | 310117 | 303729 | 1.4% |
| mixed | direct | 2003 | 0.44 | -0.089 | 0.805 | 79 | 157252 | 157252 | 8.1% |
| mixed | crop | 6125 | 0.57 | +0.029 | 0.786 | 268 | 1644338 | 1644338 | 90.4% |
| all_wrong | direct | 183 | 0.00 | +0.000 | 0.000 | 111 | 20387 | 0 | 0.0% |
| all_wrong | crop | 457 | 0.00 | +0.000 | 0.000 | 267 | 122208 | 0 | 0.0% |

## cropcost  (rollout steps 1-56, 672 groups: all_correct 146 (22%), mixed 467 (69%), all_wrong 59 (9%))
| group | type | rollouts | acc | mean adv | mean |adv| | mean tokens | sup. tokens | eff. tokens | share of Σ|adv|·tok |
|---|---|---|---|---|---|---|---|---|---|
| all_correct | direct | 1898 | 1.00 | +0.206 | 0.206 | 48 | 91412 | 33000 | 2.5% |
| all_correct | crop | 438 | 1.00 | -0.892 | 0.935 | 153 | 66991 | 60481 | 9.5% |
| mixed | direct | 5556 | 0.46 | -0.052 | 0.762 | 68 | 378703 | 378703 | 40.9% |
| mixed | crop | 1916 | 0.63 | +0.149 | 0.756 | 219 | 419216 | 419216 | 47.1% |
| all_wrong | direct | 842 | 0.00 | +0.000 | 0.000 | 62 | 52613 | 0 | 0.0% |
| all_wrong | crop | 102 | 0.00 | +0.000 | 0.000 | 209 | 21297 | 0 | 0.0% |

## floor_l025  (rollout steps 1-165, 1980 groups: all_correct 594 (30%), mixed 1266 (64%), all_wrong 120 (6%))
| group | type | rollouts | acc | mean adv | mean |adv| | mean tokens | sup. tokens | eff. tokens | share of Σ|adv|·tok |
|---|---|---|---|---|---|---|---|---|---|
| all_correct | direct | 972 | 1.00 | +0.043 | 0.043 | 51 | 49958 | 42203 | 0.1% |
| all_correct | crop | 8532 | 1.00 | -0.005 | 0.015 | 154 | 1313643 | 670764 | 1.0% |
| mixed | direct | 2554 | 0.40 | -0.064 | 0.817 | 95 | 242462 | 242462 | 5.7% |
| mixed | crop | 17702 | 0.56 | +0.009 | 0.737 | 231 | 4088458 | 4087072 | 93.2% |
| all_wrong | direct | 257 | 0.00 | +0.000 | 0.000 | 92 | 23616 | 0 | 0.0% |
| all_wrong | crop | 1663 | 0.00 | +0.000 | 0.000 | 226 | 375899 | 0 | 0.0% |

## floor_l050  (rollout steps 1-165, 1980 groups: all_correct 579 (29%), mixed 1286 (65%), all_wrong 115 (6%))
| group | type | rollouts | acc | mean adv | mean |adv| | mean tokens | sup. tokens | eff. tokens | share of Σ|adv|·tok |
|---|---|---|---|---|---|---|---|---|---|
| all_correct | direct | 1495 | 1.00 | +0.075 | 0.075 | 52 | 78137 | 63865 | 0.2% |
| all_correct | crop | 7769 | 1.00 | -0.014 | 0.030 | 157 | 1218418 | 709437 | 1.6% |
| mixed | direct | 3819 | 0.42 | -0.066 | 0.819 | 98 | 372889 | 372889 | 9.2% |
| mixed | crop | 16757 | 0.57 | +0.015 | 0.732 | 226 | 3789364 | 3787526 | 88.9% |
| all_wrong | direct | 454 | 0.00 | +0.000 | 0.000 | 94 | 42791 | 0 | 0.0% |
| all_wrong | crop | 1386 | 0.00 | +0.000 | 0.000 | 218 | 301969 | 0 | 0.0% |
