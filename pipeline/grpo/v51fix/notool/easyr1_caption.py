"""No-op caption patch for plain single-turn EasyR1 (B6: GRPO without tool).

The server's EasyR1 verl/__init__.py always imports easyr1_caption from CAPTION_GRPO_CODE and calls install();
pointing CAPTION_GRPO_CODE here keeps the stock EasyR1 rollout untouched.
"""
import types

proto = types.SimpleNamespace(__name__="none (stock single-turn EasyR1, no tool)")


def install() -> None:
    import sys
    from pathlib import Path
    parent = str(Path(__file__).resolve().parents[1])
    if parent not in sys.path:
        sys.path.append(parent)
    from training_ledger import install_direct_logger
    from verl.workers.reward.function import AutoRewardManager
    install_direct_logger(AutoRewardManager)
