"""Persist the caption sampler's Python counter alongside each rank checkpoint."""
import json
import os
from functools import wraps
from pathlib import Path


def install(worker_class, generation_function, distributed=None):
    if getattr(worker_class, "_caption_resume_state_installed", False):
        return
    if distributed is None:
        import torch.distributed as distributed
    original_save = worker_class.save_checkpoint
    original_load = worker_class.load_checkpoint

    def state_path(path):
        return Path(path) / f"caption_sampler_rank_{distributed.get_rank()}.json"

    @wraps(original_save)
    def save(self, path, save_model_only=False):
        result = original_save(self, path, save_model_only=save_model_only)
        target = state_path(path)
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"version": 1, "seq": int(getattr(generation_function, "_seq", 0))}))
        os.replace(tmp, target)
        distributed.barrier()
        return result

    @wraps(original_load)
    def load(self, path):
        target = state_path(path)
        # Reviewed runs always save this file. An older checkpoint has no exact
        # caption sampling continuation and must not silently restart at seed 0.
        if not target.is_file():
            raise FileNotFoundError(f"Missing caption sampler state: {target}")
        state = json.loads(target.read_text())
        if state.get("version") != 1 or not isinstance(state.get("seq"), int) or state["seq"] < 0:
            raise ValueError(f"Invalid caption sampler state: {target}")
        result = original_load(self, path)
        generation_function._seq = state["seq"]
        return result

    worker_class.save_checkpoint = save
    worker_class.load_checkpoint = load
    worker_class._caption_resume_state_installed = True
