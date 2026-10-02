"""LLaMA-Factory preprocessing-only check for an exported caption dataset (no model weights, no training).

Run with the LLaMA-Factory env:  /data/icml-envs/llamafactory/bin/python lf_preprocess_check.py DATASET_DIR NAME
Tokenizes the sharegpt export with the qwen2_vl template and checks, per sample: assistant turns are the
only label targets (each contains '<think>' and an action tag), receipts/system/question are masked, image
placeholders match the image count, and sequence length vs cutoff. Writes DATASET_DIR/LF_CHECK.json.
"""
import json
import sys
from pathlib import Path

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.hparams import get_train_args
from llamafactory.model import load_tokenizer

MODEL = '/data/models/Qwen2.5-VL-7B-Instruct'
CUTOFF = 16384


def main() -> None:
    ddir, name = Path(sys.argv[1]), sys.argv[2]
    args = dict(model_name_or_path=MODEL, stage='sft', do_train=True, dataset=name, dataset_dir=str(ddir),
                template='qwen2_vl', cutoff_len=CUTOFF, output_dir=str(ddir / '_lf_tmp'), overwrite_cache=True,
                preprocessing_num_workers=1, per_device_train_batch_size=1, report_to='none',
                finetuning_type='lora')
    model_args, data_args, training_args, finetuning_args, _ = get_train_args(args)
    tm = load_tokenizer(model_args)
    tok = tm['tokenizer']
    template = get_template_and_fix_tokenizer(tok, data_args)
    ds = get_dataset(template, model_args, data_args, training_args, stage='sft', **tm)['train_dataset']
    raw = json.loads((ddir / f'{name}.json').read_text())
    rows = []
    for i, ex in enumerate(ds):
        ids, labels = ex['input_ids'], ex['labels']
        target = tok.decode([t for t, l in zip(ids, labels) if l != -100])
        n_gpt = sum(c['from'] == 'gpt' for c in raw[i]['conversations'])
        rows.append(dict(
            idx=i, tokens=len(ids), truncated=len(ids) >= CUTOFF, target_tokens=sum(l != -100 for l in labels),
            gpt_turns=n_gpt, think_in_target=target.count('<think>'), actions_in_target=target.count('<grounding>')
            + target.count('<answer>'), receipt_leak='Result of Step' in target or 'Question:' in target,
            images=len(raw[i]['images']), vision_blocks=tok.decode(ids).count('<|vision_start|>')))
    ok = [r for r in rows if r['think_in_target'] == r['gpt_turns'] and r['actions_in_target'] == r['gpt_turns']
          and not r['receipt_leak'] and r['images'] == r['vision_blocks'] and not r['truncated']]
    summ = dict(samples=len(rows), passed=len(ok), max_tokens=max(r['tokens'] for r in rows),
                mean_tokens=round(sum(r['tokens'] for r in rows) / len(rows)), cutoff=CUTOFF, rows=rows)
    (ddir / 'LF_CHECK.json').write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: v for k, v in summ.items() if k != 'rows'}, indent=1))
    for r in rows:
        if r not in ok:
            print('FAIL', r)


if __name__ == '__main__':
    main()
