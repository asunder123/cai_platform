
#!/usr/bin/env python3
"""
Balance a multilabel text-classification dataset with spaCy-style `cats` labels.

Input format (either .jsonl or .json array):
{
  "text": "...",
  "cats": {"WORKFLOW_DEVIATION": 1, "EVIDENCE_CONFIDENCE": 0, ...}
}

Features
- Supports JSONL and JSON array inputs
- Computes label distribution and reports imbalance
- Balancing modes per label: undersample, oversample, or hybrid (minority up, majority down)
- Targeting strategies: align to minority, majority, or a fixed target per label
- Optional class weights computation for training
- Optional deterministic stratified split (train/val/test) using multilabel hashing buckets
- Preserves neutral language and text; shuffles deterministically with --seed

Usage
------
python balance_dataset.py \
  --input data.jsonl \
  --output balanced.jsonl \
  --mode hybrid \
  --target minority \
  --min-count 20 \
  --split 0.8 0.1 0.1 \
  --export-weights weights.json

Common targets
- minority: bring each label to the count of its minority class
- majority: bring each label to the count of its majority class
- fixed: specify --fixed-target 500 (applies to each label)

Notes
- Oversampling is done by randomized with-replacement sampling
- Undersampling is done by randomized without-replacement sampling
- Hybrid: undersample majority toward target and oversample minority toward target
- Multilabel conflicts are handled iteratively and converged with a cap on iterations
"""

import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


def load_data(path: str) -> List[dict]:
    ext = os.path.splitext(path)[1].lower()
    data = []
    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                raise ValueError("Expected a JSON array for .json input")
    else:
        raise ValueError("Input must be .jsonl or .json")
    return data


def save_data(path: str, items: List[dict]):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        with open(path, "w", encoding="utf-8") as f:
            for obj in items:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    elif ext == ".json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
    else:
        raise ValueError("Output must be .jsonl or .json")


def label_counts(data: List[dict]) -> Dict[str, Counter]:
    counts = defaultdict(Counter)
    for ex in data:
        cats = ex.get("cats", {})
        for k, v in cats.items():
            counts[k][int(v)] += 1
    return counts


def print_report(counts: Dict[str, Counter], prefix: str = "Current"):
    total = None
    try:
        # estimate total from any label
        any_label = next(iter(counts))
        total = sum(counts[any_label].values())
    except StopIteration:
        total = 0
    print(f"\n{prefix} distribution (N≈{total}):")
    print("label, zero_count, one_count, one_ratio")
    for lbl in sorted(counts.keys()):
        zeros = counts[lbl].get(0, 0)
        ones = counts[lbl].get(1, 0)
        ratio = ones / (zeros + ones) if (zeros + ones) else 0.0
        print(f"{lbl}, {zeros}, {ones}, {ratio:.3f}")


def compute_class_weights(counts: Dict[str, Counter]) -> Dict[str, Dict[int, float]]:
    weights: Dict[str, Dict[int, float]] = {}
    for lbl, c in counts.items():
        zeros = c.get(0, 0)
        ones = c.get(1, 0)
        total = zeros + ones
        # Inverse frequency style weights; add 1 to avoid div-by-zero
        w0 = total / (2 * (zeros if zeros > 0 else 1))
        w1 = total / (2 * (ones if ones > 0 else 1))
        weights[lbl] = {0: round(w0, 6), 1: round(w1, 6)}
    return weights


def decide_targets(counts: Dict[str, Counter], target: str, fixed_target: int | None) -> Dict[str, int]:
    tgt = {}
    for lbl, c in counts.items():
        z, o = c.get(0, 0), c.get(1, 0)
        if target == "minority":
            tgt[lbl] = min(z, o)
        elif target == "majority":
            tgt[lbl] = max(z, o)
        elif target == "fixed":
            if not fixed_target:
                raise ValueError("--fixed-target required when --target=fixed")
            tgt[lbl] = fixed_target
        else:
            raise ValueError("Unknown target: " + target)
    return tgt


def multilabel_hash_bucket(cats: Dict[str, int], buckets: int = 1000) -> int:
    # Deterministic bucket for pseudo-stratified splits
    key = tuple(sorted(cats.items()))
    return hash(key) % buckets


def stratified_split(data: List[dict], splits: Tuple[float, float, float], seed: int) -> Tuple[List[dict], List[dict], List[dict]]:
    assert math.isclose(sum(splits), 1.0, abs_tol=1e-6), "Splits must sum to 1.0"
    random.seed(seed)
    # Bucket by multilabel signature for coarse stratification
    buckets = defaultdict(list)
    for ex in data:
        bucket = multilabel_hash_bucket({k: int(v) for k, v in ex.get("cats", {}).items()})
        buckets[bucket].append(ex)
    train, val, test = [], [], []
    for _, group in buckets.items():
        random.shuffle(group)
        n = len(group)
        n_train = int(n * splits[0])
        n_val = int(n * splits[1])
        train.extend(group[:n_train])
        val.extend(group[n_train:n_train+n_val])
        test.extend(group[n_train+n_val:])
    return train, val, test


def balance_once(data: List[dict], label: str, target_count: int, mode: str, seed: int) -> List[dict]:
    """Balance a single label to target_count per class according to mode.
    mode in {undersample, oversample, hybrid}
    """
    random.seed(seed)
    zeros = [ex for ex in data if int(ex["cats"].get(label, 0)) == 0]
    ones = [ex for ex in data if int(ex["cats"].get(label, 0)) == 1]

    def undersample(group: List[dict], k: int) -> List[dict]:
        if k >= len(group):
            return group
        return random.sample(group, k)

    def oversample(group: List[dict], k: int) -> List[dict]:
        if len(group) == 0:
            return []
        return [random.choice(group) for _ in range(k)]

    z, o = len(zeros), len(ones)
    # Determine desired per-class counts
    if mode == "undersample":
        new_z = min(z, target_count)
        new_o = min(o, target_count)
        zeros_new = undersample(zeros, new_z)
        ones_new = undersample(ones, new_o)
    elif mode == "oversample":
        new_z = max(z, target_count)
        new_o = max(o, target_count)
        zeros_new = zeros + oversample(zeros, new_z - z)
        ones_new = ones + oversample(ones, new_o - o)
    elif mode == "hybrid":
        # Bring both classes toward target from both sides
        new_z = target_count
        new_o = target_count
        zeros_new = (undersample(zeros, min(z, target_count)) if z > target_count else zeros + oversample(zeros, target_count - z))
        ones_new  = (undersample(ones,  min(o, target_count)) if o > target_count else ones  + oversample(ones,  target_count - o))
    else:
        raise ValueError("Unknown mode: " + mode)

    balanced = zeros_new + ones_new
    random.shuffle(balanced)
    return balanced


def iterative_multilabel_balance(data: List[dict], mode: str, targets: Dict[str, int], seed: int, max_passes: int = 8) -> List[dict]:
    """Iteratively balance each label; repeat passes to reduce cross-label drift."""
    random.seed(seed)
    cur = data[:]
    for p in range(max_passes):
        changed = False
        for lbl, tgt in targets.items():
            before = label_counts(cur)[lbl]
            cur = balance_once(cur, lbl, tgt, mode, seed + p)
            after = label_counts(cur)[lbl]
            if before != after:
                changed = True
        if not changed:
            break
    random.shuffle(cur)
    return cur


def ensure_min_count_per_class(data: List[dict], min_count: int, seed: int) -> List[dict]:
    """If any label class has < min_count, oversample within that label to reach min_count."""
    random.seed(seed)
    counts = label_counts(data)
    cur = data[:]
    for lbl, c in counts.items():
        zeros_needed = max(0, min_count - c.get(0, 0))
        ones_needed  = max(0, min_count - c.get(1, 0))
        if zeros_needed > 0:
            zeros = [ex for ex in cur if int(ex['cats'].get(lbl, 0)) == 0]
            cur.extend(random.choices(zeros, k=zeros_needed) if zeros else [])
        if ones_needed > 0:
            ones = [ex for ex in cur if int(ex['cats'].get(lbl, 0)) == 1]
            cur.extend(random.choices(ones, k=ones_needed) if ones else [])
    random.shuffle(cur)
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='Path to input .jsonl or .json file')
    ap.add_argument('--output', required=True, help='Path to output .jsonl or .json file')
    ap.add_argument('--mode', choices=['undersample','oversample','hybrid'], default='hybrid')
    ap.add_argument('--target', choices=['minority','majority','fixed'], default='minority')
    ap.add_argument('--fixed-target', type=int, default=None, help='Target per label when --target=fixed')
    ap.add_argument('--min-count', type=int, default=0, help='Ensure at least this many examples per class per label after balancing')
    ap.add_argument('--seed', type=int, default=17)
    ap.add_argument('--split', nargs=3, type=float, default=None, help='train val test fractions, e.g., 0.8 0.1 0.1')
    ap.add_argument('--train-out', default=None, help='Optional path for train output when --split is provided')
    ap.add_argument('--val-out', default=None, help='Optional path for val output when --split is provided')
    ap.add_argument('--test-out', default=None, help='Optional path for test output when --split is provided')
    ap.add_argument('--export-weights', default=None, help='Optional path to save computed class weights (JSON)')

    args = ap.parse_args()

    data = load_data(args.input)
    if not data:
        raise SystemExit("No data found in input file.")

    # Report before
    counts_before = label_counts(data)
    print_report(counts_before, prefix="Original")

    # Decide targets and balance
    targets = decide_targets(counts_before, args.target, args.fixed_target)
    balanced = iterative_multilabel_balance(data, args.mode, targets, seed=args.seed)

    # Ensure minimal count per class if requested
    if args.min_count and args.min_count > 0:
        balanced = ensure_min_count_per_class(balanced, args.min_count, seed=args.seed)

    # Report after
    counts_after = label_counts(balanced)
    print_report(counts_after, prefix="Balanced")

    # Save weights if requested
    if args.export_weights:
        weights = compute_class_weights(counts_after)
        with open(args.export_weights, 'w', encoding='utf-8') as f:
            json.dump(weights, f, ensure_ascii=False, indent=2)
        print(f"Saved class weights to {args.export_weights}")

    # Split if requested
    if args.split:
        if not args.train_out or not args.val_out or not args.test_out:
            raise SystemExit("When using --split, provide --train-out, --val-out, --test-out")
        train, val, test = stratified_split(balanced, tuple(args.split), seed=args.seed)
        save_data(args.train_out, train)
        save_data(args.val_out, val)
        save_data(args.test_out, test)
        print(f"Saved train={len(train)} to {args.train_out}")
        print(f"Saved val={len(val)} to {args.val_out}")
        print(f"Saved test={len(test)} to {args.test_out}")
    else:
        save_data(args.output, balanced)
        print(f"Saved balanced dataset with N={len(balanced)} to {args.output}")


if __name__ == '__main__':
    main()
