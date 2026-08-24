"""
Label distribution visualization for curated BBB datasets.

Outputs:
  1. label_dist_three_datasets.png  - bar chart of BBB+/BBB- counts for the
     three curated splits (internal remaining, external remaining, merged
     holdout)
  2. internal_scaffold_split_dist.png - per-split (train/val/test) label
     distribution of the internal_curated_label_remaining.csv after scaffold
     split (seed=42, 80/10/10), reproducing the exact split used by
     bbb-combo1 training
  3. scaffold_split_summary.json    - numeric summary of the splits and
     scaffold overlap analysis

All paths resolved relative to /home/minji/BBB/holdout_splits/.
"""

import os, sys, json
from collections import Counter

sys.path.insert(0, '/home/minji/bbb-combo1')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

import random


HOLDOUT_DIR = '/home/minji/BBB/holdout_splits'
INT_CSV  = f'{HOLDOUT_DIR}/internal_curated_holdout_10pct_seed42/internal_curated_label_remaining.csv'
EXT_CSV  = f'{HOLDOUT_DIR}/external_cls_only_holdout_10pct_seed42/external_cls_only_label_remaining.csv'
HOLD_CSV = f'{HOLDOUT_DIR}/merged_holdout_10pct_seed42/label_holdout.csv'

OUT_DIR = '/home/minji/bbb-combo1/viz_outputs'
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1) Three-dataset label distribution
# ---------------------------------------------------------------------------

def load_csv(path):
    df = pd.read_csv(path)
    assert 'p_np' in df.columns, f'p_np column missing in {path}'
    return df


def plot_three_datasets():
    int_df  = load_csv(INT_CSV)
    ext_df  = load_csv(EXT_CSV)
    hold_df = load_csv(HOLD_CSV)

    datasets = [
        ('Internal\n(curated remaining)', int_df),
        ('External\n(cls-only remaining)', ext_df),
        ('Merged\nHoldout', hold_df),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    summary = {}
    for ax, (title, df) in zip(axes, datasets):
        cnt = df['p_np'].value_counts().sort_index()
        n0 = int(cnt.get(0, 0))
        n1 = int(cnt.get(1, 0))
        total = n0 + n1
        bars = ax.bar(['BBB- (0)', 'BBB+ (1)'], [n0, n1],
                      color=['#d8696c', '#5d8aa8'], edgecolor='black')
        for b, v in zip(bars, [n0, n1]):
            ax.text(b.get_x() + b.get_width()/2, v + total*0.01,
                    f'{v}\n({100*v/total:.1f}%)',
                    ha='center', va='bottom', fontsize=10)
        ax.set_title(f'{title}\nTotal = {total}', fontsize=11)
        ax.set_ylabel('Count')
        ax.set_ylim(0, max(n0, n1) * 1.18)
        summary[title.replace('\n', ' ')] = {
            'total': total, 'BBB-': n0, 'BBB+': n1,
            'pos_ratio': round(n1/total, 4),
        }

    fig.suptitle('Label distribution across curated BBB datasets',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out = f'{OUT_DIR}/label_dist_three_datasets.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved: {out}')
    return summary


# ---------------------------------------------------------------------------
# 2) Internal scaffold split label distribution (reproduces bbb-combo1 logic)
# ---------------------------------------------------------------------------

def murcko_scaffold(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))
    except Exception:
        return None


def reproduce_scaffold_split(df, seed=42, train_ratio=0.8, val_ratio=0.1):
    """Mirror bbb_prepare.split_then_normalize: scaffold-grouped greedy split."""
    random.seed(seed); np.random.seed(seed)
    df = df.copy().reset_index(drop=True)
    df['scaffold'] = df['smiles'].apply(murcko_scaffold)
    groups = list(df.groupby('scaffold').groups.values())
    groups = sorted(groups, key=lambda g: len(g), reverse=True)

    n = len(df)
    train_cap = int(round(train_ratio * n))
    val_cap   = int(round(val_ratio   * n))

    train_idx, val_idx, test_idx = [], [], []
    for g in groups:
        g = list(g)
        if   len(train_idx) + len(g) <= train_cap: train_idx += g
        elif len(val_idx)   + len(g) <= val_cap:   val_idx   += g
        else:                                        test_idx  += g
    return df, train_idx, val_idx, test_idx


def plot_internal_scaffold_split(seed=42):
    df = load_csv(INT_CSV)
    df, tr, va, te = reproduce_scaffold_split(df, seed=seed)

    splits = [('Train', tr), ('Val', va), ('Test', te)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    summary = {}
    for ax, (name, idx) in zip(axes, splits):
        sub = df.iloc[idx]
        cnt = sub['p_np'].value_counts().sort_index()
        n0 = int(cnt.get(0, 0)); n1 = int(cnt.get(1, 0))
        total = n0 + n1
        bars = ax.bar(['BBB- (0)', 'BBB+ (1)'], [n0, n1],
                      color=['#d8696c', '#5d8aa8'], edgecolor='black')
        for b, v in zip(bars, [n0, n1]):
            ax.text(b.get_x() + b.get_width()/2, v + total*0.01,
                    f'{v}\n({100*v/total:.1f}%)',
                    ha='center', va='bottom', fontsize=10)
        ax.set_title(f'{name}\nN={total}  (n_scaffolds={sub["scaffold"].nunique()})',
                     fontsize=11)
        ax.set_ylabel('Count')
        ax.set_ylim(0, max(n0, n1) * 1.20)
        summary[name] = {
            'total': total, 'BBB-': n0, 'BBB+': n1,
            'pos_ratio': round(n1/total, 4),
            'n_scaffolds': int(sub['scaffold'].nunique()),
        }

    fig.suptitle(f'Internal (curated remaining) — scaffold split  '
                 f'(seed={seed}, 80/10/10)', fontsize=13, y=1.02)
    fig.tight_layout()
    out = f'{OUT_DIR}/internal_scaffold_split_dist.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved: {out}')
    return df, tr, va, te, summary


# ---------------------------------------------------------------------------
# 3) Scaffold overlap analysis between splits
# ---------------------------------------------------------------------------

def scaffold_overlap_analysis(df, tr, va, te):
    s_tr = set(df.iloc[tr]['scaffold'].dropna())
    s_va = set(df.iloc[va]['scaffold'].dropna())
    s_te = set(df.iloc[te]['scaffold'].dropna())

    def jaccard(a, b):
        if not a and not b: return 0.0
        return len(a & b) / len(a | b)

    return {
        'n_unique_scaffolds': {
            'train': len(s_tr), 'val': len(s_va), 'test': len(s_te),
        },
        'intersection_counts': {
            'train_val':  len(s_tr & s_va),
            'train_test': len(s_tr & s_te),
            'val_test':   len(s_va & s_te),
            'train_val_test': len(s_tr & s_va & s_te),
        },
        'jaccard_similarity': {
            'train_val':  round(jaccard(s_tr, s_va),  6),
            'train_test': round(jaccard(s_tr, s_te),  6),
            'val_test':   round(jaccard(s_va, s_te),  6),
        },
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('[1/3] Three-dataset label distribution...')
    three_summary = plot_three_datasets()

    print('[2/3] Internal scaffold split (seed=42)...')
    df, tr, va, te, split_summary = plot_internal_scaffold_split(seed=42)

    print('[3/3] Scaffold overlap analysis...')
    overlap = scaffold_overlap_analysis(df, tr, va, te)

    out_json = f'{OUT_DIR}/scaffold_split_summary.json'
    with open(out_json, 'w') as f:
        json.dump({
            'three_dataset_label_dist': three_summary,
            'internal_scaffold_split_seed42': split_summary,
            'scaffold_overlap': overlap,
        }, f, indent=2)
    print(f'  saved: {out_json}')

    print('\n=== Summary ===')
    print(json.dumps({
        'datasets': three_summary,
        'internal_split_seed42': split_summary,
        'scaffold_overlap': overlap,
    }, indent=2))
