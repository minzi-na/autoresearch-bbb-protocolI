"""
Stereo-aware Morgan-FP UMAP comparison.

Generates chemspace_umap_stereo_aware.png with 4 panels:
  Top row    - stereo-BLIND  Morgan-FP (useChirality=False) UMAP
  Bottom row - stereo-AWARE  Morgan-FP (useChirality=True)  UMAP
  Left col   - colored by dataset
  Right col  - colored by label

Stereo-collision conflict molecules (425 in EXT_REM, 30 in holdout) are
highlighted as black-edged markers so we can directly see whether they
collapse to a single point (blind) vs separate (aware).
"""

import os, sys
sys.path.insert(0, '/home/minji/bbb-combo1')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import umap


HOLDOUT_DIR = '/home/minji/BBB/holdout_splits'
INT_CSV  = f'{HOLDOUT_DIR}/internal_curated_holdout_10pct_seed42/internal_curated_label_remaining.csv'
EXT_CSV  = f'{HOLDOUT_DIR}/external_cls_only_holdout_10pct_seed42/external_cls_only_label_remaining.csv'
HOLD_CSV = f'{HOLDOUT_DIR}/merged_holdout_10pct_seed42/label_holdout.csv'

OUT_DIR = '/home/minji/bbb-combo1/viz_outputs'
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42


def strip_stereo(s):
    m = Chem.MolFromSmiles(s)
    if m is None: return None
    Chem.RemoveStereochemistry(m)
    return Chem.MolToSmiles(m)


def load_with_mols(path, name):
    df = pd.read_csv(path).dropna(subset=['smiles']).reset_index(drop=True)
    mols, idx = [], []
    for i, smi in enumerate(df['smiles']):
        m = Chem.MolFromSmiles(smi)
        if m is None: continue
        mols.append(m); idx.append(i)
    df = df.iloc[idx].reset_index(drop=True)
    df['nostereo'] = df['smiles'].apply(strip_stereo)
    print(f'  loaded {name}: {len(df)} valid molecules')
    return df, mols


def morgan(mol, use_chirality):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024,
                                                  useChirality=use_chirality)


def fps_to_array(fps):
    arr = np.zeros((len(fps), 1024), dtype=np.uint8)
    for i, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, arr[i])
    return arr


def reproduce_internal_split(int_df, seed=42, train_ratio=0.8, val_ratio=0.1):
    import random
    random.seed(seed); np.random.seed(seed)
    df = int_df.copy()
    from rdkit.Chem.Scaffolds import MurckoScaffold
    def murcko(s):
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m)) if m else None
    df['scaffold'] = df['smiles'].apply(murcko)
    groups = list(df.groupby('scaffold').groups.values())
    groups = sorted(groups, key=lambda g: len(g), reverse=True)
    n = len(df)
    train_cap = int(round(train_ratio * n))
    val_cap   = int(round(val_ratio   * n))
    tr, va, te = [], [], []
    for g in groups:
        g = list(g)
        if   len(tr)+len(g) <= train_cap: tr += g
        elif len(va)+len(g) <= val_cap:   va += g
        else:                              te += g
    return tr


def find_conflict_molecules(int_train_df, ext_df, hold_df):
    """Identify stereo-strip label-conflict molecules.

    Returns dict mapping ('train'|'ext'|'hold', original_index_in_combined) -> True
    Combined order is concatenation [int_train, ext, hold].
    """
    # Build full pool for cross-set holdout vs train conflict
    train_lbl = defaultdict(set)
    for ns, lb in zip(int_train_df['nostereo'], int_train_df['p_np']):
        train_lbl[ns].add(int(lb))
    for ns, lb in zip(ext_df['nostereo'], ext_df['p_np']):
        train_lbl[ns].add(int(lb))

    flag_train = np.zeros(len(int_train_df), dtype=bool)
    flag_ext   = np.zeros(len(ext_df),       dtype=bool)
    flag_hold  = np.zeros(len(hold_df),      dtype=bool)

    # Within-set conflict: same nostereo with multiple labels (intra-EXT mainly)
    for df, flag in [(int_train_df, flag_train), (ext_df, flag_ext)]:
        groups = df.groupby('nostereo')['p_np'].nunique()
        conflict_ns = set(groups[groups > 1].index)
        for i, ns in enumerate(df['nostereo']):
            if ns in conflict_ns:
                flag[i] = True

    # Holdout conflict: holdout label disagrees with train pool on same 2D
    for i, (ns, lb) in enumerate(zip(hold_df['nostereo'], hold_df['p_np'])):
        if ns in train_lbl and int(lb) not in train_lbl[ns]:
            flag_hold[i] = True

    print(f'  conflict counts: train={flag_train.sum()} ext={flag_ext.sum()} '
          f'hold={flag_hold.sum()}')
    return flag_train, flag_ext, flag_hold


def compute_umap(X, name):
    print(f'  fitting UMAP on {name} ({X.shape})...')
    reducer = umap.UMAP(metric='jaccard', n_neighbors=30, min_dist=0.3,
                        random_state=SEED, n_jobs=1)
    return reducer.fit_transform(X)


def plot_panel(ax, Z, src, lbl, conflict_mask, mode):
    """mode='dataset' or 'label' """
    if mode == 'dataset':
        names = ['Internal train', 'External', 'Holdout']
        cols  = ['#5d8aa8', '#d8696c', '#88b04b']
        sizes = [10, 6, 14]
        alphas = [0.45, 0.30, 0.85]
        for s, name, c, sz, a in zip([0,1,2], names, cols, sizes, alphas):
            m = (src == s) & ~conflict_mask
            ax.scatter(Z[m,0], Z[m,1], s=sz, c=c, alpha=a,
                       label=f'{name} (n={(src==s).sum()})', edgecolor='none')
        # Conflict molecules with black edge
        m = conflict_mask
        ax.scatter(Z[m,0], Z[m,1], s=22, facecolor='yellow', alpha=0.95,
                   edgecolor='black', linewidth=0.7,
                   label=f'Conflict mol (n={m.sum()})')
    else:  # label
        pos = (lbl == 1) & ~conflict_mask
        neg = (lbl == 0) & ~conflict_mask
        ax.scatter(Z[neg,0], Z[neg,1], s=6, c='#d8696c', alpha=0.30,
                   label=f'BBB- (n={(lbl==0).sum()})', edgecolor='none')
        ax.scatter(Z[pos,0], Z[pos,1], s=6, c='#5d8aa8', alpha=0.30,
                   label=f'BBB+ (n={(lbl==1).sum()})', edgecolor='none')
        m = conflict_mask
        ax.scatter(Z[m,0], Z[m,1], s=22, facecolor='yellow', alpha=0.95,
                   edgecolor='black', linewidth=0.7,
                   label=f'Conflict mol (n={m.sum()})')
    ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')
    ax.legend(loc='best', fontsize=8)


if __name__ == '__main__':
    print('Loading datasets...')
    int_df,  int_mols  = load_with_mols(INT_CSV,  'internal')
    ext_df,  ext_mols  = load_with_mols(EXT_CSV,  'external')
    hold_df, hold_mols = load_with_mols(HOLD_CSV, 'holdout')

    print('Reproducing internal scaffold split (seed=42)...')
    tr_idx = reproduce_internal_split(int_df, seed=SEED)
    int_train_df   = int_df.iloc[tr_idx].reset_index(drop=True)
    int_train_mols = [int_mols[i] for i in tr_idx]
    print(f'  internal train n = {len(int_train_df)}')

    print('Computing FPs (stereo-blind and stereo-aware)...')
    fps_blind = ([morgan(m, False) for m in int_train_mols] +
                 [morgan(m, False) for m in ext_mols] +
                 [morgan(m, False) for m in hold_mols])
    fps_aware = ([morgan(m, True)  for m in int_train_mols] +
                 [morgan(m, True)  for m in ext_mols] +
                 [morgan(m, True)  for m in hold_mols])
    X_blind = fps_to_array(fps_blind)
    X_aware = fps_to_array(fps_aware)

    src = np.array([0]*len(int_train_df) + [1]*len(ext_df) + [2]*len(hold_df))
    lbl = np.concatenate([int_train_df['p_np'].values,
                          ext_df['p_np'].values,
                          hold_df['p_np'].values]).astype(int)

    print('Identifying stereo-collision conflict molecules...')
    flag_tr, flag_ex, flag_ho = find_conflict_molecules(int_train_df, ext_df, hold_df)
    conflict_mask = np.concatenate([flag_tr, flag_ex, flag_ho])
    print(f'  total conflict molecules highlighted: {conflict_mask.sum()}')

    Z_blind = compute_umap(X_blind, 'stereo-blind')
    Z_aware = compute_umap(X_aware, 'stereo-aware')

    fig, axes = plt.subplots(2, 2, figsize=(15, 13))

    plot_panel(axes[0,0], Z_blind, src, lbl, conflict_mask, 'dataset')
    axes[0,0].set_title('Stereo-BLIND Morgan-FP — colored by dataset',
                          fontsize=11)
    plot_panel(axes[0,1], Z_blind, src, lbl, conflict_mask, 'label')
    axes[0,1].set_title('Stereo-BLIND Morgan-FP — colored by label',
                          fontsize=11)
    plot_panel(axes[1,0], Z_aware, src, lbl, conflict_mask, 'dataset')
    axes[1,0].set_title('Stereo-AWARE Morgan-FP (useChirality=True) — colored by dataset',
                          fontsize=11)
    plot_panel(axes[1,1], Z_aware, src, lbl, conflict_mask, 'label')
    axes[1,1].set_title('Stereo-AWARE Morgan-FP (useChirality=True) — colored by label',
                          fontsize=11)

    fig.suptitle('Chemical space: stereo-blind vs stereo-aware Morgan-FP\n'
                 '(yellow markers = stereo-collision conflict molecules)',
                 fontsize=13, y=0.995)
    fig.tight_layout()
    out = f'{OUT_DIR}/chemspace_umap_stereo_aware.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nsaved: {out}')
