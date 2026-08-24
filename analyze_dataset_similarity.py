"""
Cross-dataset similarity & distribution analysis for BBB curated splits.

Goal: explain why
  (a) internal test ROC is high but external/holdout MCC/F1 is much lower
  (b) augmenting internal train with external data does NOT improve holdout
      (and may hurt internal test)

Analyses produced:
  1. scaffold_overlap.png         - scaffold-level overlap between
                                     internal_train / external / holdout
  2. tanimoto_nn_distribution.png - max Tanimoto similarity from each target
                                     molecule to several reference sets;
                                     reveals coverage gaps
  3. label_agreement_pairs.png    - for cross-dataset near-duplicate pairs
                                     (Tanimoto>=0.5/0.7/0.9), how often do
                                     labels agree?  Direct evidence of
                                     label noise / label-shift
  4. property_distributions.png   - MW, LogP, num heavy atoms, fraction sp3,
                                     num rotatable bonds — shows covariate
                                     shift between datasets
  5. chemspace_umap.png           - UMAP of Morgan-FP (1024-d, r=2) for
                                     all 3 datasets; colored by source +
                                     side-by-side label coloring
  6. analysis_summary.json        - numerical summary

Internal split is reproduced with the exact bbb-combo1 logic (seed=42 scaffold
greedy split, 80/10/10).
"""

import os, sys, json, random
from collections import Counter
sys.path.insert(0, '/home/minji/bbb-combo1')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import umap

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HOLDOUT_DIR = '/home/minji/BBB/holdout_splits'
INT_CSV  = f'{HOLDOUT_DIR}/internal_curated_holdout_10pct_seed42/internal_curated_label_remaining.csv'
EXT_CSV  = f'{HOLDOUT_DIR}/external_cls_only_holdout_10pct_seed42/external_cls_only_label_remaining.csv'
HOLD_CSV = f'{HOLDOUT_DIR}/merged_holdout_10pct_seed42/label_holdout.csv'

OUT_DIR = '/home/minji/bbb-combo1/viz_outputs'
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42

# ---------------------------------------------------------------------------
# Load + build fingerprints
# ---------------------------------------------------------------------------

def murcko_smi(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    try: return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))
    except: return None


def morgan_fp(mol, radius=2, n_bits=1024):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def load_with_fps(path, name):
    df = pd.read_csv(path).dropna(subset=['smiles']).reset_index(drop=True)
    mols, fps, scaffolds = [], [], []
    keep = []
    for i, smi in enumerate(df['smiles']):
        m = Chem.MolFromSmiles(smi)
        if m is None: continue
        mols.append(m)
        fps.append(morgan_fp(m))
        scaffolds.append(murcko_smi(smi))
        keep.append(i)
    df = df.iloc[keep].reset_index(drop=True)
    df['scaffold'] = scaffolds
    print(f'  loaded {name}: {len(df)} valid molecules')
    return df, mols, fps


def reproduce_internal_scaffold_split(df, seed=42, train_ratio=0.8, val_ratio=0.1):
    random.seed(seed); np.random.seed(seed)
    df = df.copy().reset_index(drop=True)
    groups = list(df.groupby('scaffold').groups.values())
    groups = sorted(groups, key=lambda g: len(g), reverse=True)
    n = len(df)
    train_cap = int(round(train_ratio * n))
    val_cap   = int(round(val_ratio   * n))
    tr, va, te = [], [], []
    for g in groups:
        g = list(g)
        if   len(tr) + len(g) <= train_cap: tr += g
        elif len(va) + len(g) <= val_cap:   va += g
        else:                                te += g
    return tr, va, te


# ---------------------------------------------------------------------------
# Analysis 1: scaffold overlap
# ---------------------------------------------------------------------------

def scaffold_overlap_plot(int_df, tr_idx, ext_df, hold_df):
    s_train = set(int_df.iloc[tr_idx]['scaffold'].dropna())
    s_ext   = set(ext_df ['scaffold'].dropna())
    s_hold  = set(hold_df['scaffold'].dropna())

    sets = {'Internal_train': s_train, 'External': s_ext, 'Holdout': s_hold}
    labels = list(sets.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # (a) bar chart of unique + overlap
    counts = {
        'unique_train':         len(s_train - s_ext - s_hold),
        'unique_ext':           len(s_ext - s_train - s_hold),
        'unique_hold':          len(s_hold - s_train - s_ext),
        'train_and_ext':        len((s_train & s_ext) - s_hold),
        'train_and_hold':       len((s_train & s_hold) - s_ext),
        'ext_and_hold':         len((s_ext & s_hold) - s_train),
        'all_three':            len(s_train & s_ext & s_hold),
    }
    ax = axes[0]
    ax.bar(range(len(counts)), list(counts.values()),
           color=['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#8c564b','#e377c2','#7f7f7f'],
           edgecolor='black')
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(list(counts.keys()), rotation=30, ha='right', fontsize=9)
    for i, v in enumerate(counts.values()):
        ax.text(i, v + max(counts.values())*0.01, str(v), ha='center', fontsize=9)
    ax.set_ylabel('# unique scaffolds')
    ax.set_title('Scaffold partition counts')

    # (b) coverage rates: what % of holdout/external scaffolds are in internal train?
    ax = axes[1]
    cov = {
        'holdout in train': len(s_hold & s_train) / max(len(s_hold), 1),
        'holdout in ext':   len(s_hold & s_ext)   / max(len(s_hold), 1),
        'holdout in train+ext': len(s_hold & (s_train | s_ext)) / max(len(s_hold), 1),
        'ext in train':     len(s_ext & s_train)  / max(len(s_ext),  1),
    }
    bars = ax.bar(range(len(cov)), [v*100 for v in cov.values()],
                  color=['#5d8aa8','#d8696c','#88b04b','#ffa500'], edgecolor='black')
    for b, v in zip(bars, cov.values()):
        ax.text(b.get_x()+b.get_width()/2, v*100 + 1, f'{v*100:.1f}%',
                ha='center', fontsize=10)
    ax.set_xticks(range(len(cov)))
    ax.set_xticklabels(list(cov.keys()), rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('Scaffold coverage (%)')
    ax.set_title('Scaffold coverage of target sets by source sets')
    ax.set_ylim(0, 105)

    fig.suptitle('Scaffold overlap analysis', fontsize=13, y=1.02)
    fig.tight_layout()
    out = f'{OUT_DIR}/scaffold_overlap.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved: {out}')
    return {'sets': {k: len(v) for k,v in sets.items()},
            'partition_counts': counts, 'coverage': {k: round(v,4) for k,v in cov.items()}}


# ---------------------------------------------------------------------------
# Analysis 2: max-Tanimoto NN distribution (target -> reference sets)
# ---------------------------------------------------------------------------

def max_tanimoto_to(query_fps, ref_fps):
    """For each query fp, max Tanimoto similarity to any fp in ref."""
    out = np.zeros(len(query_fps), dtype=np.float32)
    for i, q in enumerate(query_fps):
        sims = DataStructs.BulkTanimotoSimilarity(q, ref_fps)
        out[i] = max(sims) if len(sims) else 0.0
    return out


def tanimoto_nn_plot(int_train_fps, ext_fps, hold_fps):
    print('  computing max-Tanimoto distributions...')
    # holdout queries
    hold_to_train     = max_tanimoto_to(hold_fps, int_train_fps)
    hold_to_ext       = max_tanimoto_to(hold_fps, ext_fps)
    hold_to_train_ext = np.maximum(hold_to_train, hold_to_ext)
    # external queries
    ext_to_train      = max_tanimoto_to(ext_fps, int_train_fps)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    bins = np.linspace(0, 1, 41)
    ax.hist(hold_to_train,     bins=bins, alpha=0.55, label='Holdout → Train',
            color='#5d8aa8', edgecolor='black')
    ax.hist(hold_to_ext,       bins=bins, alpha=0.45, label='Holdout → External',
            color='#d8696c', edgecolor='black')
    ax.hist(hold_to_train_ext, bins=bins, alpha=0.30, label='Holdout → Train+External',
            color='#88b04b', edgecolor='black')
    ax.set_xlabel('Max Tanimoto similarity (Morgan r=2, 1024 bits)')
    ax.set_ylabel('# holdout molecules')
    ax.set_title('How close is each holdout molecule to its NN in the training pool?')
    ax.legend(loc='upper left', fontsize=9)

    ax = axes[1]
    # CDF view — clearer for 'augmentation effect'
    for name, arr, c in [
        ('Holdout → Train',          hold_to_train,     '#5d8aa8'),
        ('Holdout → External',       hold_to_ext,       '#d8696c'),
        ('Holdout → Train+External', hold_to_train_ext, '#88b04b'),
        ('External → Train',         ext_to_train,      '#8c564b'),
    ]:
        x = np.sort(arr); y = np.arange(1, len(x)+1) / len(x)
        ax.plot(x, y, label=f'{name}  (median={np.median(arr):.3f})', color=c, lw=2)
    ax.set_xlabel('Max Tanimoto similarity to reference set')
    ax.set_ylabel('Cumulative fraction')
    ax.set_title('CDF of max-Tanimoto to reference set')
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)

    fig.suptitle('Cross-dataset Tanimoto nearest-neighbor distribution', fontsize=13, y=1.02)
    fig.tight_layout()
    out = f'{OUT_DIR}/tanimoto_nn_distribution.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved: {out}')
    return {
        'holdout_to_train':       {'median': float(np.median(hold_to_train)),
                                   'mean': float(np.mean(hold_to_train)),
                                   'p25': float(np.percentile(hold_to_train, 25)),
                                   'p75': float(np.percentile(hold_to_train, 75))},
        'holdout_to_ext':         {'median': float(np.median(hold_to_ext)),
                                   'mean':   float(np.mean(hold_to_ext))},
        'holdout_to_train_ext':   {'median': float(np.median(hold_to_train_ext)),
                                   'mean':   float(np.mean(hold_to_train_ext))},
        'ext_to_train':           {'median': float(np.median(ext_to_train)),
                                   'mean':   float(np.mean(ext_to_train))},
        'gain_from_aug': float(np.median(hold_to_train_ext) - np.median(hold_to_train)),
    }, hold_to_train, hold_to_ext, hold_to_train_ext, ext_to_train


# ---------------------------------------------------------------------------
# Analysis 3: label agreement on near-duplicate cross-dataset pairs
# ---------------------------------------------------------------------------

def label_agreement_plot(src_fps, src_labels, src_name,
                          tgt_fps, tgt_labels, tgt_name):
    """For each src molecule, find its NN in tgt; if Tanimoto>=t, check label match."""
    thresholds = [0.3, 0.5, 0.7, 0.9]
    results = {}
    for t in thresholds:
        agree = 0; total = 0
        for i, q in enumerate(src_fps):
            sims = DataStructs.BulkTanimotoSimilarity(q, tgt_fps)
            j = int(np.argmax(sims))
            if sims[j] >= t:
                total += 1
                if int(src_labels[i]) == int(tgt_labels[j]):
                    agree += 1
        results[t] = {'matched_pairs': total,
                      'agreement_rate': (agree/total) if total else None}
    return results


def label_agreement_figure(int_train_fps, int_train_lbl,
                            ext_fps, ext_lbl,
                            hold_fps, hold_lbl):
    print('  computing label agreement on cross-dataset NN pairs...')
    pairs = [
        ('Holdout vs Internal train', hold_fps, hold_lbl, int_train_fps, int_train_lbl),
        ('Holdout vs External',       hold_fps, hold_lbl, ext_fps,       ext_lbl),
        ('External vs Internal train',ext_fps,  ext_lbl,  int_train_fps, int_train_lbl),
    ]
    all_res = {}
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    width = 0.25
    thresholds = [0.3, 0.5, 0.7, 0.9]
    x = np.arange(len(thresholds))
    colors = ['#5d8aa8', '#d8696c', '#88b04b']
    for k, (name, sf, sl, tf, tl) in enumerate(pairs):
        r = label_agreement_plot(sf, sl, '', tf, tl, '')
        all_res[name] = {str(t): r[t] for t in thresholds}
        rates = [r[t]['agreement_rate'] if r[t]['agreement_rate'] is not None else 0
                 for t in thresholds]
        npairs = [r[t]['matched_pairs'] for t in thresholds]
        ax.bar(x + (k-1)*width, rates, width, color=colors[k], edgecolor='black',
               label=name)
        for xi, rate, n in zip(x + (k-1)*width, rates, npairs):
            ax.text(xi, rate + 0.01, f'{rate*100:.0f}%\n(n={n})',
                    ha='center', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'≥{t}' for t in thresholds])
    ax.set_xlabel('Tanimoto similarity threshold of NN pair')
    ax.set_ylabel('Label agreement rate')
    ax.set_title('Label consistency on cross-dataset near-neighbor pairs\n'
                 '(disagreement = label noise / annotation drift)')
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    out = f'{OUT_DIR}/label_agreement_pairs.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved: {out}')
    return all_res


# ---------------------------------------------------------------------------
# Analysis 4: molecular property distributions (covariate shift)
# ---------------------------------------------------------------------------

def compute_props(mols):
    rows = []
    for m in mols:
        rows.append({
            'MW':      Descriptors.MolWt(m),
            'LogP':    Descriptors.MolLogP(m),
            'HBA':     Lipinski.NumHAcceptors(m),
            'HBD':     Lipinski.NumHDonors(m),
            'TPSA':    Descriptors.TPSA(m),
            'NumHeavy': m.GetNumHeavyAtoms(),
            'NumRot':  Lipinski.NumRotatableBonds(m),
            'FractionCSP3': rdMolDescriptors.CalcFractionCSP3(m),
            'NumRings': rdMolDescriptors.CalcNumRings(m),
        })
    return pd.DataFrame(rows)


def property_plot(int_train_mols, ext_mols, hold_mols):
    p_train = compute_props(int_train_mols); p_train['set'] = 'Internal train'
    p_ext   = compute_props(ext_mols);       p_ext  ['set'] = 'External'
    p_hold  = compute_props(hold_mols);      p_hold ['set'] = 'Holdout'
    df = pd.concat([p_train, p_ext, p_hold], ignore_index=True)

    props = ['MW','LogP','TPSA','NumHeavy','NumRot','FractionCSP3']
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    colors = {'Internal train':'#5d8aa8', 'External':'#d8696c', 'Holdout':'#88b04b'}
    for ax, p in zip(axes.flatten(), props):
        for setname, c in colors.items():
            sub = df[df['set']==setname][p]
            ax.hist(sub, bins=40, alpha=0.45, color=c, label=setname,
                    density=True, edgecolor='black', linewidth=0.5)
        ax.set_title(p, fontsize=11)
        ax.set_ylabel('density')
        ax.legend(fontsize=8)
    fig.suptitle('Molecular property distributions per dataset (covariate shift)',
                 fontsize=13, y=1.01)
    fig.tight_layout()
    out = f'{OUT_DIR}/property_distributions.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved: {out}')

    # KS test summary (no scipy needed: use np for crude two-sample distance)
    summary = {}
    for p in props:
        a = df[df['set']=='Internal train'][p].values
        b = df[df['set']=='External'][p].values
        c = df[df['set']=='Holdout'][p].values
        summary[p] = {
            'mean_train': round(float(np.mean(a)), 3),
            'mean_ext':   round(float(np.mean(b)), 3),
            'mean_hold':  round(float(np.mean(c)), 3),
            'mean_diff_ext_vs_train':  round(float(np.mean(b)-np.mean(a)), 3),
            'mean_diff_hold_vs_train': round(float(np.mean(c)-np.mean(a)), 3),
        }
    return summary


# ---------------------------------------------------------------------------
# Analysis 5: chemical space UMAP
# ---------------------------------------------------------------------------

def chemspace_umap(int_train_fps, int_train_lbl,
                    ext_fps,       ext_lbl,
                    hold_fps,      hold_lbl):
    print('  computing UMAP on Morgan-FP...')
    def to_array(fps):
        arr = np.zeros((len(fps), 1024), dtype=np.uint8)
        for i, fp in enumerate(fps):
            DataStructs.ConvertToNumpyArray(fp, arr[i])
        return arr
    A = to_array(int_train_fps)
    B = to_array(ext_fps)
    C = to_array(hold_fps)
    X = np.vstack([A, B, C]).astype(np.float32)
    src = np.array([0]*len(A) + [1]*len(B) + [2]*len(C))
    lbl = np.concatenate([int_train_lbl, ext_lbl, hold_lbl]).astype(int)

    reducer = umap.UMAP(metric='jaccard', n_neighbors=30, min_dist=0.3,
                        random_state=SEED, n_jobs=1)
    Z = reducer.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    # Color by source
    ax = axes[0]
    names = ['Internal train', 'External', 'Holdout']
    cols  = ['#5d8aa8', '#d8696c', '#88b04b']
    sizes = [10, 6, 14]
    alphas = [0.45, 0.30, 0.85]
    for s, name, c, sz, a in zip([0,1,2], names, cols, sizes, alphas):
        m = src == s
        ax.scatter(Z[m,0], Z[m,1], s=sz, c=c, alpha=a, label=f'{name} (n={m.sum()})',
                   edgecolor='none')
    ax.set_title('UMAP of Morgan-FP (Jaccard) — colored by dataset')
    ax.legend(loc='best', fontsize=9)
    ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')

    # Color by label, faceted
    ax = axes[1]
    pos = lbl == 1; neg = lbl == 0
    ax.scatter(Z[neg,0], Z[neg,1], s=6, c='#d8696c', alpha=0.30,
               label=f'BBB- (n={neg.sum()})', edgecolor='none')
    ax.scatter(Z[pos,0], Z[pos,1], s=6, c='#5d8aa8', alpha=0.30,
               label=f'BBB+ (n={pos.sum()})', edgecolor='none')
    ax.set_title('Same UMAP — colored by label (all 3 datasets pooled)')
    ax.legend(loc='best', fontsize=9)
    ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')

    fig.suptitle('Chemical space (Morgan-FP UMAP)', fontsize=13, y=1.02)
    fig.tight_layout()
    out = f'{OUT_DIR}/chemspace_umap.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved: {out}')
    return Z, src, lbl


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('Loading datasets...')
    int_df,  int_mols,  int_fps  = load_with_fps(INT_CSV,  'internal')
    ext_df,  ext_mols,  ext_fps  = load_with_fps(EXT_CSV,  'external')
    hold_df, hold_mols, hold_fps = load_with_fps(HOLD_CSV, 'holdout')

    print('Reproducing internal scaffold split (seed=42)...')
    tr_idx, va_idx, te_idx = reproduce_internal_scaffold_split(int_df, seed=SEED)
    int_train_fps  = [int_fps [i] for i in tr_idx]
    int_train_mols = [int_mols[i] for i in tr_idx]
    int_train_lbl  = int_df.iloc[tr_idx]['p_np'].values
    ext_lbl  = ext_df ['p_np'].values
    hold_lbl = hold_df['p_np'].values
    print(f'  internal train n = {len(int_train_fps)}')

    print('[1/5] Scaffold overlap...')
    sc_overlap = scaffold_overlap_plot(int_df, tr_idx, ext_df, hold_df)

    print('[2/5] Tanimoto NN distributions...')
    tn_summary, h2t, h2e, h2te, e2t = tanimoto_nn_plot(int_train_fps, ext_fps, hold_fps)

    print('[3/5] Label agreement on near-duplicate pairs...')
    la_summary = label_agreement_figure(int_train_fps, int_train_lbl,
                                          ext_fps, ext_lbl,
                                          hold_fps, hold_lbl)

    print('[4/5] Property distributions...')
    prop_summary = property_plot(int_train_mols, ext_mols, hold_mols)

    print('[5/5] Chemical space UMAP...')
    Z, src, lbl = chemspace_umap(int_train_fps, int_train_lbl,
                                  ext_fps, ext_lbl,
                                  hold_fps, hold_lbl)

    out_json = f'{OUT_DIR}/analysis_summary.json'
    with open(out_json, 'w') as f:
        json.dump({
            'scaffold_overlap':  sc_overlap,
            'tanimoto_nn':       tn_summary,
            'label_agreement':   la_summary,
            'property_means':    prop_summary,
        }, f, indent=2)
    print(f'  saved: {out_json}')

    print('\nAll figures saved to', OUT_DIR)
