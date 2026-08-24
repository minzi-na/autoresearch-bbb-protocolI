"""
TabPFN-2.5 inference on holdout subsets total(888) / nn05(329), combo1 features.
Per seed: scaffold-split internal -> fit scaler+TabPFN on train -> predict on the
1089 merged holdout -> slice to 888/329. Soft-vote ensemble + per-seed mean/std.

Run: /home/minji/anaconda3/envs/tabpfn/bin/python eval_tabpfn_888_329.py
"""
import os, sys, json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (matthews_corrcoef, accuracy_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score)
from rdkit import Chem

sys.path.insert(0, '/home/minji/bbb-combo1')
from tabpfn25_baseline import (load_dataset, scaffold_split, canon,
    INT_LABEL, INT_MOLE, HOLD_LABEL, HOLD_MOLE, MACCS_DIM, AVALON_BITS, RDKIT_DIM, SEEDS)

SUB = {
    'total': '/home/minji/holdout_subset/merged_holdout_10pct_seed42_simfilter09_total/label_holdout.csv',
    'nn05':  '/home/minji/holdout_subset/merged_holdout_10pct_seed42_simfilter09_nn05/label_holdout.csv',
}

def metrics(yt, yp):
    yt = np.asarray(yt, int); yp = np.asarray(yp, float); pred = (yp > 0.5).astype(int)
    cm = confusion_matrix(yt, pred)
    spec = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if cm.size == 4 and (cm[0, 0] + cm[0, 1]) > 0 else 0.0
    hb = len(set(yt.tolist())) > 1
    return dict(roc_auc=float(roc_auc_score(yt, yp)) if hb else 0.0,
                mcc=float(matthews_corrcoef(yt, pred)),
                f1=float(f1_score(yt, pred, zero_division=0)),
                accuracy=float(accuracy_score(yt, pred)),
                auprc=float(average_precision_score(yt, yp)) if hb else 0.0,
                specificity=float(spec))

import pandas as pd
def load_want(p):
    return set(canon(s) for s in pd.read_csv(p)['smiles'])

from tabpfn import TabPFNClassifier

X_int, y_int, smi_int, _ = load_dataset(INT_LABEL, INT_MOLE)
X_hold, y_hold, smi_hold, _ = load_dataset(HOLD_LABEL, HOLD_MOLE)
print(f'internal={X_int.shape}  holdout={X_hold.shape}', flush=True)

masks = {'full': np.ones(len(smi_hold), bool)}
for name, p in SUB.items():
    want = load_want(p)
    masks[name] = np.array([s in want for s in smi_hold], bool)
    print(f'  subset {name}: target={len(want)}  matched={int(masks[name].sum())}', flush=True)

rd_start = MACCS_DIM + AVALON_BITS
rd_end   = rd_start + RDKIT_DIM

hold_seed_probs = []
for seed in SEEDS:
    tr, va, te = scaffold_split(smi_int, y_int, seed=seed)
    X_tr = X_int[tr].copy(); y_tr = y_int[tr]
    X_h = X_hold.copy()
    scaler = StandardScaler().fit(X_tr[:, rd_start:rd_end])
    for arr in [X_tr, X_h]:
        arr[:, rd_start:rd_end] = scaler.transform(arr[:, rd_start:rd_end]).astype(np.float32)
    clf = TabPFNClassifier(n_estimators=8, random_state=seed, ignore_pretraining_limits=False)
    clf.fit(X_tr, y_tr)
    p = clf.predict_proba(X_h)[:, 1]
    hold_seed_probs.append(np.asarray(p, dtype=np.float64))
    print(f'  seed={seed:>4d}  hold_full_roc={roc_auc_score(y_hold, p):.4f}', flush=True)

ps = np.stack(hold_seed_probs)  # (10, 1089)
y = np.asarray(y_hold, int)
out = {'tabpfn': {}}
for name, mask in masks.items():
    ens = metrics(y[mask], ps[:, mask].mean(0))
    per = [metrics(y[mask], ps[i, mask]) for i in range(len(SEEDS))]
    psm = {k: round(float(np.mean([x[k] for x in per])), 4) for k in ens}
    pss = {k: round(float(np.std([x[k] for x in per])), 4) for k in ens}
    out['tabpfn'][name] = {'n': int(mask.sum()),
                           'ensemble': {k: round(float(v), 4) for k, v in ens.items()},
                           'per_seed_mean': psm, 'per_seed_std': pss}
    print(f'[{name}] ENS roc={ens["roc_auc"]:.4f}  per-seed roc={psm["roc_auc"]:.4f}±{pss["roc_auc"]:.4f}', flush=True)

json.dump(out, open('/home/minji/exp1_tabpfn_888_329_results.json', 'w'), indent=2)
print('saved -> /home/minji/exp1_tabpfn_888_329_results.json')
