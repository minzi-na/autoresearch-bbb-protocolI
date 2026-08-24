"""
TabPFN-2.5 inference on the GM-BBB external set (8109), combo1 features.
Per seed: scaffold-split internal -> fit scaler+TabPFN on train -> predict on
external 8109. Per-seed mean±std + soft-vote ensemble (threshold prob > 0.5).

Run: /home/minji/anaconda3/envs/tabpfn/bin/python eval_tabpfn_ext8109.py
"""
import sys, json
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (matthews_corrcoef, accuracy_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score)

sys.path.insert(0, '/home/minji/bbb-combo1')
from tabpfn25_baseline import (load_dataset, scaffold_split,
    INT_LABEL, INT_MOLE, MACCS_DIM, AVALON_BITS, RDKIT_DIM, SEEDS)
from tabpfn import TabPFNClassifier

EXT_LABEL='/home/minji/BBB/holdout_splits/external_cls_only_holdout_10pct_seed42/external_cls_only_label_remaining.csv'
EXT_MOLE ='/home/minji/BBB/holdout_splits/external_cls_only_holdout_10pct_seed42/external_cls_only_mole_remaining.csv'

def metrics(yt,yp):
    yt=np.asarray(yt,int); yp=np.asarray(yp,float); pred=(yp>0.5).astype(int)
    cm=confusion_matrix(yt,pred)
    spec=cm[0,0]/(cm[0,0]+cm[0,1]) if cm.size==4 and (cm[0,0]+cm[0,1])>0 else 0.0
    hb=len(set(yt.tolist()))>1
    return dict(roc_auc=float(roc_auc_score(yt,yp)) if hb else 0.0,
                mcc=float(matthews_corrcoef(yt,pred)), f1=float(f1_score(yt,pred,zero_division=0)),
                accuracy=float(accuracy_score(yt,pred)),
                auprc=float(average_precision_score(yt,yp)) if hb else 0.0, specificity=float(spec))

X_int,y_int,smi_int,_=load_dataset(INT_LABEL,INT_MOLE)
X_ext,y_ext,smi_ext,_=load_dataset(EXT_LABEL,EXT_MOLE)
print(f'internal={X_int.shape}  external={X_ext.shape} pos={int(np.sum(y_ext))}',flush=True)
rd_start=MACCS_DIM+AVALON_BITS; rd_end=rd_start+RDKIT_DIM

seed_probs=[]
for seed in SEEDS:
    tr,va,te=scaffold_split(smi_int,y_int,seed=seed)
    X_tr=X_int[tr].copy(); y_tr=y_int[tr]; X_e=X_ext.copy()
    scaler=StandardScaler().fit(X_tr[:,rd_start:rd_end])
    X_tr[:,rd_start:rd_end]=scaler.transform(X_tr[:,rd_start:rd_end]).astype(np.float32)
    X_e[:,rd_start:rd_end]=scaler.transform(X_e[:,rd_start:rd_end]).astype(np.float32)
    clf=TabPFNClassifier(n_estimators=8,random_state=seed,ignore_pretraining_limits=False)
    clf.fit(X_tr,y_tr)
    p=clf.predict_proba(X_e)[:,1]
    seed_probs.append(np.asarray(p,dtype=np.float64))
    print(f'  seed={seed:>4d}  ext_roc={roc_auc_score(y_ext,p):.4f}',flush=True)

ps=np.stack(seed_probs); y=np.asarray(y_ext,int)
ens=metrics(y,ps.mean(0)); per=[metrics(y,ps[i]) for i in range(len(SEEDS))]
psm={k:round(float(np.mean([x[k] for x in per])),4) for k in ens}
pss={k:round(float(np.std([x[k] for x in per])),4) for k in ens}
out={'tabpfn':{'n':int(len(y)),'ensemble':{k:round(float(v),4) for k,v in ens.items()},
               'per_seed_mean':psm,'per_seed_std':pss}}
print(f'[ext8109] ENS roc={ens["roc_auc"]:.4f}  per-seed roc={psm["roc_auc"]:.4f}±{pss["roc_auc"]:.4f}  auprc={psm["auprc"]:.4f}')
json.dump(out,open('/home/minji/exp1_tabpfn_ext8109_results.json','w'),indent=2)
print('saved -> /home/minji/exp1_tabpfn_ext8109_results.json')
