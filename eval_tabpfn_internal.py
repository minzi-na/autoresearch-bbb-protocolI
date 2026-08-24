"""
TabPFN-2.5 per-seed internal-test evaluation (combo1), to obtain mean±std
consistent with the other baselines. Per seed: scaffold-split internal ->
fit scaler+TabPFN on train -> predict on that seed's internal TEST split.

Run: /home/minji/anaconda3/envs/tabpfn/bin/python eval_tabpfn_internal.py
"""
import sys, json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (matthews_corrcoef, accuracy_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score)

sys.path.insert(0, '/home/minji/bbb-combo1')
from tabpfn25_baseline import (load_dataset, scaffold_split,
    INT_LABEL, INT_MOLE, MACCS_DIM, AVALON_BITS, RDKIT_DIM, SEEDS)
from tabpfn import TabPFNClassifier

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
rd_start=MACCS_DIM+AVALON_BITS; rd_end=rd_start+RDKIT_DIM
print(f'internal={X_int.shape}',flush=True)

per=[]
for seed in SEEDS:
    tr,va,te=scaffold_split(smi_int,y_int,seed=seed)
    X_tr=X_int[tr].copy(); y_tr=y_int[tr]; X_te=X_int[te].copy(); y_te=y_int[te]
    scaler=StandardScaler().fit(X_tr[:,rd_start:rd_end])
    X_tr[:,rd_start:rd_end]=scaler.transform(X_tr[:,rd_start:rd_end]).astype(np.float32)
    X_te[:,rd_start:rd_end]=scaler.transform(X_te[:,rd_start:rd_end]).astype(np.float32)
    clf=TabPFNClassifier(n_estimators=8,random_state=seed,ignore_pretraining_limits=False)
    clf.fit(X_tr,y_tr)
    p=clf.predict_proba(X_te)[:,1]
    m=metrics(y_te,p); per.append(m)
    print(f'  seed={seed:>4d}  int_roc={m["roc_auc"]:.4f} n_test={len(te)}',flush=True)

MK=["roc_auc","auprc","mcc","f1","accuracy","precision","recall","specificity"] if "precision" in per[0] else list(per[0].keys())
mean={k:round(float(np.mean([x[k] for x in per])),6) for k in per[0]}
std ={k:round(float(np.std ([x[k] for x in per])),6) for k in per[0]}
out={'tabpfn':{'eval_set':'internal_test','n_seeds':len(SEEDS),'per_seed_mean':mean,'per_seed_std':std}}
print(f'\n[internal] ROC {mean["roc_auc"]:.4f}±{std["roc_auc"]:.4f}  AUPRC {mean["auprc"]:.4f}±{std["auprc"]:.4f}  '
      f'MCC {mean["mcc"]:.4f}±{std["mcc"]:.4f}  F1 {mean["f1"]:.4f}  ACC {mean["accuracy"]:.4f}')
json.dump(out,open('/home/minji/exp1_tabpfn_internal_results.json','w'),indent=2)
print('saved -> /home/minji/exp1_tabpfn_internal_results.json')
