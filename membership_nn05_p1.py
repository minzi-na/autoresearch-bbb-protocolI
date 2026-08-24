"""Protocol I (COMBO1) nn05 per-molecule correctness, split by pretraining-corpus
membership (SCAGE-5M union GuacaMol). Mirrors eval_holdout_subset_888_329.py inference.
"""
import os, numpy as np, torch
import torch.utils.data as data
from collections import OrderedDict
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')

from bbb_prepare import set_seed, split_then_normalize, LABEL_PATH, EMBED_PATHS, FP_TYPES, device
from bbb_iter90 import load_cached_dataset, apply_scaler, predict_probs, SEEDS, SPLIT_MODE
from eval_all_models import load_model

ARTIFACT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bbb_artifacts')
EXP1 = OrderedDict([('pw008', os.path.join(ARTIFACT_BASE,'phase2_best')),
                    ('auto',  os.path.join(ARTIFACT_BASE,'phase2_best_auto'))])
HOLD_LABEL='/home/minji/BBB/holdout_splits/merged_holdout_10pct_seed42/label_holdout.csv'
HOLD_EMB={'scage1':'/home/minji/BBB/holdout_splits/merged_holdout_10pct_seed42/scage1_holdout.csv',
          'scage2':'/home/minji/BBB/holdout_splits/merged_holdout_10pct_seed42/scage2_holdout.csv',
          'mole':'/home/minji/BBB/holdout_splits/merged_holdout_10pct_seed42/mole_holdout.csv'}
NN05='/home/minji/holdout_subset/merged_holdout_10pct_seed42_simfilter09_nn05/label_holdout.csv'

internal = load_cached_dataset('internal', LABEL_PATH, EMBED_PATHS, fp_types=FP_TYPES)
expected = internal.expected_dims
mod_dims = OrderedDict((t, expected[t]) for t in FP_TYPES)
holdout = load_cached_dataset('holdout', HOLD_LABEL, HOLD_EMB, fp_types=FP_TYPES, expected_dims=expected)
hold_smiles=list(holdout.df['smiles']); y_hold=holdout.labels.numpy().astype(int)

import pandas as pd
nn05_smiles=set(pd.read_csv(NN05)['smiles'])
def canon(s,iso):
    m=Chem.MolFromSmiles(str(s)); return Chem.MolToSmiles(m,isomericSmiles=iso) if m else None
# nn05 mask within 1089 (match by raw or canonical)
nn05_canon={canon(s,True) for s in nn05_smiles}|{canon(s,False) for s in nn05_smiles}|set(nn05_smiles)
mask=np.array([ (s in nn05_smiles) or (canon(s,True) in nn05_canon) for s in hold_smiles ], dtype=bool)
print('nn05 matched in cache:', int(mask.sum()))

# membership: build lookup for nn05 cache smiles, scan corpora
idxs=np.where(mask)[0]
v2pos={}
for pos in idxs:
    s=hold_smiles[pos]
    for v in {str(s),canon(s,True),canon(s,False)}:
        if v: v2pos.setdefault(v,set()).add(pos)
def scan(p,e):
    seen=set()
    with open(p,encoding=e) as f:
        for ln in f:
            t=ln.strip()
            if t in v2pos: seen|=v2pos[t]
    return seen
seen=scan('/tmp/scage_corpus/pretrain-5m.txt','utf-16')|scan('/home/minji/Desktop/guacamol_v1_all.smiles','utf-8')
seen_mask=np.array([p in seen for p in idxs])
print(f'nn05 seen(union)={seen_mask.sum()}  unseen={(~seen_mask).sum()}')

ens_by_variant={}; perseed_by_variant={}
for name,art in EXP1.items():
    seed_probs=[]
    for seed in SEEDS:
        set_seed(seed)
        _,_,_,scaler,(rs,re),_=split_then_normalize(internal,split_mode=SPLIT_MODE,train_ratio=0.8,val_ratio=0.1,seed=seed)
        hX=apply_scaler(holdout.features,scaler,rs,re)
        model,_,_=load_model(art,seed,mod_dims); model=model.to(device)
        loader=data.DataLoader(data.TensorDataset(hX,holdout.labels),batch_size=256,shuffle=False)
        seed_probs.append(predict_probs(model,loader)); del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    stack=np.stack(seed_probs,0)                   # (10, 1089)
    ens=stack.mean(0)                              # per-seed mean prob (soft-vote)
    ens_by_variant[name]=ens[idxs]
    perseed_by_variant[name]=stack[:,idxs]         # (10, 329)
    e=ens[idxs]; y=y_hold[idxs]; correct=((e>0.5).astype(int)==y).astype(int)
    cs=correct[seen_mask]; cu=correct[~seen_mask]
    print(f'\n[Protocol I {name}] nn05 per-seed-mean prob>0.5 accuracy')
    print(f'  seen  : {cs.sum()}/{len(cs)} = {cs.mean():.4f}')
    print(f'  unseen: {cu.sum()}/{len(cu)} = {cu.mean():.4f}')
    print(f'  overall: {correct.sum()}/{len(correct)} = {correct.mean():.4f}')

# save per-molecule predictions (nn05) keyed by smiles for downstream near-membership analysis
out=pd.DataFrame({'smiles':[hold_smiles[i] for i in idxs],
                  'p_np':y_hold[idxs],
                  'ens_pw008':np.round(ens_by_variant['pw008'],6),
                  'ens_auto':np.round(ens_by_variant['auto'],6)})
out.to_csv('/home/minji/BBB_paper/shared_splits/predictions/protocolI_nn05_predictions.csv',index=False)
print('\nsaved /home/minji/BBB_paper/shared_splits/predictions/protocolI_nn05_predictions.csv', out.shape)

# per-seed probabilities (pw0.08 primary) for per-seed-mean membership analysis
ps=pd.DataFrame({'smiles':[hold_smiles[i] for i in idxs],'p_np':y_hold[idxs]})
for j,seed in enumerate(SEEDS):
    ps[f'prob_seed{seed}']=np.round(perseed_by_variant['pw008'][j],6)
ps.to_csv('/home/minji/BBB_paper/shared_splits/predictions/protocolI_nn05_perseed_pw008.csv',index=False)
print('saved protocolI_nn05_perseed_pw008.csv', ps.shape)
