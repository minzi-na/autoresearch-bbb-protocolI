"""
GateMol-BBB internal test metrics + ext/holdout ensemble ROC-AUC.
Run from each worktree directory.

Usage:
    conda run -n rapids-25.02 python gmlp_eval.py \
        --model_dir bbb_artifacts/phase2_best \
        --output gmlp_metrics.json
"""

import os, sys, json, pickle, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef, f1_score, accuracy_score
)
from bbb_prepare import (
    split_then_normalize, eval_model,
    LABEL_PATH, EMBED_PATHS, FP_TYPES,
)
# Use best-commit architecture (cbfe537) to match saved model weights
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location('bbb_train_best',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bbb_train_best.py'))
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
MultiModalGMLPFromFlat = _mod.MultiModalGMLPFromFlat
predict_probs       = _mod.predict_probs
roc_auc_from_probs  = _mod.roc_auc_from_probs
load_cached_dataset = _mod.load_cached_dataset
apply_scaler        = _mod.apply_scaler
SEEDS               = _mod.SEEDS
SPLIT_MODE          = _mod.SPLIT_MODE
BASE_CONFIG         = _mod.BASE_CONFIG
EXT_LABEL_PATH      = _mod.EXT_LABEL_PATH
EXT_EMBED_PATHS     = _mod.EXT_EMBED_PATHS
HOLDOUT_LABEL_PATH  = _mod.HOLDOUT_LABEL_PATH
HOLDOUT_EMBED_PATHS = _mod.HOLDOUT_EMBED_PATHS
from collections import OrderedDict
import torch.utils.data as data

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
THRESHOLD = 0.5


def load_model(seed_dir, mod_dims):
    cfg_path = os.path.join(seed_dir, 'config.json')
    with open(cfg_path) as f:
        cfg = json.load(f)
    hparams = cfg.get('hparams', cfg.get('base_config', {}))

    import inspect
    init_params = inspect.signature(MultiModalGMLPFromFlat.__init__).parameters
    kwargs = dict(
        mod_dims=mod_dims,
        d_model=hparams.get('d_model', BASE_CONFIG['d_model']),
        d_ffn=hparams.get('d_ffn', BASE_CONFIG['d_ffn']),
        depth=hparams.get('depth', BASE_CONFIG['depth']),
        dropout=hparams.get('dropout', BASE_CONFIG['dropout']),
        use_gated_pool=True,
    )
    if 'stochastic_depth_rate' in init_params:
        kwargs['stochastic_depth_rate'] = hparams.get('stochastic_depth_rate', 0.05)

    model = MultiModalGMLPFromFlat(**kwargs).to(DEVICE)
    state = torch.load(os.path.join(seed_dir, 'model.pth'),
                       map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def get_probs(model, X: torch.Tensor) -> np.ndarray:
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            xb = X[i:i+256].to(DEVICE)
            probs.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(probs)


def compute_cls_metrics(y_true, probs, threshold=THRESHOLD):
    preds = (probs >= threshold).astype(int)
    return {
        'roc_auc':  round(float(roc_auc_score(y_true, probs)), 4),
        'mcc':      round(float(matthews_corrcoef(y_true, preds)), 4),
        'f1':       round(float(f1_score(y_true, preds, zero_division=0)), 4),
        'accuracy': round(float(accuracy_score(y_true, preds)), 4),
    }


def main(model_dir, output_path):
    print(f'Loading datasets...')
    dataset = load_cached_dataset('internal', LABEL_PATH, EMBED_PATHS, fp_types=FP_TYPES)
    expected_dims = dataset.expected_dims
    mod_dims = OrderedDict((t, expected_dims[t]) for t in FP_TYPES)

    ext_dataset = load_cached_dataset(
        'external', EXT_LABEL_PATH, EXT_EMBED_PATHS,
        fp_types=FP_TYPES, expected_dims=expected_dims,
    )
    holdout_dataset = load_cached_dataset(
        'holdout', HOLDOUT_LABEL_PATH, HOLDOUT_EMBED_PATHS,
        fp_types=FP_TYPES, expected_dims=expected_dims,
    )

    seed_metrics = []
    ext_probs_list, holdout_probs_list = [], []

    for seed in SEEDS:
        seed_dir = os.path.join(model_dir, f'seed_{seed}')
        if not os.path.exists(os.path.join(seed_dir, 'model.pth')):
            print(f'  [skip] seed={seed} not found')
            continue

        # Load scaler and split
        scaler_path = os.path.join(seed_dir, 'rdkit_scaler.pkl')
        scaler = None
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)

        from bbb_prepare import set_seed
        set_seed(seed)
        train_ds, val_ds, test_ds, scaler_fit, (rd_start, rd_end), _ = split_then_normalize(
            dataset, split_mode=SPLIT_MODE,
            train_ratio=0.8, val_ratio=0.1, seed=seed,
        )
        # Use saved scaler (fitted on training data during original training)
        if scaler is not None:
            scaler_fit = scaler

        # Internal test
        X_test = test_ds.tensors[0]
        y_test  = test_ds.tensors[1].numpy().astype(int)
        # test_ds already normalized by split_then_normalize

        # Ext / holdout
        X_ext     = apply_scaler(ext_dataset.features,     scaler_fit, rd_start, rd_end)
        X_holdout = apply_scaler(holdout_dataset.features, scaler_fit, rd_start, rd_end)

        model = load_model(seed_dir, mod_dims)
        test_probs    = get_probs(model, X_test)
        ext_probs     = get_probs(model, X_ext)
        holdout_probs = get_probs(model, X_holdout)
        del model

        m = compute_cls_metrics(y_test, test_probs)
        seed_metrics.append(m)
        ext_probs_list.append(ext_probs)
        holdout_probs_list.append(holdout_probs)
        print(f'  seed={seed:4d}  roc={m["roc_auc"]:.4f}  mcc={m["mcc"]:.4f}'
              f'  f1={m["f1"]:.4f}  acc={m["accuracy"]:.4f}')

    # Mean internal metrics
    mean_metrics = {
        k: round(float(np.mean([m[k] for m in seed_metrics])), 4)
        for k in ['roc_auc', 'mcc', 'f1', 'accuracy']
    }

    # Soft voting ensemble
    y_ext     = ext_dataset.labels.numpy().astype(int)
    y_holdout = holdout_dataset.labels.numpy().astype(int)
    roc_ext     = round(float(roc_auc_score(y_ext,     np.mean(ext_probs_list,     axis=0))), 4)
    roc_holdout = round(float(roc_auc_score(y_holdout, np.mean(holdout_probs_list, axis=0))), 4)

    result = {
        'model_dir':    model_dir,
        'fp_types':     FP_TYPES,
        'n_seeds':      len(seed_metrics),
        'internal':     mean_metrics,
        'roc_ext':      roc_ext,
        'roc_holdout':  roc_holdout,
        'seed_metrics': seed_metrics,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f'\n  [mean] roc={mean_metrics["roc_auc"]}  mcc={mean_metrics["mcc"]}'
          f'  f1={mean_metrics["f1"]}  acc={mean_metrics["accuracy"]}')
    print(f'  [ensemble] roc_ext={roc_ext}  roc_holdout={roc_holdout}')
    print(f'  Saved -> {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--output',    required=True)
    args = parser.parse_args()
    main(args.model_dir, args.output)
