"""
bbb_project1.py — Project 1: Feature Combination Search

model_development.md § 5-1에 따라, 8개 피쳐의 2개 이상 조합(총 247개)을
고정 Baseline 모델로 체계적으로 탐색한다.

- scaffold split × 10 seed + random_scaffold split × 10 seed = 조합당 20회 학습
- 전체 데이터셋은 1회만 로드 (컬럼 슬라이싱으로 조합별 피쳐 선택)
- 매 seed 결과를 CSV에 즉시 저장 (중단 후 재개 가능)
- 탐색 완료 후 scaffold AUC-ROC 기준 최적 조합을 선정하여 모델/스플릿 아티팩트 저장

Usage:
    conda run -n rapids-25.02 python /home/minji/autoresearch/bbb_project1.py

Output (auto-created):
    /home/minji/BBB/project1/results_per_seed.csv    — 모든 seed별 상세 결과
    /home/minji/BBB/project1/results_summary.csv     — 조합 × split_mode 평균 요약
    /home/minji/BBB/project1/best_combo_info.json    — 최적 조합 정보
    /home/minji/BBB/project1/best_combo/             — 최적 조합 모델/스플릿 아티팩트
"""

import itertools
import json
import os
import pickle
import sys
import time
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.metrics import average_precision_score

# autoresearch 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bbb_prepare import (
    LABEL_PATH, EMBED_PATHS, FP_TYPES,
    ScageConcatDataset, split_then_normalize, eval_model,
    set_seed, device,
)
from bbb_train import (
    SpatialGatingUnit, gMLPBlock, gMLP,
    MultiModalGMLPFromFlat, train_model,
)

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------
SEEDS = [42, 100, 200, 300, 400, 500, 600, 700, 800, 900]
SPLIT_MODES = ['scaffold', 'random_scaffold']

# model_development.md § 5-1 Baseline Model 파라미터
BASELINE_CONFIG = {
    'd_model':        512,
    'd_ffn':          1048,
    'depth':          4,
    'dropout':        0.2,
    'use_gated_pool': True,
    'lr':             1e-4,
    'weight_decay':   1e-5,
    'num_epochs':     50,
    'patience':       10,
    'batch_size':     128,
}

OUT_DIR  = '/home/minji/BBB/project1'
SEED_CSV = os.path.join(OUT_DIR, 'results_per_seed.csv')
SUMM_CSV = os.path.join(OUT_DIR, 'results_summary.csv')
BEST_DIR = os.path.join(OUT_DIR, 'best_combo')

METRIC_COLS = [
    'accuracy', 'precision', 'recall', 'f1', 'roc_auc',
    'mcc', 'auprc', 'sensitivity', 'specificity',
    'train_pos_ratio', 'val_pos_ratio', 'test_pos_ratio',
]

# ---------------------------------------------------------------------------
# 피쳐 서브셋 래퍼
# ---------------------------------------------------------------------------
class SubsetDataset:
    """전체 데이터셋에서 지정한 피쳐 조합의 컬럼만 슬라이싱하는 경량 래퍼.
    split_then_normalize가 요구하는 .features / .labels / .df /
    .fp_types / .expected_dims 인터페이스를 제공한다."""

    def __init__(self, full_dataset, combo: list, all_offsets: dict):
        col_idxs = []
        for t in combo:
            start, end = all_offsets[t]
            col_idxs.extend(range(start, end))
        self.features      = full_dataset.features[:, col_idxs].clone()
        self.labels        = full_dataset.labels
        self.df            = full_dataset.df
        self.fp_types      = list(combo)
        self.expected_dims = {t: full_dataset.expected_dims[t] for t in combo}

# ---------------------------------------------------------------------------
# AUPRC 포함 평가 함수
# ---------------------------------------------------------------------------
def eval_model_full(model, loader) -> dict:
    """bbb_prepare.eval_model에 AUPRC를 추가한 버전."""
    metrics = eval_model(model, loader)

    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            probs = torch.sigmoid(model(x)).cpu().numpy()
            y_prob.extend(probs.tolist())
            y_true.extend(y.numpy().tolist())

    has_both = len(set(y_true)) > 1
    metrics['auprc'] = round(
        average_precision_score(y_true, y_prob) if has_both else 0.0, 4
    )
    return metrics

# ---------------------------------------------------------------------------
# 단일 seed 실행
# ---------------------------------------------------------------------------
def run_one_seed(
    subset_ds: SubsetDataset,
    combo: list,
    seed: int,
    split_mode: str,
    cfg: dict,
    save_dir: str = None,
) -> dict:
    """(combo, seed, split_mode) 조합으로 baseline 모델을 학습하고 test 지표를 반환.
    save_dir이 주어지면 모델/스플릿/스케일러 아티팩트를 저장한다."""

    set_seed(seed)

    train_ds, val_ds, test_ds, scaler, _, split_indices = split_then_normalize(
        subset_ds, split_mode=split_mode,
        train_ratio=0.8, val_ratio=0.1, seed=seed,
    )
    train_idx, val_idx, test_idx = split_indices

    train_loader = data.DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,  drop_last=False)
    val_loader   = data.DataLoader(val_ds,   batch_size=cfg['batch_size'], shuffle=False, drop_last=False)
    test_loader  = data.DataLoader(test_ds,  batch_size=cfg['batch_size'], shuffle=False, drop_last=False)

    # 클래스 불균형 처리 (pos_weight)
    y_train = train_ds.tensors[1].cpu().numpy()
    y_val   = val_ds.tensors[1].cpu().numpy()
    y_test  = test_ds.tensors[1].cpu().numpy()
    n_pos = float((y_train == 1).sum())
    n_neg = float((y_train == 0).sum())
    pos_weight = (
        torch.tensor([max(n_neg / n_pos, 1.0)], dtype=torch.float32, device=device)
        if n_pos > 0 else None
    )
    loss_fn = (
        nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        if pos_weight is not None else nn.BCEWithLogitsLoss()
    )

    # 모델 초기화
    set_seed(seed)
    mod_dims = OrderedDict((t, subset_ds.expected_dims[t]) for t in combo)
    model = MultiModalGMLPFromFlat(
        mod_dims       = mod_dims,
        d_model        = cfg['d_model'],
        d_ffn          = cfg['d_ffn'],
        depth          = cfg['depth'],
        dropout        = cfg['dropout'],
        use_gated_pool = cfg['use_gated_pool'],
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay']
    )

    model = train_model(
        model, optimizer, train_loader, val_loader, loss_fn,
        num_epochs=cfg['num_epochs'], patience=cfg['patience'],
    )

    # 평가
    metrics = eval_model_full(model, test_loader)
    metrics['train_pos_ratio'] = round(float(np.mean(y_train)), 4)
    metrics['val_pos_ratio']   = round(float(np.mean(y_val)),   4)
    metrics['test_pos_ratio']  = round(float(np.mean(y_test)),  4)

    # 아티팩트 저장 (최적 조합에만 사용)
    if save_dir:
        os.makedirs(os.path.join(save_dir, 'models'),   exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'splits'),   exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'scalers'),  exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'configs'),  exist_ok=True)

        tag = f"{split_mode}_seed{seed}"
        torch.save(model.state_dict(),
                   os.path.join(save_dir, 'models', f"model_{tag}.pth"))
        np.save(os.path.join(save_dir, 'splits', f"train_idx_{tag}.npy"),
                np.array(train_idx, dtype=np.int64))
        np.save(os.path.join(save_dir, 'splits', f"val_idx_{tag}.npy"),
                np.array(val_idx,   dtype=np.int64))
        np.save(os.path.join(save_dir, 'splits', f"test_idx_{tag}.npy"),
                np.array(test_idx,  dtype=np.int64))
        if scaler is not None:
            with open(os.path.join(save_dir, 'scalers', f"scaler_{tag}.pkl"), 'wb') as f:
                pickle.dump(scaler, f)

        cfg_record = {
            **cfg,
            'combo': combo,
            'split_mode': split_mode,
            'seed': seed,
            'mod_dims': {k: int(v) for k, v in mod_dims.items()},
            **metrics,
        }
        with open(os.path.join(save_dir, 'configs', f"config_{tag}.json"), 'w') as f:
            json.dump(cfg_record, f, indent=2)

    return metrics

# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Step 1. 전체 데이터셋 1회 로드 ──────────────────────────────────────
    print('[Step 1] 전체 데이터셋 로드 (8개 피쳐 전부)...')
    t0 = time.time()
    full_ds = ScageConcatDataset(LABEL_PATH, EMBED_PATHS, fp_types=FP_TYPES)
    print(f'  로드 완료: {len(full_ds)}개 샘플 | '
          f'feature_dim={full_ds.features.shape[1]} | {time.time()-t0:.1f}s')

    # 피쳐별 컬럼 오프셋 사전 계산
    all_offsets: dict = {}
    offset = 0
    for t in FP_TYPES:
        dim = full_ds.expected_dims[t]
        all_offsets[t] = (offset, offset + dim)
        offset += dim
    print(f'  피쳐 오프셋: { {t: all_offsets[t] for t in FP_TYPES} }')

    # ── Step 2. 모든 조합 열거 (2개 이상) ────────────────────────────────────
    all_combos = []
    for r in range(2, len(FP_TYPES) + 1):
        for combo in itertools.combinations(FP_TYPES, r):
            all_combos.append(list(combo))
    print(f'\n[Step 2] 총 피쳐 조합 수: {len(all_combos)}개')

    # ── Step 3. 재개 상태 확인 ────────────────────────────────────────────────
    if os.path.exists(SEED_CSV):
        done_df  = pd.read_csv(SEED_CSV)
        done_keys = set(
            zip(done_df['combo_str'], done_df['split_mode'], done_df['seed'])
        )
        print(f'  재개: {len(done_keys)}개 seed-run 이미 완료')
    else:
        done_keys = set()
        # CSV 헤더 초기화
        header = pd.DataFrame(columns=['combo_str', 'n_features', 'split_mode', 'seed'] + METRIC_COLS)
        header.to_csv(SEED_CSV, index=False)
        print('  새 실험 시작 (results_per_seed.csv 초기화)')

    # ── Step 4. 탐색 루프 ─────────────────────────────────────────────────────
    print(f'\n[Step 4] 피쳐 조합 탐색 시작')
    total = len(all_combos)
    t_search_start = time.time()

    for ci, combo in enumerate(all_combos):
        combo_str  = '+'.join(combo)
        n_features = len(combo)

        # 이 조합에 대한 미완료 seed-run 수 계산
        pending = [
            (sm, s)
            for sm in SPLIT_MODES for s in SEEDS
            if (combo_str, sm, s) not in done_keys
        ]
        if not pending:
            print(f'[{ci+1}/{total}] SKIP (완료): {combo_str}')
            continue

        print(f'\n[{ci+1}/{total}] combo={combo_str} ({n_features}개 피쳐) | '
              f'미완료 {len(pending)}/20 runs')

        subset_ds = SubsetDataset(full_ds, combo, all_offsets)
        combo_rows = []

        for split_mode in SPLIT_MODES:
            for seed in SEEDS:
                if (combo_str, split_mode, seed) in done_keys:
                    continue
                try:
                    t_run = time.time()
                    metrics = run_one_seed(
                        subset_ds, combo, seed, split_mode, BASELINE_CONFIG
                    )
                    elapsed = time.time() - t_run

                    row = {
                        'combo_str':  combo_str,
                        'n_features': n_features,
                        'split_mode': split_mode,
                        'seed':       seed,
                        **{k: metrics.get(k, float('nan')) for k in METRIC_COLS},
                    }
                    combo_rows.append(row)
                    print(f'  {split_mode} seed={seed:3d}: '
                          f'AUC={metrics.get("roc_auc", 0):.4f} '
                          f'F1={metrics.get("f1", 0):.4f} '
                          f'MCC={metrics.get("mcc", 0):.4f} '
                          f'({elapsed:.1f}s)')

                except Exception as e:
                    import traceback
                    print(f'  ERROR {split_mode} seed={seed}: {e}')
                    traceback.print_exc()

        # 즉시 CSV에 추가 저장
        if combo_rows:
            pd.DataFrame(combo_rows).to_csv(SEED_CSV, mode='a', header=False, index=False)

    elapsed_total = time.time() - t_search_start
    print(f'\n탐색 완료. 소요 시간: {elapsed_total/3600:.1f}시간')

    # ── Step 5. 요약 & 최적 조합 선정 ────────────────────────────────────────
    print('\n[Step 5] 결과 요약 및 최적 조합 선정...')
    df = pd.read_csv(SEED_CSV)

    summ = (
        df.groupby(['combo_str', 'n_features', 'split_mode'])[METRIC_COLS]
        .mean()
        .round(4)
        .reset_index()
    )
    summ.to_csv(SUMM_CSV, index=False)
    print(f'  요약 저장 → {SUMM_CSV}')

    # scaffold AUC-ROC 기준 1차 정렬, random_scaffold AUC-ROC 기준 2차 정렬
    scaf = (
        summ[summ['split_mode'] == 'scaffold'][['combo_str', 'roc_auc']]
        .rename(columns={'roc_auc': 'scaffold_auc'})
    )
    rs = (
        summ[summ['split_mode'] == 'random_scaffold'][['combo_str', 'roc_auc']]
        .rename(columns={'roc_auc': 'rs_auc'})
    )
    merged = scaf.merge(rs, on='combo_str')
    best_row     = merged.sort_values(['scaffold_auc', 'rs_auc'], ascending=False).iloc[0]
    best_combo_str = best_row['combo_str']
    best_combo     = best_combo_str.split('+')

    print(f'\n  ★ 최적 피쳐 조합: {best_combo_str}')
    print(f'    scaffold AUC-ROC       = {best_row["scaffold_auc"]:.4f}')
    print(f'    random_scaffold AUC-ROC = {best_row["rs_auc"]:.4f}')

    # 상위 10개 조합 출력
    top10 = merged.sort_values(['scaffold_auc', 'rs_auc'], ascending=False).head(10)
    print('\n  [Top 10 조합]')
    print(top10.to_string(index=False))

    best_info = {
        'combo':        best_combo,
        'combo_str':    best_combo_str,
        'scaffold_auc': float(best_row['scaffold_auc']),
        'rs_auc':       float(best_row['rs_auc']),
    }
    with open(os.path.join(OUT_DIR, 'best_combo_info.json'), 'w') as f:
        json.dump(best_info, f, indent=2, ensure_ascii=False)

    # ── Step 6. 최적 조합 아티팩트 저장 (재학습) ─────────────────────────────
    print(f'\n[Step 6] 최적 조합 아티팩트 저장 (재학습): {best_combo_str}')
    os.makedirs(BEST_DIR, exist_ok=True)

    subset_ds = SubsetDataset(full_ds, best_combo, all_offsets)

    for split_mode in SPLIT_MODES:
        for seed in SEEDS:
            print(f'  저장: {split_mode} seed={seed}')
            run_one_seed(
                subset_ds, best_combo, seed, split_mode,
                BASELINE_CONFIG, save_dir=BEST_DIR,
            )

    # best_combo 전체 config 저장
    with open(os.path.join(BEST_DIR, 'best_combo_config.json'), 'w') as f:
        json.dump({
            **best_info,
            'baseline_config': BASELINE_CONFIG,
            'seeds': SEEDS,
            'split_modes': SPLIT_MODES,
        }, f, indent=2, ensure_ascii=False)

    print(f'\n[Done] Project 1 완료.')
    print(f'  per-seed 결과: {SEED_CSV}')
    print(f'  요약:          {SUMM_CSV}')
    print(f'  최적 조합:     {best_combo_str}')
    print(f'  아티팩트:      {BEST_DIR}')


if __name__ == '__main__':
    main()
