# Combo1 저장 모델 요약

## 모델 목록 및 디렉토리

| 모델 ID | 저장 경로 | 아키텍처 | 하이퍼파라미터 | pos_weight |
|---------|-----------|----------|----------------|------------|
| `iter90_pw008` | `bbb_artifacts/phase2_best/` | iter90 | Phase 2 최적 (dropout=0.046, lr=1.10e-4, wd=3.88e-6) | **0.08 고정** |
| `iter90_p2_auto` | `bbb_artifacts/phase2_best_auto/` | iter90 | Phase 2 최적 (동일) | **auto** (n_neg/n_pos per seed) |
| `iter90_base_auto` | `bbb_artifacts/iter90_base_auto/` | iter90 | BASE (dropout=0.1, lr=1e-4, wd=1e-5) | **auto** (n_neg/n_pos per seed, clamp 0.1~10) |
| `iter55_base_auto` | `bbb_artifacts/best/` | iter55 | BASE (dropout=0.1, lr=1e-4, wd=1e-5) | **auto** (n_neg/n_pos per seed) |
| `tabpfn_full` (baseline) | `tabpfn_baseline_results.tsv` | TabPFN-2.5 (in-context, no gradient) | maccs+avalon+rdkit+mole (1663-d) | N/A (메타 학습된 분류기) |

---

## 모델 간 차이

### 아키텍처 차이: iter90 vs iter55

| 구성요소 | iter90 | iter55 |
|----------|--------|--------|
| Norm | RMSNorm (`scale` 파라미터) | LayerNorm (`weight/bias` 파라미터) |
| Activation | SiLU | GELU |
| Pooling | `pool_query`: 입력 의존적 attention (동적 가중치) | `alpha`: 학습된 고정 softmax 가중치 (정적) |
| Skip connection | `skip_gate` (sigmoid scalar) | 동일 |
| Stochastic depth | inline Bernoulli mask | DropPath 모듈 |

### pos_weight 차이

- **pw=0.08 고정**: Phase 2 튜닝 당시 설정값. Internal ROC-AUC는 높지만, BCEWithLogitsLoss의 양성 편향으로 인해 외부/홀드아웃 데이터에서 threshold 보정이 무너져 MCC/F1이 급락함.
- **pw=auto**: 각 seed의 학습 데이터에서 `n_neg / n_pos`를 계산해 적용. 실제 클래스 비율을 반영하므로 일반화 성능(MCC, F1)이 안정적.

### 하이퍼파라미터 차이: Phase2 최적 vs BASE

| 파라미터 | Phase2 최적 | BASE |
|----------|-------------|------|
| dropout | 0.046 | 0.1 |
| lr | 1.10e-4 | 1e-4 |
| weight_decay | 3.88e-6 | 1e-5 |
| d_model | 512 | 512 |
| d_ffn | 1048 | 1048 |
| depth | 4 | 4 |

---

## 성능 비교 (10-seed / soft voting ensemble)

### Internal Test Set (10-seed mean ± std)

| 모델 | ROC-AUC | MCC | F1 | ACC |
|------|---------|-----|----|-----|
| iter90_pw008 | **0.8785** ± 0.0165 | 0.4369 ± 0.0387 | 0.9062 ± 0.0141 | 0.8437 ± 0.0205 |
| iter90_p2_auto | 0.8622 ± 0.0193 | 0.4339 ± 0.0561 | 0.9256 ± 0.0075 | 0.8701 ± 0.0126 |
| iter90_base_auto | 0.8614 ± 0.0224 | 0.4415 ± 0.0570 | 0.9266 ± 0.0077 | 0.8719 ± 0.0129 |
| iter55_base_auto | 0.8647 ± 0.0206 | **0.4571** ± 0.0502 | **0.9283** ± 0.0072 | **0.8749** ± 0.0121 |
| tabpfn_full (baseline) | 0.8499 | 0.3492 | 0.9240 | — |

### External Dataset (soft voting ensemble)

| 모델 | ROC-AUC | MCC | F1 | ACC |
|------|---------|-----|----|-----|
| iter90_pw008 | 0.7779 | 0.2730 | 0.5090 | 0.5325 |
| iter90_p2_auto | **0.7788** | 0.3438 | 0.7038 | **0.6552** |
| iter90_base_auto | **0.7788** | 0.3447 | 0.7016 | 0.6540 |
| iter55_base_auto | 0.7787 | 0.3280 | 0.6734 | 0.6328 |
| tabpfn_full (baseline) | 0.7683 | **0.3966** | **0.7960** | — |

### Holdout Set (soft voting ensemble)

| 모델 | ROC-AUC | MCC | F1 | ACC |
|------|---------|-----|----|-----|
| iter90_pw008 | 0.8170 | 0.3237 | 0.5945 | 0.5803 |
| iter90_p2_auto | 0.8155 | 0.4148 | 0.7568 | 0.7025 |
| iter90_base_auto | 0.8163 | 0.4188 | 0.7583 | **0.7043** |
| iter55_base_auto | **0.8171** | 0.3892 | 0.7276 | 0.6768 |
| tabpfn_full (baseline) | 0.8055 | **0.4508** | **0.8261** | — |

> TabPFN-2.5는 ACC를 별도 보고하지 않아 비워둠. TabPFN은 단일 inference (10-seed 메타분포 내재화) → soft voting 미적용.

---

## 주요 관찰

1. **pw=0.08은 Internal AUC가 가장 높지만 일반화 실패**: External F1=0.51, Holdout MCC=0.32로 threshold 편향이 심함. AUC는 순위 기반 지표라 편향에 둔감하지만 실제 분류 품질(MCC/F1)은 급락.

2. **pw=auto 전환 시 MCC/F1 대폭 개선**: External MCC 0.27→0.34, Holdout MCC 0.32→0.42 수준으로 향상. ROC-AUC는 거의 동일하게 유지.

3. **iter90 BASE vs P2_BEST (pw=auto)**: 성능 차이 미미 (ROC-AUC 0.8614 vs 0.8622). Phase 2 튜닝의 효과가 pw=auto 상태에서는 거의 사라짐.

4. **iter55 vs iter90 (pw=auto)**: Internal에서 iter55가 MCC/F1/ACC 모두 약간 우세. External에서는 iter90이 MCC/F1 우세. Holdout AUC는 iter55가 미세하게 높음.

5. **TabPFN-2.5 baseline 비교**:
   - **ROC-AUC**: 모든 데이터셋에서 combo1 모델 < TabPFN. Internal 0.8499, External 0.7683, Holdout 0.8055로 가장 낮음 → 분리력(ranking 품질) 면에서 combo1이 우위.
   - **MCC/F1**: External과 Holdout에서 TabPFN이 가장 높음 (Ext MCC 0.3966 vs combo1 최고 0.3447, Hold MCC 0.4508 vs 0.4188; Ext F1 0.7960 vs 0.7038, Hold F1 0.8261 vs 0.7583) → threshold 0.5 기준 분류 품질은 TabPFN이 더 안정적.
   - **해석**: TabPFN은 메타학습된 in-context 분류기로 calibration이 자연스럽게 잘 맞아 분류 품질이 좋지만, 상대적 분리력(AUC)은 도메인 특화 멀티모달 모델보다 낮음. combo1은 ranking은 잘하지만 threshold 보정 여지가 남아있음.
