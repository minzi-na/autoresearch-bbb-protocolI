# BBB 실험 구조 개요

## 1. 전체 목표

혈뇌장벽(BBB) 투과성 이진 분류 모델의 **composite score를 최대화**한다.

```
composite_score = f1 + mcc + roc_auc   (최대 3.0)
```

두 가지 scaffold split 모드 모두에서 점수를 최대화하며,
트레이드오프가 있을 경우 평균 `(score_rs + score_s) / 2`를 기준으로 판단한다.

---

## 2. 파일 역할 분리

| 파일 | 수정 가능 여부 | 역할 |
|------|:-----------:|------|
| `bbb_prepare.py` | **불가** | 데이터 로딩, 피처 엔지니어링, 데이터셋, split, 평가 |
| `bbb_train.py`   | **가능** | 모델 아키텍처, Optuna 서치 스페이스, 학습 루프 |
| `bbb_results.tsv` | 기록 전용 | 실험 결과 로그 (git 커밋 금지) |
| `bbb_program.md`  | 참조용 | 실험 루프 지시서 |

---

## 3. 데이터 구성 (`bbb_prepare.py`)

### 3-1. 입력 데이터

- **레이블**: `bench_label.csv` — SMILES + `p_np` 컬럼 (`BBB+`=1, `BBB-`=0)
- **외부 임베딩** (3종):
  - `scage1`: 분자 수준 임베딩 (bench_embed.csv)
  - `scage2`: 원자 수준 임베딩 (bench_atom_embed.csv)
  - `mole`: MolE 기반 임베딩 (MolE_embed_base_bbb.csv)

### 3-2. 피처 벡터 구성 (8 모달리티)

각 분자를 아래 8종의 표현으로 변환하여 순서대로 **concatenation**:

| 모달리티 | 타입 | 차원 |
|---------|------|:----:|
| `ecfp`   | Morgan Fingerprint (radius=2) | 1024 |
| `maccs`  | MACCS Keys | 166 |
| `avalon` | Avalon Fingerprint | 512 |
| `tt`     | Topological Torsion FP | 1024 |
| `rdkit`  | RDKit 2D Descriptors | ~209 |
| `scage1` | 학습된 분자 임베딩 | 가변 |
| `scage2` | 학습된 원자 임베딩 | 가변 |
| `mole`   | MolE 언어모델 임베딩 | 768 |

→ **전체 입력 차원 ≈ 5,000+ (모달리티 수: 8개)**

`rdkit` 2D 디스크립터만 StandardScaler로 정규화 (train 기준 fit).

### 3-3. 데이터 분할

- **두 가지 split 모드** (모두 scaffold 기반, RDKit Murcko scaffold):
  - `scaffold`: scaffold 크기 내림차순으로 정렬 후 greedy split (분포 편향이 강한 hard split)
  - `random_scaffold`: scaffold 목록을 무작위로 섞은 후 greedy split (상대적으로 쉬운 split)
- 비율: train 80% / val 10% / test 10%
- 고정 시드: `BASE_SEED = 600`

### 3-4. 고정 학습 상수

| 상수 | 값 |
|------|:--:|
| `N_OPTUNA_TRIALS` | 30 |
| `NUM_EPOCHS` | 50 |
| `PATIENCE` | 10 (early stopping) |
| `BATCH_SIZE` | 128 |

---

## 4. 모델 아키텍처 (`bbb_train.py`)

### 4-1. 전체 흐름

```
입력 flat vector (B, total_dim)
    │
    ▼  [모달리티별 Linear projection]
8개 token (B, 8, d_model)
    │
    ▼  [gMLP backbone (depth 레이어)]
8개 token (B, 8, d_model)
    │
    ▼  [Gated Pool 또는 mean]
(B, d_model)
    │
    ▼  LayerNorm + Dropout
    │
    ▼  Linear(d_model → 1)
logit (B,)
```

### 4-2. 모달리티 프로젝션

각 모달리티 i에 대해 독립적인 `Linear(mod_dim_i → d_model)`
→ 8개의 동일한 크기 토큰을 생성 (`seq_len = 8`)

### 4-3. gMLP Backbone

**gMLPBlock** (depth번 반복):
```
x → LayerNorm → Linear(d_model → d_ffn×2) → GELU
                                               │
                                    ┌──────────┘
                                    ▼
                           SpatialGatingUnit
                             u * LayerNorm(v)
                             Conv1d(seq_len, seq_len, k=1)
                                    │
                                    ▼
                        Linear(d_ffn → d_model)
                                    │
                                    + residual
```

**SpatialGatingUnit (SGU)**:
- 채널을 반으로 쪼개 `u`, `v`
- `v`에 LayerNorm 후 `Conv1d`(1×1, 시퀀스 축) 적용
- `u * v`로 게이팅 → 모달리티 간 상호작용 모델링

### 4-4. Pooling

- `use_gated_pool=True`: 학습 가능한 softmax 가중치 `α ∈ R^{seq_len}` 으로 가중 평균
- `use_gated_pool=False`: 단순 mean pooling

---

## 5. Optuna 하이퍼파라미터 서치

### 현재 서치 스페이스

| 파라미터 | 타입 | 범위 / 후보 |
|---------|------|------------|
| `d_model` | categorical | 256, 512, 768, 1024 |
| `depth` | int (step=2) | 2, 4, 6, 8 |
| `ffn_multiplier` | categorical | 2, 3, 4 |
| `dropout` | float | 0.1 ~ 0.4 |
| `lr` | float (log) | 1e-5 ~ 5e-4 |
| `weight_decay` | float (log) | 5e-6 ~ 5e-4 |

→ `d_ffn = d_model × ffn_multiplier`

### Optuna 설정

- Sampler: **TPE** (seed 고정)
- Pruner: **Median Pruner** (startup=5 trials, warmup=30 steps)
- 방향: `maximize` (composite score)
- 클래스 불균형 처리: `pos_weight = n_neg / n_pos` (BCEWithLogitsLoss)

---

## 6. 학습 루프

- 옵티마이저: **Adam** (lr, weight_decay)
- 스케줄러: 없음 (현재 베이스라인)
- Early stopping: val loss 기준, `patience=10`
- Best 모델 state 복원

---

## 7. 실험 루프 (autoresearch 방식)

```
1. bbb_results.tsv + git log 확인
2. bbb_train.py에 아키텍처 변경 제안
3. git commit
4. 실행: conda run -n rapids-25.02 python bbb_train.py > bbb_run.log 2>&1
5. 결과 파싱: grep "^composite_" bbb_run.log
6. bbb_results.tsv에 기록 (git commit 안 함)
7. avg(score_rs, score_s) 개선 → keep / 아니면 → git reset --hard HEAD~1
8. 반복
```

### 결과 파싱 포맷 (bbb_run.log 끝부분)

```
---
composite_randomscaffold:   2.345678
composite_scaffold:         2.198765
total_seconds:     1847.3
peak_vram_mb:      3204.1
n_trials:          30
```

---

## 8. 탐색 예정 아이디어

- [ ] SGU를 모달리티 간 cross-attention으로 교체
- [ ] 블록별 학습 가능한 residual scaling (λ)
- [ ] CLS 토큰 / attention pooling / multi-head pooling
- [ ] 모달리티 dropout (훈련 시 랜덤으로 하나 제거)
- [ ] depth=8, d_model=256 조합 (얕은 채널 + 깊은 레이어)
- [ ] Batch/Instance Norm in projection layer
- [ ] Cosine Annealing LR 스케줄러
- [ ] AdamW + gradient clipping
- [ ] GELU → SwiGLU 또는 ReLU²
- [ ] Gated pool → 단순 학습 가능한 선형 결합

---

## 9. 현재 상태

- **베이스라인**: 미실행 (bbb_results.tsv에 결과 없음)
- **예상 베이스라인 점수**: composite ~2.2 ~ 2.6
- **목표**: 가능한 한 3.0에 근접
