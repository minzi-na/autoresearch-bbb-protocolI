# BBB autoresearch — Per-Combo Architecture Optimization

Autonomous architecture optimization for a multi-modal g-MLP Blood-Brain Barrier (BBB) prediction model.

## Task

Hold-out test set ROC-AUC 상위 3개 피쳐 조합 각각에 대해 독립적으로 모델 아키텍처를 최적화한다.
**하이퍼파라미터는 baseline으로 완전 고정**하고, 아키텍처 구조 변경만으로 **ROC-AUC**를 최대화한다.

판단 기준: **internal test ROC-AUC** (scaffold split, 10 seeds 평균) — keep/discard 결정
참고 지표: **external ROC-AUC**, **holdout ROC-AUC** (동일 10개 모델로 평가) — generalization 검증, 결정에는 사용 안 함

random scaffold split은 사용하지 않는다. scaffold split으로 학습된 모델이 일반화 성능에서도 충분히 검증됨.

Baseline (original g-MLP, d_model=512, d_ffn=1048, depth=4, dropout=0.2)

## Target Feature Combinations

아래 3개 조합에 대해 각각 독립적인 autoresearch 루프를 실행한다.
Hold-out test set ROC-AUC 기준 상위 3개 (result_table_top10.xlsx 기준):

| 순위 | combo_str | Hold-out ROC-AUC |
|------|-----------|-----------------|
| 1 | `maccs+avalon+rdkit+mole` | 0.8082 |
| 2 | `ecfp+maccs+avalon+tt+rdkit+scage1+mole` | 0.8082 |
| 3 | `ecfp+maccs+avalon+tt+rdkit+mole` | 0.8079 |

## Data

원본 BBBP benchmark 대신 **holdout을 제외한 curated remaining 데이터**를 사용한다.

```
LABEL_PATH  = /home/minji/BBB/holdout_splits/internal_curated_holdout_10pct_seed42/internal_curated_label_remaining.csv
EMBED_PATHS (콤보별로 필요한 것만 포함):
  combo1 (maccs+avalon+rdkit+mole):
    mole   = /home/minji/BBB/holdout_splits/internal_curated_holdout_10pct_seed42/internal_curated_mole_remaining.csv
  combo2 (ecfp+maccs+avalon+tt+rdkit+scage1+mole):
    scage1 = /home/minji/BBB/holdout_splits/internal_curated_holdout_10pct_seed42/internal_curated_scage1_remaining.csv
    mole   = /home/minji/BBB/holdout_splits/internal_curated_holdout_10pct_seed42/internal_curated_mole_remaining.csv
  combo3 (ecfp+maccs+avalon+tt+rdkit+mole):
    mole   = /home/minji/BBB/holdout_splits/internal_curated_holdout_10pct_seed42/internal_curated_mole_remaining.csv
```

## Baseline Model Parameters (실험 전체 고정)

아키텍처 탐색 전 과정에서 이 파라미터는 절대 변경하지 않는다.
성능 변화는 오직 아키텍처 구조 변경에만 기인해야 한다.

| 파라미터 | 값 |
|----------|-----|
| d_model | 512 |
| d_ffn | 1048 |
| depth | 4 |
| dropout | 0.2 |
| use_gated_pool | True |
| lr | 1e-4 |
| weight_decay | 1e-5 |
| num_epochs | 50 |
| patience | 10 |
| batch_size | 128 |
| seeds | [42, 100, 200, 300, 400, 500, 600, 700, 800, 900] |
| split_mode | scaffold |

**평가 방식**: scaffold split 10 seeds 평균
- `roc_auc_scaffold`: internal test ROC-AUC → keep/discard 기준
- `roc_auc_external`: 동일 모델로 external remaining 전체 평가
- `roc_auc_holdout`: 동일 모델로 merged holdout 전체 평가

## Setup

1. **Agree on a run tag**: 형식 `bbb-<combo_abbr>-<date>`. 예: `bbb-combo1-apr07`.
   - combo1 = `maccs+avalon+rdkit+mole`
   - combo2 = `ecfp+maccs+avalon+tt+rdkit+scage1+mole`
   - combo3 = `ecfp+maccs+avalon+tt+rdkit+mole`
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files** for full context:
   - `bbb_prepare.py` — 데이터 로딩, 피쳐 엔지니어링, 평가 로직. **LABEL_PATH, EMBED_PATHS, FP_TYPES 수정 필요.**
   - `bbb_train.py` — 모델 아키텍처, 학습 루프. **아키텍처 수정 대상.**
4. **Modify `bbb_prepare.py`**: LABEL_PATH, EMBED_PATHS를 위 Data 섹션 경로로 교체하고, FP_TYPES를 현재 콤보에 맞게 설정.
5. **Verify environment**: `conda run -n rapids-25.02 python bbb_train.py` (smoke test).
6. **Initialize `bbb_results_<combo_abbr>.tsv`** with just the header row.
7. **Confirm and go**.

## Running Each Experiment

콤보별 실행 명령 (각 콤보마다 독립적으로 실행):

```bash
# 실행
conda run -n rapids-25.02 python bbb_train.py > bbb_run_combo<N>.log 2>&1

# 결과 확인
grep "^roc_auc_" bbb_run_combo<N>.log
```

**What you CAN change in `bbb_train.py`:**
- Model architecture: `SpatialGatingUnit`, `gMLPBlock`, `gMLP`, `MultiModalGMLPFromFlat`
  - New gating mechanisms, normalization strategies, residual connections, pooling methods
- Training loop inside `train_model()`: optimizer type, LR scheduler, gradient clipping, etc.

**What you CANNOT change:**
- `bbb_prepare.py` — 완전 고정 파일 (data loading, split, evaluation)
- The composite_score metric formula: always `f1 + mcc + roc_auc`
- The `---` output block format at the end of `__main__` (it is parsed to extract results)
- Any hyperparameter value (d_model, d_ffn, depth, dropout, lr, weight_decay, batch_size, seeds)

**The goal**: maximize `roc_auc_scaffold` (internal test ROC-AUC, 10 seeds 평균).
roc_ext, roc_holdout가 함께 개선되지 않으면 description에 명시 (internal-only gain 주의).

**Simplicity criterion**: simpler is better if scores are equal.

## Output format

A successful run prints:
```
---
roc_auc_scaffold:    0.798765
roc_auc_external:    0.776543
roc_auc_holdout:     0.808123
total_seconds:       1847.3
peak_vram_mb:        3204.1
n_seeds:             10
```

- `roc_auc_scaffold`: internal test ROC-AUC (scaffold split, 10 seeds 평균) → **keep/discard 기준**
- `roc_auc_external`: external remaining ROC-AUC (동일 10개 모델)
- `roc_auc_holdout`: merged holdout ROC-AUC (동일 10개 모델)

Extract key metrics:
```bash
grep "^roc_auc_" bbb_run_combo<N>.log
```

## Logging results

콤보별로 별도 TSV 파일에 기록한다: `bbb_results_combo1.tsv`, `bbb_results_combo2.tsv`, `bbb_results_combo3.tsv`

```
commit	roc_s	roc_ext	roc_holdout	status	description
```

1. `commit`      — short git hash (7 chars)
2. `roc_s`       — `roc_auc_scaffold` (10 seeds 평균), use 0.000000 for crashes
3. `roc_ext`     — `roc_auc_external` (10 seeds 평균), use 0.000000 for crashes
4. `roc_holdout` — `roc_auc_holdout` (10 seeds 평균), use 0.000000 for crashes
5. `status`      — `keep`, `discard`, or `crash`
6. `description` — short description of what was tried (no tabs)

**keep 기준**: roc_s 개선 시 keep.
roc_ext 또는 roc_holdout이 함께 개선되지 않으면 description에 명시 (internal-only gain 주의).

Example (`bbb_results_combo1.tsv`, combo: `maccs+avalon+rdkit+mole`):
```
commit	roc_s	roc_ext	roc_holdout	status	description
a1b2c3d	0.798765	0.776543	0.808123	keep	baseline (maccs+avalon+rdkit+mole, curated remaining)
b2c3d4e	0.801000	0.779000	0.810000	keep	add LayerNorm after projection
c3d4e5f	0.793000	0.771000	0.804000	discard	replace SGU with simple mean gate
```

## Parallel Execution (3 combos simultaneously)

3개 콤보는 **별도 프로세스로 동시에 병렬 실행**한다. 각 콤보는 독립적인 git worktree와 결과 파일을 갖는다.

### Worktree 구성

```bash
# 콤보별 worktree 생성
git worktree add ../bbb-combo1 -b autoresearch/bbb-combo1-<date>
git worktree add ../bbb-combo2 -b autoresearch/bbb-combo2-<date>
git worktree add ../bbb-combo3 -b autoresearch/bbb-combo3-<date>
```

각 worktree에서 `bbb_prepare.py`와 `bbb_train.py`를 해당 콤보에 맞게 독립적으로 수정한다.

### 실행 로그 분리

| 콤보 | 실행 명령 | 결과 TSV |
|------|-----------|---------|
| combo1 (`maccs+avalon+rdkit+mole`) | `python bbb_train.py > bbb_run_combo1.log 2>&1` | `bbb_results_combo1.tsv` |
| combo2 (`ecfp+maccs+avalon+tt+rdkit+scage1+mole`) | `python bbb_train.py > bbb_run_combo2.log 2>&1` | `bbb_results_combo2.tsv` |
| combo3 (`ecfp+maccs+avalon+tt+rdkit+mole`) | `python bbb_train.py > bbb_run_combo3.log 2>&1` | `bbb_results_combo3.tsv` |

**VRAM 주의**: 3개 프로세스가 동시에 GPU를 사용하므로, 전체 VRAM 여유를 모니터링한다. batch_size와 d_model은 고정 파라미터이므로 변경 불가.

## The experiment loop (per combo)

각 콤보에 대해 아래 루프를 독립적으로, 동시에 실행한다.

**LOOP FOREVER (until the human interrupts):**

1. `bbb_results_<combo>.tsv`와 git 상태를 보고 지금까지 시도한 내용을 파악한다.
2. 다음 시도할 아키텍처 변경을 구체적으로 제안한다 (`bbb_train.py` 수정 대상 명시).
3. `bbb_train.py`를 수정하고 `git commit`한다.
4. Run (각 worktree 디렉토리 내에서):
   ```bash
   cd /home/minji/bbb-combo<N>
   conda run -n rapids-25.02 python bbb_train.py > bbb_run_combo<N>.log 2>&1
   ```
5. Read results: `grep "^roc_auc_" bbb_run_combo<N>.log`
6. If grep is empty → crash. `tail -n 50 bbb_run_combo<N>.log`로 스택 확인 후 수정 또는 건너뜀.
7. `bbb_results_<combo>.tsv`에 기록 (git commit 하지 않음).
8. roc_s가 개선됐으면 → **keep**:
   ```bash
   rm -rf bbb_artifacts/best && cp -r bbb_artifacts/current bbb_artifacts/best
   ```
   roc_ext, roc_holdout 방향도 확인하여 TSV에 기록.
9. 개선 없으면 → **discard**:
   ```bash
   git reset --hard HEAD~1
   ```
   (`bbb_artifacts/current/`는 다음 실험에서 자동으로 덮어씌워짐)

**Timeout**: 실행 1회 최대 90분 (10 seeds × ~9분/run: scaffold split + external + holdout 평가 포함). 초과 시 kill 후 crash 처리.

**방향 전환 규칙**: 같은 컴포넌트 방향에서 3번 연속 discard면 해당 방향을 포기하고 우선순위 테이블의 다음 항목으로 이동한다.
- 예: attention pooling 시도 → discard → multi-head pooling 변형 → discard → pooling 관련 미세조정 → discard → 다음 항목(Residual scaling)으로 이동
- crash는 연속 discard 카운트에 포함하지 않는다 (코드 오류이므로 수정 후 재시도)
- 테이블을 모두 소진한 뒤에는 지금까지 keep된 변경들의 조합 또는 변형을 자율적으로 생성한다

**NEVER STOP**: 사람이 중단할 때까지 루프를 계속 실행.

## Architecture Optimization Strategy

### 현재 아키텍처 구조

```
flat feature vector
    ↓ split by modality
[Linear(in_dim → d_model)] × N modalities   (per-modality projection)
    ↓ stack
(B, seq_len, d_model)
    ↓
[gMLPBlock × depth]                          (backbone)
  └─ LayerNorm → Linear(d_model → d_ffn*2) → GELU
     → SGU: split u,v; norm v; Conv1d mixing; u*v
     → Linear(d_ffn → d_model) → residual add
    ↓
gated weighted pool (learned softmax α)      (aggregation)
    ↓
Dropout → Linear(d_model → 1)
```

각 컴포넌트의 약점과 개선 방향을 이해하고 실험 순서를 결정한다.

---

### 컴포넌트별 약점과 개선 아이디어

#### A. Per-modality Projection (리스크: 낮음)

**현재 문제:**
ECFP/MACCS/Avalon/TT는 binary bits (0/1), RDKit은 정규화된 연속형, MolE/Scage는
pretrained embedding — 분포가 완전히 다른 입력을 단순 `Linear` 하나로 d_model 공간에 매핑.
정규화 없으면 모달리티 간 token scale 차이 → 학습 불안정.

**개선 방향:**
1. **projection 후 LayerNorm 추가** (간단, 효과적):
   ```python
   self.proj_norm = nn.ModuleDict({name: nn.LayerNorm(d_model) for name in mod_names})
   # forward: token = self.proj_norm[name](self.proj[name](chunk))
   ```
2. **2-layer projection** (표현력 향상):
   ```python
   Linear(in_dim → d_model) → GELU → Linear(d_model → d_model)
   ```
   이질적인 입력 공간을 비선형 변환으로 더 잘 매핑.

---

#### B. Spatial Gating Unit (리스크: 중간)

**현재 문제:**
`Conv1d(seq_len, seq_len, kernel_size=1)` = seq_len×seq_len 학습 가능한 mixing matrix.
모든 modality를 동등하게 선형 결합. 어떤 modality가 어떤 modality에 주목해야 하는지
학습 초기에 signal이 약함. bias=1.0으로 identity-like 초기화되어 있지만
비선형성 없이 단순 선형 혼합.

**개선 방향:**
1. **Residual scaling (learnable λ)** — SGU 결과의 영향 강도를 학습:
   ```python
   self.gate_scale = nn.Parameter(torch.zeros(1))  # 초기값 0 → 안전한 시작
   return u * (v + self.gate_scale.exp() * spatial_proj(v))
   ```
   초기엔 identity에 가깝게, 점차 cross-modal mixing 학습.

2. **Multi-head SGU** — 여러 개의 독립적인 mixing matrix:
   ```python
   # n_heads개의 Conv1d → 각각 다른 modality interaction 학습
   # 결과를 평균 or concat → project
   ```

3. **Attention-based gating** — content-dependent mixing (리스크 높음):
   ```python
   # Q, K, V = linear projections of v
   # attention_weights = softmax(QK^T / sqrt(d)) ; out = attention_weights @ V
   ```
   입력에 따라 동적으로 modality 간 중요도 결정.

---

#### C. Pooling (리스크: 낮음~중간)

**현재 문제:**
`α = softmax(learnable param)` → 가중합. 모든 샘플에 동일한 가중치.
분자마다 어떤 modality가 중요한지 다를 수 있는데 입력을 보지 않고 pooling.

**개선 방향:**
1. **Attention pooling** — 입력을 보고 동적으로 가중치 계산:
   ```python
   # query = learned vector (d_model,)
   # scores = softmax(X @ query / sqrt(d_model))  shape (B, seq_len)
   # pool = (scores.unsqueeze(-1) * X).sum(dim=1)
   ```
2. **CLS token** — 학습 가능한 토큰을 prepend, backbone 통과 후 CLS만 사용:
   ```python
   self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
   # seq_len이 N+1이 되므로 SGU도 seq_len+1로 초기화 필요
   ```
3. **Multi-head pooling** — k개의 서로 다른 pooling vector → concat → project (표현력 max):
   ```python
   # k개의 query로 각각 attention pooling → (B, k*d_model) → Linear → (B, d_model)
   ```

---

#### D. Training Dynamics (리스크: 낮음)

**현재 문제:**
Adam + lr=1e-4 고정, no scheduler, no gradient clipping.
Adam의 weight_decay는 adaptive gradient에 곱해져 실질적 decay 효과가 약함.
고정 lr은 후반 수렴이 느리거나 과적합 방지 어려움.

또한 scaffold split 특성상 val set의 scaffold가 train과 완전히 달라 **distribution shift** 발생.
이로 인해 val_loss가 노이즈가 크고 early stopping이 너무 일찍 발동하거나 불안정해짐.

**개선 방향:**

1. **Early stopping 기준을 val_loss → val_roc_auc로 변경** (scaffold split 핵심 문제 해결):
   val_loss는 분포 차이에 민감하지만 val_roc_auc는 상대적 순위 기반이라 scaffold shift에 더 강건.
   최종 평가 지표와도 일치하여 직접적 최적화 가능.
   ```python
   # 매 epoch val set에서 inference → roc_auc 계산
   model.eval()
   y_true, y_prob = [], []
   with torch.no_grad():
       for x, y in val_loader:
           prob = torch.sigmoid(model(x.to(device))).cpu().numpy()
           y_prob.extend(prob); y_true.extend(y.numpy())
   val_auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0
   if val_auc > best_val_auc:
       best_state = deepcopy(model.state_dict())
       bad = 0
   else:
       bad += 1
   ```

2. **Label smoothing** (val_loss 안정화 + 과적합 억제):
   Hard 0/1 target 대신 soft target으로 loss landscape를 부드럽게.
   scaffold split의 distribution shift에서 과도한 확신을 억제.
   ```python
   smooth = 0.1
   y_smooth = y * (1 - smooth) + smooth * 0.5   # 1→0.95, 0→0.05
   loss = F.binary_cross_entropy_with_logits(pred, y_smooth)
   ```

3. **Warmup + cosine annealing** (학습 초기 불안정 방지):
   처음 몇 epoch 동안 lr을 선형으로 올리고 이후 cosine decay.
   scaffold split에서 초기 val_loss spike를 줄여 early stopping 오발동 방지.
   ```python
   warmup_epochs = 5
   def lr_lambda(epoch):
       if epoch < warmup_epochs:
           return (epoch + 1) / warmup_epochs
       progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
       return 0.5 * (1 + math.cos(math.pi * progress)) * 0.99 + 0.01  # eta_min=lr*0.01
   scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
   # train loop 끝에: scheduler.step()
   ```

4. **AdamW + gradient clipping** (optimizer 개선):
   ```python
   optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

5. **pos_weight 조정** (클래스 불균형 보정):
   BBB 데이터는 BBB+ >> BBB- (약 2.26:1). 기본 BCEWithLogitsLoss는 pos_weight=1.0으로 불균형 무시.
   학습 데이터 비율에서 자동 계산하거나 수동으로 탐색.
   ```python
   # 자동 계산 (train set 기준, seed마다 재계산)
   n_pos = (y_train == 1).float().sum()
   n_neg = (y_train == 0).float().sum()
   pos_weight = torch.tensor([n_neg / n_pos]).to(device)  # ~0.44
   loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
   ```
   변형 시도: 데이터 비율 자동값 → 0.5 → 1.5 → 2.0 순으로 탐색.
   단, label smoothing과 함께 적용 시 y_smooth에 pos_weight를 적용해야 함에 주의.

---

#### E. Cross-modal FiLM / AdaLN Conditioning (리스크: 중간)

**개념:**
backbone 진입 **전에** 각 modality token이 다른 modality들의 요약 정보를 conditioning signal로
받아 자신의 scale/bias를 조정. SGU가 backbone **안에서** token 값을 섞는 것과 달리,
여기서는 **"어떻게 처리될 것인가"** 자체를 사전에 modality 간 관계로 조정.

```
[proj된 token들] (B, N, d_model)
       ↓
Cross-modal FiLM / AdaLN
  fingerprint tokens → embedding tokens를 조건화
  embedding tokens  → fingerprint tokens를 조건화
       ↓
gMLP backbone (이미 cross-modal 정보를 반영한 token들로 처리)
```

**설계 옵션:**

1. **Category-based FiLM (권장)** — 도메인 의미 있음:
   ```python
   class CrossModalFiLM(nn.Module):
       def __init__(self, d_model, fp_indices, embed_indices):
           super().__init__()
           self.embed_to_fp  = nn.Linear(d_model, d_model * 2)  # embedding → FP 조건화
           self.fp_to_embed  = nn.Linear(d_model, d_model * 2)  # FP → embedding 조건화
           self.fp_idx, self.emb_idx = fp_indices, embed_indices

       def forward(self, X):
           # X: (B, N, d_model)
           fp_summary  = X[:, self.fp_idx, :].mean(1)   # (B, d_model)
           emb_summary = X[:, self.emb_idx, :].mean(1)  # (B, d_model)

           gamma_fp,  beta_fp  = self.embed_to_fp(emb_summary).chunk(2, dim=-1)
           gamma_emb, beta_emb = self.fp_to_embed(fp_summary).chunk(2, dim=-1)

           X = X.clone()
           X[:, self.fp_idx,  :] = gamma_fp.unsqueeze(1)  * X[:, self.fp_idx,  :] + beta_fp.unsqueeze(1)
           X[:, self.emb_idx, :] = gamma_emb.unsqueeze(1) * X[:, self.emb_idx, :] + beta_emb.unsqueeze(1)
           return X
   ```
   → "구조 지문이 embedding 해석을 조건화하고, embedding이 구조 지문의 중요도를 조정"

2. **Global conditioning FiLM (단순)** — 모든 token의 평균을 conditioning으로:
   ```python
   cond = X.mean(dim=1)                        # (B, d_model)
   # 각 modality별 Linear(d_model → 2*d_model) → γ, β
   token_i = γ_i * token_i + β_i
   ```
   → 자기 자신도 conditioning에 포함되지만 구현 단순

3. **AdaLN 변형 (DiT 방식)** — LayerNorm 이후에 적용해 더 안정적:
   ```python
   x_norm = nn.LayerNorm(d_model, elementwise_affine=False)(token)
   gamma, beta = Linear(cond, d_model*2).chunk(2)
   token = (1 + gamma) * x_norm + beta         # scale 초기값 0 → 안전한 시작
   ```

**combo별 fp/embed 분류:**
```
combo1: fp=[maccs, avalon, rdkit],  embed=[mole]
combo2: fp=[ecfp, maccs, avalon, tt, rdkit],  embed=[scage1, mole]
combo3: fp=[ecfp, maccs, avalon, tt, rdkit],  embed=[mole]
```

---

#### F. Regularization (리스크: 낮음)

**현재 문제:**
Dropout(0.2)만 head 직전에 적용. 소규모 데이터셋(~1677개)에서 multi-modal 입력은
특정 modality에 과적합하기 쉬움.

**개선 방향:**
1. **Modality dropout** — 학습 중 무작위로 특정 modality token을 0으로:
   ```python
   # training 중: 각 modality를 p=0.1~0.2 확률로 zero out
   # 각 modality가 독립적으로 유용하도록 강제
   # combo2(7개 modality)에서 특히 효과적
   ```
2. **Label smoothing** — hard 0/1 대신 soft target:
   ```python
   loss_fn = nn.BCEWithLogitsLoss(reduction='none')
   # target = target * (1 - 0.1) + 0.05  (smoothing=0.1)
   ```
3. **Stochastic depth** — 일부 gMLPBlock을 학습 중 무작위로 skip:
   ```python
   # 각 block에 drop_prob (e.g., 0.1) 적용
   # 소규모 데이터셋 과적합 억제에 효과적
   ```

---

### 실험 우선순위 (권장 순서)

| 순서 | 변경 내용 | 예상 효과 | 리스크 | 비고 |
|------|-----------|-----------|--------|------|
| 1 | Early stopping: val_loss → val_roc_auc | scaffold shift 대응, 안정적 수렴 | 낮음 | scaffold split 핵심 문제 직접 해결 |
| 2 | Label smoothing (smooth=0.1) | loss 안정화, 과확신 억제 | 낮음 | 1번과 함께 적용 권장 |
| 3 | Warmup (5 epoch) + cosine annealing | 초기 불안정 방지, 후반 수렴 개선 | 낮음 | early stopping 오발동 감소 |
| 4 | AdamW + gradient clipping | optimizer 개선, 수렴 안정성 ↑ | 낮음 | 거의 항상 이득 |
| 5 | pos_weight 조정 (데이터 비율 기반 자동 계산) | 클래스 불균형 보정, 학습 안정화 | 낮음 | seed마다 train set 비율로 계산 |
| 6 | Per-modality LayerNorm after projection | 이질적 입력 안정화 | 낮음 | 독립 변수로 검증 |
| 7 | Modality dropout (p=0.15) | 과적합 억제 | 낮음 | combo2에 특히 효과적 |
| 8 | Cross-modal FiLM (category-based) | backbone 진입 전 modality 간 상호 조건화 | 중간 | fp↔embed 분리 gating |
| 9 | Attention pooling | 동적 modality weighting | 중간 | gated_pool 대체 |
| 10 | Residual scaling (learnable λ) | SGU 안정화 | 중간 | 0 초기화로 안전 |
| 11 | 2-layer per-modality projection | 표현력 ↑ | 중간 | 파라미터 증가 |
| 12 | FiLM → AdaLN 변형 | normalization 후 conditioning으로 안정성 ↑ | 중간 | 8번 개선 버전 |
| 13 | Multi-head SGU | cross-modal attention ↑ | 중간 | seq_len별 효과 다름 |
| 14 | CLS token pooling | pooling 표현력 ↑ | 중간 | seq_len+1 주의 |
| 15 | Stochastic depth | 과적합 억제 | 중간 | depth=4라 효과 제한적 |
| 16 | Attention-based SGU | 동적 cross-modal mixing | 높음 | 파라미터 많음 |
| 17 | SwiGLU/GeGLU activation | channel proj 개선 | 높음 | SGU 구조와 충돌 주의 |

**실험 원칙:**
- 변경은 **하나씩** 적용 (독립 변수 통제)
- 단, 1+2 (AdamW + cosine)는 training dynamics 패키지로 묶어 한 번에 시도
- roc_s 개선 없으면 즉시 `git reset --hard HEAD~1`
- combo별로 같은 순서를 따르되, 결과가 달리면 combo 특성 분석 후 분기

### Combo별 특성 고려

| Combo | seq_len | 특성 | 주목할 아이디어 |
|-------|---------|------|----------------|
| combo1 `maccs+avalon+rdkit+mole` | 4 | 적은 modality, 안정적 | LayerNorm, attention pool |
| combo2 `ecfp+maccs+avalon+tt+rdkit+scage1+mole` | 7 | 많은 modality | modality dropout, multi-head SGU |
| combo3 `ecfp+maccs+avalon+tt+rdkit+mole` | 6 | combo2에서 scage1 제거 | combo2 결과 참고해 선택적 적용 |
