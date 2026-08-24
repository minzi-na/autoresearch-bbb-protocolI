# GateMol-BBB Architecture Improvement Ideas

Literature-based survey and architecture improvement proposals for GateMol-BBB.  
Generated: 2026-04-28

---

## 현재 baseline 아키텍처 요약

**Phase 1 v2 reset baseline** (commit `a489d67`, roc_s=0.8481):

- 입력: maccs(166) + avalon(512) + rdkit(217) + mole(768) = 1663-d flat concat
- Per-modality projection: `Linear(in_dim → d_model=512)` × 4 (독립)
- Backbone: gMLP × 4 layers
  - gMLPBlock: pre-LN → `Linear(512→2096)` → GELU → SGU → `Linear(1048→512)` → +residual
  - SGU: `u, v = chunk(x, dim=-1)` → `v ← LN(Conv1d(LN(v)))` → `u * v`
- Pooling: `softmax(α)` gated weighted sum (α ∈ R^4, init=0)
- Head: `LN → Dropout(0.2) → Linear(512→1)`
- Loss: `BCEWithLogitsLoss(pos_weight=max(n_neg/n_pos, 1))`

**현재 HEAD** (commit `6504576`, roc_s=0.8647, iter55):

- SGU가 Conv1d → enhanced attention SGU로 교체됨 (iter52):
  - pre-norm u+v, Q/K/V projection (V eye init), learnable log-temp, diagonal mask, attn dropout
- iter55: skip connection sigmoid convex combine 추가 (`gate = sigmoid(skip_gate)`, init=0)

---

## Hard Constraints

- gMLP backbone 기반 유지 필수
- SGU 유지 (replacement 아닌 enhancement/variant)
- SGU의 u/v split, token-axis mixing, element-wise gating 구조 가능한 한 보존
- Self-attention/Transformer로 전면 대체 금지
- GNN, 3D equivariant 모델 금지
- Small/imbalanced BBB dataset → 과도한 복잡도 회피
- 단순성, 효율성, multi-modal fusion 구조 유지

---

## 1. Literature-derived Architecture Ideas

### RQ1: Input-level Modality Gating

---

#### Idea 1.1 — Modality-level Scalar Gate (SE-style)

**관련 문헌**
- Hu et al., *CVPR 2018* — Squeeze-and-Excitation Networks (원형 SE block)
- Li et al., *IEEE J Biomed Health Inform 2026* — PAM-CDR: property-aware multi-modal drug representation. property-guided attention으로 fingerprint/graph token을 enrich하는 1단계가 유사한 패턴.
  DOI: [10.1109/JBHI.2026.3658090](https://doi.org/10.1109/JBHI.2026.3658090)

**적용 방법**  
per-modality projection 직후 각 token `t_m ∈ R^512`마다:
```python
g_m = sigmoid(Linear(d_model -> 1)(t_m))   # scalar gate per modality
t_m = t_m * (2.0 * g_m)                    # init=identity (sigmoid(0)=0.5, 2x scale)
```

**장점**: 모달리티 importance 명시적 학습, 파라미터 최소 (~수천), MACCS↔Avalon 같은 redundant fingerprint 간 자동 재분배 가능

**리스크**: 기존 `softmax(α)` pooling과 기능 중복 가능성

**구현 난이도**: 1/5  
**Overfitting 위험**: 1/5  
**단순성/효율성**: 5/5

> ⚠️ **이미 시도됨**: iter48 (identity-init=1, exact tie), iter49 (logit=0, exact tie), iter50 (channel gate randn, -0.017/-0.048), iter51 (scalar gate randn, -0.056 mcc) — identity/zero init은 gradient flow 없음, randn init은 너무 공격적. **이 형태의 SE-gate는 BBB combo1에서 효과 없음 확인됨.**

---

#### Idea 1.2 — Cross-modality Context FiLM (zero-init)

**관련 문헌**
- Perez et al., *AAAI 2018* — FiLM: Visual Reasoning with a General Conditioning Layer
  arXiv: [1709.07871](https://arxiv.org/abs/1709.07871)
- Brockschmidt, *ICML 2020* — GNN-FiLM: Graph Neural Networks with Feature-wise Linear Modulation. small data에서 generalization 좋음.
  [proceedings.mlr.press](http://proceedings.mlr.press/v119/brockschmidt20a/brockschmidt20a.pdf)

**적용 방법**  
모든 토큰의 평균을 condition으로 사용:
```python
ctx = mean(tokens)                          # (B, d_model)
gamma_m, beta_m = MLP(ctx)                  # gamma init=ones, beta init=zeros
t_m = gamma_m * t_m + beta_m               # FiLM modulation
```

**장점**: zero/identity-init이라 baseline 성능 보존; cross-modal context를 활용한 self-conditioning

**리스크**: iter3 (category-based FiLM, roc -0.024), iter7 (AdaLN, roc -0.015) 실패 이력 — conditioning 자체가 BBB에서 일관되게 어려움

**구현 난이도**: 2/5  
**Overfitting 위험**: 3/5  
**단순성/효율성**: 4/5

> ⚠️ **유사 시도 실패**: iter3/7 모두 conditioning 방식에서 일관된 하락. Condition source가 다르나 신중 접근 필요.

---

### RQ2: SGU Enhancement

---

#### Idea 2.1 — Tiny Attention Add-on Inside SGU (aMLP pattern)

**관련 문헌**
- Liu et al., *NeurIPS 2021* — Pay Attention to MLPs (gMLP/aMLP 원논문).
  "SGU의 static mixing과 tiny attention은 complementary" — aMLP = gMLP + 64-d single-head attention.
  arXiv: [2105.08050](https://arxiv.org/abs/2105.08050)
- Xu et al., *J Pharm Anal 2025* — 3D-EDiffMG: gMLP block + tiny attention을 분자 생성 모델에 적용.
  DOI: [10.1016/j.jpha.2025.101257](https://doi.org/10.1016/j.jpha.2025.101257)

**적용 방법**  
SGU 내 v 분기에 Conv1d 와 tiny attention의 residual 합산:
```python
v_conv = Conv1d(v)                          # original spatial mixing
v_attn = SingleHeadSelfAttn(v, d=64)        # tiny content-dependent mixing
v_mixed = v_conv + alpha * v_attn           # alpha init=0 (safe start)
return u * v_mixed
```

**장점**: SGU constraint 완벽 부합 (add-on이지 replacement 아님); 원논문에서 content-dependent interaction 보완 확인; iter52와 달리 Conv1d를 유지

**리스크**: "self-attention 금지" constraint의 경계선 — Transformer 대체가 아닌 add-on이므로 기술적으로 미충돌. 다만 iter11 (attention-based SGU)이 실패한 이력 있음 (그쪽은 Conv1d **대체**였고 본 제안은 **추가**)

**구현 난이도**: 2/5  
**Overfitting 위험**: 3/5 (seq_len=4라 attention matrix 4×4로 매우 경량)  
**단순성/효율성**: 4/5

---

#### Idea 2.2 — Multi-Gated SGU (Intra-token + Cross-token Cascade)

**관련 문헌**
- Koh et al., *BMVC 2023* — MG-MLP: Multi-Gated MLP for Image Restoration.
  "intra-token + cross-token cascade gating → third-order interaction; Transformer 대비 효율적"
  [papers.bmvc2023.org](https://papers.bmvc2023.org/0283.pdf)

**적용 방법**  
SGU에 두 번째 intra-token gate를 cascade로 추가:
```python
u, v = x.chunk(2, dim=-1)
v = LayerNorm(v)

# Stage 1: intra-token gating (NEW)
v_gate = v * sigmoid(Linear_SE(v))         # channel-wise, init=zeros → identity start
                                            # Linear_SE: d_ffn -> d_ffn // 8 -> d_ffn

# Stage 2: cross-token mixing (original)
v_mixed = Conv1d(v_gate)

return u * v_mixed
```

**장점**: SGU u/v/gate 구조 완전 보존; attention 없음; third-order interaction 가능; 파라미터 증가 가능성 제어 가능 (reduction ratio r=8 사용 시 +~550K total)

**리스크**: 작은 데이터에서 학습 신호 분산 우려; intra-gate가 identity-init이면 tied 위험 (iter48/49 패턴)

**구현 난이도**: 2/5  
**Overfitting 위험**: 2/5  
**단순성/효율성**: 4/5

> ℹ️ **미시도** — iter 이력에서 intra+cross cascade 형태는 아직 시도된 바 없음.

---

#### Idea 2.3 — Multi-axis SGU (Group Split: local + global mixing)

**관련 문헌**
- Tu et al., *CVPR 2022* — MAXIM: Multi-Axis MLP for Image Processing.
  "local + global mixing을 분리해 병렬 처리 후 합산"
  arXiv: [2201.02973](https://arxiv.org/abs/2201.02973)
- Liu & Lin, *ScienceDirect 2025* — CAgMLP: Cross-Axis Gated Token Mixer

**적용 방법**  
v를 fingerprint-그룹 vs embedding-그룹으로 나눠 별도 mixing 후 합산:
```python
# modality groups: fp_group = [maccs, avalon], desc_group = [rdkit, mole]
v_global = Conv1d(seq=4, seq=4, k=1)(v)    # original all-token mixing
v_local  = Conv1d(seq=2, seq=2, k=1)(v[:, :2]) concatenated with v[:, 2:]
v_mixed  = gate * v_global + (1-gate) * v_local  # gate init=1 → starts as global
```

**장점**: 모달리티 그룹 prior(fp vs descriptor) 반영; init=1이면 baseline과 동일

**리스크**: 그룹 정의가 combo 의존적; combo가 바뀌면 재정의 필요

**구현 난이도**: 3/5  
**Overfitting 위험**: 2/5  
**단순성/효율성**: 3/5

---

#### Idea 2.4 — Depthwise Conv Augmentation in SGU (CgMLP-lite)

**관련 문헌**
- Rajagopal & Nirmala, *arXiv 2021* — Convolutional Gated MLP: Combining Convolutions & gMLP.
  "gMLP의 overfitting 경향을 conv inductive bias로 완화"
  arXiv: [2111.03940](https://arxiv.org/abs/2111.03940)

**적용 방법**  
SGU의 v 분기에 depthwise conv 1-D (kernel=3, padding=1) residual 추가:
```python
v_mixed = Conv1d(v) + alpha * DepthwiseConv(v, kernel=3, padding=1)
# alpha init=0, depthwise operates on channel dim
```

**장점**: 추가 파라미터 최소 (per-block ~3K); inductive bias 추가; init=0으로 safe

**리스크**: fingerprint의 channel 순서에 의미 있는 인접성이 없음 → 효과 미미 예상. 현재 HEAD는 이미 Conv1d를 attention으로 교체했으므로 적용 대상이 달라짐.

**구현 난이도**: 2/5  
**Overfitting 위험**: 1/5  
**단순성/효율성**: 5/5

---

### RQ3: Molecule-specific Gated Pooling

---

#### Idea 3.1 — Hyper-pooling (Pooled-context Predicts Per-token Weights)

**관련 문헌**
- Buterez et al., *NeurIPS 2022* — GNNs with Adaptive Readouts. PyG의 `MLPAggregation` 등 adaptive readout이 mean-pool 대비 일관 향상.
  [proceedings.neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2022/file/7caf9d251b546bc78078b35b4a6f3b7e-Paper-Conference.pdf)
- Zheng et al., *J Chem Inf Model 2026* — NeuMTL: attention pooling + cross-modal feature integration for BBB.
  DOI: [10.1021/acs.jcim.5c02725](https://doi.org/10.1021/acs.jcim.5c02725)

**적용 방법**  
기존 고정 α에 분자별 delta를 hyper-network로 residual 추가:
```python
ctx = X.mean(dim=1)                         # (B, d_model), pooled context
delta = MLP(ctx)                            # (B, seq_len), zero-init last layer
w = softmax(alpha + delta, dim=-1)          # alpha: global learnable baseline
Xp = (X * w.unsqueeze(-1)).sum(dim=1)
```

**장점**: 분자별 modality importance 적응; residual 구조로 baseline과 동일한 init; iter4/34의 fixed-init 실패 패턴과 다름

**리스크**: 작은 데이터에서 hyper-network가 noisy weight 학습 위험; 추가 ~67K 파라미터

**구현 난이도**: 2/5  
**Overfitting 위험**: 3/5  
**단순성/효율성**: 4/5

> ℹ️ **미시도** — iter56 (attention pool zero-init, tied)과는 구조적으로 다름: 본 제안은 기존 α에 context-conditioned residual을 더하는 형태.

---

#### Idea 3.2 — Single Learnable Query (Set Transformer-lite)

**관련 문헌**
- Lee et al., *ICML 2019* — Set Transformer: 1-query attention pool이 mean 대비 일관 향상.
  [proceedings.mlr.press](http://proceedings.mlr.press/v97/lee19d/lee19d.pdf)
- Zheng et al., *J Chem Inf Model 2026* — NeuMTL attention pooling module.
  DOI: [10.1021/acs.jcim.5c02725](https://doi.org/10.1021/acs.jcim.5c02725)

**적용 방법**  
```python
q = nn.Parameter(torch.randn(d_model) * 0.01)  # small randn init
attn = softmax(X @ q / sqrt(d_model), dim=1)    # (B, seq_len)
Xp = (X * attn.unsqueeze(-1)).sum(dim=1)
```

**장점**: 단순; 분자별 다른 weight; 파라미터 +d_model

**리스크**: iter4 (zero-init, tied), iter34 (randn-init, -0.006/-0.029), iter44/45 (multi-head, 큰 하락), iter56 (tied) — **pool query 기반 approach는 BBB combo1에서 모두 실패.** 구조적으로 새로운 변형 없이는 재시도 가치 낮음.

**구현 난이도**: 2/5  
**Overfitting 위험**: 2/5  
**단순성/효율성**: 4/5

> ⚠️ **이미 시도됨 (여러 변형 포함)**: iter4/34/44/45/56 모두 실패 또는 tied.

---

#### Idea 3.3 — Gated Cross-attention Pooling

**관련 문헌**
- de Oliveira & Saeed, *bioRxiv 2026* — TITAN-BBB: attention-based multimodal BBB prediction with tabular + image + text.
  DOI: [10.64898/2026.02.15.706007](https://doi.org/10.64898/2026.02.15.706007)
- ACS Omega 2025 — MCMPP: Multimodal Cross-Attention Molecular Property Prediction.
  [pubs.acs.org](https://pubs.acs.org/doi/10.1021/acsomega.5c07964)
- PMC2023 — Multi-modal adaptive gated mechanism for VQA. gated cross-attention pooling with tanh.

**적용 방법**  
```python
q_pool = nn.Parameter(torch.zeros(d_model))    # learnable query
scores = X @ q_pool / sqrt(d_model)            # (B, seq_len)
attn_w = softmax(scores, dim=1)
Xp_ca = (X * attn_w.unsqueeze(-1)).sum(1)
gate = sigmoid(nn.Parameter(torch.zeros(1)))   # init=0.5
Xp = (1-gate) * Xp_gated + gate * Xp_ca       # convex combine with existing pool
```

**장점**: BBB 특화 multimodal 모델들에서 일관된 효과; sigmoid gate로 안전한 init 가능

**리스크**: 추가 파라미터 ~d_model; iter56과 유사한 zero-init 함정 가능성; overfitting 위험 큼

**구현 난이도**: 3/5  
**Overfitting 위험**: 4/5  
**단순성/효율성**: 3/5

---

### RQ4: Modality Token Integration

---

#### Idea 4.1 — CLS-style Summary Token

**관련 문헌**
- BERT/ViT CLS 패턴 — 학습 가능한 summary token을 sequence 앞에 추가
- Feng et al., *J Phys Chem Lett 2025* — BBBP multimodal ensemble; Multi-Head Attention fusion 후 token-level summary 활용.
  DOI: [10.1021/acs.jpclett.5c01077](https://doi.org/10.1021/acs.jpclett.5c01077)

**적용 방법**  
```python
summary = nn.Parameter(torch.zeros(d_model))   # learnable, init=0
X = torch.cat([summary.expand(B,1,-1), X], dim=1)  # seq_len: 4 -> 5
X = backbone(X)                                # gMLP backbone
Xp = X[:, 0]  # or concat with gated_pool
```

**장점**: backbone에서 summary token이 모든 modality와 cross-modal mixing

**리스크**: iter9 (CLS pool, roc -0.006 mcc -0.013) 실패. gated_pool이 이미 동일 역할 수행

**구현 난이도**: 2/5  
**Overfitting 위험**: 3/5  
**단순성/효율성**: 4/5

> ⚠️ **이미 시도됨**: iter9 실패.

---

#### Idea 4.2 — Hierarchical Multi-token per Modality (k=2 split)

**관련 문헌**
- MMFRL, *arXiv 2024* — early/intermediate/late fusion 비교에서 intermediate fusion이 일반적으로 best.
  arXiv: [2410.12128](https://arxiv.org/abs/2410.12128)
- MulAFNet, *ACS Omega 2024* — multiple molecular representations for property prediction.
  [pubs.acs.org](https://pubs.acs.org/doi/10.1021/acsomega.4c09884)

**적용 방법**  
```python
# per-modality: Linear(in_dim -> 2 * d_model) then reshape -> 2 tokens
# seq_len: 4 -> 8
```

**장점**: backbone에 더 풍부한 token 다양성; SGU token-axis mixing이 더 의미 있음

**리스크**: **overfitting 위험 최대** — iter6 (2-layer projection, roc -0.007), iter58 (2-layer projection, roc -0.034) 모두 실패. token 2배 = 1.5K 훈련 데이터에 과부하.

**구현 난이도**: 2/5  
**Overfitting 위험**: 5/5  
**단순성/효율성**: 3/5

> ⚠️ **강한 유사 패턴 이미 실패**: iter6/58.

---

#### Idea 4.3 — Modality Token + Group-level Summary Token

**관련 문헌**
- MMFRL arXiv [2410.12128](https://arxiv.org/abs/2410.12128) — intermediate fusion: 4 modality tokens + 1 group token = partial intermediate fusion
- Multimodal Fusion with Relational Learning — stage별 fusion 비교

**적용 방법**  
```python
group_summary = X.mean(dim=1, keepdim=True)        # (B, 1, d_model)
group_residual = nn.Parameter(torch.zeros(d_model)) # init=0
X = torch.cat([X, group_summary + group_residual], dim=1)  # seq_len: 4 -> 5
```

**장점**: token +1만 추가; 모달리티가 group-summary와 직접 cross-modal mixing; init=0으로 baseline과 동일

**리스크**: iter9 (CLS) 실패와 유사한 위험 구조

**구현 난이도**: 2/5  
**Overfitting 위험**: 3/5  
**단순성/효율성**: 4/5

---

## 2. Prioritized Recommendation

| 순위 | 아이디어 | 근거 |
|------|----------|------|
| **★1** | **Idea 3.1** — Hyper-pooling | 미시도 아이디어; residual 구조로 zero-disturbance; NeuMTL/Adaptive Readouts 에서 검증; 기존 iter56과 달리 context-conditioned |
| **★2** | **Idea 2.2** — Multi-Gated SGU cascade | 미시도; SGU u/v/gate 완전 보존; attention 없음; MG-MLP 검증됨; intra+cross cascade가 baseline 대비 명확한 구조 차이 |
| **★3** | **Idea 2.3** — Multi-axis SGU (group split) | 미시도; 모달리티 그룹 prior 활용; init=1이면 baseline과 동일 시작 |
| 4 | Idea 2.1 — Tiny attention add-on | aMLP/3D-EDiffMG 검증됨; SGU add-on 형태; soft constraint 경계선 |
| 5 | Idea 4.3 — Group-summary token | 미시도; iter9와 유사하나 다른 형태 |
| 6 | Idea 1.2 — Self-context FiLM | iter3/7 실패 이력; condition source가 달라 재시도 가치 있으나 리스크 높음 |
| **X** | Idea 1.1 (SE-gate) | iter48/49/50/51 모두 실패. 재시도 가치 없음 |
| **X** | Idea 3.2 (Query pool) | iter4/34/44/45/56 모두 실패. 재시도 가치 없음 |
| **X** | Idea 4.1 (CLS token) | iter9 실패. 재시도 가치 없음 |
| **X** | Idea 4.2 (Hierarchical k=2) | iter6/58 강한 유사 패턴 실패. 재시도 가치 없음 |

---

## 3. Minimal-change Implementation Plan

현재 `bbb_train.py`의 `MultiModalGMLPFromFlat` 및 `SpatialGatingUnit` 클래스에 **partial edit**으로 추가 가능한 형태.  
모든 변경은 zero/identity-init이라 baseline 동작을 보존.  
현재 HEAD는 iter55(attention SGU + skip gate)이므로 아래 pseudo-code는 그 위에 추가하는 형태.

---

### Plan A — Idea 3.1: Hyper-pooling (residual α prediction)

`MultiModalGMLPFromFlat.__init__` 추가:
```python
# NEW: hyper-network predicts per-molecule pooling weight delta
self.alpha_hyper = nn.Sequential(
    nn.Linear(d_model, d_model // 4),
    nn.GELU(),
    nn.Linear(d_model // 4, self.seq_len),
)
nn.init.zeros_(self.alpha_hyper[-1].weight)
nn.init.zeros_(self.alpha_hyper[-1].bias)    # delta=0 at init -> reduces to baseline
```

`MultiModalGMLPFromFlat.forward` 수정 (pooling 부분):
```python
if self.use_gated_pool:
    ctx = X.mean(dim=1)                          # (B, d_model)
    delta = self.alpha_hyper(ctx)                # (B, seq_len), init=0
    w = torch.softmax(self.alpha + delta, dim=-1)    # (B, seq_len)
    Xp = (X * w.unsqueeze(-1)).sum(dim=1)
```

추가 파라미터: ~67K (d_model=512 기준)

---

### Plan B — Idea 2.2: Multi-Gated SGU (cascade intra+cross)

`SpatialGatingUnit.__init__` 추가:
```python
# NEW: intra-token gating stage (cascade before cross-token mixing)
self.intra_gate = nn.Sequential(
    nn.Linear(d_ffn, d_ffn // 8),
    nn.GELU(),
    nn.Linear(d_ffn // 8, d_ffn),
)
nn.init.zeros_(self.intra_gate[-1].weight)
nn.init.zeros_(self.intra_gate[-1].bias)     # sigmoid(0)=0.5, 2x scale -> identity
```

`SpatialGatingUnit.forward` 수정:
```python
u, v = x.chunk(2, dim=-1)
v = self.norm_v(v)

# Stage 1: intra-token gating (NEW, init=identity)
v = v * (2.0 * torch.sigmoid(self.intra_gate(v)))

# Stage 2: cross-token mixing (original attention-based or Conv1d)
# ... existing mixing code unchanged ...

return u * v_out
```

추가 파라미터: ~4 layers × (1048/8×1048 + 1048/8×1048) ≈ ~550K

---

### Plan C — Idea 2.1: Tiny Attention Add-on in SGU (aMLP pattern)

`SpatialGatingUnit.__init__` 추가:
```python
# NEW: tiny 64-d single-head attention, complementary to spatial mixing
d_tiny = 64
self.tiny_q = nn.Linear(d_ffn, d_tiny)
self.tiny_k = nn.Linear(d_ffn, d_tiny)
self.tiny_v = nn.Linear(d_ffn, d_ffn)
nn.init.zeros_(self.tiny_v.weight)
nn.init.eye_(self.tiny_v.weight[:d_ffn, :d_ffn])  # identity init (if d_ffn compatible)
self.tiny_scale = nn.Parameter(torch.zeros(1))     # init=0 -> no contribution at start
```

`SpatialGatingUnit.forward` 수정:
```python
# after v_out from existing mixing:
Q = self.tiny_q(v)
K = self.tiny_k(v)
V = self.tiny_v(v)
tiny_attn = torch.softmax(Q @ K.transpose(-1,-2) / (64**0.5), dim=-1)
v_tiny = tiny_attn @ V
v_out = v_out + torch.sigmoid(self.tiny_scale) * v_tiny  # gated add-on
```

추가 파라미터: ~4 layers × (64×1048×2 + 1048²) ≈ ~4.6M  
*(이 경우 파라미터 부하가 크므로 실제 적용 시 d_tiny=32로 줄여서 시작 권장)*

---

**권장 진행 순서**: Plan A → Plan B → Plan C (1개씩 단독 적용, 평가 후 누적)

---

## 4. What Not to Change (모델 정체성 보존)

| 항목 | 이유 |
|------|------|
| **gMLP backbone 전체 구조** | channel-mixing FFN + SGU sequential 구조 교체 금지 |
| **SGU 핵심 3요소** | u/v split (channel-axis), v에 token-axis mixing, `u ⊙ v_mixed` element-wise gate |
| **Per-modality 독립 Linear projection** | 모달리티 identity 보존; 공유 projection 또는 cross-modal projection 금지 |
| **Token sequence 크기 폭증 금지** | per-feature token (예: 512 fp → 512 tokens) 등 극단적 증가 금지 |
| **softmax(α) gated pooling base** | mean-pool, GAP, [CLS]-only로의 단순 대체 금지; adaptive readout은 *augment* 형태로만 |
| **Single-layer linear head** | iter29 (2-layer head) 실패. `LN → Dropout → Linear(d_model→1)` 유지 |
| **Loss 및 pos_weight** | `BCEWithLogitsLoss(pos_weight=max(n_neg/n_pos,1))` 고정; focal loss 등 도입 금지 |
| **BASE_CONFIG hyperparameters** | d_model=512, d_ffn=1048, depth=4, dropout=0.2, lr=1e-4, wd=1e-5, bs=128 — Phase 2 HPO 영역 |
| **Multi-modal fusion paradigm** | late fusion, early flat concat MLP, GNN/3D 등 다른 fusion paradigm으로 회귀 금지 |
| **파라미터 증가 한계** | 현재 7.32M → 10% 이내 증가 권장 (≤8M) |

---

## 참고 문헌

### PubMed

*(본 섹션은 PubMed 검색 결과를 기반으로 작성됨)*

- Feng et al. (2025) — Interpretable Multimodal Deep Ensemble for BBBP. *J Phys Chem Lett*.
  DOI: [10.1021/acs.jpclett.5c01077](https://doi.org/10.1021/acs.jpclett.5c01077)
- de Oliveira & Saeed (2026) — TITAN-BBB Multi-Modal Deep-Learning. *bioRxiv*.
  DOI: [10.64898/2026.02.15.706007](https://doi.org/10.64898/2026.02.15.706007)
- Zheng et al. (2026) — NeuMTL: Unified Multimodal Framework for CNS Drug Discovery. *J Chem Inf Model*.
  DOI: [10.1021/acs.jcim.5c02725](https://doi.org/10.1021/acs.jcim.5c02725)
- Li et al. (2026) — PAM-CDR: Property-Aware Multi-Modal Drug Representation. *IEEE J Biomed Health Inform*.
  DOI: [10.1109/JBHI.2026.3658090](https://doi.org/10.1109/JBHI.2026.3658090)
- Xu et al. (2025) — 3D-EDiffMG: gMLP+tiny attention for molecular generation. *J Pharm Anal*.
  DOI: [10.1016/j.jpha.2025.101257](https://doi.org/10.1016/j.jpha.2025.101257)

### ML / arXiv

- Liu et al. (2021) — Pay Attention to MLPs (gMLP, aMLP). NeurIPS 2021.
  arXiv: [2105.08050](https://arxiv.org/abs/2105.08050)
- Koh et al. (2023) — MG-MLP: Multi-Gated MLP for Image Restoration. BMVC 2023.
  [papers.bmvc2023.org](https://papers.bmvc2023.org/0283.pdf)
- Rajagopal & Nirmala (2021) — CgMLP: Convolutional Gated MLP. arXiv.
  arXiv: [2111.03940](https://arxiv.org/abs/2111.03940)
- Tu et al. (2022) — MAXIM: Multi-Axis MLP for Image Processing. CVPR 2022.
  arXiv: [2201.02973](https://arxiv.org/abs/2201.02973)
- Perez et al. (2018) — FiLM: Visual Reasoning with a General Conditioning Layer.
  arXiv: [1709.07871](https://arxiv.org/abs/1709.07871)
- Brockschmidt (2020) — GNN-FiLM: Graph Neural Networks with Feature-wise Linear Modulation. ICML 2020.
  [proceedings.mlr.press](http://proceedings.mlr.press/v119/brockschmidt20a/brockschmidt20a.pdf)
- Lee et al. (2019) — Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks. ICML 2019.
  [proceedings.mlr.press](http://proceedings.mlr.press/v97/lee19d/lee19d.pdf)
- Buterez et al. (2022) — Graph Neural Networks with Adaptive Readouts. NeurIPS 2022.
  [proceedings.neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2022/file/7caf9d251b546bc78078b35b4a6f3b7e-Paper-Conference.pdf)
- arXiv (2024) — MMFRL: Multimodal Fusion with Relational Learning for Molecular Property Prediction.
  arXiv: [2410.12128](https://arxiv.org/abs/2410.12128)
- ACS Omega (2025) — MCMPP: Multimodal Cross-Attention Molecular Property Prediction.
  [pubs.acs.org](https://pubs.acs.org/doi/10.1021/acsomega.5c07964)
- ACS Omega (2024) — MulAFNet: Multiple Molecular Representations for Property Prediction.
  [pubs.acs.org](https://pubs.acs.org/doi/10.1021/acsomega.4c09884)

---

*이 문서는 autoresearch 진행 중 2026-04-28 기준 iter65까지의 실험 이력을 반영함.*
*현재 HEAD: iter55 (commit 6504576), roc_s=0.8647, mcc_s=0.4571*
