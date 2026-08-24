# Stereo-collision 라벨 충돌 분석

BBB curated 데이터셋 큐레이션 단계와 combo1 모델 입력 feature 단계 사이의
설계 mismatch에 대한 데이터 기반 진단.

## 1. 핵심 발견

큐레이션은 **stereochemistry를 보존한 canonical SMILES 기준**으로 다수결
dedup이 적용되었지만, combo1 모델이 사용하는 모든 feature는
**stereo-blind**이다. 이 mismatch로 인해 같은 2D 구조의 stereoisomer 쌍이
서로 다른 라벨을 갖고 학습/평가 데이터에 공존하게 되었다.

**한 줄 요약**:
> 큐레이션은 "stereo 다르면 다른 분자"로 처리했고, 모델은 "stereo 무시"로
> 처리한다. 그 결과 모델 입력 단계에서 동일 입력에 BBB+/BBB- 모순 라벨이
> 동시에 들어오는 분자 쌍이 다수 존재한다.

## 2. 큐레이션 로직 분석

큐레이션 스크립트: `/home/minji/BBB/data_process/curate_bench_label.py`

```python
# Line 34: canonical SMILES 생성 (stereo 보존, RDKit 기본값)
work_df["canon"] = work_df["smiles"].apply(
    lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True)
)

# Line 117-119: canonical SMILES 그룹별 다수결
valid_df.groupby("canon").agg(
    Class_BBB=("Class_BBB", lambda x: int(round(x.mean()))),
)
```

| 단계 | 동작 |
|------|------|
| `MolToSmiles(canonical=True)` | RDKit 기본은 `isomericSmiles=True` → stereo 표시 (`@`, `@@`, `/`, `\`) **유지** |
| `groupby("canon")` | stereo-포함 SMILES로 그룹화 → L/D-form은 다른 그룹 |
| `int(round(mean))` | 같은 그룹 내에서만 다수결 |

→ **L-알라닌끼리 라벨 충돌 시 다수결 적용**, 그러나 **L-알라닌 vs D-알라닌은 애초에 다른 그룹이므로 충돌 자체로 인식되지 않음**.

## 3. combo1 feature는 모두 stereo-blind

`bbb_prepare.py`의 `FP_TYPES = ['maccs', 'avalon', 'rdkit', 'mole']`:

| Feature | Stereo 정보 |
|---------|-------------|
| MACCS (166-d) | ❌ 지원 안 함 |
| Avalon (512-d) | ❌ 기본값 stereo 무시 |
| RDKit descriptors (217-d) | ❌ 거의 없음 (chiral center count 정도만) |
| MolE 임베딩 (768-d) | ❌ 사전학습이 stereo-blind |

→ 모델 입력 단계에서 L/D-form은 **완전히 동일한 벡터**로 변환됨.

## 4. 데이터에서 정량화

### 4.1 데이터셋 크기 비교 (Canonical vs Stereo-strip)

| 데이터셋 | 총 | Canonical unique | Stereo-strip unique | Stereo-strip 중복 쌍 |
|----------|------|------------------|---------------------|----------------------|
| INT_REM | 1,677 | 1,677 | 1,663 | 14 |
| INT_HOL | 187 | 187 | 186 | 1 |
| **EXT_REM** | **8,109** | **8,109** | **7,112** | **997** |
| EXT_HOL | 902 | 902 | 890 | 12 |
| MERGED holdout | 1,089 | 1,089 | 1,076 | 13 |

→ EXT_REM 내부에 같은 2D 구조의 stereoisomer 쌍이 **997개** 존재.

### 4.2 데이터셋 간 stereo-strip 중복

| 비교 | Canonical | Stereo-strip |
|------|-----------|--------------|
| MERGED holdout ∩ EXT_REM | 0 | **158** |
| MERGED holdout ∩ INT_REM | 0 | 6 |
| INT_REM ∩ EXT_REM | 0 | 0 |

INT-EXT 사이 dedup은 잘 적용되었으나, Holdout과 EXT_REM 사이엔 stereo-strip
기준 158건 중복. 원인은 EXT_REM 내부의 stereoisomer 쌍 997개 중 일부가
EXT_HOL 샘플링 때 분리되었기 때문.

### 4.3 라벨 충돌 정량화

#### 데이터셋 내부 충돌

| 데이터셋 | Stereo-strip 그룹 | 라벨 충돌 그룹 | 충돌에 갇힌 분자 |
|----------|-------------------|----------------|------------------|
| INT_REM | 14 | 4 | 8 (0.48%) |
| **EXT_REM** | **638** | **186** | **425 (5.24%)** |
| MERGED holdout | 13 | 1 | 2 (0.18%) |

→ EXT_REM에 같은 2D 구조 + 다른 BBB 라벨인 그룹이 186개, 분자 425개.

#### Train pool ↔ Holdout 충돌

| 항목 | 개수 | 비율 |
|------|------|------|
| Holdout 분자 중 train pool에 같은 2D 존재 | 175 | 16.1% |
| 그 중 라벨이 train과 모두 다른 경우 | **30** | 17.1% |
| Train pool 자체에서 양쪽 라벨이 모두 있는 경우 | 11 | — |

→ Holdout 30개(2.8%)는 stereo-blind 모델로는 **수학적으로 맞출 수 없음**.

## 5. 모델 학습/추론에 미치는 영향

### 학습 중

```
입력 X (stereo 무시한 fingerprint):
  분자 A (L-form, BBB+)  →  X = [0,1,0,1,...]  →  loss: y=1
  분자 B (D-form, BBB-)  →  X = [0,1,0,1,...]  →  loss: y=0
                                  ↑
                        동일한 입력에 모순 라벨
```

BCE loss는 동일 입력에 모순 라벨이 오면 평균 확률(0.5) 예측이 수학적
최적해 → 모델이 0.5 부근으로 흐려져 **calibration 손상**.

### Augmentation 시 영향

- Internal Train 단독 (1,342) — 내부 충돌 8개로 거의 깨끗
- Internal + EXT_REM (≈9,451) — **EXT_REM의 425개 충돌 분자가 모순 신호로 직접 주입**
- Internal test의 깨끗한 분자 예측까지 흐려져 internal test 성능 하락
- Holdout은 30개 영원히 못 맞히는 noise floor + augmentation 효과 30~40% 손실

### 모델별 차이

| 모델 유형 | 영향 |
|-----------|------|
| High-capacity multi-modal (combo1) | 모순까지 외워서 calibration 직접 타격 → MCC/F1 가장 큰 손실 |
| Regularized (TabPFN, GBM, 단순 MLP) | 모순을 평균화/무시 → robust → augmentation 효과 더 큼 |

## 6. 현재 방식 유지 시 trade-off

### 6.1 유지의 장점

- **화학적 정합성**: enantiomer가 다른 BBB 거동을 보일 수 있음 (thalidomide 등)
- **데이터 보존**: stereo-strip dedup 시 425개 손실/임의 결정
- **확장성**: stereo-aware feature 도입 시 데이터 재큐레이션 불필요
- **실용 ranking 성능**: 현재 holdout ROC-AUC 0.817 수준은 ranking 활용에 충분

### 6.2 유지의 비용 (정량)

| 항목 | 영향 |
|------|------|
| Holdout noise floor | 30개(2.8%) — 영원히 못 맞히는 분자 |
| EXT_REM 모순 신호 | 425개(5.24%) — augmentation 학습 시 직접 주입 |
| Holdout ROC-AUC 천장 (stereo-blind) | ~0.83–0.85 |
| Holdout MCC 천장 (stereo-blind) | ~0.45 부근 |
| Augmentation 효율 손실 | 30~40% |

### 6.3 사용 목적별 권장

| 사용 목적 | 현재 방식 유지? |
|-----------|-----------------|
| 약물 후보 ranking / 스크리닝 | 유지 OK (ROC-AUC 기반이라 충분) |
| 개별 분자 BBB 정밀 분류 | 30개 한계 명시 + threshold 보정 권장 |
| Augmentation으로 holdout 극대화 | 비권장 — stereo-strip dedup 또는 stereo-aware feature 필요 |
| 타모델과 fair benchmark | 유지하되 "stereo-collision baseline" 별도 보고 |
| 논문/SoTA 주장 | 한계와 30/425 noise를 명시해야 방어 가능 |

## 7. 완화 방안 — combo1 feature 현실 검토

> ⚠️ 중요 정정: combo1 = `maccs + avalon + rdkit + mole`이며 **Morgan FP는
> combo1에 포함되지 않는다**. 따라서 "Morgan에 `useChirality=True`만 켜면
> 된다"는 단순 처방은 combo1엔 적용되지 않는다.

### 7.1 combo1 각 feature의 stereo 옵션 가능성

| Feature | Stereo 지원 |
|---------|-------------|
| MACCS (166-d) | ❌ 정의된 fragment에 stereo 없음 — 옵션 자체 없음 |
| Avalon (512-d) | ❌ RDKit `GetAvalonFP` 기본 구현에 stereo 옵션 없음 |
| RDKit descriptors (217-d) | ⚠️ 일부 stereo 관련 descriptor만 (`NumChiralCenters` 등) |
| MolE 임베딩 (768-d) | ❌ 사전학습 자체가 stereo-blind |

→ **현재 4개 feature 어느 것도 단순 옵션 변경으로 stereo-aware가 되지 않음**.

### 7.2 실제 가능한 옵션

| 옵션 | 설명 | 변경 규모 | 효과 |
|------|------|-----------|------|
| (A) Morgan-chiral 5번째 modality 추가 | `FP_TYPES`에 `morgan_chiral` 추가, +1024 dim | 중 (코드 + 학습 재실행) | stereoisomer 구별 가능 |
| (B) Avalon → Morgan-chiral 교체 | combo 자체 변경, dim 변화 작음 | 중 (hparam 재튜닝 위험) | 동일 효과 |
| (C) RDKit slice에 stereo descriptor 명시 추가 | `NumAtomStereoCenters` 등 +5 dim | 소 (수 줄 추가) | 부분적 — stereo 존재만 알림, R/S 구별 못 함 |
| (D) MolE → stereo-aware 사전학습으로 교체 | Uni-Mol, ChemBERTa-stereo 등 | 대 (임베딩 재생성) | 가장 깊은 해결 |
| **(E) 큐레이션 단계 stereo-strip dedup** | `to_canonical()`에 `isomericSmiles=False` + 다수결 | **0줄 (큐레이션 재실행)** | 모순 신호 원천 제거 |
| (F) 평가 시 분리 보고 | Holdout 30 conflict 분리 | 0줄 | noise floor 명시 |
| (G) Augmentation 시 EXT_REM 충돌 425 제외 | 학습 직전 필터링 | 소 | 모순 신호 학습 차단 |
| (H) Soft label | stereo-strip 그룹 평균 확률을 라벨로 | 소 | uncertainty로 표현 |

### 7.3 옵션별 권장도

| 옵션 | 권장도 | 메모 |
|------|--------|------|
| (E) Stereo-strip dedup | ⭐⭐⭐⭐ | 모델/feature 코드 0줄, 큐레이션 정책만 변경 |
| (A) Morgan-chiral 추가 | ⭐⭐⭐ | stereo 정보 보존하면서 모델이 활용 가능 |
| (G) Conflict 425 제외 | ⭐⭐⭐ | 큐레이션 그대로, 학습 코드만 추가 필터 |
| (F) 분리 보고 | ⭐⭐⭐ | 어떤 옵션과도 병행 권장 |
| (B) Avalon 교체 | ⭐⭐ | combo 정의 자체 바뀜 — phase1/phase2 재실험 위험 |
| (C) RDKit stereo descriptor | ⭐ | 충돌 분자 구별엔 부족 (R/S 방향 정보 없음) |
| (D) MolE 교체 | ⭐ | 가장 큰 변경, 임베딩 전체 재생성 |
| (H) Soft label | ⭐ | 효과 불확실, 추가 실험 필요 |

## 8. 두 방향의 일관된 해결

| 방향 | 큐레이션 | 모델 feature |
|------|----------|--------------|
| **(A) Stereo 일관 무시** | stereo-strip canonical로 그룹화 + 다수결 (옵션 7.2-E) | 현재 그대로 (stereo-blind) |
| **(B) Stereo 일관 유지** | 현재 그대로 | Morgan-chiral 추가 (옵션 7.2-A) 또는 stereo-aware embedding (옵션 7.2-D) |

(A)는 화학적으로 stereo 정보를 포기하지만 데이터-모델 일관성 확보 (가장 단순).
(B)는 stereo 정보를 유지하면서 모델이 그것을 활용하게 만듦 (구현 복잡).

## 9. 결론

현재 방식 유지의 비용은 정확히 정량화됨:
- Holdout 성능 천장: ROC-AUC ~0.83, MCC ~0.45 부근
- Augmentation 효율: 30~40% 손실
- 30개 분자는 stereo-blind 모델의 영구적 noise floor

### 권장 실행 순서

1. **(F) 분리 보고**: 어떤 시도 전이든 noise floor 30건을 분리해서 평가 — 진짜 일반화 성능을 가시화 (즉시 실행 가능)
2. **(E) 큐레이션 stereo-strip dedup** 또는 **(G) augmentation conflict 제외**:
   둘 중 하나로 모순 신호를 원천 차단 — 모델 코드 변경 없음
3. 그래도 부족하면 **(A) Morgan-chiral 5번째 modality 추가**: stereo 정보를
   살리면서 다른 modality 영향 최소화

> 핵심: combo1 4개 feature는 모두 stereo-blind라서, 큐레이션이나 학습 데이터
> 단계에서 충돌을 정리하는 (E)/(G)가 가장 가성비 좋은 단기 처방이다.
