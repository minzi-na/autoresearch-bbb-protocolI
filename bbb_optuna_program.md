# BBB autoresearch — Phase 2 HPO Optimization

Autonomous hyperparameter optimization for the best architecture selected in Phase 1 of the BBB autoresearch workflow.

## Task

Phase 1에서 선정된 **best architecture를 고정**한 뒤, Optuna 기반 hyperparameter optimization(HPO)을 통해 성능을 추가 개선한다.
이 단계에서는 **아키텍처 구조를 더 이상 변경하지 않는다.**

판단 기준: **3-seed validation scaffold mean ROC-AUC**
- Optuna trial 비교는 validation/internal scaffold ROC-AUC를 사용한다.
- 기본 objective는 3개 seed 평균 validation ROC-AUC다.

최종 검증 지표:
- **10-seed scaffold test ROC-AUC**
- **external ROC-AUC**
- **holdout ROC-AUC**

참고 지표:
- **탐색 효율** (trial 수 대비 최고 성능, pruning 효율)
- **안정성** (seed variance, 재현성)

## Scope

이 문서는 **Phase 2: HPO optimization** 전용 운영 문서다.

- `bbb_train.py`:
  - Phase 1에서 고른 best architecture를 고정한 backend
  - 모델 구조 변경 금지
- `bbb_optuna.py`:
  - Optuna search space / objective / sampler / pruner / study execution 관리
  - **이 파일이 Phase 2에서 agent가 수정하는 주 대상**

## Objective

목표는 “새 구조를 발명”하는 것이 아니라, **고정된 best architecture에 대해 더 나은 hyperparameter setting을 더 효율적으로 찾는 것**이다.
Trial 단계에서는 validation ROC-AUC를 기준으로 탐색하고, 최종 채택 전에는 반드시 full reevaluation을 수행한다.

즉, Phase 2 autoresearch agent의 역할은:

- architecture modification engineer 가 아니라
- **Optuna search-space design & modification engineer**

이다.

## Best Architecture Input

Phase 2 시작 전에 사람은 다음을 확정해야 한다.

1. 어떤 combo를 대상으로 할지
2. Phase 1에서 선택된 최종 best architecture commit 또는 파일 상태
3. 해당 best architecture가 고정 반영된 `bbb_train.py`

Phase 2에서는 이 상태를 baseline으로 사용한다.

## Files In Scope

반드시 먼저 읽고 시작한다.

- `bbb_prepare.py`
  - 데이터 로딩, 피처 생성, split, 평가 로직
  - **수정 금지**
- `bbb_train.py`
  - best architecture가 고정된 학습 backend
  - 모델 구조 확인용
  - **아키텍처 수정 금지**
- `bbb_optuna.py`
  - Optuna objective / study / search space / pruning / logging
  - **Phase 2의 수정 대상**
- `bbb_optuna_program.md`
  - 현재 문서
- `bbb_results_<combo>.tsv`
  - Phase 1 architecture search 기록
- `bbb_hpo_results_<combo>.tsv`
  - Phase 2 HPO search 기록

## What You CAN Change

`bbb_optuna.py` 안에서 다음 항목을 수정할 수 있다.

- search space 범위
  - `d_model`
  - `d_ffn`
  - `dropout`
  - `lr`
  - `weight_decay`
  - `batch_size`
  - 기타 명시적으로 HPO 대상으로 연 파라미터
- search space의 형태
  - categorical
  - int range
  - float range
  - log scale
- Optuna sampler
  - 예: `TPESampler`, `RandomSampler`
- Optuna pruner
  - 예: `MedianPruner`, `SuccessiveHalvingPruner`
- objective 계산 방식
  - single seed
  - small multi-seed mean
  - staged evaluation
- intermediate reporting / pruning 기준
- study budget
  - `n_trials`
  - timeout
- final reevaluation protocol
  - top-k trial 재평가
  - full multi-seed test reevaluation
- 결과 로깅 형식
  - 단, TSV/summary 포맷은 parsing 가능하게 일관되게 유지

## What You CANNOT Change

- `bbb_prepare.py`
- `bbb_train.py`의 best architecture 구조
- Phase 1에서 확정한 구조 자체
- 데이터 split 규칙
- external / holdout 데이터 정의
- 출력 포맷을 깨는 변경
- holdout을 tuning objective로 직접 사용하는 것

중요:
- holdout은 최종 일반화 확인용이지, tuning target이 아니다.
- Phase 2의 직접 최적화 대상은 internal scaffold 성능이어야 한다.

## Recommended HPO Parameters

초기 search space 예시:

| parameter | suggested range |
|-----------|-----------------|
| `d_model` | `[256, 384, 512, 768]` |
| `d_ffn` | `[512, 768, 1048, 1536, 2048]` |
| `dropout` | `0.0 ~ 0.4` |
| `lr` | `1e-5 ~ 5e-4` (log) |
| `weight_decay` | `1e-6 ~ 1e-3` (log) |
| `batch_size` | `[64, 128, 256]` |

주의:
- search space는 한 번에 너무 넓히지 않는다.
- 구조적으로 말이 안 되는 조합은 제한한다.
- GPU/VRAM 예산을 고려한다.

## Recommended Evaluation Strategy

full 10-seed scaffold test + external + holdout 평가를 모든 trial에 적용하면 비용이 너무 크므로 단계적으로 평가한다.

권장 방식:

1. **search phase**
   - 3 seeds로 빠르게 탐색
   - validation/internal scaffold mean ROC-AUC를 objective로 사용
   - pruning 적극 활용

2. **confirmation phase**
   - top-k trial 선택
   - 10-seed scaffold test로 재평가
   - external / holdout 함께 확인

3. **final phase**
   - 최종 후보만 full evaluation 결과로 채택
   - 성능과 안정성 확인

## Setup

1. 대상 combo를 확정한다.
2. Phase 1 best architecture가 반영된 `bbb_train.py` 상태를 고정한다.
3. `bbb_optuna.py`를 준비한다.
   - objective: 3-seed validation scaffold mean ROC-AUC
   - reevaluation: 10-seed scaffold test + external + holdout
4. `bbb_hpo_results_<combo>.tsv`를 header만 있는 상태로 초기화한다.
5. smoke test를 수행한다.

예시:

```bash
conda run -n rapids-25.02 python bbb_optuna.py
```

## Output format

HPO run은 최소한 아래 정보를 출력해야 한다.

```text
---
best_val_roc_auc:    0.801234
best_trial:          37
total_trials:        80
total_seconds:       5421.8
peak_vram_mb:        3520.4
```

선택적으로 아래도 함께 출력 가능:

```text
best_external_roc_auc:    0.778901
best_holdout_roc_auc:     0.810456
reeval_n_seeds:           10
```

## Logging results

콤보별 Phase 2 결과 파일:

- `bbb_hpo_results_combo1.tsv`
- `bbb_hpo_results_combo2.tsv`
- `bbb_hpo_results_combo3.tsv`

형식:

```text
commit	best_val_roc	best_ext	best_holdout	best_trial	total_trials	status	description
```

설명:

1. `commit`
   - short git hash
2. `best_val_roc`
   - best validation/internal scaffold ROC-AUC
3. `best_ext`
   - optional external reevaluation ROC-AUC
4. `best_holdout`
   - optional holdout reevaluation ROC-AUC
5. `best_trial`
   - selected best trial id
6. `total_trials`
   - total executed trials
7. `status`
   - `keep`, `discard`, or `crash`
8. `description`
   - 이번 HPO 설계 변경 요약

## The experiment loop

**LOOP FOREVER (until the human interrupts):**

1. `bbb_hpo_results_<combo>.tsv`와 git 상태를 확인해 지금까지의 search-space 변경 이력을 파악한다.
2. 다음 시도할 HPO 설계 변경을 구체적으로 제안한다.
   - 예: `lr` 범위를 좁히기
   - 예: `d_model` 후보를 줄이기
   - 예: `MedianPruner` → `SuccessiveHalvingPruner`
   - 예: objective를 validation 1-seed에서 validation 3-seed mean으로 강화
3. `bbb_optuna.py`를 수정하고 git commit한다.
4. Run:
   ```bash
   conda run -n rapids-25.02 python bbb_optuna.py > bbb_hpo_run_<combo>.log 2>&1
   ```
5. 결과를 읽는다.
   - best validation ROC-AUC
   - best trial
   - total trials
   - reevaluated 10-seed scaffold test / external / holdout metrics
6. 결과가 비어 있거나 실패하면 crash로 처리하고 로그 tail로 원인을 파악한다.
7. `bbb_hpo_results_<combo>.tsv`에 기록한다.
8. best trial 성능 또는 탐색 효율이 개선됐으면 **keep**
   - best config artifact 갱신
9. 개선이 없으면 **discard**
   - 직전 HPO 설계 변경 revert

## Keep / Discard policy

### keep

다음 중 하나 이상 만족하면 keep 고려:

- best validation/internal scaffold ROC-AUC 개선
- reevaluated 10-seed scaffold test ROC-AUC 개선
- 같은 성능에서 trial 수 감소
- 같은 성능에서 탐색 안정성 개선
- pruning이 더 효율적으로 작동
- 재평가 성능이 더 안정적

### discard

다음이면 discard:

- best 성능 개선 없음
- search space가 지나치게 넓어져 비효율 증가
- pruning이 너무 공격적이라 유망 trial 손실
- 재현성 악화
- reevaluated test/external/holdout 결과가 불안정

## Direction-switch rule

같은 방향에서 3번 연속 discard면 그 방향을 포기하고 다음 우선순위로 이동한다.

예:
- search range 확장 → discard
- range 재조정 → discard
- log scale 미세조정 → discard
- 그 다음에는 sampler/pruner 방향으로 이동

crash는 discard 연속 카운트에 포함하지 않는다.

## HPO Optimization Strategy

### A. Search Space Refinement

초기엔 넓게 시작하고, 유망한 영역이 보이면 좁힌다.

예:
- `lr`: `1e-5 ~ 5e-4` → `3e-5 ~ 2e-4`
- `dropout`: `0.0 ~ 0.4` → `0.05 ~ 0.25`
- `d_model`: `[256, 384, 512, 768]` → `[384, 512]`

### B. Sampler Strategy

기본은 `TPESampler` 권장.

변경 후보:
- startup trial 수 조정
- multivariate TPE 사용 여부
- random seed 고정 여부

### C. Pruner Strategy

기본은 `MedianPruner` 또는 `SuccessiveHalvingPruner`.

변경 후보:
- warmup step 조정
- pruning aggressiveness 조정
- intermediate metric reporting 빈도 조정

### D. Objective Design

비용과 안정성의 균형이 중요하다. 기본 objective는 validation 기준이고, 최종 선정은 reevaluation 기준으로 이뤄진다.

변경 후보:
- 1-seed validation objective
- 3-seed validation mean objective
- staged objective
- top trial만 full reevaluation

### E. Reevaluation Protocol

최종 후보 검증을 위한 별도 단계. 10-seed scaffold test + external + holdout를 사용한다.

변경 후보:
- top-1만 full reevaluation
- top-3 reevaluation 후 최종 선택
- external / holdout은 reevaluation 단계에서만 확인

## Recommended Priority Order

| priority | change | expected gain | risk | note |
|----------|--------|---------------|------|------|
| 1 | 초기 search space 정의 | 탐색 기반 확보 | 낮음 | 너무 좁지 않게 시작 |
| 2 | `lr`, `weight_decay` log-range 조정 | 수렴 개선 | 낮음 | 가장 영향 큼 |
| 3 | `dropout`, `batch_size` 후보 조정 | 일반화/안정성 개선 | 낮음 | VRAM 주의 |
| 4 | `d_model`, `d_ffn` 후보 정제 | 구조-용량 정합성 개선 | 중간 | 비용 증가 가능 |
| 5 | sampler 튜닝 | 탐색 효율 개선 | 낮음 | TPE 우선 |
| 6 | pruner 튜닝 | 시간 절약 | 중간 | 과도한 pruning 주의 |
| 7 | objective를 multi-seed로 강화 | 안정성 개선 | 중간 | 비용 증가 |
| 8 | top-k reevaluation 프로토콜 조정 | 최종 선택 신뢰도 개선 | 낮음 | 마지막 단계 |

## Experimental Principles

- 아키텍처는 고정한다.
- holdout은 tuning objective에 사용하지 않는다.
- search space는 한 번에 하나의 방향으로 조정한다.
- 비용이 큰 설정은 staged evaluation으로 완화한다.
- 같은 성능이면 더 단순한 search space / 더 적은 trial 수를 선호한다.
- 결과는 항상 누적 기록하고, 로그는 보존한다.

## Final Goal

Phase 2의 최종 산출물은 다음이다.

1. combo별 best architecture에 대한 best hyperparameter setting
2. 해당 setting의 internal scaffold 성능
3. external / holdout 일반화 성능
4. 재현 가능한 Optuna search configuration

이 단계의 목적은 “더 좋은 모델 구조”가 아니라, **“선정된 구조를 가장 잘 작동시키는 설정을 찾는 것”**이다.
