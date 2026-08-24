# Phase 1 Process Notes

## 목적

이 문서는 BBB autoresearch의 **1단계(architecture optimization 단계)** 를 어떻게 운영할지에 대한 기록이다.

1단계는 각 input feature combination에 대해
**모델 아키텍처를 최적화**하는 단계이고,
2단계는 1단계에서 고른 **best architecture를 고정한 상태에서 hyperparameter optimization(HPO)** 을 수행하는 단계다.

## 핵심 결론

- 1단계는 `bbb_train.py`를 수정하며 아키텍처를 탐색한다.
- 1단계에서는 각 combo를 독립적으로 다룬다.
- 1단계에서는 하이퍼파라미터를 baseline으로 고정한다.
- 1단계의 autoresearch 역할은 **architecture modification engineer** 이다.
- 1단계가 끝난 뒤에만 2단계로 넘어간다.

## 1단계와 2단계의 차이

### 1단계

- 목적: 더 좋은 모델 구조 찾기
- 수정 대상: `bbb_train.py`
- combo별 데이터 설정을 위해 `bbb_prepare.py`를 현재 combo에 맞게 고정
- 고정 대상:
  - `bbb_prepare.py`의 split/evaluation 로직
  - baseline 하이퍼파라미터
- 실험 단위:
  - 아키텍처 변경 1개
  - `python bbb_train.py` 실행
  - 10-seed 결과 비교 후 keep/discard

### 2단계

- 목적: 고정된 best architecture에 대해 더 좋은 hyperparameter setting 찾기
- 수정 대상: `bbb_optuna.py`
- 고정 대상:
  - `bbb_prepare.py`
  - `bbb_train.py`의 best architecture
- 실험 단위:
  - Optuna search space / objective / sampler / pruner 설계 변경
  - `python bbb_optuna.py` 실행
  - best trial 성능과 탐색 효율 비교 후 keep/discard

## 1단계에서 autoresearch가 바꾸는 것

1단계에서 autoresearch는 다음과 같은 **모델 구조 관련 요소**를 제안하고 수정한다.

- projection 구조
- SGU / gating 구조
- residual 연결 방식
- normalization 방식
- pooling 방식
- backbone block 세부 구조
- optimizer / scheduler / gradient clipping 같은 학습 루프 요소
  - 단, baseline 고정 원칙을 깨지 않는 범위에서만 다룬다

즉, 1단계에서 바뀌는 것은 **탐색기 설계**가 아니라 **모델 구조와 학습 구조**다.

## 왜 combo별로 분리해야 하는가

Phase 1의 출발점은
"어떤 단일 구조가 모든 feature combination에 대해 최선일 것이라고 가정하지 않는다"
는 점이다.

따라서 아래 상위 combo들을 각각 독립 실험으로 다룬다.

- `maccs+avalon+rdkit+mole`
- `ecfp+maccs+avalon+tt+rdkit+scage1+mole`
- `ecfp+maccs+avalon+tt+rdkit+mole`

이 분리는 다음 이유 때문에 필요하다.

- 모달리티 구성이 다르면 projection과 cross-modal interaction의 최적 형태가 달라질 수 있다.
- 어떤 구조 개선은 특정 combo에서는 효과가 있지만 다른 combo에서는 무효일 수 있다.
- combo별 best architecture를 따로 확보해야 이후 Phase 2 HPO도 combo별로 독립 진행할 수 있다.

## 1단계의 판단 기준

1단계에서 keep/discard의 주 판단 지표는:

- **internal test ROC-AUC (`roc_auc_scaffold`)**

이다.

보조 지표는:

- `roc_auc_external`
- `roc_auc_holdout`

이다.

의미는 다음과 같다.

- `roc_auc_scaffold`는 현재 구조가 internal scaffold 환경에서 일관되게 좋아졌는지 판단하는 기준이다.
- `roc_auc_external`, `roc_auc_holdout`은 generalization이 같이 좋아지는지 확인하는 보조 신호다.
- internal만 좋아지고 external/holdout이 악화되면 keep는 가능하지만 그 사실을 반드시 기록해야 한다.

## 왜 하이퍼파라미터를 고정하는가

1단계에서는 아키텍처 효과만 보고 싶기 때문에 baseline 하이퍼파라미터를 고정한다.

그 이유는 다음과 같다.

- 구조 변화와 하이퍼파라미터 변화가 섞이면 개선 원인을 분리할 수 없다.
- 1단계의 목적은 "더 좋은 search space"를 찾는 것이 아니라 "더 좋은 구조"를 찾는 것이다.
- 구조가 확정되기 전에 HPO까지 같이 섞으면 탐색 공간이 너무 커져 autoresearch 루프가 불안정해진다.

즉 1단계는
"architecture search under fixed training conditions"
로 이해하는 것이 맞다.

## Step 1 문구 설계

1단계용 외부 루프는 아래처럼 정의한다.

### Step 1. 현재 combo와 기존 기록 확인

- 현재 combo의 `bbb_results_<combo>.tsv`를 읽는다.
- 어떤 구조 변경이 이미 시도되었는지 파악한다.
- 현재 `bbb_artifacts/best` 상태를 기준선으로 삼는다.

### Step 2. Agent가 다음 아키텍처 변경 제안

- `bbb_train.py`에서 다음에 손댈 구조 요소를 하나 정한다.
- 예:
  - projection 뒤 LayerNorm 추가
  - pooling을 attention pooling으로 변경
  - SGU에 residual scaling 추가
  - multi-head style mixing 추가
- 변경은 가급적 한 번에 한 방향씩 한다.

### Step 3. `bbb_train.py` 수정 후 실행

- 구조 변경을 `bbb_train.py`에 반영한다.
- git commit 후 실행한다.
- `python bbb_train.py`는 10 seeds를 돌며 internal/external/holdout 결과를 낸다.

### Step 4. 결과 파싱 및 keep/discard 판단

- 주 판단 지표:
  - `roc_auc_scaffold`
- 보조 지표:
  - `roc_auc_external`
  - `roc_auc_holdout`
- 개선이면 keep, 아니면 discard한다.

## 1단계 Step 3의 의미

중요한 점은, 1단계의 Step 3는

- "`bbb_train.py`를 수정하고 한 번 실행"

이라는 뜻이지만,
그 한 번의 실행은 실제로는

- 고정된 seed 목록 전체에 대한 학습/평가
- internal scaffold test 집계
- external / holdout ensemble 평가

를 포함하는 꽤 큰 단위의 실험이라는 점이다.

즉 `bbb_train.py`는 1단계에서
직접적인 실험 엔트리포인트이자
architecture search의 중심 파일이다.

## 필요한 파일

1단계에서 필요한 핵심 파일은 다음과 같다.

- `bbb_prepare.py`
- `bbb_train.py`
- `bbb_program.md`
- `bbb_results_combo1.tsv`
- `bbb_results_combo2.tsv`
- `bbb_results_combo3.tsv`

### 각 파일의 역할

#### `bbb_prepare.py`

- 데이터 로딩
- 피처 생성
- scaffold split
- RDKit normalization
- 평가 함수 제공

1단계에서 combo별 데이터 설정은 필요하지만,
split/evaluation 로직 자체는 고정 인프라로 취급한다.

#### `bbb_train.py`

- 모델 생성
- 학습 루프
- architecture component 정의
- 10-seed 평가 및 metric 출력

1단계의 주 수정 대상이다.

#### `bbb_program.md`

- 1단계용 operating 문서
- agent가 무엇을 바꾸고 무엇을 바꾸지 말아야 하는지 정의

#### `bbb_results_<combo>.tsv`

- combo별 architecture search 기록
- keep/discard 판단 이력
- 다음 시도 설계의 참고 근거

## 왜 `bbb_prepare.py`를 함부로 바꾸지 않는가

1단계에서도 `bbb_prepare.py`는 중요하지만,
아키텍처 탐색의 핵심 변수는 아니다.

이유:

- `bbb_prepare.py`가 자주 바뀌면 데이터/평가 조건 자체가 변한다.
- 그러면 구조 개선인지 데이터 조건 변화인지 분리할 수 없다.
- 따라서 combo별 feature selection이나 경로 고정은 필요하더라도,
  split/evaluation 규칙은 안정적으로 유지해야 한다.

즉 1단계에서 비교하고 싶은 것은

- 데이터 파이프라인의 변화가 아니라
- 같은 데이터 조건에서의 구조 차이

다.

## 왜 1단계를 먼저 끝내야 하는가

1단계를 먼저 끝내야 하는 이유는 명확하다.

- 구조가 확정되지 않은 상태에서 HPO를 시작하면,
  HPO 결과가 특정 구조의 우연한 세팅에 묶인다.
- 나중에 구조가 바뀌면 HPO를 다시 해야 한다.
- 따라서 먼저 combo별 best architecture를 확보하고,
  그 다음에야 해당 구조에 대해 HPO를 여는 것이 맞다.

즉 순서는:

1. combo별 architecture search
2. combo별 best architecture 확정
3. 그 구조를 backend로 고정
4. Phase 2 HPO 시작

이다.

## 최종 운영 원칙

- 1단계에서는 `bbb_train.py`를 중심으로 architecture search를 수행한다.
- combo별로 독립적인 worktree와 결과 파일을 사용한다.
- keep/discard 기준은 `roc_auc_scaffold`다.
- external / holdout은 보조 검증 지표로 기록한다.
- 하이퍼파라미터와 HPO 설계는 1단계에 섞지 않는다.
- combo별 best architecture가 확정된 뒤에만 2단계로 넘어간다.

## 요약

- 1단계: `bbb_train.py`를 수정하며 combo별 architecture search
- 주 판단 기준: `roc_auc_scaffold`
- 보조 기준: `roc_auc_external`, `roc_auc_holdout`
- `bbb_prepare.py`는 고정 인프라
- `bbb_program.md`는 운영 지침서
- `phase1.md`는 그 운영 지침의 설계 설명 문서
