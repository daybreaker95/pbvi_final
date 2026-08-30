# Archive: 6-State POMDP Research History & Early Results

이 디렉토리는 최신 **11-state x 6-risk class MOMDP (`dp/` 파이프라인)** 구축 이전에 수행되었던 초기 **6-state POMDP 기반 연구 산출물 및 문서**들을 기록/보존(archive)한 공간입니다.

---

## 1. 6-State 모델의 개요 및 한계

### 초기 6-State 임상 모델 정의
- **임상 상태 (6개)**:
  `[0: Normal, 1: Early Adenoma, 2: Advanced Adenoma, 3: Preclinical Cancer, 4: Clinical Cancer, 5: Dead]`
- **특징**:
  CMOST 마이크로시뮬레이션에서 10만 명의 자연사 이력을 추출하여 6x6 전이 행렬을 추정하고, 이를 기반으로 PBVI(Point-Based Value Iteration)를 적용하여 적응형 선별검사 정책을 도출했습니다.

### 최신 11-state MOMDP로 대체(superseded)된 이유
1. **용종 진행 메모리 소실**: CMOST 엔진의 원본 6단계 용종 성장(P1~P6)을 Early/Advanced 2단계로 단순화(pooling)함에 따라, 용종 절제(polypectomy)가 가진 실제 암 예방 효과와 환자별 진행 속도 메모리를 충분히 반영하지 못했습니다 (엔진 대비 10년 주기 검진의 암 발생 억제율을 약 10%p 과소평가).
2. **관측 메모리 부재**: 최신 모델의 (tau, last finding)(마지막 검진 후 경과 연수 및 소견) 구조가 없어, 과거 선별검사 이력에 따른 조건부 자연사 전이를 정밀하게 모사하지 못했습니다.
3. **최신 모델 현황**: 현재는 **11개 임상 상태 (`N`, `P1~P6`, `U1~U4`) x 6개 잠재 위험군(latent risk class) = 66개 상태**의 MOMDP 모델(`dp/`)로 전면 개편되어, 100만 명 규모 시뮬레이션에서 모든 고정 검진 스케줄에 대해 엄격한 파레토 우위(Strict Pareto Dominance)를 달성하였습니다.

---

## 2. 아카이브 파일 목록 및 설명

| 파일명 | 설명 | 비고 |
| :--- | :--- | :--- |
| [`manuscript_v1_6state_superseded.md`](./manuscript_v1_6state_superseded.md) | 초기 6-state POMDP 기반 연구의 논문 초안 원본 | 최신 메인 논문인 `paper/manuscript.md`로 완전히 승계됨 |
| [`results_comparison.md`](./results_comparison.md) | 초기 6-state PBVI 정책과 최적 고정 스케줄 비교 (n = 30,000) | 당시 life-years gained (LYG) 및 대장암 사망률 비교표 수록 |
| [`results_nonadherence.md`](./results_nonadherence.md) | 초기 비순응(non-adherence / no-show) 시나리오 실험 초안 | 적응형 정책의 재계획(re-planning) 효과 검증 |
| [`results_auroc_sweep.md`](./results_auroc_sweep.md) | 위험도 판별력(AUROC 0.50 ~ 0.85) 스윕 실험 초안 | 6-state 기반 위험군 타겟팅 조건 분석 |
| [`results_prs.md`](./results_prs.md) | 다유전자 위험 점수(PRS) 연계 선별검사 분석 초안 | 초기 PRS 4분위/Oracle 비교 |
| [`results_risk_factors.md`](./results_risk_factors.md) | 한국인 역학 기반 위험요인 및 상대위험도(RR) 모델링 초안 | 식이 및 생활습관 인자 분석 |
| [`results_subgroups.md`](./results_subgroups.md) | 성별 및 위험계층별 서브그룹 분석 초안 | 초기 서브그룹별 예산 할당 결과 |

---

## 3. 최신 연구 파일 안내

최신 연구 결과 및 완성된 논문 원고는 상위 디렉토리의 아래 파일들을 참조하십시오:
- **메인 논문 (11-state MOMDP 100만 명 검증)**: [`../manuscript.md`](../manuscript.md)
- **최신 방법론 세부 초안**: [`../dp_methods.md`](../dp_methods.md)
- **최신 결과 세부 초안**: [`../dp_results.md`](../dp_results.md)
- **최신 논문 Figures 인덱스**: [`../figures_index.md`](../figures_index.md)
- **최신 DP 파이프라인 코드 및 결과**: `dp/` 및 `results/dp/`
