#!/usr/bin/env python3
"""
state34.py — POMDP 상태공간 단일 정의 (single source of truth)
=====================================================================
[2026-07-07 개정] P 상태축을 burden → **dwell(폴립 나이)** 로 교체.
근거: P→U 전환 hazard가 폴립 dwell에 강하게 의존(0-2y 0.00012 → 4-10y 0.00054, 4.6×,
CV=0.56 peaked). 단일 Markov P-state(상수 hazard)는 오래된 폴립의 빠른 전환을 못 담아
검진 과도예방=발생률 갭을 유발. dwell tunnel-state로 semi-Markov 재현.
burden은 headroom 7%로 미미해 제거(다중성 잔존은 콜로연산자 marginal B2로 유지).

레이아웃 (총 40개; 모듈명은 legacy)
------------------
  0..2     Normal   : naive / post_LR / post_HR              (surveillance 이력)
  3..20    P        : 병기(1-6) × dwell-phase(0,1,2)          (폴립 나이 tunnel)
  21..24   U        : preclinical 암 병기 I-IV
  25..36   D        : 검출 암 병기(1-4) × 진단후 phase(0,1,2)  (경과연수)
  37..39   Dead     : CRC / Comp / Other

dwell-phase 경계(폴립 나이 '연'): <2y →0, 2–4y →1, ≥4y →2   (P_DWELL_BOUNDS)
D phase 경계(진단후 '연'): <1y →0, 1–3y →1, ≥3y →2           (D_PHASE_BOUNDS)
tier: Last_AdvPolyp 있으면 post_HR(2), Last_Polyp 있으면 post_LR(1), else naive(0)
"""
import numpy as np

# ── 그룹 base offset ────────────────────────────────────────────────────────
# [2026-07-08] U에 sojourn(발병후 경과) 축 추가 → U 4→12, 총 40→48.
NORMAL0 = 0      # 0,1,2
P0      = 3      # 3..20   = (stage-1)*3 + dwell
U0      = 21     # 21..32  = (stage-1)*3 + sojourn   (병기4 × sojourn3)
D0      = 33     # 33..44  = (stage-1)*3 + phase
DEAD0   = 45     # 45,46,47
N_STATES34 = 48                       # (legacy 이름; 실제 48)

N_PDWELL       = 3
P_DWELL_BOUNDS = (2.0, 4.0)           # 폴립 나이(년)
N_USOJ         = 3
U_SOJ_BOUNDS   = (2.0, 4.0)           # preclinical 암 발병후 경과(년); median~3.25, peaked
N_DPHASE       = 3
D_PHASE_BOUNDS = (1.0, 3.0)           # 진단후 경과(년)

DEAD_CRC   = DEAD0
DEAD_COMP  = DEAD0 + 1
DEAD_OTHER = DEAD0 + 2


# ── 인덱스 계산 ─────────────────────────────────────────────────────────────
def normal_idx(tier):                 # tier: 0 naive, 1 post_LR, 2 post_HR
    return NORMAL0 + tier


def p_dwell(years):
    if years < P_DWELL_BOUNDS[0]:
        return 0
    if years < P_DWELL_BOUNDS[1]:
        return 1
    return 2


def p_idx(stage, dwell):              # stage 1-6, dwell 0-2
    return P0 + (stage - 1) * N_PDWELL + dwell


def u_soj(years):                     # 발병후 경과 → sojourn phase 0-2
    if years < U_SOJ_BOUNDS[0]:
        return 0
    if years < U_SOJ_BOUNDS[1]:
        return 1
    return 2


def u_idx(stage, soj=0):              # stage 1-4, soj 0-2
    return U0 + (stage - 1) * N_USOJ + soj


def d_phase(years_since_dx):
    if years_since_dx < D_PHASE_BOUNDS[0]:
        return 0
    if years_since_dx < D_PHASE_BOUNDS[1]:
        return 1
    return 2


def d_idx(stage, phase):              # stage 1-4, phase 0-2
    return D0 + (stage - 1) * N_DPHASE + phase


# ── 라벨 ────────────────────────────────────────────────────────────────────
def labels():
    L = [None] * N_STATES34
    L[0], L[1], L[2] = 'Normal_naive', 'Normal_postLR', 'Normal_postHR'
    for s in range(1, 7):
        for dw in range(N_PDWELL):
            L[p_idx(s, dw)] = f'P{s}_dw{dw}'
    for s in range(1, 5):
        for sj in range(N_USOJ):
            L[u_idx(s, sj)] = f'U{s}_sj{sj}'
    for s in range(1, 5):
        for ph in range(N_DPHASE):
            L[d_idx(s, ph)] = f'D{s}_ph{ph}'
    L[DEAD_CRC], L[DEAD_COMP], L[DEAD_OTHER] = 'Dead_CRC', 'Dead_Comp', 'Dead_Other'
    return L


# ── 40 → 18 collapse (기존 18-state 회귀검증용) ─────────────────────────────
# 18-state: Normal0, P1-6=1..6, U1-4=7..10, D1-4=11..14, Dead 15/16/17
def _collapse_one(i):
    if i <= 2:
        return 0
    if i < U0:                        # P: (stage-1)*3+dwell → stage
        stage = (i - P0) // N_PDWELL + 1
        return stage                  # 1..6
    if i < D0:                        # U: (stage-1)*N_USOJ+soj → stage
        stage = (i - U0) // N_USOJ + 1
        return 6 + stage              # U1->7 .. U4->10
    if i < DEAD0:                     # D
        stage = (i - D0) // N_DPHASE + 1
        return 10 + stage             # D1->11 .. D4->14
    return 15 + (i - DEAD0)           # Dead 15/16/17


COLLAPSE_34_TO_18 = np.array([_collapse_one(i) for i in range(N_STATES34)],
                             dtype=np.int64)


def collapse_to_18(arr):
    """40-state 인덱스 배열 → 18-state 인덱스 배열."""
    return COLLAPSE_34_TO_18[np.asarray(arr, dtype=np.int64)]


if __name__ == '__main__':
    L = labels()
    assert all(x is not None for x in L), 'label gap'
    print(f'N_STATES34 = {N_STATES34}')
    for i, name in enumerate(L):
        print(f'  {i:2d}  {name:14s} -> 18:{COLLAPSE_34_TO_18[i]}')
