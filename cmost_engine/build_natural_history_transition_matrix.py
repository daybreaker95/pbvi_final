#!/usr/bin/env python3
"""
build_natural_history_transition_matrix.py
===========================================
CMOST13 기반 10만명 시뮬레이션으로 Natural History (Wait) Transition Matrix 생성

개요:
  - Screening OFF, Polyp/Cancer Surveillance OFF (= 아무 개입 없는 natural history)
  - 10만 명 × 100년 시뮬레이션 실행
  - 매년(annual) 말 POMDP 상태 스냅샷 → annual transition matrix 구성
  - 행(row) = 현재 상태, 열(col) = 다음 연도 상태
  - T[i, j] = P(s_{t+1} = j | s_t = i, a = wait)

상태 공간 (18개, _pomdp_state_idx 기준):
  0  : Normal
  1  : P1   (Polyp stage 1)
  2  : P2
  3  : P3
  4  : P4
  5  : P5
  6  : P6
  7  : U1   (Undetected Cancer stage I)
  8  : U2
  9  : U3
  10 : U4
  11 : D1   (Detected Cancer stage I, symptomatic detection 포함)
  12 : D2
  13 : D3
  14 : D4
  15 : Dead_CRC   (암 사망)
  16 : Dead_Comp  (내시경 합병증 사망)
  17 : Dead_Other (자연사)

출력 파일 (python/Results/ 폴더):
  transition_matrix_wait.npy   - 18×18 transition matrix (numpy)
  transition_matrix_wait.csv   - 동일 내용 CSV
  transition_matrix_wait.npz   - 상세 데이터 포함 (T, counts, state_recorder)

사용법:
  cd .../python
  python build_natural_history_transition_matrix.py
"""

import os
import sys
import copy
import importlib.util
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup: python/ 디렉토리를 import 경로에 추가
# ---------------------------------------------------------------------------
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from NumberCrunching_100000 import NumberCrunching_100000

# ---------------------------------------------------------------------------
# 상태 정의
# ---------------------------------------------------------------------------
STATE_NAMES = [
    'Normal',
    'P1', 'P2', 'P3', 'P4', 'P5', 'P6',
    'U1', 'U2', 'U3', 'U4',
    'D1', 'D2', 'D3', 'D4',
    'Dead_CRC', 'Dead_Comp', 'Dead_Other',
]
N_STATES = len(STATE_NAMES)  # 18
DEAD_STATES = {15, 16, 17}   # absorbing states


# ---------------------------------------------------------------------------
# 1. CMOST13 파라미터 로드
# ---------------------------------------------------------------------------
def load_cmost13():
    """CMOST13.py에서 파라미터 딕셔너리 로드 (deepcopy)."""
    settings_path = os.path.join(_this_dir, 'settings', 'CMOST13.py')
    spec = importlib.util.spec_from_file_location('_cmost13', settings_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return copy.deepcopy(mod.settings)


# ---------------------------------------------------------------------------
# 2. Natural history 파라미터 준비 (calculate_sub.py 로직 그대로 인라인)
# ---------------------------------------------------------------------------
def prepare_simulation_params(n_patients=100_000):
    """
    CMOST13 settings를 로드하고 natural history 시뮬레이션에 필요한
    모든 파라미터를 prepare한다.
    (calculate_sub.py의 파라미터 준비 로직을 독립 재현)

    Returns
    -------
    dict containing all arrays/values needed for NumberCrunching_100000
    """
    variables = load_cmost13()
    n = n_patients
    variables['Number_patients'] = n

    # [핵심] Natural history: 정기검진 OFF, 추적관찰(Surveillance) OFF
    variables['Screening']['Mode'] = 'off'
    variables['Polyp_Surveillance'] = 'off'
    variables['Cancer_Surveillance'] = 'off'
    variables['SpecialText'] = ''
    variables['SpecialFlag'] = 'off'

    # -----------------------------------------------------------------------
    # Direct Cancer Rate 보간 (20-element → 150-element, linear, step=5)
    # -----------------------------------------------------------------------
    direct_cancer_rate = np.zeros((2, 150))
    src_rate = np.atleast_2d(np.array(variables['DirectCancerRate'], dtype=float))
    counter = 0
    for x1 in range(19):
        for x2 in range(1, 6):
            direct_cancer_rate[0, counter] = (
                src_rate[0, x1] * (5 - x2) + src_rate[0, x1 + 1] * (x2 - 1)
            ) / 4.0
            direct_cancer_rate[1, counter] = (
                src_rate[1, x1] * (5 - x2) + src_rate[1, x1 + 1] * (x2 - 1)
            ) / 4.0
            counter += 1
    direct_cancer_rate[0, counter:] = src_rate[0, -1]
    direct_cancer_rate[1, counter:] = src_rate[1, -1]
    direct_cancer_speed = variables['DirectCancerSpeed']

    # -----------------------------------------------------------------------
    # Stage Variables
    # -----------------------------------------------------------------------
    stage_variables = {}
    stage_variables['Progression'] = np.array(variables['Progression'], dtype=float)

    fast_cancer_src = np.array(variables['FastCancer'], dtype=float)
    fast_cancer = np.zeros(10)
    fast_cancer[:len(fast_cancer_src)] = fast_cancer_src
    fast_cancer[5:10] = 0   # MATLAB: FastCancer(6:10) = 0
    stage_variables['FastCancer'] = fast_cancer

    stage_variables['Healing'] = np.array(variables['Healing'], dtype=float)
    stage_variables['Symptoms'] = np.array(variables['Symptoms'], dtype=float)
    stage_variables['Colo_Detection'] = np.array(variables['Colo_Detection'], dtype=float)
    stage_variables['RectoSigmo_Detection'] = np.array(variables['RectoSigmo_Detection'], dtype=float)
    stage_variables['Mortality'] = np.array(variables['Mortality'], dtype=float)

    # calculate_sub.py:132 — DwellSpeed는 항상 'Slow'로 override
    dwell_speed = 'Slow'

    # -----------------------------------------------------------------------
    # Location
    # -----------------------------------------------------------------------
    location = {
        'NewPolyp':           np.array(variables['Location_NewPolyp'],          dtype=float),
        'DirectCa':           np.array(variables['Location_DirectCa'],          dtype=float),
        'EarlyProgression':   np.array(variables['Location_EarlyProgression'],  dtype=float),
        'AdvancedProgression':np.array(variables['Location_AdvancedProgression'],dtype=float),
        'CancerProgression':  np.array(variables['Location_CancerProgression'], dtype=float),
        'CancerSymptoms':     np.array(variables['Location_CancerSymptoms'],    dtype=float),
        'ColoDetection':      np.array(variables['Location_ColoDetection'],     dtype=float),
        'RectoSigmoDetection':np.array(variables['Location_RectoSigmoDetection'],dtype=float),
        'ColoReach':          np.array(variables['Location_ColoReach'],         dtype=float),
        'RectoSigmoReach':    np.array(variables['Location_RectoSigmoReach'],   dtype=float),
    }

    # -----------------------------------------------------------------------
    # Female/Gender
    # -----------------------------------------------------------------------
    female = {
        'fraction_female':            variables['fraction_female'],
        'new_polyp_female':           variables['new_polyp_female'],
        'early_progression_female':   variables['early_progression_female'],
        'advanced_progression_female':variables['advanced_progression_female'],
        'symptoms_female':            variables['symptoms_female'],
    }

    # -----------------------------------------------------------------------
    # Costs (필요하지만 natural history에서는 실제로 사용 안 함)
    # -----------------------------------------------------------------------
    cost = dict(variables['Cost'])
    cost_src = variables['Cost']
    cost_stage = {
        'Initial':    [cost_src['Initial_I'],    cost_src['Initial_II'],    cost_src['Initial_III'],    cost_src['Initial_IV']],
        'Cont':       [cost_src['Cont_I'],        cost_src['Cont_II'],        cost_src['Cont_III'],        cost_src['Cont_IV']],
        'Final':      [cost_src['Final_I'],       cost_src['Final_II'],       cost_src['Final_III'],       cost_src['Final_IV']],
        'Final_oc':   [cost_src['Final_oc_I'],    cost_src['Final_oc_II'],    cost_src['Final_oc_III'],    cost_src['Final_oc_IV']],
        'FutInitial': [cost_src['FutInitial_I'],  cost_src['FutInitial_II'],  cost_src['FutInitial_III'],  cost_src['FutInitial_IV']],
        'FutCont':    [cost_src['FutCont_I'],     cost_src['FutCont_II'],     cost_src['FutCont_III'],     cost_src['FutCont_IV']],
        'FutFinal':   [cost_src['FutFinal_I'],    cost_src['FutFinal_II'],    cost_src['FutFinal_III'],    cost_src['FutFinal_IV']],
        'FutFinal_oc':[cost_src['FutFinal_oc_I'], cost_src['FutFinal_oc_II'], cost_src['FutFinal_oc_III'], cost_src['FutFinal_oc_IV']],
    }

    # -----------------------------------------------------------------------
    # Complications (risc)
    # -----------------------------------------------------------------------
    risc = {
        'Colonoscopy_RiscPerforation':       variables['Colonoscopy_RiscPerforation'],
        'Rectosigmo_Perforation':            variables['Rectosigmo_Perforation'],
        'Colonoscopy_RiscSerosaBurn':        variables['Colonoscopy_RiscSerosaBurn'],
        'Colonoscopy_RiscBleedingTransfusion':variables['Colonoscopy_RiscBleedingTransfusion'],
        'Colonoscopy_RiscBleeding':          variables['Colonoscopy_RiscBleeding'],
        'DeathPerforation':                  variables['DeathPerforation'],
        'DeathBleedingTransfusion':          variables['DeathBleedingTransfusion'],
    }

    # -----------------------------------------------------------------------
    # Flags: 정기검진 OFF, 추적관찰은 variables 설정에 따름
    # -----------------------------------------------------------------------
    special_text = '                         '  # 25 spaces (blank)
    flag = {
        'Polyp_Surveillance': (variables.get('Polyp_Surveillance', 'off') == 'on'),
        'Cancer_Surveillance': (variables.get('Cancer_Surveillance', 'off') == 'on'),
        'SpecialFlag': False,
        'Screening': False,
        'Correlation': (variables.get('RiskCorrelation', 'on') == 'on'),
        'Schoen': False, 'Holme': False, 'Segnan': False, 'Atkin': False,
        'perfect': False, 'Mock': False, 'Kolo1': False, 'Kolo2': False,
        'Kolo3': False, 'Po55': False, 'treated': False, 'AllPolypFollowUp': False,
    }

    # -----------------------------------------------------------------------
    # Screening (OFF → matrix all zeros)
    # -----------------------------------------------------------------------
    screening_test = np.zeros((7, 8))
    col_vars = list(variables['Screening']['Colonoscopy'])
    screening_test[0, :] = [col_vars[0], col_vars[1], 0,
                             col_vars[2], col_vars[3], col_vars[4], col_vars[5], col_vars[6]]
    screening_test[1, :] = variables['Screening']['Rectosigmoidoscopy']
    screening_test[2, :] = variables['Screening']['FOBT']
    screening_test[3, :] = variables['Screening']['I_FOBT']
    screening_test[4, :] = variables['Screening']['Sept9_HiSens']
    screening_test[5, :] = variables['Screening']['Sept9_HiSpec']
    screening_test[6, :] = variables['Screening']['other']

    # ScreeningMatrix all zeros = no screening assigned to any patient
    screening_matrix = np.zeros(1000, dtype=int)

    sensitivity = np.zeros((8, 10))
    sensitivity[2, :] = np.array(variables['Screening']['FOBT_Sens'],         dtype=float)
    sensitivity[3, :] = np.array(variables['Screening']['I_FOBT_Sens'],       dtype=float)
    sensitivity[4, :] = np.array(variables['Screening']['Sept9_HiSens_Sens'], dtype=float)
    sensitivity[5, :] = np.array(variables['Screening']['Sept9_HiSpec_Sens'], dtype=float)
    sensitivity[6, :] = np.array(variables['Screening']['other_Sens'],        dtype=float)

    # -----------------------------------------------------------------------
    # Age Progression (6 polyp types × 150 age-years)
    # -----------------------------------------------------------------------
    prog    = np.array(variables['Progression'],       dtype=float)
    early_p = np.array(variables['EarlyProgression'],  dtype=float)
    adv_p   = np.array(variables['AdvancedProgression'], dtype=float)
    age_progression = np.zeros((6, 150))
    age_progression[0, :] = early_p * prog[0]
    age_progression[1, :] = early_p * prog[1]
    age_progression[2, :] = early_p * prog[2]
    age_progression[3, :] = early_p * prog[3]
    age_progression[4, :] = adv_p   * prog[4]
    age_progression[5, :] = adv_p   * prog[5]

    new_polyp = np.array(variables['NewPolyp'], dtype=float)
    colonoscopy_likelyhood = np.array(variables['ColonoscopyLikelyhood'], dtype=float)

    # -----------------------------------------------------------------------
    # Patient distribution (IndividualRisk, Gender, ScreeningPreference)
    # -----------------------------------------------------------------------
    src_individual_risk = np.array(variables['IndividualRisk'], dtype=float)
    rand_indices = np.random.randint(0, len(src_individual_risk), size=n)
    individual_risk = src_individual_risk[rand_indices]

    rand_gender = np.random.random(n)
    gender_arr = np.where(rand_gender < female['fraction_female'], 2, 1).astype(float)

    rand_pref = np.random.randint(0, 1000, size=n)
    screening_preference = screening_matrix[rand_pref]  # all zeros

    risk_dist = {
        'EarlyRisk':    np.array(variables['EarlyRisk'], dtype=float),
        'AdvancedRisk': np.array(variables['AdvRisk'],   dtype=float),
    }

    # -----------------------------------------------------------------------
    # Mortality Matrix (4×100×1000, SEER 기반)
    # -----------------------------------------------------------------------
    survival_tmp = np.array([100, 82.4, 74.6, 69.5, 65.9, 63.3,
                              61.5, 60, 58.9, 58, 57.3]) / 100.0
    surf = np.zeros(21)
    counter = 0
    for x1 in range(5):
        for x2 in range(1, 5):
            surf[counter] = (
                survival_tmp[x1] * (5 - x2) / 4.0
                + survival_tmp[x1 + 1] * (x2 - 1) / 4.0
            )
            counter += 1
    surf[counter] = survival_tmp[5]
    surf = 1.0 - surf

    mortality_correction = (
        np.array(variables['MortalityCorrectionGraph'], dtype=float) - 1.0
    )
    mortality_matrix = np.full((4, 100, 1000), 25, dtype=int)
    mortality_params = stage_variables['Mortality']

    for f in range(4):
        factor = mortality_params[f + 6] / (1.0 - survival_tmp[5])
        surf2 = surf * factor
        surf4 = surf2 * surf2
        for y_idx in range(100):
            corr_val = mortality_correction[y_idx]
            denom = (surf2 * corr_val) + surf2
            term = np.zeros_like(surf2)
            mask = denom != 0
            term[mask] = (surf2[mask] * corr_val) / denom[mask]
            mort_temp = surf2 + term * (1.0 - surf4)
            mort_temp2 = np.clip(mort_temp[1:21], None, 1.0)
            ind_start = 0
            for g in range(20):
                ind_end = int(round(mort_temp2[g] * 1000))
                ind_end = max(1, min(1000, ind_end))
                if ind_start < ind_end:
                    mortality_matrix[f, y_idx, ind_start:ind_end] = g + 1
                ind_start = ind_end
                if ind_start >= 1000:
                    break
                if g == 19:
                    val_limit = int(round(mort_temp2[g] * 1000))
                    if val_limit < 1000:
                        mortality_matrix[f, y_idx, val_limit:] = 25
            mortality_matrix[f, y_idx, :] = np.random.permutation(
                mortality_matrix[f, y_idx, :]
            )

    life_table = np.array(variables['LifeTable'], dtype=float)

    # -----------------------------------------------------------------------
    # Stage Duration (hardcoded constants from MATLAB)
    # -----------------------------------------------------------------------
    stage_duration = np.array([
        [1,     0,     0,     0    ],
        [0.468, 0.532, 0,     0    ],
        [0.25,  0.398, 0.352, 0    ],
        [0.162, 0.22,  0.275, 0.343],
    ])

    # -----------------------------------------------------------------------
    # tx1 (25×4 Weibull parameters for SojournMatrix)
    # -----------------------------------------------------------------------
    tx1 = np.array([
        [0.442, 0.490, 0.010, 0.003],
        [0.413, 0.515, 0.017, 0.006],
        [0.385, 0.533, 0.028, 0.010],
        [0.716, 1.091, 0.083, 0.032],
        [0.662, 1.101, 0.118, 0.050],
        [0.913, 1.645, 0.243, 0.111],
        [0.833, 1.616, 0.321, 0.158],
        [1.004, 2.087, 0.546, 0.288],
        [0.899, 1.992, 0.675, 0.380],
        [0.996, 2.344, 1.012, 0.605],
        [1.223, 3.049, 1.654, 1.047],
        [1.670, 4.396, 2.960, 1.979],
        [1.571, 4.352, 3.598, 2.532],
        [1.233, 3.587, 3.604, 2.663],
        [0.668, 2.036, 2.464, 1.907],
        [0.405, 1.289, 1.864, 1.508],
        [0.274, 0.910, 1.560, 1.317],
        [0.231, 0.800, 1.615, 1.420],
        [0.146, 0.527, 1.243, 1.137],
        [0.123, 0.461, 1.267, 1.204],
        [0.069, 0.270, 0.856, 0.843],
        [0.059, 0.236, 0.863, 0.881],
        [0.025, 0.104, 0.434, 0.458],
        [0.021, 0.091, 0.434, 0.473],
        [0.018, 0.080, 0.434, 0.488],
    ])

    # -----------------------------------------------------------------------
    # Location Matrix (2×1000, CDF 기반 랜덤 위치 조회용)
    # -----------------------------------------------------------------------
    location_matrix = np.zeros((2, 1000), dtype=int)

    # Row 0: New Polyp 위치
    loc_counter = 0
    total_np = np.sum(location['NewPolyp'])
    for f in range(13):
        current_sum = np.sum(location['NewPolyp'][:f + 1])
        ende = int(round((current_sum / total_np) * 1000))
        ende = min(ende, 1000)
        if ende > loc_counter:
            location_matrix[0, loc_counter:ende] = f + 1
            loc_counter = ende

    # Row 1: Direct Cancer 위치
    loc_counter = 0
    total_dc = np.sum(location['DirectCa'])
    for f in range(13):
        current_sum = np.sum(location['DirectCa'][:f + 1])
        ende = int(round((current_sum / total_dc) * 1000))
        ende = min(ende, 1000)
        if ende > loc_counter:
            location_matrix[1, loc_counter:ende] = f + 1
            loc_counter = ende

    return dict(
        p=10,
        n=n,
        stage_variables=stage_variables,
        location=location,
        cost=cost,
        cost_stage=cost_stage,
        risc=risc,
        flag=flag,
        special_text=special_text,
        female=female,
        sensitivity=sensitivity,
        screening_test=screening_test,
        screening_preference=screening_preference,
        age_progression=age_progression,
        new_polyp=new_polyp,
        colonoscopy_likelyhood=colonoscopy_likelyhood,
        individual_risk=individual_risk,
        risk_dist=risk_dist,
        gender_arr=gender_arr,
        life_table=life_table,
        mortality_matrix=mortality_matrix,
        location_matrix=location_matrix,
        stage_duration=stage_duration,
        tx1=tx1,
        direct_cancer_rate=direct_cancer_rate,
        direct_cancer_speed=direct_cancer_speed,
        dwell_speed=dwell_speed,
    )


# ---------------------------------------------------------------------------
# 3. Annual state_recorder → Transition Matrix 구성
# ---------------------------------------------------------------------------
def build_transition_matrix(state_recorder, n_patients):
    """
    Annual state snapshots(shape: 100×n)로부터 annual transition matrix를 구성한다.

    Parameters
    ----------
    state_recorder : np.ndarray, shape (100, n_patients), dtype int
        state_recorder[yi, z] = year y+1 말 환자 z의 POMDP 상태 인덱스
    n_patients : int

    Returns
    -------
    T : np.ndarray (18, 18)  — 정규화된 전이확률행렬
    counts : np.ndarray (18, 18) — raw 카운트
    """
    counts = np.zeros((N_STATES, N_STATES), dtype=np.int64)

    # -----------------------------------------------------------------------
    # Year 0 (시뮬레이션 시작) → Year 1 말 전이
    # 시뮬레이션 시작 시점 모든 환자는 Normal(0) 상태
    # -----------------------------------------------------------------------
    s_year0 = np.zeros(n_patients, dtype=int)  # all Normal
    s_year1 = state_recorder[0, :].astype(int)
    np.add.at(counts, (s_year0, s_year1), 1)

    # -----------------------------------------------------------------------
    # Year yi 말 → Year yi+1 말 전이 (yi = 0..98)
    # -----------------------------------------------------------------------
    for yi in range(99):
        s_from = state_recorder[yi,     :].astype(int)
        s_to   = state_recorder[yi + 1, :].astype(int)
        # 유효 범위 체크 (안전장치)
        valid = (s_from >= 0) & (s_from < N_STATES) & (s_to >= 0) & (s_to < N_STATES)
        np.add.at(counts, (s_from[valid], s_to[valid]), 1)

    # -----------------------------------------------------------------------
    # Row 정규화: T[i, j] = counts[i, j] / sum_j counts[i, j]
    # 관측이 없는 상태는 자기 자신으로 유지 (self-loop)
    # -----------------------------------------------------------------------
    T = np.zeros((N_STATES, N_STATES), dtype=float)
    row_sums = counts.sum(axis=1)
    for i in range(N_STATES):
        if row_sums[i] > 0:
            T[i, :] = counts[i, :] / row_sums[i]
        else:
            T[i, i] = 1.0  # 관측 없음 → absorbing self-loop

    return T, counts


# ---------------------------------------------------------------------------
# 4. 출력 및 검증
# ---------------------------------------------------------------------------
def print_summary(T, counts):
    """Transition matrix 요약 출력."""
    print()
    print("=" * 80)
    print("TRANSITION MATRIX SUMMARY  (natural history / wait)")
    print("=" * 80)
    print(f"{'State':<14} | {'Top-3 Next States':<55} | {'N obs':>10}")
    print("-" * 80)
    for i, name in enumerate(STATE_NAMES):
        row = T[i]
        n_obs = int(counts[i].sum())
        top3_idx = np.argsort(row)[::-1][:3]
        top3_str = "  ".join(
            f"{STATE_NAMES[j]}:{row[j]:.4f}"
            for j in top3_idx
            if row[j] > 0.0001
        )
        print(f"{name:<14} | {top3_str:<55} | {n_obs:>10,}")

    print()
    print("Row-sum check (all should be ≈ 1.0):")
    row_sums = T.sum(axis=1)
    all_ok = True
    for i, name in enumerate(STATE_NAMES):
        rs = row_sums[i]
        status = "" if abs(rs - 1.0) < 1e-9 else " ← WARN"
        print(f"  {name:<14}: {rs:.10f}{status}")
        if abs(rs - 1.0) >= 1e-9:
            all_ok = False
    if all_ok:
        print("  All rows sum to 1.0 ✓")
    print()


def save_results(T, counts, state_recorder, n_patients, y_result):
    """결과 파일 저장."""
    out_dir = os.path.join(_this_dir, 'Results')
    os.makedirs(out_dir, exist_ok=True)

    npy_path = os.path.join(out_dir, 'transition_matrix_wait.npy')
    csv_path = os.path.join(out_dir, 'transition_matrix_wait.csv')
    npz_path = os.path.join(out_dir, 'transition_matrix_wait.npz')

    xlsx_path = os.path.join(out_dir, 'transition_matrix_wait.xlsx')

    # .npy
    np.save(npy_path, T)

    # .csv
    df = pd.DataFrame(T, index=STATE_NAMES, columns=STATE_NAMES)
    df.to_csv(csv_path)

    # .npz (상세 포함)
    np.savez_compressed(
        npz_path,
        T=T,
        counts=counts,
        state_recorder=state_recorder,
        state_names=np.array(STATE_NAMES),
        n_patients=np.array([n_patients]),
        n_years=np.array([y_result]),
    )

    # .xlsx (시트 2개: 전이확률 + raw counts)
    df_counts = pd.DataFrame(counts, index=STATE_NAMES, columns=STATE_NAMES)
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Transition_Probability')
        df_counts.to_excel(writer, sheet_name='Raw_Counts')

    print("Saved:")
    print(f"  {npy_path}")
    print(f"  {csv_path}")
    print(f"  {npz_path}")
    print(f"  {xlsx_path}")
    print()


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main(n_patients=200_000, seed=42):
    np.random.seed(seed)

    print("=" * 80)
    print("Natural History Transition Matrix  —  CMOST13 / Wait Condition")
    print("=" * 80)
    print(f"  Patients       : {n_patients:,}")
    print(f"  Screening      : OFF  (natural history)")
    print(f"  Surveillance   : OFF  (no follow-up)")
    print(f"  Symptomatic Dx : ON   (part of natural history)")
    print(f"  Random seed    : {seed}")
    print()

    # -----------------------------------------------------------------------
    # 파라미터 준비
    # -----------------------------------------------------------------------
    print("Preparing simulation parameters...")
    params = prepare_simulation_params(n_patients)
    print("Done.")
    print()

    # -----------------------------------------------------------------------
    # state_recorder 초기화 (100 years × n_patients, int8 — 0..17 fit)
    # -----------------------------------------------------------------------
    state_recorder = np.zeros((100, n_patients), dtype=np.int8)

    # -----------------------------------------------------------------------
    # NumberCrunching_100000 직접 호출
    # -----------------------------------------------------------------------
    print(f"Running CMOST13 simulation ({n_patients:,} patients × 100 years)...")
    (y_result, gender_out, death_cause, last, death_year, natural_death_year,
     direct_cancer_out, direct_cancer_r, direct_cancer2, direct_cancer2_r,
     progressed_cancer, progressed_cancer_r, tumor_record,
     dwell_time_progression, dwell_time_fast_cancer,
     has_cancer, num_polyps, max_polyps, all_polyps, num_cancer, max_cancer,
     payment_type, money, number,
     early_polyps_removed, diagnosed_cancer, advanced_polyps_removed,
     year_included, year_alive,
     ) = NumberCrunching_100000(
        params['p'],
        params['stage_variables'],
        params['location'],
        params['cost'],
        params['cost_stage'],
        params['risc'],
        params['flag'],
        params['special_text'],
        params['female'],
        params['sensitivity'],
        params['screening_test'],
        params['screening_preference'],
        params['age_progression'],
        params['new_polyp'],
        params['colonoscopy_likelyhood'],
        params['individual_risk'],
        params['risk_dist'],
        params['gender_arr'],
        params['life_table'],
        params['mortality_matrix'],
        params['location_matrix'],
        params['stage_duration'],
        params['tx1'],
        params['direct_cancer_rate'],
        params['direct_cancer_speed'],
        params['dwell_speed'],
        state_recorder=state_recorder,
    )
    print(f"Simulation complete. Simulated {y_result} years.")
    print()

    # -----------------------------------------------------------------------
    # Transition Matrix 구성
    # -----------------------------------------------------------------------
    print("Building transition matrix from annual state snapshots...")
    T, counts = build_transition_matrix(state_recorder, n_patients)
    print("Done.")

    # -----------------------------------------------------------------------
    # 요약 출력 및 저장
    # -----------------------------------------------------------------------
    print_summary(T, counts)
    save_results(T, counts, state_recorder, n_patients, y_result)

    # 기본 통계 출력
    ca_deaths   = int(np.sum(death_cause == 2))
    other_deaths = int(np.sum(death_cause == 1))
    comp_deaths  = int(np.sum(death_cause == 3))
    total_prog  = int(np.sum(progressed_cancer))
    total_dir   = int(np.sum(direct_cancer2))

    print("--- Simulation Statistics ---")
    print(f"  Cancer deaths   : {ca_deaths:,}")
    print(f"  Natural deaths  : {other_deaths:,}")
    print(f"  Comp. deaths    : {comp_deaths:,}")
    print(f"  Progressed Ca   : {total_prog:,}")
    print(f"  Direct Ca       : {total_dir:,}")
    print()

    return T, counts, state_recorder


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Build natural history transition matrix from CMOST simulation.')
    parser.add_argument('--n_patients', type=int, default=200_000,
                        help='Number of patients to simulate (default: 200000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()
    T, counts, state_recorder = main(n_patients=args.n_patients, seed=args.seed)
