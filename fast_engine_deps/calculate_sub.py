###############################################################################
#
#     CMOST: Colon Modeling with Open Source Tool
#     created by Meher Prakash and Benjamin Misselwitz 2012 - 2016
#
#     This program is part of free software package CMOST for colo-rectal
#     cancer simulations: You can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
###############################################################################

"""
calculate_sub.py -- Python translation of CalculateSub.m

This is the core orchestration module for the CMOST simulation. It:
  1. Prepares all simulation variables from the settings dictionary
  2. Calls NumberCrunching_100000 (the Monte Carlo simulation engine)
  3. Calls Evaluation (the results analysis/benchmarking module)

MATLAB equivalent: function [handles, BM] = CalculateSub(handles)

Parameters:
    handles : dict
        Dictionary with key 'Variables' containing all simulation parameters.

Returns:
    tuple: (handles, BM) where handles is updated with results and BM is
           the benchmark comparison dictionary.
"""

import os
import sys
import numpy as np

# Ensure the python/ directory is on the import path so sibling modules
# can be found regardless of the working directory.
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from NumberCrunching_100000_fastengine import NumberCrunching_100000
from Evaluation import Evaluation


def prepare_parameters(handles):
    """
    Extract and transform all CMOST settings into the argument bundle
    consumed by NumberCrunching_100000.

    This block was previously inline at the top of calculate_sub().  It is
    now exposed as a standalone function so the individual-level environment
    (pbvi_thlee/env) can build the *exact* same CMOST parameters and drive a
    per-patient stepping engine, guaranteeing fidelity with the population
    microsimulation.

    IMPORTANT: this function consumes the RNG (individual-risk sampling,
    gender assignment, screening-preference assignment, mortality-matrix
    shuffling) in the original order, so calculate_sub() still produces
    bit-identical results to the pre-refactor code.

    Parameters
    ----------
    handles : dict
        Must contain key 'Variables' with a dict of all simulation parameters.

    Returns
    -------
    dict
        params bundle with every array/kwarg required by
        NumberCrunching_100000 (keys documented at the return statement).
    """

    # ---------------------------------------------------------
    # 1. Preparation of Variables
    # ---------------------------------------------------------

    p = 10   # types of polyps
    n = handles['Variables']['Number_patients']

    # --- Direct Cancer Rate Interpolation ---
    # MATLAB: interpolates 20-element DirectCancerRate into 150-element array
    # using linear interpolation with 5 sub-steps between each pair of points
    direct_cancer_rate = np.zeros((2, 150))

    src_rate = np.atleast_2d(np.array(handles['Variables']['DirectCancerRate'], dtype=float))

    counter = 0  # 0-based index
    for x1 in range(19):       # MATLAB 1:19
        for x2 in range(1, 6): # MATLAB 1:5
            # Linear interpolation: (Rate[x1]*(5-x2) + Rate[x1+1]*(x2-1)) / 4
            direct_cancer_rate[0, counter] = (src_rate[0, x1] * (5 - x2) + src_rate[0, x1 + 1] * (x2 - 1)) / 4.0
            direct_cancer_rate[1, counter] = (src_rate[1, x1] * (5 - x2) + src_rate[1, x1 + 1] * (x2 - 1)) / 4.0
            counter += 1

    # Fill remaining positions with last value
    # counter is now 95 (19*5)
    direct_cancer_rate[0, counter:150] = src_rate[0, -1]
    direct_cancer_rate[1, counter:150] = src_rate[1, -1]

    direct_cancer_speed = handles['Variables']['DirectCancerSpeed']

    # --- Stage Variables ---
    stage_variables = {}
    stage_variables['Progression'] = np.array(handles['Variables']['Progression'], dtype=float)

    fast_cancer_src = np.array(handles['Variables']['FastCancer'], dtype=float)
    # MATLAB FastCancer has 10 elements (one per polyp type). Settings files
    # may store fewer elements. Pad to 10 if needed, then zero out indices 5-9.
    fast_cancer = np.zeros(10)
    fast_cancer[:len(fast_cancer_src)] = fast_cancer_src
    fast_cancer[5:10] = 0  # MATLAB: FastCancer(6:10) = 0
    stage_variables['FastCancer'] = fast_cancer

    stage_variables['Healing'] = np.array(handles['Variables']['Healing'], dtype=float)
    stage_variables['Symptoms'] = np.array(handles['Variables']['Symptoms'], dtype=float)
    stage_variables['Colo_Detection'] = np.array(handles['Variables']['Colo_Detection'], dtype=float)
    stage_variables['RectoSigmo_Detection'] = np.array(handles['Variables']['RectoSigmo_Detection'], dtype=float)
    stage_variables['Mortality'] = np.array(handles['Variables']['Mortality'], dtype=float)

    if 'DwellSpeed' in handles['Variables']:
        dwell_speed = handles['Variables']['DwellSpeed']
    else:
        dwell_speed = 'Fast'

    # MATLAB CalculateSub.m:66 unconditionally overrides DwellSpeed to 'Slow'
    # (regardless of what the settings file contains).  Match that behaviour.
    dwell_speed = 'Slow'

    # --- Location ---
    location = {}
    location['NewPolyp'] = np.array(handles['Variables']['Location_NewPolyp'], dtype=float)
    location['DirectCa'] = np.array(handles['Variables']['Location_DirectCa'], dtype=float)
    location['EarlyProgression'] = np.array(handles['Variables']['Location_EarlyProgression'], dtype=float)
    location['AdvancedProgression'] = np.array(handles['Variables']['Location_AdvancedProgression'], dtype=float)
    location['CancerProgression'] = np.array(handles['Variables']['Location_CancerProgression'], dtype=float)
    location['CancerSymptoms'] = np.array(handles['Variables']['Location_CancerSymptoms'], dtype=float)
    location['ColoDetection'] = np.array(handles['Variables']['Location_ColoDetection'], dtype=float)
    location['RectoSigmoDetection'] = np.array(handles['Variables']['Location_RectoSigmoDetection'], dtype=float)
    location['ColoReach'] = np.array(handles['Variables']['Location_ColoReach'], dtype=float)
    location['RectoSigmoReach'] = np.array(handles['Variables']['Location_RectoSigmoReach'], dtype=float)

    # --- Female/Male ---
    female = {}
    female['fraction_female'] = handles['Variables']['fraction_female']
    female['new_polyp_female'] = handles['Variables']['new_polyp_female']
    female['early_progression_female'] = handles['Variables']['early_progression_female']
    female['advanced_progression_female'] = handles['Variables']['advanced_progression_female']
    female['symptoms_female'] = handles['Variables']['symptoms_female']

    # --- Costs ---
    cost = dict(handles['Variables']['Cost'])

    cost_src = handles['Variables']['Cost']
    cost_stage = {}

    # Current treatment costs
    cost_stage['Initial'] = [cost_src['Initial_I'], cost_src['Initial_II'], cost_src['Initial_III'], cost_src['Initial_IV']]
    cost_stage['Cont'] = [cost_src['Cont_I'], cost_src['Cont_II'], cost_src['Cont_III'], cost_src['Cont_IV']]
    cost_stage['Final'] = [cost_src['Final_I'], cost_src['Final_II'], cost_src['Final_III'], cost_src['Final_IV']]
    cost_stage['Final_oc'] = [cost_src['Final_oc_I'], cost_src['Final_oc_II'], cost_src['Final_oc_III'], cost_src['Final_oc_IV']]

    # Future treatment costs
    cost_stage['FutInitial'] = [cost_src['FutInitial_I'], cost_src['FutInitial_II'], cost_src['FutInitial_III'], cost_src['FutInitial_IV']]
    cost_stage['FutCont'] = [cost_src['FutCont_I'], cost_src['FutCont_II'], cost_src['FutCont_III'], cost_src['FutCont_IV']]
    cost_stage['FutFinal'] = [cost_src['FutFinal_I'], cost_src['FutFinal_II'], cost_src['FutFinal_III'], cost_src['FutFinal_IV']]
    cost_stage['FutFinal_oc'] = [cost_src['FutFinal_oc_I'], cost_src['FutFinal_oc_II'], cost_src['FutFinal_oc_III'], cost_src['FutFinal_oc_IV']]

    # --- Complications (Risc) ---
    risc = {}
    risc['Colonoscopy_RiscPerforation'] = handles['Variables']['Colonoscopy_RiscPerforation']
    risc['Rectosigmo_Perforation'] = handles['Variables']['Rectosigmo_Perforation']
    risc['Colonoscopy_RiscSerosaBurn'] = handles['Variables']['Colonoscopy_RiscSerosaBurn']
    risc['Colonoscopy_RiscBleedingTransfusion'] = handles['Variables']['Colonoscopy_RiscBleedingTransfusion']
    risc['Colonoscopy_RiscBleeding'] = handles['Variables']['Colonoscopy_RiscBleeding']
    risc['DeathPerforation'] = handles['Variables']['DeathPerforation']
    risc['DeathBleedingTransfusion'] = handles['Variables']['DeathBleedingTransfusion']

    # --- Special Scenarios ---
    special_text = str(handles['Variables'].get('SpecialText', ''))
    padding = '                         '  # 25 spaces
    if len(special_text) >= 25:
        special_text = special_text[:25]
    else:
        special_text = (special_text + padding)[:25]

    flag = {}
    flag['Polyp_Surveillance'] = (handles['Variables'].get('Polyp_Surveillance', 'off') == 'on')
    flag['Cancer_Surveillance'] = (handles['Variables'].get('Cancer_Surveillance', 'off') == 'on')
    flag['SpecialFlag'] = (handles['Variables'].get('SpecialFlag', 'off') == 'on')
    flag['Screening'] = (handles['Variables'].get('Screening', {}).get('Mode', 'off') == 'on')
    flag['Correlation'] = (handles['Variables'].get('RiskCorrelation', 'on') == 'on')

    # Default False flags
    for k in ['Schoen', 'Holme', 'Segnan', 'Atkin', 'perfect', 'Mock',
              'Kolo1', 'Kolo2', 'Kolo3', 'Po55', 'treated', 'AllPolypFollowUp']:
        flag[k] = False

    # String comparisons (matching MATLAB isequal checks)
    if special_text.startswith('RS-Schoen'):
        flag['Schoen'] = True
    elif special_text.startswith('RS-Holme'):
        flag['Holme'] = True
    elif special_text.startswith('RS-Segnan'):
        flag['Segnan'] = True
    elif special_text.startswith('RS-Atkin'):
        flag['Atkin'] = True
    elif special_text.startswith('perfect'):
        flag['perfect'] = True
    elif special_text.startswith('AllPolypFollowUp'):
        flag['AllPolypFollowUp'] = True
    elif special_text.startswith('Kolo1'):
        flag['Kolo1'] = True
    elif special_text.startswith('Kolo2'):
        flag['Kolo2'] = True
    elif special_text.startswith('Kolo3'):
        flag['Kolo3'] = True
    elif special_text[:6] == 'Po+-55':
        flag['Po55'] = True
    elif 'treated' in special_text:
        flag['treated'] = True

    if 'Mock' in special_text:
        flag['Mock'] = True

    # ---------------------------------------------------------
    # 2. Screening Variables
    # ---------------------------------------------------------

    # ScreeningTest matrix: 7 tests x 8 parameters
    # Row 0: Colonoscopy, Row 1: Rectosigmoidoscopy, Row 2: FOBT,
    # Row 3: I_FOBT, Row 4: Sept9_HiSens, Row 5: Sept9_HiSpec, Row 6: other
    screening_test = np.zeros((7, 8))

    # Colonoscopy: MATLAB inserts 0 as third element
    # MATLAB: [handles.Variables.Screening.Colonoscopy(1:2), 0, handles.Variables.Screening.Colonoscopy(3:7)]
    col_vars = list(handles['Variables']['Screening']['Colonoscopy'])
    screening_test[0, :] = [col_vars[0], col_vars[1], 0,
                            col_vars[2], col_vars[3], col_vars[4], col_vars[5], col_vars[6]]

    screening_test[1, :] = handles['Variables']['Screening']['Rectosigmoidoscopy']
    screening_test[2, :] = handles['Variables']['Screening']['FOBT']
    screening_test[3, :] = handles['Variables']['Screening']['I_FOBT']
    screening_test[4, :] = handles['Variables']['Screening']['Sept9_HiSens']
    screening_test[5, :] = handles['Variables']['Screening']['Sept9_HiSpec']
    screening_test[6, :] = handles['Variables']['Screening']['other']

    screening_handles = ['Colonoscopy', 'Rectosigmoidoscopy', 'FOBT', 'I_FOBT',
                         'Sept9_HiSens', 'Sept9_HiSpec', 'other']

    # Construct ScreeningMatrix: a 1000-element lookup table for screening type assignment
    # MATLAB: ScreeningMatrix = zeros(1, 1000) then fills with test index (1-based)
    # Python: we fill with 1-based test index (0 = no screening), matching MATLAB
    screening_matrix = np.zeros(1000, dtype=int)
    start_idx = 0
    for f_idx, name in enumerate(screening_handles):
        prob = handles['Variables']['Screening'][name][0]
        if prob > 0:
            length = int(round(prob * 1000))
            end_idx = min(start_idx + length, 1000)
            # MATLAB stores 1-based test index (f=1..7) in ScreeningMatrix.
            # NumberCrunching_100000 checks ScreeningPreference(z) != 0 to
            # determine whether a patient has screening assigned (0 = none),
            # then uses  pi = preference - 1  to get a 0-based row index
            # into ScreeningTest.  We must store 1-based values here to
            # match that convention.
            screening_matrix[start_idx:end_idx] = f_idx + 1
            start_idx = end_idx

    # Sensitivity arrays for stool/blood tests
    # MATLAB: Sensitivity(3,:) = FOBT_Sens (10 elements)
    # Python: row 2 = FOBT, row 3 = I_FOBT, etc. (0-based rows)
    # The Sens arrays have 10 elements: P1-P6, Ca1-Ca4
    sens_fobt = np.array(handles['Variables']['Screening']['FOBT_Sens'], dtype=float)
    sens_ifobt = np.array(handles['Variables']['Screening']['I_FOBT_Sens'], dtype=float)
    sens_sept9hs = np.array(handles['Variables']['Screening']['Sept9_HiSens_Sens'], dtype=float)
    sens_sept9hp = np.array(handles['Variables']['Screening']['Sept9_HiSpec_Sens'], dtype=float)
    sens_other = np.array(handles['Variables']['Screening']['other_Sens'], dtype=float)

    # NumberCrunching_100000 indexes Sensitivity as Sensitivity[pi, max_p-1]
    # where pi is the test index (0-based) and max_p is a 1-based polyp type (1-10).
    # So we need Sensitivity to have shape (num_tests, 10).
    # MATLAB: rows 3-7 (1-based), Python: rows 2-6 (0-based)
    sensitivity = np.zeros((8, 10))
    sensitivity[2, :] = sens_fobt
    sensitivity[3, :] = sens_ifobt
    sensitivity[4, :] = sens_sept9hs
    sensitivity[5, :] = sens_sept9hp
    sensitivity[6, :] = sens_other

    # --- Age Progression ---
    # 6 polyp types x 150 age-years
    age_progression = np.zeros((6, 150))
    prog = np.array(handles['Variables']['Progression'], dtype=float)
    early_p = np.array(handles['Variables']['EarlyProgression'], dtype=float)
    adv_p = np.array(handles['Variables']['AdvancedProgression'], dtype=float)

    age_progression[0, :] = early_p * prog[0]
    age_progression[1, :] = early_p * prog[1]
    age_progression[2, :] = early_p * prog[2]
    age_progression[3, :] = early_p * prog[3]
    age_progression[4, :] = adv_p * prog[4]
    age_progression[5, :] = adv_p * prog[5]

    new_polyp = np.array(handles['Variables']['NewPolyp'], dtype=float)
    colonoscopy_likelyhood = np.array(handles['Variables']['ColonoscopyLikelyhood'], dtype=float)

    # --- Patient Distribution ---
    individual_risk = np.zeros(n)
    risk_dist = {}
    risk_dist['EarlyRisk'] = np.array(handles['Variables']['EarlyRisk'], dtype=float)
    risk_dist['AdvancedRisk'] = np.array(handles['Variables']['AdvRisk'], dtype=float)

    src_individual_risk = np.array(handles['Variables']['IndividualRisk'], dtype=float)

    # Vectorized: MATLAB: round(rand*499)+1 gives 1..500 (1-based)
    # Python: randint(0,500) gives 0..499 (0-based)
    rand_indices = np.random.randint(0, len(src_individual_risk), size=n)
    individual_risk = src_individual_risk[rand_indices]

    # Gender: 1=male, 2=female
    rand_gender = np.random.random(n)
    gender_arr = np.where(rand_gender < female['fraction_female'], 2, 1).astype(float)

    # Screening Preference
    rand_pref = np.random.randint(0, 1000, size=n)
    screening_preference = screening_matrix[rand_pref]

    # ---------------------------------------------------------
    # 3. Mortality Calculation
    # ---------------------------------------------------------

    # Source: SEER relative survival rates
    survival_tmp = np.array([100, 82.4, 74.6, 69.5, 65.9, 63.3, 61.5, 60, 58.9, 58, 57.3])
    survival_tmp = survival_tmp / 100.0

    # Create smooth curve via interpolation
    surf = np.zeros(21)
    counter = 0
    for x1 in range(5):           # MATLAB 1:5
        for x2 in range(1, 5):    # MATLAB 1:4
            surf[counter] = survival_tmp[x1] * (5 - x2) / 4.0 + survival_tmp[x1 + 1] * (x2 - 1) / 4.0
            counter += 1
    surf[counter] = survival_tmp[5]

    # MATLAB: Surf = ones(1,21) - Surf
    surf = 1.0 - surf

    mortality_correction = np.array(handles['Variables']['MortalityCorrectionGraph'], dtype=float) - 1.0

    # Mortality Matrix: (4 stages, 100 years, 1000 random slots), filled with 25 (= survived)
    mortality_matrix = np.full((4, 100, 1000), 25, dtype=int)

    mortality_params = stage_variables['Mortality']

    try:
        for f in range(4):  # 4 cancer stages
            # MATLAB: Mortality(f+6) with f=1..4 -> indices 7,8,9,10 (1-based)
            # Python: indices 6,7,8,9 (0-based)
            factor = mortality_params[f + 6] / (1.0 - survival_tmp[5])
            surf2 = surf * factor
            surf4 = surf2 * surf2

            for y_idx in range(100):  # 100 age-years
                corr_val = mortality_correction[y_idx]
                denom = (surf2 * corr_val) + surf2

                term = np.zeros_like(surf2)
                mask = denom != 0
                term[mask] = (surf2[mask] * corr_val) / denom[mask]

                mort_temp = surf2 + term * (1.0 - surf4)

                # MATLAB: MortTemp2 = MortTemp(2:21) -> Python indices 1:21
                mort_temp2 = np.clip(mort_temp[1:21], None, 1.0)

                ind_start = 0
                for g in range(20):
                    ind_end = int(round(mort_temp2[g] * 1000))
                    if ind_end < 1:
                        ind_end = 1
                    if ind_end > 1000:
                        ind_end = 1000

                    if ind_start < ind_end:
                        mortality_matrix[f, y_idx, ind_start:ind_end] = g + 1

                    ind_start = ind_end
                    if ind_start >= 1000:
                        ind_start = 1000
                        break

                    if g == 19:
                        val_limit = int(round(mort_temp2[g] * 1000))
                        if val_limit < 1000:
                            mortality_matrix[f, y_idx, val_limit:1000] = 25

                # Shuffle the 1000 slots for this year/stage
                mortality_matrix[f, y_idx, :] = np.random.permutation(mortality_matrix[f, y_idx, :])

    except Exception as e:
        print(f"Error in Mortality Matrix generation: {e}")
        raise

    life_table = np.array(handles['Variables']['LifeTable'], dtype=float)
    if flag['Po55']:
        life_table = np.zeros_like(life_table)

    # ---------------------------------------------------------
    # 4. Stages & Duration (hardcoded constants from MATLAB)
    # ---------------------------------------------------------

    stage_duration = np.array([
        [1, 0, 0, 0],
        [0.468, 0.532, 0, 0],
        [0.25, 0.398, 0.352, 0],
        [0.162, 0.22, 0.275, 0.343]
    ])

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
        [0.018, 0.080, 0.434, 0.488]
    ])

    # Location Matrix (2 rows, 1000 cols) for fast random lookup
    # Row 0: new polyp location, Row 1: direct cancer location
    # Values are 1-based location indices (1..13)
    location_matrix = np.zeros((2, 1000), dtype=int)

    # New Polyp Location (CDF-based assignment)
    loc_counter = 0
    total_new_polyp = np.sum(location['NewPolyp'])
    for f in range(13):
        current_sum = np.sum(location['NewPolyp'][:f + 1])
        ende = int(round((current_sum / total_new_polyp) * 1000))
        if ende > 1000:
            ende = 1000
        if ende > loc_counter:
            location_matrix[0, loc_counter:ende] = f + 1  # 1-based location
            loc_counter = ende

    # Direct Cancer Location (CDF-based assignment)
    loc_counter = 0
    total_direct_ca = np.sum(location['DirectCa'])
    for f in range(13):
        current_sum = np.sum(location['DirectCa'][:f + 1])
        ende = int(round((current_sum / total_direct_ca) * 1000))
        if ende > 1000:
            ende = 1000
        if ende > loc_counter:
            location_matrix[1, loc_counter:ende] = f + 1
            loc_counter = ende

    # Bundle every prepared parameter for NumberCrunching_100000.
    return {
        'p': p, 'n': n,
        'stage_variables': stage_variables, 'location': location,
        'cost': cost, 'cost_stage': cost_stage, 'risc': risc, 'flag': flag,
        'special_text': special_text, 'female': female, 'sensitivity': sensitivity,
        'screening_test': screening_test, 'screening_preference': screening_preference,
        'age_progression': age_progression, 'new_polyp': new_polyp,
        'colonoscopy_likelyhood': colonoscopy_likelyhood,
        'individual_risk': individual_risk, 'risk_dist': risk_dist,
        'gender_arr': gender_arr, 'life_table': life_table,
        'mortality_matrix': mortality_matrix, 'location_matrix': location_matrix,
        'stage_duration': stage_duration, 'tx1': tx1,
        'direct_cancer_rate': direct_cancer_rate,
        'direct_cancer_speed': direct_cancer_speed, 'dwell_speed': dwell_speed,
    }


def calculate_sub(handles):
    """
    Prepare simulation variables and run the CMOST simulation pipeline.

    Behaviour is unchanged from the original translation of CalculateSub.m;
    the parameter-preparation block is now factored out into
    prepare_parameters() so it can be reused by the individual-level
    environment.  This function still:
      - builds the parameter bundle (via prepare_parameters)
      - calls NumberCrunching_100000 (the Monte Carlo simulation engine)
      - calls Evaluation (results analysis / benchmarking)

    Returns
    -------
    tuple : (handles, BM)
    """
    params = prepare_parameters(handles)
    p = params['p']; n = params['n']
    stage_variables = params['stage_variables']; location = params['location']
    cost = params['cost']; cost_stage = params['cost_stage']; risc = params['risc']
    flag = params['flag']; special_text = params['special_text']
    female = params['female']; sensitivity = params['sensitivity']
    screening_test = params['screening_test']
    screening_preference = params['screening_preference']
    age_progression = params['age_progression']; new_polyp = params['new_polyp']
    colonoscopy_likelyhood = params['colonoscopy_likelyhood']
    individual_risk = params['individual_risk']; risk_dist = params['risk_dist']
    gender_arr = params['gender_arr']; life_table = params['life_table']
    mortality_matrix = params['mortality_matrix']
    location_matrix = params['location_matrix']
    stage_duration = params['stage_duration']; tx1 = params['tx1']
    direct_cancer_rate = params['direct_cancer_rate']
    direct_cancer_speed = params['direct_cancer_speed']
    dwell_speed = params['dwell_speed']

    # ---------------------------------------------------------
    # 5. Running Calculations
    # ---------------------------------------------------------

    print(f"Running CMOST simulation with {n} patients...")

    try:
        (y_result, gender_out, death_cause, last, death_year, natural_death_year,
         direct_cancer_out, direct_cancer_r, direct_cancer2, direct_cancer2_r,
         progressed_cancer, progressed_cancer_r, tumor_record,
         dwell_time_progression, dwell_time_fast_cancer,
         has_cancer, num_polyps, max_polyps, all_polyps, num_cancer, max_cancer,
         payment_type, money, number,
         early_polyps_removed, diagnosed_cancer, advanced_polyps_removed,
         year_included, year_alive
         ) = NumberCrunching_100000(
            p, stage_variables, location, cost, cost_stage, risc,
            flag, special_text, female, sensitivity,
            screening_test, screening_preference, age_progression,
            new_polyp, colonoscopy_likelyhood, individual_risk,
            risk_dist, gender_arr, life_table, mortality_matrix,
            location_matrix, stage_duration, tx1, direct_cancer_rate,
            direct_cancer_speed, dwell_speed
        )

        print(f"Simulation complete. Simulated {y_result} years.")

    except Exception as e:
        print(f"Error running NumberCrunching_100000: {e}")
        import traceback
        traceback.print_exc()
        return handles, None

    # Pack results into data dictionary (matching MATLAB data struct fields)
    data = {
        'n': n,
        'y': y_result,
        'Gender': gender_out,
        'DeathCause': death_cause,
        'Last': last,
        'DeathYear': death_year,
        'NaturalDeathYear': natural_death_year,
        'DirectCancer': direct_cancer_out,
        'DirectCancerR': direct_cancer_r,
        'DirectCancer2': direct_cancer2,
        'DirectCancer2R': direct_cancer2_r,
        'ProgressedCancer': progressed_cancer,
        'ProgressedCancerR': progressed_cancer_r,
        'TumorRecord': tumor_record,
        'DwellTimeProgression': dwell_time_progression,
        'DwellTimeFastCancer': dwell_time_fast_cancer,
        'HasCancer': has_cancer,
        'NumPolyps': num_polyps,
        'MaxPolyps': max_polyps,
        'AllPolyps': all_polyps,
        'NumCancer': num_cancer,
        'MaxCancer': max_cancer,
        'PaymentType': payment_type,
        'Money': money,
        'Number': number,
        'EarlyPolypsRemoved': early_polyps_removed,
        'DiagnosedCancer': diagnosed_cancer,
        'AdvancedPolypsRemoved': advanced_polyps_removed,
        'YearIncluded': year_included,
        'YearAlive': year_alive,
        'InputCost': cost,
        'InputCostStage': cost_stage,
    }

    # ---------------------------------------------------------
    # 6. Evaluation
    # ---------------------------------------------------------

    # MATLAB: [data, BM] = Evaluation(data, handles.Variables);
    bm = None
    try:
        data, bm = Evaluation(data, handles['Variables'])
        print("Evaluation complete.")
    except Exception as e:
        print(f"Error in Evaluation: {e}")
        import traceback
        traceback.print_exc()

    # Store data in handles (matching MATLAB behavior: handles.data = data)
    handles['data'] = data

    return handles, bm
