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

import numpy as np
import math
import os

# ---------------------------------------------------------------------------
# DEBUG_TRACE: set to True (or set env var CMOST_DEBUG_TRACE=1) to log every
# cancer creation event to a CSV file for comparison with MATLAB.
# The trace file is written to python/debug_cancer_trace.csv.
# ---------------------------------------------------------------------------
DEBUG_TRACE = os.environ.get('CMOST_DEBUG_TRACE', '0') == '1'
_trace_rows = []  # populated when DEBUG_TRACE is True

# ---------------------------------------------------------------------------
# INDEX CONVENTION NOTES:
#
# In the original MATLAB code, y goes from 1 to 100 (1-based) and is used
# directly as an array index.  In this Python translation we keep y as a
# 1-based *semantic* year counter (1..100) and use  yi = y - 1  whenever we
# need a 0-based array index.  Similarly, patient index z goes 1..n in
# MATLAB; here z goes 0..n-1.  Cancer stages remain 7-10 and locations
# remain 1-13 as semantic values; we subtract the appropriate offset only
# when indexing into 0-based arrays (e.g. stage-7 for a 4-element array,
# location-1 for a 13-element array).
#
# MATLAB's  rand  is replaced by  np.random.rand() .
# MATLAB's  round(rand*999)+1  (giving 1..1000) becomes
#           int(round(np.random.rand()*999))  (giving 0..999) for 0-based
#           array access into 1000-element lookup arrays.
# ---------------------------------------------------------------------------


def _rand_idx_1000():
    """Return a random 0-based index in [0, 999] matching MATLAB round(rand*999+1) -> 1..1000."""
    return int(round(np.random.rand() * 999))


def _find_last_nonzero(arr):
    """Return 0-based index of last non-zero element, or -1 if none."""
    nz = np.flatnonzero(arr)
    if len(nz) == 0:
        return -1
    return int(nz[-1])


def _count_nonzero(arr):
    """Count non-zero elements."""
    return int(np.count_nonzero(arr))


def _shift_left_polyp(Polyp_Polyps, Polyp_PolypYear, Polyp_PolypLocation,
                       Polyp_EarlyProgression, Polyp_AdvProgression, z, f, l):
    """
    Replicate MATLAB:
        Polyp.Polyps(z, f:l) = Polyp.Polyps(z, f+1:l+1);
    for all five polyp arrays.  f and l are 0-based Python indices.
    In MATLAB f:l has length l-f+1 and f+1:l+1 also has length l-f+1.
    In Python [f:l+1] has length l-f+1 and [f+1:l+2] also has length l-f+1.
    """
    Polyp_Polyps[z, f:l+1]           = np.append(Polyp_Polyps[z, f+1:l+1], 0)
    Polyp_PolypYear[z, f:l+1]       = np.append(Polyp_PolypYear[z, f+1:l+1], 0)
    Polyp_PolypLocation[z, f:l+1]   = np.append(Polyp_PolypLocation[z, f+1:l+1], 0)
    Polyp_EarlyProgression[z, f:l+1] = np.append(Polyp_EarlyProgression[z, f+1:l+1], 0)
    Polyp_AdvProgression[z, f:l+1]   = np.append(Polyp_AdvProgression[z, f+1:l+1], 0)


def _shift_left_cancer(Ca_Cancer, Ca_CancerYear, Ca_CancerLocation,
                        Ca_DwellTime, Ca_SympTime, Ca_SympStage,
                        Ca_TimeStage_I, Ca_TimeStage_II, Ca_TimeStage_III,
                        z, f, l):
    """
    Replicate MATLAB shift-left for all cancer arrays.
    f and l are 0-based Python indices.
    """
    Ca_Cancer[z, f:l+1]         = np.append(Ca_Cancer[z, f+1:l+1], 0)
    Ca_CancerYear[z, f:l+1]     = np.append(Ca_CancerYear[z, f+1:l+1], 0)
    Ca_CancerLocation[z, f:l+1] = np.append(Ca_CancerLocation[z, f+1:l+1], 0)
    Ca_DwellTime[z, f:l+1]      = np.append(Ca_DwellTime[z, f+1:l+1], 0)
    Ca_SympTime[z, f:l+1]       = np.append(Ca_SympTime[z, f+1:l+1], 0)
    Ca_SympStage[z, f:l+1]      = np.append(Ca_SympStage[z, f+1:l+1], 0)
    Ca_TimeStage_I[z, f:l+1]    = np.append(Ca_TimeStage_I[z, f+1:l+1], 0)
    Ca_TimeStage_II[z, f:l+1]   = np.append(Ca_TimeStage_II[z, f+1:l+1], 0)
    Ca_TimeStage_III[z, f:l+1]  = np.append(Ca_TimeStage_III[z, f+1:l+1], 0)


# ===================================================================
#  COLONOSCOPY  (sub-function)
# ===================================================================

def Colonoscopy(z, y, q, Modus, Gender,
                Polyp_Polyps, Polyp_PolypYear, Polyp_PolypLocation,
                Polyp_EarlyProgression, Polyp_AdvProgression,
                Ca_Cancer, Ca_CancerYear, Ca_CancerLocation,
                Ca_DwellTime, Ca_SympTime, Ca_SympStage,
                Ca_TimeStage_I, Ca_TimeStage_II, Ca_TimeStage_III,
                Detected_Cancer, Detected_CancerYear, Detected_CancerLocation,
                Detected_MortTime,
                Included, DeathCause, DeathYear,
                DiagnosedCancer, AdvancedPolypsRemoved, EarlyPolypsRemoved,
                Last_Colonoscopy, Last_Polyp, Last_AdvPolyp, Last_Cancer,
                TumorRecord_Stage, TumorRecord_Location, TumorRecord_Sojourn,
                TumorRecord_DwellTime, TumorRecord_Gender,
                TumorRecord_Detection, TumorRecord_PatientNumber,
                PaymentType_Colonoscopy, PaymentType_ColonoscopyPolyp,
                PaymentType_Colonoscopy_Cancer,
                PaymentType_Perforation, PaymentType_Serosa,
                PaymentType_Bleeding, PaymentType_BleedingTransf,
                PaymentType_Cancer_ini, PaymentType_Cancer_con,
                PaymentType_Cancer_fin,
                PaymentType_QCancer_ini, PaymentType_QCancer_con,
                PaymentType_QCancer_fin,
                Money_Screening, Money_Treatment, Money_FutureTreatment,
                Money_FollowUp, Money_Other,
                StageVariables, Cost, Location, risc,
                ColoReachMatrix, MortalityMatrix, CostStage):
    """
    Perform a colonoscopy for patient z.
    z is 0-based patient index.
    y is 1-based year (1..100).  yi = y-1 for array indexing.
    q is 1-based quarter (1..4).
    """
    yi = y - 1  # 0-based year index

    # in this function we do a colonoscopy for the respective patient (number
    # z). We cure all detected polyps, and handle the case if a cancer was
    # detected

    # we determine the reach of this colonoscopy (cecum = 1, rectum = 13))
    CurrentReach = int(ColoReachMatrix[_rand_idx_1000()])
    CurrentReachMatrix = np.zeros(13)
    # MATLAB: CurrentReachMatrix(CurrentReach:13) = 1  (1-based)
    CurrentReachMatrix[CurrentReach - 1:13] = 1

    counter = 0
    # MATLAB: for f=length(find(Polyp.Polyps(z, :))) : -1 : 1
    l_polyp = _count_nonzero(Polyp_Polyps[z, :])
    for f in range(l_polyp - 1, -1, -1):  # backwards, 0-based
        Tumor = Polyp_Polyps[z, f]
        p_loc = Polyp_PolypLocation[z, f]  # 1-based location
        # MATLAB: rand < StageVariables.Colo_Detection(Tumor) * Location.ColoDetection(loc)
        #         AND CurrentReachMatrix(loc) == 1
        if (np.random.rand() < StageVariables['Colo_Detection'][int(Tumor) - 1] *
                Location['ColoDetection'][int(p_loc) - 1] and
                CurrentReachMatrix[int(p_loc) - 1] == 1):
            # we delete the current polyp
            l = _count_nonzero(Polyp_Polyps[z, :])
            _shift_left_polyp(Polyp_Polyps, Polyp_PolypYear, Polyp_PolypLocation,
                              Polyp_EarlyProgression, Polyp_AdvProgression, z, f, l - 1)
            counter += 1
            if Tumor > 4:
                AdvancedPolypsRemoved[yi] += 1
                Last_AdvPolyp[z] = y
            else:
                Last_Polyp[z] = y
                EarlyPolypsRemoved[yi] += 1

    if counter > 2:
        Last_AdvPolyp[z] = y  # 3 polyps counts as an advanced polyp

    StageTmp = 0
    EStage = 0
    ELocation = 0
    ESojourn = 0
    EDwellTime = 0
    LocationTmp = 0
    SojournTmp = 0
    DwellTimeTmp = 0

    m = 0
    # m2 moved the switch statement up. otherwise without cancer m remains 0
    if Modus == 'Scre':
        m = 1
    elif Modus == 'Symp':
        m = 2
    elif Modus == 'Foll':
        m = 3
    elif Modus == 'Base':
        m = 4

    l_ca = _count_nonzero(Ca_Cancer[z, :])
    for f in range(l_ca - 1, -1, -1):
        Tumor = Ca_Cancer[z, f]
        ca_loc = Ca_CancerLocation[z, f]  # 1-based
        # MATLAB: rand < StageVariables.Colo_Detection(Tumor) AND CurrentReachMatrix(loc)==1
        if (np.random.rand() < StageVariables['Colo_Detection'][int(Tumor) - 1] and
                CurrentReachMatrix[int(ca_loc) - 1] == 1):

            if counter == 0:
                counter = -1
            # the cancer is now a detected cancer
            pos = _count_nonzero(Detected_Cancer[z, :])
            Detected_Cancer[z, pos] = Ca_Cancer[z, f]
            Detected_CancerYear[z, pos] = y + (q - 1) / 4.0
            Detected_CancerLocation[z, pos] = Ca_CancerLocation[z, f]
            Detected_MortTime[z, pos] = MortalityMatrix[int(Ca_Cancer[z, f]) - 7, yi, _rand_idx_1000()]

            # we need keep track of key parameters
            StageTmp = Tumor
            LocationTmp = Ca_CancerLocation[z, f]
            SojournTmp = y + (q - 1) / 4.0 - Ca_CancerYear[z, f]
            DwellTimeTmp = Ca_DwellTime[z, f]

            # the original cancer is removed from the database
            l = _count_nonzero(Ca_Cancer[z, :])
            _shift_left_cancer(Ca_Cancer, Ca_CancerYear, Ca_CancerLocation,
                               Ca_DwellTime, Ca_SympTime, Ca_SympStage,
                               Ca_TimeStage_I, Ca_TimeStage_II, Ca_TimeStage_III,
                               z, f, l - 1)

            DiagnosedCancer[yi, z] = max(DiagnosedCancer[yi, z], Tumor)
            Last_Cancer[z] = y

            # re-determine m (same as above, but MATLAB has it here too)
            if Modus == 'Scre':
                m = 1
            elif Modus == 'Symp':
                m = 2
            elif Modus == 'Foll':
                m = 3
            elif Modus == 'Base':
                m = 4

        if StageTmp != 0:
            if StageTmp > EStage:
                EStage = StageTmp
                ELocation = LocationTmp
                ESojourn = SojournTmp
                EDwellTime = DwellTimeTmp

    if StageTmp != 0:
        pos = _count_nonzero(TumorRecord_Stage[yi, :])
        TumorRecord_Stage[yi, pos] = EStage
        TumorRecord_Location[yi, pos] = ELocation
        TumorRecord_DwellTime[yi, pos] = EDwellTime
        TumorRecord_Sojourn[yi, pos] = ESojourn
        TumorRecord_Gender[yi, pos] = Gender[z]
        TumorRecord_Detection[yi, pos] = m
        TumorRecord_PatientNumber[yi, pos] = z + 1  # store 1-based patient number

    Last_Colonoscopy[z] = y

    if counter == 0:  # no tumor or polyp
        factor = 0.75
        moneyspent = Cost['Colonoscopy']
        PaymentType_Colonoscopy[m - 1, yi] += 1
    elif counter == -1:
        factor = 1.5
        moneyspent = Cost['Colonoscopy_Cancer']
        PaymentType_Colonoscopy_Cancer[m - 1, yi] += 1
    else:
        factor = 1.5
        moneyspent = Cost['Colonoscopy_Polyp']
        PaymentType_ColonoscopyPolyp[m - 1, yi] += 1

    # Complications
    if np.random.rand() < risc['Colonoscopy_RiscPerforation'] * factor:
        # a perforation happened
        moneyspent += Cost['Colonoscopy_Perforation']
        PaymentType_Perforation[m - 1, yi] += 1
        if np.random.rand() < risc['DeathPerforation']:
            # patient died during colonoscopy from a perforation
            Included[z] = False
            DeathCause[z] = 3
            DeathYear[z] = y
            # we add the costs
            AddCosts(Detected_Cancer, Detected_CancerYear, Detected_CancerLocation,
                     Detected_MortTime, CostStage,
                     PaymentType_Cancer_ini, PaymentType_Cancer_con,
                     PaymentType_Cancer_fin,
                     PaymentType_QCancer_ini, PaymentType_QCancer_con,
                     PaymentType_QCancer_fin,
                     Money_Treatment, Money_FutureTreatment,
                     y + (q - 1) / 4.0, z, 'oc')
    elif np.random.rand() < risc['Colonoscopy_RiscSerosaBurn'] * factor:
        # serosal burn
        moneyspent += Cost['Colonoscopy_Serosal_burn']
        PaymentType_Serosa[m - 1, yi] += 1
    elif np.random.rand() < risc['Colonoscopy_RiscBleeding'] * factor:
        # a bleeding episode (no transfusion)
        moneyspent += Cost['Colonoscopy_bleed']
        PaymentType_Bleeding[m - 1, yi] += 1
    elif np.random.rand() < risc['Colonoscopy_RiscBleedingTransfusion'] * factor:
        # bleeding requiring transfusion
        moneyspent += Cost['Colonoscopy_bleed_transfusion']
        PaymentType_BleedingTransf[m - 1, yi] += 1
        if np.random.rand() < risc['DeathBleedingTransfusion']:
            # patient died during colonoscopy from a bleeding complication
            Included[z] = False
            DeathCause[z] = 3
            DeathYear[z] = y

    if Modus == 'Scre':
        Money_Screening[yi] += moneyspent
    elif Modus == 'Symp':
        Money_Treatment[yi] += moneyspent
    elif Modus == 'Foll':
        Money_FollowUp[yi] += moneyspent
    elif Modus == 'Base':
        Money_Other[yi] += moneyspent


# ===================================================================
#  RECTOSIGMOIDOSCOPY  (sub-function)
# ===================================================================

def RectoSigmo(z, y, Polyp_Polyps, Polyp_PolypYear, Polyp_PolypLocation,
               Polyp_EarlyProgression, Polyp_AdvProgression,
               Ca_Cancer, Ca_CancerLocation,
               Included, DeathCause, DeathYear,
               PaymentType_RS, PaymentType_RSPolyp, PaymentType_Perforation,
               Money_Screening,
               StageVariables, Cost, Location, risc,
               RectoSigmoReachMatrix, flag):
    """
    Perform a rectosigmoidoscopy for patient z.
    Returns (PolypFlag, AdvPolypFlag, CancerFlag).
    z is 0-based, y is 1-based.
    """
    yi = y - 1

    # we determine the reach (cecum = 1, rectum = 13)
    CurrentReach = int(RectoSigmoReachMatrix[_rand_idx_1000()])
    CurrentReachMatrix = np.zeros(13)
    CurrentReachMatrix[CurrentReach - 1:13] = 1

    counter = 0
    PolypFlag = 0
    AdvPolypFlag = 0
    CancerFlag = 0
    PolypMax = 0

    l_polyp = _count_nonzero(Polyp_Polyps[z, :])

    if flag.get('Schoen', False):
        # Schoen study
        for f in range(l_polyp - 1, -1, -1):
            Tumor = Polyp_Polyps[z, f]
            p_loc = Polyp_PolypLocation[z, f]
            if (np.random.rand() < StageVariables['RectoSigmo_Detection'][int(Tumor) - 1] *
                    Location['RectoSigmoDetection'][int(p_loc) - 1] and
                    CurrentReachMatrix[int(p_loc) - 1] == 1):
                # in this scenario we only do follow up for larger polyps
                if Tumor > 2:
                    PolypFlag = 1.5
                elif PolypFlag == 1.5:
                    PolypFlag = 1.5
                else:
                    PolypFlag = 1
                counter += 1
                PolypMax = max(PolypMax, Tumor)

    elif flag.get('Atkin', False):
        # Atkin study
        for f in range(l_polyp - 1, -1, -1):
            Tumor = Polyp_Polyps[z, f]
            p_loc = Polyp_PolypLocation[z, f]
            if (np.random.rand() < StageVariables['RectoSigmo_Detection'][int(Tumor) - 1] *
                    Location['RectoSigmoDetection'][int(p_loc) - 1] and
                    CurrentReachMatrix[int(p_loc) - 1] == 1):
                if Tumor > 2:
                    PolypFlag = 1
                counter += 1
                # we delete the current polyp only in the Atkins study
                if Tumor > 2:
                    l = _count_nonzero(Polyp_Polyps[z, :])
                    _shift_left_polyp(Polyp_Polyps, Polyp_PolypYear, Polyp_PolypLocation,
                                      Polyp_EarlyProgression, Polyp_AdvProgression, z, f, l - 1)
                PolypMax = max(PolypMax, Tumor)

    elif flag.get('Segnan', False):
        # Italian / Segnan study
        for f in range(l_polyp - 1, -1, -1):
            Tumor = Polyp_Polyps[z, f]
            p_loc = Polyp_PolypLocation[z, f]
            if (np.random.rand() < StageVariables['RectoSigmo_Detection'][int(Tumor) - 1] *
                    Location['RectoSigmoDetection'][int(p_loc) - 1] and
                    CurrentReachMatrix[int(p_loc) - 1] == 1):
                if Tumor > 2:
                    PolypFlag = 1
                    PolypMax = max(PolypMax, Tumor)
                    counter += 1
                else:
                    # in this study we only delete small polyps; larger polyps are referred to colonoscopy
                    l = _count_nonzero(Polyp_Polyps[z, :])
                    _shift_left_polyp(Polyp_Polyps, Polyp_PolypYear, Polyp_PolypLocation,
                                      Polyp_EarlyProgression, Polyp_AdvProgression, z, f, l - 1)

    else:
        # Default
        for f in range(l_polyp - 1, -1, -1):
            Tumor = Polyp_Polyps[z, f]
            p_loc = Polyp_PolypLocation[z, f]
            if (np.random.rand() < StageVariables['RectoSigmo_Detection'][int(Tumor) - 1] *
                    Location['RectoSigmoDetection'][int(p_loc) - 1] and
                    CurrentReachMatrix[int(p_loc) - 1] == 1):
                PolypFlag = 1
                counter += 1
                PolypMax = max(PolypMax, Tumor)

    if PolypMax > 4 or counter > 2:
        AdvPolypFlag = 1

    # Cancer detection
    l_ca = _count_nonzero(Ca_Cancer[z, :])
    for f in range(l_ca - 1, -1, -1):
        Tumor = Ca_Cancer[z, f]
        ca_loc = Ca_CancerLocation[z, f]
        if (np.random.rand() < StageVariables['RectoSigmo_Detection'][int(Tumor) - 1] and
                CurrentReachMatrix[int(ca_loc) - 1] == 1):
            counter += 1
            CancerFlag = 1

    if counter == 0:
        Money_Screening[yi] += Cost['Sigmoidoscopy']
        PaymentType_RS[0, yi] += 1
    else:
        Money_Screening[yi] += Cost['Sigmoidoscopy_Polyp']
        PaymentType_RSPolyp[0, yi] += 1

    # Complications
    if np.random.rand() < risc['Rectosigmo_Perforation']:
        Money_Screening[yi] += Cost['Colonoscopy_Perforation']
        PaymentType_Perforation[0, yi] += 1
        if np.random.rand() < risc['DeathPerforation']:
            Included[z] = False
            DeathCause[z] = 3
            DeathYear[z] = y
            CancerFlag = 0
            PolypFlag = 0

    return PolypFlag, AdvPolypFlag, CancerFlag


# ===================================================================
#  ADD COSTS  (sub-function)
# ===================================================================

def AddCosts(Detected_Cancer, Detected_CancerYear, Detected_CancerLocation,
             Detected_MortTime, CostStage,
             PaymentType_Cancer_ini, PaymentType_Cancer_con, PaymentType_Cancer_fin,
             PaymentType_QCancer_ini, PaymentType_QCancer_con, PaymentType_QCancer_fin,
             Money_Treatment, Money_FutureTreatment,
             time, z, mode):
    """
    Calculate and add treatment costs for a patient who has died.
    z is 0-based patient index.
    time is the continuous year (e.g. 5.25).
    mode is 'oc' (other causes) or 'tu' (tumor).
    """
    SubCost = np.zeros((25, 404))
    SubCostFut = np.zeros((25, 404))

    Ende = time
    l = _count_nonzero(Detected_Cancer[z, :])

    for x1 in range(l):
        Start = Detected_CancerYear[z, x1]
        Difference = Ende - Start
        stage_idx = int(Detected_Cancer[z, x1]) - 7  # 0-based stage index (0..3)
        start_q = int(Start * 4)  # 0-based quarter from year start
        ende_q = int(Ende * 4)

        # MATLAB uses 1-based indexing for SubCost columns: Start*4+1 etc.
        # In Python 0-based: Start*4 corresponds to MATLAB Start*4+1
        # We map MATLAB column index C to Python C-1.

        if Difference <= 1.0 / 4:
            # first quarter costs
            SubCost[x1, start_q] = CostStage['Initial'][stage_idx]
            SubCostFut[x1, start_q] = CostStage['FutInitial'][stage_idx]
            PaymentType_Cancer_ini[stage_idx, int(math.floor(Start)) - 1] += 1
            PaymentType_QCancer_ini[stage_idx, int(math.floor(Start)) - 1, 0] += 1

        elif Difference > 1.0 / 4 and Difference <= 1.25:
            SubCost[x1, start_q] = CostStage['Initial'][stage_idx]
            SubCostFut[x1, start_q] = CostStage['FutInitial'][stage_idx]
            PaymentType_Cancer_ini[stage_idx, int(math.floor(Start)) - 1] += 1
            PaymentType_QCancer_ini[stage_idx, int(math.floor(Start)) - 1, 0] += 1

            if mode != 'oc':
                SubCost[x1, start_q + 1:ende_q] = 1.0 / 4 * CostStage['Final'][stage_idx]
                SubCostFut[x1, start_q + 1:ende_q] = 1.0 / 4 * CostStage['FutFinal'][stage_idx]
                nq = ende_q - (start_q + 1)
                PaymentType_Cancer_fin[stage_idx, int(math.floor(Ende - 1)) - 1] += nq / 4.0
                for Qcount in range(nq):
                    PaymentType_QCancer_fin[stage_idx, int(math.floor(Ende - 1)) - 1, Qcount] += 1
            else:
                SubCost[x1, start_q + 1:ende_q] = 1.0 / 4 * CostStage['Cont'][stage_idx]
                SubCostFut[x1, start_q + 1:ende_q] = 1.0 / 4 * CostStage['FutCont'][stage_idx]
                nq = ende_q - (start_q + 1)
                PaymentType_Cancer_con[stage_idx, int(math.floor(Ende - 1)) - 1] += nq / 4.0
                for Qcount in range(nq):
                    PaymentType_QCancer_con[stage_idx, int(math.floor(Ende - 1)) - 1, Qcount] += 1

        elif Difference > 1.25 and Difference <= 5.0:
            SubCost[x1, start_q:start_q + 1] = CostStage['Initial'][stage_idx]
            SubCostFut[x1, start_q:start_q + 1] = CostStage['FutInitial'][stage_idx]
            cont_end_q = int((Ende - 1) * 4)
            SubCost[x1, start_q + 1:cont_end_q] = 1.0 / 4 * CostStage['Cont'][stage_idx]
            SubCostFut[x1, start_q + 1:cont_end_q] = 1.0 / 4 * CostStage['FutCont'][stage_idx]
            PaymentType_Cancer_ini[stage_idx, int(math.floor(Start)) - 1] += 1
            PaymentType_QCancer_ini[stage_idx, int(math.floor(Start)) - 1, 0] += 1

            yyears = int(math.floor(Difference - 1.25))
            qquarters = (Difference - 1.25) - math.floor(Difference - 1.25)

            for con_y in range(1, yyears + 1):
                PaymentType_Cancer_con[stage_idx, int(math.floor(Start)) + con_y - 1] += 1
                for Qcount in range(con_y * 4 - 3 - 1, con_y * 4):  # MATLAB (con_y*4-3):(con_y*4) -> Python 0-based
                    PaymentType_QCancer_con[stage_idx, int(math.floor(Start)) + con_y - 1, Qcount] += 1

            if yyears > 0:
                con_y_val = yyears
            else:
                con_y_val = 0
            PaymentType_Cancer_con[stage_idx, int(math.floor(Start)) + con_y_val] += qquarters
            for Qcount in range(int(4 * qquarters)):
                PaymentType_QCancer_con[stage_idx, int(math.floor(Start)) + con_y_val,
                                        con_y_val * 4 + Qcount] += 1

            if mode != 'oc':
                SubCost[x1, cont_end_q:ende_q] = 1.0 / 4 * CostStage['Final'][stage_idx]
                SubCostFut[x1, cont_end_q:ende_q] = 1.0 / 4 * CostStage['FutFinal'][stage_idx]
                PaymentType_Cancer_fin[stage_idx, int(math.floor(Ende)) - 2] += 1
                for Qcount in range(4):
                    PaymentType_QCancer_fin[stage_idx, int(math.floor(Ende - 1)) - 1, Qcount] += 1
            else:
                SubCost[x1, cont_end_q:ende_q] = 1.0 / 4 * CostStage['Cont'][stage_idx]
                SubCostFut[x1, cont_end_q:ende_q] = 1.0 / 4 * CostStage['FutCont'][stage_idx]
                PaymentType_Cancer_con[stage_idx, int(math.floor(Ende)) - 2] += 1
                for Qcount in range(4):
                    PaymentType_QCancer_con[stage_idx, int(math.floor(Ende - 1)) - 1,
                                            con_y_val * 4 + int(qquarters * 4) + Qcount] += 1

        elif Difference > 5:
            SubCost[x1, start_q:start_q + 1] = CostStage['Initial'][stage_idx]
            SubCostFut[x1, start_q:start_q + 1] = CostStage['FutInitial'][stage_idx]
            SubCost[x1, start_q + 1:ende_q] = 1.0 / 4 * CostStage['Cont'][stage_idx]
            SubCostFut[x1, start_q + 1:ende_q] = 1.0 / 4 * CostStage['FutCont'][stage_idx]
            PaymentType_Cancer_ini[stage_idx, int(math.floor(Start)) - 1] += 1
            PaymentType_QCancer_ini[stage_idx, int(math.floor(Start)) - 1, 0] += 1

            for con_y in range(1, 5):  # MATLAB for con_y=1:4
                PaymentType_Cancer_con[stage_idx, int(math.floor(Start)) + con_y - 1] += 1
                for Qcount in range(con_y * 4 - 3 - 1, con_y * 4):
                    PaymentType_QCancer_con[stage_idx, int(math.floor(Start)) + con_y - 1, Qcount] += 1
            con_y_val = 4
            PaymentType_Cancer_con[stage_idx, int(math.floor(Start)) + con_y_val] += 0.75
            for Qcount in range(3):
                PaymentType_QCancer_con[stage_idx, int(math.floor(Start)) + con_y_val, Qcount] += 1

    SubCostAll = np.sum(SubCost, axis=0)
    SubCostAllFut = np.sum(SubCostFut, axis=0)
    Counter = 0
    for x1 in range(100):
        for x2 in range(4):
            Money_Treatment[x1] += SubCostAll[Counter]
            Money_FutureTreatment[x1] += SubCostAllFut[Counter]
            Counter += 1


# ===================================================================
#  MAIN FUNCTION: NumberCrunching_100000
# ===================================================================

def NumberCrunching_100000(p, StageVariables, Location, Cost, CostStage, risc,
                           flag, SpecialText, female, Sensitivity,
                           ScreeningTest, ScreeningPreference, AgeProgression,
                           NewPolyp, ColonoscopyLikelyhood, IndividualRisk,
                           RiskDistribution, Gender, LifeTable, MortalityMatrix,
                           LocationMatrix_in, StageDuration, tx1,
                           DirectCancerRate, DirectCancerSpeed, DwellSpeed):
    """
    Main simulation function.
    All input arrays use the same conventions as the MATLAB caller.
    Returns a tuple of all output variables matching the MATLAB signature.
    """

    # to do:
    # write subfunction for adjustment of costs if patient died, to be included
    # for colonoscopy and rectosigmoidoscopy

    # INITIALIZE
    # n is derived from the Gender array length, which was sized for the
    # requested Number_patients by calculate_sub. The original MATLAB code
    # hardcoded n=100000; here we support variable population sizes.
    n = len(Gender)
    Included = np.ones(n, dtype=bool)       # initially all patients are included
    Alive = np.ones(n, dtype=bool)          # initially all patients are alive
    DeathCause = np.zeros(n)
    DeathYear = np.zeros(n)
    NaturalDeathYear = np.zeros(n)

    DirectCancer = np.zeros((5, 100))
    DirectCancerR = np.zeros(100)
    DirectCancer2 = np.zeros(100)
    DirectCancer2R = np.zeros(100)
    ProgressedCancer = np.zeros(100)
    ProgressedCancerR = np.zeros(100)

    tr_cols = round(n / 10)
    TumorRecord_Stage = np.zeros((100, tr_cols))
    TumorRecord_Location = np.zeros((100, tr_cols))
    TumorRecord_Sojourn = np.zeros((100, tr_cols))
    TumorRecord_DwellTime = np.zeros((100, tr_cols))
    TumorRecord_Gender = np.zeros((100, tr_cols))
    TumorRecord_Detection = np.zeros((100, tr_cols))
    TumorRecord_PatientNumber = np.zeros((100, tr_cols))

    DwellTimeProgression = np.zeros((100, tr_cols))
    DwellTimeFastCancer = np.zeros((100, tr_cols))

    Last_Colonoscopy = np.zeros(n)
    Last_Polyp = np.ones(n) * -100
    Last_AdvPolyp = np.ones(n) * -100
    Last_Cancer = np.ones(n) * -100
    Last_ScreenTest = np.zeros(n)
    Last_Included = np.zeros(n)
    Last_TestDone = np.zeros(n)
    Last_TestYear = np.zeros(n)
    Last_TestYear2 = np.zeros(n)

    AusschlussPolyp = 0
    AusschlussCa = 0
    AusschlussKolo = 0
    PosPolyp = 0
    PosCa = 0
    PosPolypCa = 0

    Polyp_Polyps = np.zeros((n, 51))
    Polyp_PolypYear = np.zeros((n, 51))
    Polyp_PolypLocation = np.zeros((n, 51))
    Polyp_AdvProgression = np.zeros((n, 51))
    Polyp_EarlyProgression = np.zeros((n, 51))

    Ca_Cancer = np.zeros((n, 25))
    Ca_CancerYear = np.zeros((n, 25))
    Ca_CancerLocation = np.zeros((n, 25))
    Ca_TimeStage_I = np.zeros((n, 25))
    Ca_TimeStage_II = np.zeros((n, 25))
    Ca_TimeStage_III = np.zeros((n, 25))
    Ca_SympTime = np.zeros((n, 25))
    Ca_SympStage = np.zeros((n, 25))
    Ca_DwellTime = np.zeros((n, 25))

    Detected_Cancer = np.zeros((n, 50))
    Detected_CancerYear = np.zeros((n, 50))
    Detected_CancerLocation = np.zeros((n, 50))
    Detected_MortTime = np.zeros((n, 50))

    HasCancer = np.zeros((100, n))
    NumPolyps = np.zeros((100, n))
    MaxPolyps = np.zeros((100, n))
    AllPolyps = np.zeros((6, 100))

    DiagnosedCancer = np.zeros((100, n))
    NumCancer = np.zeros((100, n))
    MaxCancer = np.zeros((100, n))

    Money_AllCost = np.zeros(100)
    Money_AllCostFuture = np.zeros(100)
    Money_Treatment = np.zeros(100)
    Money_FutureTreatment = np.zeros(100)
    Money_Screening = np.zeros(100)
    Money_FollowUp = np.zeros(100)
    Money_Other = np.zeros(100)

    Number_Screening_Colonoscopy = np.zeros(100)
    Number_Symptoms_Colonoscopy = np.zeros(100)
    Number_Follow_Up_Colonoscopy = np.zeros(100)
    Number_Baseline_Colonoscopy = np.zeros(100)
    Number_RectoSigmo = np.zeros(100)
    Number_FOBT = np.zeros(100)
    Number_I_FOBT = np.zeros(100)
    Number_Sept9 = np.zeros(100)
    Number_other = np.zeros(100)

    EarlyPolypsRemoved = np.zeros(100)
    AdvancedPolypsRemoved = np.zeros(100)

    YearIncluded = np.zeros((100, n), dtype=bool)
    YearAlive = np.zeros((100, n), dtype=bool)

    # Payment types
    PaymentType_FOBT = np.zeros((1, 100))
    PaymentType_I_FOBT = np.zeros((1, 100))
    PaymentType_Sept9_HighSens = np.zeros((1, 100))
    PaymentType_Sept9_HighSpec = np.zeros((1, 100))
    PaymentType_RS = np.zeros((1, 100))
    PaymentType_RSPolyp = np.zeros((1, 100))
    PaymentType_Colonoscopy = np.zeros((4, 100))
    PaymentType_ColonoscopyPolyp = np.zeros((4, 100))
    PaymentType_Colonoscopy_Cancer = np.zeros((4, 100))
    PaymentType_Perforation = np.zeros((4, 100))
    PaymentType_Serosa = np.zeros((4, 100))
    PaymentType_Bleeding = np.zeros((4, 100))
    PaymentType_BleedingTransf = np.zeros((4, 100))
    PaymentType_Cancer_ini = np.zeros((4, 101))
    PaymentType_Cancer_con = np.zeros((4, 101))
    PaymentType_Cancer_fin = np.zeros((4, 101))
    PaymentType_QCancer_ini = np.zeros((4, 101, 4))
    PaymentType_QCancer_con = np.zeros((4, 101, 20))
    PaymentType_QCancer_fin = np.zeros((4, 101, 4))
    PaymentType_Other = np.zeros((1, 100))

    # matrix for fast indexing
    GenderProgression = np.ones((10, 2))
    # MATLAB: GenderProgression(1:4, 2) = female.early_progression_female
    GenderProgression[0:4, 1] = female['early_progression_female']
    # MATLAB: GenderProgression(5:6, 2) = female.advanced_progression_female
    GenderProgression[4:6, 1] = female['advanced_progression_female']

    # matrix for fast indexing
    LocationProgression = np.zeros((10, 13))
    # MATLAB rows 1-5 -> Python rows 0-4
    LocationProgression[0, :] = Location['EarlyProgression']
    LocationProgression[1, :] = Location['EarlyProgression']
    LocationProgression[2, :] = Location['EarlyProgression']
    LocationProgression[3, :] = Location['EarlyProgression']
    LocationProgression[4, :] = Location['EarlyProgression']
    # MATLAB row 6 -> Python row 5
    LocationProgression[5, :] = Location['AdvancedProgression']
    # MATLAB rows 7-10 -> Python rows 6-9
    LocationProgression[6, :] = Location['CancerProgression']
    LocationProgression[7, :] = Location['CancerProgression']
    LocationProgression[8, :] = Location['CancerProgression']
    LocationProgression[9, :] = Location['CancerProgression']

    # reach of rectosigmoidoscopy
    TmpLoc = np.zeros((13, 1000))
    for f in range(13):
        limit = int(round(1000 * Location['RectoSigmoReach'][f]))
        TmpLoc[f, 0:limit] = 1
    for f in range(12):
        TmpLoc[f + 1, :] = np.logical_or(TmpLoc[f + 1, :], TmpLoc[f, :])
    RectoSigmoReachMatrix = -np.sum(TmpLoc, axis=0) + 14

    # reach of colonoscopy
    TmpLoc = np.zeros((13, 1000))
    for f in range(13):
        limit = int(round(1000 * Location['ColoReach'][f]))
        TmpLoc[f, 0:limit] = 1
    for f in range(12):
        TmpLoc[f + 1, :] = np.logical_or(TmpLoc[f + 1, :], TmpLoc[f, :])
    ColoReachMatrix = -np.sum(TmpLoc, axis=0) + 14

    # LocationMatrix: Use the 2D input matrix (row 0: NewPolyp, row 1: DirectCa)
    # This preserves the correct distributions for both polyp and direct cancer locations
    LocationMatrix = LocationMatrix_in

    # Cancer progression stage matrix
    StageMatrix = np.zeros(1000)
    # MATLAB: StageMatrix(1:150) = 7; etc. (1-based)
    StageMatrix[0:150] = 7
    StageMatrix[150:506] = 8
    StageMatrix[506:785] = 9
    StageMatrix[785:1000] = 10

    # Sojourn time matrix
    tx2 = np.arange(0.25, 6.50, 0.25)  # 0.25 : 0.25 : 6.25  (25 elements)
    SojournMatrix = np.zeros((1000, 4))
    for f in range(4):
        tx3 = tx1[:, f].copy()
        tx3 = np.round(tx3 / np.sum(tx3) * 10 * 1000)
        Counter = 0
        for f2 in range(25):
            if int(round(tx3[f2] / 10)) != 0:
                end_idx = int(round(np.sum(tx3[0:f2 + 1]) / 10))
                SojournMatrix[Counter:end_idx, f] = tx2[f2]
                Counter = end_idx
                if f2 == 24 and Counter != 1000:
                    SojournMatrix[Counter:1000, f] = tx2[f2]

    CaSurv = np.zeros(4)
    CaDeath = np.zeros(4)

    # Make a mutable copy of ScreeningPreference
    ScreeningPreference = ScreeningPreference.copy()

    # ===================================================================
    #  MAIN SIMULATION LOOP
    # ===================================================================
    y = 0  # year counter; incremented at start of loop to become 1-based
    while np.sum(Included) > 0 and y < 100:
        y += 1
        yi = y - 1  # 0-based year index for arrays

        # for speed we make this calculation in advance
        PolypRate = np.ones(n)
        # the individual risk
        PolypRate = PolypRate * IndividualRisk
        # the age specific risk
        PolypRate = PolypRate * NewPolyp[yi]
        # the gender specific risk
        PolypRate[Gender == 2] = PolypRate[Gender == 2] * female['new_polyp_female']

        for z in range(n):  # z is 0-based (MATLAB z=1:n)
            for q in range(1, 5):  # q = 1,2,3,4
                time = y + (q - 1) / 4.0

                #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                #  people die of natural causes     %
                #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                if Alive[z]:
                    # divided by 4 since this is a quarterly calculation
                    # MATLAB: LifeTable(y, Gender(z))  -- y and Gender are 1-based
                    if np.random.rand() < (LifeTable[yi, int(Gender[z]) - 1] / 4.0):
                        Alive[z] = False
                        NaturalDeathYear[z] = time

                        # in these cases the patient was really alive
                        if Included[z]:
                            Included[z] = False
                            DeathCause[z] = 1
                            DeathYear[z] = time

                            # we need to calculate the costs
                            if np.sum(Detected_Cancer[z, :]) > 0:
                                AddCosts(Detected_Cancer, Detected_CancerYear,
                                         Detected_CancerLocation, Detected_MortTime,
                                         CostStage,
                                         PaymentType_Cancer_ini, PaymentType_Cancer_con,
                                         PaymentType_Cancer_fin,
                                         PaymentType_QCancer_ini, PaymentType_QCancer_con,
                                         PaymentType_QCancer_fin,
                                         Money_Treatment, Money_FutureTreatment,
                                         time, z, 'oc')

                #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                #    people die of cancer           %
                #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                if Included[z]:
                    first_det = np.flatnonzero(Detected_Cancer[z, :])
                    if len(first_det) > 0:
                        l = len(first_det)
                        for fi in range(l):
                            f = fi  # 0-based index into Detected arrays
                            if Detected_MortTime[z, f] < 21:
                                if (time - Detected_CancerYear[z, f]) >= Detected_MortTime[z, f] / 4.0:
                                    # patient died of cancer
                                    Included[z] = False
                                    DeathCause[z] = 2
                                    DeathYear[z] = time

                                    # we need to calculate the costs
                                    AddCosts(Detected_Cancer, Detected_CancerYear,
                                             Detected_CancerLocation, Detected_MortTime,
                                             CostStage,
                                             PaymentType_Cancer_ini, PaymentType_Cancer_con,
                                             PaymentType_Cancer_fin,
                                             PaymentType_QCancer_ini, PaymentType_QCancer_con,
                                             PaymentType_QCancer_fin,
                                             Money_Treatment, Money_FutureTreatment,
                                             time, z, 'tu')
                                    # MATLAB: CaDeath(Detected.Cancer(z,f)-6)
                                    CaDeath[int(Detected_Cancer[z, f]) - 7] += 1
                                    break  # we leave the loop
                            elif (time - Detected_CancerYear[z, f]) == 21.0 / 4:
                                CaSurv[int(Detected_Cancer[z, f]) - 7] += 1

                #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                # a NEW POLYP appears               %
                #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                if Included[z]:
                    # a new polyp appears
                    if np.random.rand() < PolypRate[z]:
                        if Polyp_Polyps[z, 0] > 0:
                            pos = _find_last_nonzero(Polyp_Polyps[z, :]) + 1
                        else:
                            pos = 0
                        if pos < 50:  # number polyps limited to 50
                            Polyp_Polyps[z, pos] = 1
                            Polyp_PolypYear[z, pos] = time
                            # MATLAB: LocationMatrix(1, round(rand*999)+1) -- row 1 in MATLAB = row 0 in Python
                            Polyp_PolypLocation[z, pos] = LocationMatrix[0, _rand_idx_1000()]

                            # we just save the percentile of the risk
                            Polyp_EarlyProgression[z, pos] = int(round(np.random.rand() * 499)) + 1

                            # if correlation applies, both percentiles are identical
                            if flag.get('Correlation', False):
                                Polyp_AdvProgression[z, pos] = Polyp_EarlyProgression[z, pos]
                            else:
                                Polyp_AdvProgression[z, pos] = int(round(np.random.rand() * 499)) + 1

                    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    # a NEW Cancer appears DIRECTLY     %
                    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    # MATLAB: DirectCancerRate(Gender(z), y)  -- both 1-based
                    if np.random.rand() < DirectCancerRate[int(Gender[z]) - 1, yi] * DirectCancerSpeed:
                        l2 = _count_nonzero(Ca_Cancer[z, :])
                        if l2 < 25:
                            Ca_Cancer[z, l2] = 7
                            Ca_CancerYear[z, l2] = time
                            # MATLAB: LocationMatrix(2, round(rand*999)+1) -- row 2 in MATLAB = row 1 in Python
                            Ca_CancerLocation[z, l2] = LocationMatrix[1, _rand_idx_1000()]
                            Ca_DwellTime[z, l2] = 0

                            # a random number for stage and sojourn time
                            tmp1 = int(StageMatrix[_rand_idx_1000()])
                            # MATLAB: SojournMatrix(round(rand*999+1), tmp1-6)
                            tmp2 = SojournMatrix[_rand_idx_1000(), tmp1 - 7]

                            Ca_SympTime[z, l2] = time + tmp2
                            Ca_SympStage[z, l2] = tmp1
                            if tmp1 > 7:
                                Ca_TimeStage_I[z, l2] = time + round(tmp2 * StageDuration[tmp1 - 7, 0] * 4) / 4.0
                            else:
                                Ca_TimeStage_I[z, l2] = 1000
                            if tmp1 > 8:
                                Ca_TimeStage_II[z, l2] = time + round(tmp2 * np.sum(StageDuration[tmp1 - 7, 0:2]) * 4) / 4.0
                            else:
                                Ca_TimeStage_II[z, l2] = 1000
                            if tmp1 > 9:
                                Ca_TimeStage_III[z, l2] = time + round(tmp2 * np.sum(StageDuration[tmp1 - 7, 0:3]) * 4) / 4.0
                            else:
                                Ca_TimeStage_III[z, l2] = 1000

                            # we keep track
                            dt_pos = _count_nonzero(DwellTimeProgression[yi, :])
                            DwellTimeProgression[yi, dt_pos] = 0
                            # MATLAB: HasCancer(y:100, z) = 1
                            HasCancer[yi:100, z] = 1
                            DirectCancer2[yi] += 1
                            if Ca_CancerLocation[z, l2] < 4:
                                DirectCancer2R[yi] += 1

                            if DEBUG_TRACE:
                                _trace_rows.append({
                                    'patient': z, 'year': y, 'quarter': q,
                                    'time': time, 'pathway': 'direct',
                                    'polyp_stage': 0,
                                    'polyp_year': 0,
                                    'dwell_time': 0,
                                    'location': int(Ca_CancerLocation[z, l2]),
                                    'gender': int(Gender[z]),
                                })

                    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    #      a polyp progresses           %
                    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    l_poly = _count_nonzero(Polyp_Polyps[z, :])
                    for f in range(l_poly - 1, -1, -1):
                        polyp_stage = int(Polyp_Polyps[z, f])
                        p_loc = int(Polyp_PolypLocation[z, f])
                        # MATLAB: AgeProgression(Polyp.Polyps(z, f), y) -- 1-based
                        # LocationProgression(Polyp.Polyps(z,f), Polyp.PolypLocation(z, f))
                        # GenderProgression(Polyp.Polyps(z, f), Gender(z))
                        risk_mult = ((polyp_stage < 5) * RiskDistribution['EarlyRisk'][int(Polyp_EarlyProgression[z, f]) - 1] +
                                     (polyp_stage > 4) * RiskDistribution['AdvancedRisk'][int(Polyp_AdvProgression[z, f]) - 1])
                        tmp = (AgeProgression[polyp_stage - 1, yi] *
                               LocationProgression[polyp_stage - 1, p_loc - 1] *
                               GenderProgression[polyp_stage - 1, int(Gender[z]) - 1] *
                               risk_mult)

                        if np.random.rand() < tmp:
                            Polyp_Polyps[z, f] += 1
                            if Polyp_Polyps[z, f] > 6:
                                # this is cancer now
                                l2 = _count_nonzero(Ca_Cancer[z, :])
                                Ca_Cancer[z, l2] = 7
                                Ca_CancerYear[z, l2] = time
                                Ca_CancerLocation[z, l2] = Polyp_PolypLocation[z, f]
                                Ca_DwellTime[z, l2] = time - Polyp_PolypYear[z, f]

                                tmp1 = int(StageMatrix[_rand_idx_1000()])
                                tmp2 = SojournMatrix[_rand_idx_1000(), tmp1 - 7]

                                Ca_SympTime[z, l2] = time + tmp2
                                Ca_SympStage[z, l2] = tmp1
                                if tmp1 > 7:
                                    Ca_TimeStage_I[z, l2] = time + round(tmp2 * StageDuration[tmp1 - 7, 0] * 4) / 4.0
                                else:
                                    Ca_TimeStage_I[z, l2] = 1000
                                if tmp1 > 8:
                                    Ca_TimeStage_II[z, l2] = time + round(tmp2 * np.sum(StageDuration[tmp1 - 7, 0:2]) * 4) / 4.0
                                else:
                                    Ca_TimeStage_II[z, l2] = 1000
                                if tmp1 > 9:
                                    Ca_TimeStage_III[z, l2] = time + round(tmp2 * np.sum(StageDuration[tmp1 - 7, 0:3]) * 4) / 4.0
                                else:
                                    Ca_TimeStage_III[z, l2] = 1000

                                dt_pos = _count_nonzero(DwellTimeProgression[yi, :])
                                DwellTimeProgression[yi, dt_pos] = time - Polyp_PolypYear[z, f]
                                HasCancer[yi:100, z] = 1
                                ProgressedCancer[yi] += 1
                                if Ca_CancerLocation[z, l2] < 4:
                                    ProgressedCancerR[yi] += 1

                                if DEBUG_TRACE:
                                    _trace_rows.append({
                                        'patient': z, 'year': y, 'quarter': q,
                                        'time': time, 'pathway': 'progressed',
                                        'polyp_stage': polyp_stage,
                                        'polyp_year': Polyp_PolypYear[z, f],
                                        'dwell_time': time - Polyp_PolypYear[z, f],
                                        'location': int(Ca_CancerLocation[z, l2]),
                                        'gender': int(Gender[z]),
                                    })

                                # delete the polyp
                                l_now = _count_nonzero(Polyp_Polyps[z, :])
                                _shift_left_polyp(Polyp_Polyps, Polyp_PolypYear,
                                                  Polyp_PolypLocation, Polyp_EarlyProgression,
                                                  Polyp_AdvProgression, z, f, l_now - 1)

                        elif np.random.rand() < (
                            (DwellSpeed == 'Slow') * (
                                StageVariables['FastCancer'][polyp_stage - 1] *
                                AgeProgression[5, yi] *
                                LocationProgression[5, p_loc - 1] *
                                GenderProgression[5, int(Gender[z]) - 1]
                            ) +
                            (DwellSpeed == 'Fast') * (
                                StageVariables['FastCancer'][polyp_stage - 1] *
                                AgeProgression[5, yi] *
                                LocationProgression[5, p_loc - 1] *
                                GenderProgression[5, int(Gender[z]) - 1]
                            ) * risk_mult
                        ):
                            # this is fast progressed cancer now
                            l2 = _count_nonzero(Ca_Cancer[z, :])
                            Ca_Cancer[z, l2] = 7
                            Ca_CancerYear[z, l2] = time
                            Ca_CancerLocation[z, l2] = Polyp_PolypLocation[z, f]
                            Ca_DwellTime[z, l2] = time - Polyp_PolypYear[z, f]

                            tmp1 = int(StageMatrix[_rand_idx_1000()])
                            tmp2 = SojournMatrix[_rand_idx_1000(), tmp1 - 7]

                            Ca_SympTime[z, l2] = time + tmp2
                            Ca_SympStage[z, l2] = tmp1
                            if tmp1 > 7:
                                Ca_TimeStage_I[z, l2] = time + round(tmp2 * StageDuration[tmp1 - 7, 0] * 4) / 4.0
                            else:
                                Ca_TimeStage_I[z, l2] = 1000
                            if tmp1 > 8:
                                Ca_TimeStage_II[z, l2] = time + round(tmp2 * np.sum(StageDuration[tmp1 - 7, 0:2]) * 4) / 4.0
                            else:
                                Ca_TimeStage_II[z, l2] = 1000
                            if tmp1 > 9:
                                Ca_TimeStage_III[z, l2] = time + round(tmp2 * np.sum(StageDuration[tmp1 - 7, 0:3]) * 4) / 4.0
                            else:
                                Ca_TimeStage_III[z, l2] = 1000

                            dt_pos = _count_nonzero(DwellTimeFastCancer[yi, :])
                            DwellTimeFastCancer[yi, dt_pos] = time - Polyp_PolypYear[z, f]
                            HasCancer[yi:100, z] = 1
                            # MATLAB: DirectCancer(Polyp.Polyps(z, f), y)
                            DirectCancer[polyp_stage - 1, yi] += 1
                            if Ca_CancerLocation[z, l2] < 4:
                                DirectCancerR[yi] += 1

                            if DEBUG_TRACE:
                                _trace_rows.append({
                                    'patient': z, 'year': y, 'quarter': q,
                                    'time': time, 'pathway': 'fast_cancer',
                                    'polyp_stage': polyp_stage,
                                    'polyp_year': Polyp_PolypYear[z, f],
                                    'dwell_time': time - Polyp_PolypYear[z, f],
                                    'location': int(Ca_CancerLocation[z, l2]),
                                    'gender': int(Gender[z]),
                                })

                            # delete the polyp
                            l_now = _count_nonzero(Polyp_Polyps[z, :])
                            _shift_left_polyp(Polyp_Polyps, Polyp_PolypYear,
                                              Polyp_PolypLocation, Polyp_EarlyProgression,
                                              Polyp_AdvProgression, z, f, l_now - 1)

                    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    #   a polyp shrinks or disappears      %
                    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    # MATLAB computes l once BEFORE the loop and reuses it for all
                    # shift operations:  Polyp.Polyps(z, f:l) = Polyp.Polyps(z, f+1:l+1)
                    # We must do the same — using _count_nonzero AFTER decrementing
                    # to 0 gives a value that is 1 too small, leaving stranded
                    # "ghost polyps" with PolypYear=0 and PolypLocation=0.
                    l_poly = _count_nonzero(Polyp_Polyps[z, :])  # recalculate
                    for f in range(l_poly - 1, -1, -1):
                        polyp_stage = int(Polyp_Polyps[z, f])
                        if np.random.rand() < StageVariables['Healing'][polyp_stage - 1]:
                            Polyp_Polyps[z, f] -= 1
                            if Polyp_Polyps[z, f] == 0:
                                # polyp disappears — shift using l_poly (the count
                                # before any deletions in this loop), matching MATLAB
                                _shift_left_polyp(Polyp_Polyps, Polyp_PolypYear,
                                                  Polyp_PolypLocation, Polyp_EarlyProgression,
                                                  Polyp_AdvProgression, z, f, l_poly - 1)
                                l_poly -= 1

                    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    # symptom development               %
                    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    l_ca = _count_nonzero(Ca_Cancer[z, :])
                    for f in range(l_ca - 1, -1, -1):
                        if time >= Ca_SympTime[z, f]:
                            # if symptoms appear we do colonoscopy
                            Number_Symptoms_Colonoscopy[yi] += 1
                            Colonoscopy(z, y, q, 'Symp', Gender,
                                        Polyp_Polyps, Polyp_PolypYear, Polyp_PolypLocation,
                                        Polyp_EarlyProgression, Polyp_AdvProgression,
                                        Ca_Cancer, Ca_CancerYear, Ca_CancerLocation,
                                        Ca_DwellTime, Ca_SympTime, Ca_SympStage,
                                        Ca_TimeStage_I, Ca_TimeStage_II, Ca_TimeStage_III,
                                        Detected_Cancer, Detected_CancerYear,
                                        Detected_CancerLocation, Detected_MortTime,
                                        Included, DeathCause, DeathYear,
                                        DiagnosedCancer, AdvancedPolypsRemoved, EarlyPolypsRemoved,
                                        Last_Colonoscopy, Last_Polyp, Last_AdvPolyp, Last_Cancer,
                                        TumorRecord_Stage, TumorRecord_Location, TumorRecord_Sojourn,
                                        TumorRecord_DwellTime, TumorRecord_Gender,
                                        TumorRecord_Detection, TumorRecord_PatientNumber,
                                        PaymentType_Colonoscopy, PaymentType_ColonoscopyPolyp,
                                        PaymentType_Colonoscopy_Cancer,
                                        PaymentType_Perforation, PaymentType_Serosa,
                                        PaymentType_Bleeding, PaymentType_BleedingTransf,
                                        PaymentType_Cancer_ini, PaymentType_Cancer_con,
                                        PaymentType_Cancer_fin,
                                        PaymentType_QCancer_ini, PaymentType_QCancer_con,
                                        PaymentType_QCancer_fin,
                                        Money_Screening, Money_Treatment, Money_FutureTreatment,
                                        Money_FollowUp, Money_Other,
                                        StageVariables, Cost, Location, risc,
                                        ColoReachMatrix, MortalityMatrix, CostStage)
                            break

                    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    # Cancer Progression                %
                    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    l_ca = _count_nonzero(Ca_Cancer[z, :])
                    for f in range(l_ca):
                        if Ca_Cancer[z, f] == 7:
                            if time >= Ca_TimeStage_I[z, f]:
                                Ca_Cancer[z, f] = 8
                        elif Ca_Cancer[z, f] == 8:
                            if time >= Ca_TimeStage_II[z, f]:
                                Ca_Cancer[z, f] = 9
                        elif Ca_Cancer[z, f] == 9:
                            if time >= Ca_TimeStage_III[z, f]:
                                Ca_Cancer[z, f] = 10

                    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    #    baseline colonoscopy           %
                    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    # (commented out in MATLAB source)

                    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    # polyp and cancer surveillance     %
                    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    if q == 1:
                        SurveillanceFlag = 0
                        if flag.get('Polyp_Surveillance', False):
                            if (y - Last_Polyp[z] == 5) and (y - Last_Colonoscopy[z] >= 5):
                                SurveillanceFlag = 1
                            elif (y - Last_Polyp[z] > 5) and (y - Last_Polyp[z] <= 9) and (y - Last_Colonoscopy[z] >= 5):
                                SurveillanceFlag = 1
                            elif (y - Last_AdvPolyp[z] == 3) and (y - Last_Colonoscopy[z] >= 3):
                                SurveillanceFlag = 1
                            elif Last_AdvPolyp[z] != -100:
                                if (y - Last_AdvPolyp[z] >= 5) and (y - Last_Colonoscopy[z] >= 5):
                                    SurveillanceFlag = 1
                            elif flag.get('AllPolypFollowUp', False):
                                if Last_Polyp[z] != -100:
                                    if (y - Last_Polyp[z] >= 5) and (y - Last_Colonoscopy[z] >= 5):
                                        SurveillanceFlag = 1

                        if flag.get('Cancer_Surveillance', False):
                            if Last_Cancer[z] != -100:
                                if (y - Last_Cancer[z] == 1) and (y - Last_Colonoscopy[z] == 1):
                                    SurveillanceFlag = 1
                                elif (y - Last_Cancer[z] == 4) and (y - Last_Colonoscopy[z] == 3):
                                    SurveillanceFlag = 1
                                elif (y - Last_Cancer[z] >= 5) and (y - Last_Colonoscopy[z] >= 5):
                                    SurveillanceFlag = 1

                        if SurveillanceFlag == 1:
                            Number_Follow_Up_Colonoscopy[yi] += 1
                            Colonoscopy(z, y, q, 'Foll', Gender,
                                        Polyp_Polyps, Polyp_PolypYear, Polyp_PolypLocation,
                                        Polyp_EarlyProgression, Polyp_AdvProgression,
                                        Ca_Cancer, Ca_CancerYear, Ca_CancerLocation,
                                        Ca_DwellTime, Ca_SympTime, Ca_SympStage,
                                        Ca_TimeStage_I, Ca_TimeStage_II, Ca_TimeStage_III,
                                        Detected_Cancer, Detected_CancerYear,
                                        Detected_CancerLocation, Detected_MortTime,
                                        Included, DeathCause, DeathYear,
                                        DiagnosedCancer, AdvancedPolypsRemoved, EarlyPolypsRemoved,
                                        Last_Colonoscopy, Last_Polyp, Last_AdvPolyp, Last_Cancer,
                                        TumorRecord_Stage, TumorRecord_Location, TumorRecord_Sojourn,
                                        TumorRecord_DwellTime, TumorRecord_Gender,
                                        TumorRecord_Detection, TumorRecord_PatientNumber,
                                        PaymentType_Colonoscopy, PaymentType_ColonoscopyPolyp,
                                        PaymentType_Colonoscopy_Cancer,
                                        PaymentType_Perforation, PaymentType_Serosa,
                                        PaymentType_Bleeding, PaymentType_BleedingTransf,
                                        PaymentType_Cancer_ini, PaymentType_Cancer_con,
                                        PaymentType_Cancer_fin,
                                        PaymentType_QCancer_ini, PaymentType_QCancer_con,
                                        PaymentType_QCancer_fin,
                                        Money_Screening, Money_Treatment, Money_FutureTreatment,
                                        Money_FollowUp, Money_Other,
                                        StageVariables, Cost, Location, risc,
                                        ColoReachMatrix, MortalityMatrix, CostStage)

                        # perhaps we do screening?
                        if flag.get('Screening', False):
                            # we only screen patients who are alive
                            if Included[z] and ScreeningPreference[z] != 0:
                                preference = int(ScreeningPreference[z])  # 1-based
                                pi = preference - 1  # 0-based for ScreeningTest rows
                                # MATLAB ScreeningTest columns (1-based):
                                # 1:PercentPop, 2:Adherence, 3:FollowUp, 4:y-start, 5:y-end,
                                # 6:interval, 7:y after colo, 8:specificity
                                if y >= ScreeningTest[pi, 3] and y < ScreeningTest[pi, 4]:
                                    if y - Last_Colonoscopy[z] >= ScreeningTest[pi, 6]:

                                        if preference == 1:  # Colonoscopy
                                            if y - Last_Colonoscopy[z] >= ScreeningTest[pi, 5]:
                                                Number_Screening_Colonoscopy[yi] += 1
                                                Colonoscopy(z, y, q, 'Scre', Gender,
                                                    Polyp_Polyps, Polyp_PolypYear, Polyp_PolypLocation,
                                                    Polyp_EarlyProgression, Polyp_AdvProgression,
                                                    Ca_Cancer, Ca_CancerYear, Ca_CancerLocation,
                                                    Ca_DwellTime, Ca_SympTime, Ca_SympStage,
                                                    Ca_TimeStage_I, Ca_TimeStage_II, Ca_TimeStage_III,
                                                    Detected_Cancer, Detected_CancerYear,
                                                    Detected_CancerLocation, Detected_MortTime,
                                                    Included, DeathCause, DeathYear,
                                                    DiagnosedCancer, AdvancedPolypsRemoved, EarlyPolypsRemoved,
                                                    Last_Colonoscopy, Last_Polyp, Last_AdvPolyp, Last_Cancer,
                                                    TumorRecord_Stage, TumorRecord_Location, TumorRecord_Sojourn,
                                                    TumorRecord_DwellTime, TumorRecord_Gender,
                                                    TumorRecord_Detection, TumorRecord_PatientNumber,
                                                    PaymentType_Colonoscopy, PaymentType_ColonoscopyPolyp,
                                                    PaymentType_Colonoscopy_Cancer,
                                                    PaymentType_Perforation, PaymentType_Serosa,
                                                    PaymentType_Bleeding, PaymentType_BleedingTransf,
                                                    PaymentType_Cancer_ini, PaymentType_Cancer_con,
                                                    PaymentType_Cancer_fin,
                                                    PaymentType_QCancer_ini, PaymentType_QCancer_con,
                                                    PaymentType_QCancer_fin,
                                                    Money_Screening, Money_Treatment, Money_FutureTreatment,
                                                    Money_FollowUp, Money_Other,
                                                    StageVariables, Cost, Location, risc,
                                                    ColoReachMatrix, MortalityMatrix, CostStage)

                                        elif preference == 2:  # Rectosigmoidoscopy
                                            if y - Last_ScreenTest[z] >= ScreeningTest[pi, 5]:
                                                if np.random.rand() < ScreeningTest[pi, 1]:
                                                    Number_RectoSigmo[yi] += 1
                                                    Last_ScreenTest[z] = y
                                                    PolypFlag, AdvPolypFlag, CancerFlag = RectoSigmo(
                                                        z, y, Polyp_Polyps, Polyp_PolypYear, Polyp_PolypLocation,
                                                        Polyp_EarlyProgression, Polyp_AdvProgression,
                                                        Ca_Cancer, Ca_CancerLocation,
                                                        Included, DeathCause, DeathYear,
                                                        PaymentType_RS, PaymentType_RSPolyp,
                                                        PaymentType_Perforation, Money_Screening,
                                                        StageVariables, Cost, Location, risc,
                                                        RectoSigmoReachMatrix, flag)
                                                    if PolypFlag or CancerFlag or AdvPolypFlag:
                                                        if np.random.rand() < ScreeningTest[pi, 2]:
                                                            Number_Screening_Colonoscopy[yi] += 1
                                                            ScreeningPreference[z] = 1
                                                            Colonoscopy(z, y, q, 'Scre', Gender,
                                                                Polyp_Polyps, Polyp_PolypYear, Polyp_PolypLocation,
                                                                Polyp_EarlyProgression, Polyp_AdvProgression,
                                                                Ca_Cancer, Ca_CancerYear, Ca_CancerLocation,
                                                                Ca_DwellTime, Ca_SympTime, Ca_SympStage,
                                                                Ca_TimeStage_I, Ca_TimeStage_II, Ca_TimeStage_III,
                                                                Detected_Cancer, Detected_CancerYear,
                                                                Detected_CancerLocation, Detected_MortTime,
                                                                Included, DeathCause, DeathYear,
                                                                DiagnosedCancer, AdvancedPolypsRemoved, EarlyPolypsRemoved,
                                                                Last_Colonoscopy, Last_Polyp, Last_AdvPolyp, Last_Cancer,
                                                                TumorRecord_Stage, TumorRecord_Location, TumorRecord_Sojourn,
                                                                TumorRecord_DwellTime, TumorRecord_Gender,
                                                                TumorRecord_Detection, TumorRecord_PatientNumber,
                                                                PaymentType_Colonoscopy, PaymentType_ColonoscopyPolyp,
                                                                PaymentType_Colonoscopy_Cancer,
                                                                PaymentType_Perforation, PaymentType_Serosa,
                                                                PaymentType_Bleeding, PaymentType_BleedingTransf,
                                                                PaymentType_Cancer_ini, PaymentType_Cancer_con,
                                                                PaymentType_Cancer_fin,
                                                                PaymentType_QCancer_ini, PaymentType_QCancer_con,
                                                                PaymentType_QCancer_fin,
                                                                Money_Screening, Money_Treatment, Money_FutureTreatment,
                                                                Money_FollowUp, Money_Other,
                                                                StageVariables, Cost, Location, risc,
                                                                ColoReachMatrix, MortalityMatrix, CostStage)

                                        else:  # other test (FOBT, I_FOBT, Sept9, etc.)
                                            if y - Last_ScreenTest[z] >= ScreeningTest[pi, 5]:
                                                if np.random.rand() < ScreeningTest[pi, 1]:
                                                    Last_ScreenTest[z] = y
                                                    Limit = 0
                                                    last_polyp_idx = _find_last_nonzero(Polyp_Polyps[z, :])
                                                    if last_polyp_idx >= 0:
                                                        # MATLAB: Sensitivity(preference, max(Polyp.Polyps(z,:)))
                                                        max_p = int(np.max(Polyp_Polyps[z, :]))
                                                        Limit = Sensitivity[pi, max_p - 1]
                                                    last_ca_idx = _find_last_nonzero(Ca_Cancer[z, :])
                                                    if last_ca_idx >= 0:
                                                        max_c = int(np.max(Ca_Cancer[z, :]))
                                                        Limit = Sensitivity[pi, max_c - 1]
                                                    Limit = max(Limit, 1 - ScreeningTest[pi, 7])
                                                    if np.random.rand() < Limit:
                                                        if np.random.rand() < ScreeningTest[pi, 2]:
                                                            Number_Screening_Colonoscopy[yi] += 1
                                                            ScreeningPreference[z] = 1
                                                            Colonoscopy(z, y, q, 'Scre', Gender,
                                                                Polyp_Polyps, Polyp_PolypYear, Polyp_PolypLocation,
                                                                Polyp_EarlyProgression, Polyp_AdvProgression,
                                                                Ca_Cancer, Ca_CancerYear, Ca_CancerLocation,
                                                                Ca_DwellTime, Ca_SympTime, Ca_SympStage,
                                                                Ca_TimeStage_I, Ca_TimeStage_II, Ca_TimeStage_III,
                                                                Detected_Cancer, Detected_CancerYear,
                                                                Detected_CancerLocation, Detected_MortTime,
                                                                Included, DeathCause, DeathYear,
                                                                DiagnosedCancer, AdvancedPolypsRemoved, EarlyPolypsRemoved,
                                                                Last_Colonoscopy, Last_Polyp, Last_AdvPolyp, Last_Cancer,
                                                                TumorRecord_Stage, TumorRecord_Location, TumorRecord_Sojourn,
                                                                TumorRecord_DwellTime, TumorRecord_Gender,
                                                                TumorRecord_Detection, TumorRecord_PatientNumber,
                                                                PaymentType_Colonoscopy, PaymentType_ColonoscopyPolyp,
                                                                PaymentType_Colonoscopy_Cancer,
                                                                PaymentType_Perforation, PaymentType_Serosa,
                                                                PaymentType_Bleeding, PaymentType_BleedingTransf,
                                                                PaymentType_Cancer_ini, PaymentType_Cancer_con,
                                                                PaymentType_Cancer_fin,
                                                                PaymentType_QCancer_ini, PaymentType_QCancer_con,
                                                                PaymentType_QCancer_fin,
                                                                Money_Screening, Money_Treatment, Money_FutureTreatment,
                                                                Money_FollowUp, Money_Other,
                                                                StageVariables, Cost, Location, risc,
                                                                ColoReachMatrix, MortalityMatrix, CostStage)
                                                    # cost accounting for the screening test itself
                                                    if preference == 3:
                                                        Number_FOBT[yi] += 1
                                                        Money_Screening[yi] += Cost['FOBT']
                                                        PaymentType_FOBT[0, yi] += 1
                                                    elif preference == 4:
                                                        Number_I_FOBT[yi] += 1
                                                        Money_Screening[yi] += Cost['I_FOBT']
                                                        PaymentType_I_FOBT[0, yi] += 1
                                                    elif preference == 5:
                                                        Number_Sept9[yi] += 1
                                                        Money_Screening[yi] += Cost['Sept9_HighSens']
                                                        PaymentType_Sept9_HighSens[0, yi] += 1
                                                    elif preference == 6:
                                                        Number_Sept9[yi] += 1
                                                        Money_Screening[yi] += Cost['Sept9_HighSpec']
                                                        PaymentType_Sept9_HighSpec[0, yi] += 1
                                                    elif preference == 7:
                                                        Number_other[yi] += 1
                                                        Money_Screening[yi] += Cost['other']
                                                        PaymentType_Other[0, yi] += 1

                        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        #    special scenarios              %
                        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        # Helper: define a local function to call Colonoscopy with all args
                        def _do_colonoscopy(modus):
                            Colonoscopy(z, y, q, modus, Gender,
                                Polyp_Polyps, Polyp_PolypYear, Polyp_PolypLocation,
                                Polyp_EarlyProgression, Polyp_AdvProgression,
                                Ca_Cancer, Ca_CancerYear, Ca_CancerLocation,
                                Ca_DwellTime, Ca_SympTime, Ca_SympStage,
                                Ca_TimeStage_I, Ca_TimeStage_II, Ca_TimeStage_III,
                                Detected_Cancer, Detected_CancerYear,
                                Detected_CancerLocation, Detected_MortTime,
                                Included, DeathCause, DeathYear,
                                DiagnosedCancer, AdvancedPolypsRemoved, EarlyPolypsRemoved,
                                Last_Colonoscopy, Last_Polyp, Last_AdvPolyp, Last_Cancer,
                                TumorRecord_Stage, TumorRecord_Location, TumorRecord_Sojourn,
                                TumorRecord_DwellTime, TumorRecord_Gender,
                                TumorRecord_Detection, TumorRecord_PatientNumber,
                                PaymentType_Colonoscopy, PaymentType_ColonoscopyPolyp,
                                PaymentType_Colonoscopy_Cancer,
                                PaymentType_Perforation, PaymentType_Serosa,
                                PaymentType_Bleeding, PaymentType_BleedingTransf,
                                PaymentType_Cancer_ini, PaymentType_Cancer_con,
                                PaymentType_Cancer_fin,
                                PaymentType_QCancer_ini, PaymentType_QCancer_con,
                                PaymentType_QCancer_fin,
                                Money_Screening, Money_Treatment, Money_FutureTreatment,
                                Money_FollowUp, Money_Other,
                                StageVariables, Cost, Location, risc,
                                ColoReachMatrix, MortalityMatrix, CostStage)

                        def _do_recto_sigmo():
                            return RectoSigmo(
                                z, y, Polyp_Polyps, Polyp_PolypYear, Polyp_PolypLocation,
                                Polyp_EarlyProgression, Polyp_AdvProgression,
                                Ca_Cancer, Ca_CancerLocation,
                                Included, DeathCause, DeathYear,
                                PaymentType_RS, PaymentType_RSPolyp,
                                PaymentType_Perforation, Money_Screening,
                                StageVariables, Cost, Location, risc,
                                RectoSigmoReachMatrix, flag)

                        if flag.get('SpecialFlag', False) and q == 1:
                            if flag.get('Atkin', False):
                                if y == 1:
                                    if z == 0:  # MATLAB z==1, first patient
                                        # randomly one test year between 55 and 64
                                        tmp_years = np.arange(55, 65)
                                        for ff in range(n):
                                            Last_Included[ff] = tmp_years[int(round(np.random.rand() * (len(tmp_years) - 1)))]
                                            if np.random.rand() < 0.71:
                                                Last_TestYear[ff] = Last_Included[ff]
                                else:
                                    if Last_Included[z] == y:
                                        StudyFlag = True
                                        StudyFlag = StudyFlag and (y - Last_Colonoscopy[z] > 3)
                                        StudyFlag = StudyFlag and (Last_Cancer[z] == -100)
                                        StudyFlag = StudyFlag and (Last_Polyp[z] == -100)
                                        StudyFlag = StudyFlag and (Last_AdvPolyp[z] == -100)
                                        if not StudyFlag:
                                            Last_TestYear[z] = -1
                                            Last_TestYear2[z] = -1
                                            Last_Included[z] = -1

                                    if Last_TestYear[z] == y:
                                        Last_TestDone[z] = 1
                                        if not flag.get('Mock', False):
                                            Number_RectoSigmo[yi] += 1
                                            PolypFlag, AdvPolypFlag, CancerFlag = _do_recto_sigmo()
                                            if AdvPolypFlag:
                                                Last_AdvPolyp[z] = y
                                            elif PolypFlag:
                                                Last_Polyp[z] = y
                                            if AdvPolypFlag or CancerFlag:
                                                if PolypFlag:
                                                    PosPolyp += 1
                                                if CancerFlag:
                                                    PosCa += 1
                                                if CancerFlag and PolypFlag:
                                                    PosPolypCa += 1
                                                Number_Screening_Colonoscopy[yi] += 1
                                                _do_colonoscopy('Scre')

                            elif flag.get('Schoen', False):
                                if y == 1:
                                    if z == 0:
                                        tmp_years = np.arange(55, 75)
                                        for ff in range(n):
                                            Last_Included[ff] = tmp_years[int(round(np.random.rand() * (len(tmp_years) - 1)))]
                                            if np.random.rand() < 0.83:
                                                Last_TestYear[ff] = Last_Included[ff]
                                                if np.random.rand() < 0.65:
                                                    if np.random.rand() < 0.25:
                                                        Last_TestYear2[ff] = Last_Included[ff] + 3
                                                    else:
                                                        Last_TestYear2[ff] = Last_Included[ff] + 5
                                                elif np.random.rand() < 0.035:
                                                    if np.random.rand() < 0.25:
                                                        Last_TestYear2[ff] = Last_Included[ff] + 3
                                                    else:
                                                        Last_TestYear2[ff] = Last_Included[ff] + 5
                                else:
                                    if Last_Included[z] == y:
                                        StudyFlag = True
                                        StudyFlag = StudyFlag and (y - Last_Colonoscopy[z] > 3)
                                        if y - Last_Colonoscopy[z] <= 3:
                                            AusschlussKolo += 1
                                        StudyFlag = StudyFlag and (Last_Cancer[z] == -100)
                                        if Last_Cancer[z] != -100:
                                            AusschlussCa += 1
                                        StudyFlag = StudyFlag and (Last_Polyp[z] == -100)
                                        if Last_Polyp[z] != -100 or Last_AdvPolyp[z] != -100:
                                            AusschlussPolyp += 1
                                        StudyFlag = StudyFlag and (Last_AdvPolyp[z] == -100)
                                        if not StudyFlag:
                                            Last_TestYear[z] = -1
                                            Last_TestYear2[z] = -1
                                            Last_Included[z] = -1

                                    if Last_TestYear[z] == y or Last_TestYear2[z] == y:
                                        if q == 1:
                                            Last_TestDone[z] = 1
                                            if not flag.get('Mock', False):
                                                Number_RectoSigmo[yi] += 1
                                                PolypFlag, AdvPolypFlag, CancerFlag = _do_recto_sigmo()
                                                if AdvPolypFlag:
                                                    Last_AdvPolyp[z] = y
                                                elif PolypFlag > 0:
                                                    Last_Polyp[z] = y
                                                if PolypFlag > 1 or AdvPolypFlag or CancerFlag:
                                                    if PolypFlag:
                                                        PosPolyp += 1
                                                    if CancerFlag:
                                                        PosCa += 1
                                                    if CancerFlag and PolypFlag:
                                                        PosPolypCa += 1
                                                    Number_Screening_Colonoscopy[yi] += 1
                                                    _do_colonoscopy('Scre')

                            elif flag.get('Segnan', False):
                                if y == 1:
                                    if z == 0:
                                        tmp_years = np.arange(55, 65)
                                        for ff in range(n):
                                            Last_Included[ff] = tmp_years[int(round(np.random.rand() * (len(tmp_years) - 1)))]
                                            if np.random.rand() < 0.583:
                                                Last_TestYear[ff] = Last_Included[ff]
                                else:
                                    if Last_Included[z] == y:
                                        StudyFlag = True
                                        StudyFlag = StudyFlag and (y - Last_Colonoscopy[z] > 2)
                                        StudyFlag = StudyFlag and (Last_Cancer[z] == -100)
                                        if not StudyFlag:
                                            Last_TestYear[z] = -1
                                            Last_TestYear2[z] = -1
                                            Last_Included[z] = -1

                                    if Last_TestYear[z] == y:
                                        if q == 1:
                                            Last_TestDone[z] = 1
                                            if not flag.get('Mock', False):
                                                Number_RectoSigmo[yi] += 1
                                                PolypFlag, AdvPolypFlag, CancerFlag = _do_recto_sigmo()
                                                if AdvPolypFlag:
                                                    Last_AdvPolyp[z] = y
                                                elif PolypFlag > 0:
                                                    Last_Polyp[z] = y
                                                if PolypFlag > 1 or AdvPolypFlag or CancerFlag:
                                                    if PolypFlag:
                                                        PosPolyp += 1
                                                    if CancerFlag:
                                                        PosCa += 1
                                                    if CancerFlag and PolypFlag:
                                                        PosPolypCa += 1
                                                    Number_Screening_Colonoscopy[yi] += 1
                                                    _do_colonoscopy('Scre')

                            elif flag.get('Holme', False):
                                if y == 1:
                                    if z == 0:
                                        tmp_years = np.arange(51, 66)
                                        for ff in range(n):
                                            Last_Included[ff] = tmp_years[int(round(np.random.rand() * (len(tmp_years) - 1)))]
                                            if np.random.rand() < 0.651:
                                                Last_TestYear[ff] = Last_Included[ff]
                                else:
                                    if Last_TestYear[z] == y:
                                        if Last_TestDone[z] != 1:
                                            StudyFlag = True
                                            StudyFlag = StudyFlag and (Last_Cancer[z] == -100)
                                            if Last_Cancer[z] != -100:
                                                AusschlussCa += 1
                                            if StudyFlag:
                                                Last_TestDone[z] = 1
                                                if not flag.get('Mock', False):
                                                    Number_RectoSigmo[yi] += 1
                                                    PolypFlag, AdvPolypFlag, CancerFlag = _do_recto_sigmo()
                                                    if AdvPolypFlag:
                                                        Last_AdvPolyp[z] = y
                                                    elif PolypFlag:
                                                        Last_Polyp[z] = y
                                                    if PolypFlag or AdvPolypFlag or CancerFlag:
                                                        if PolypFlag:
                                                            PosPolyp += 1
                                                        if CancerFlag:
                                                            PosCa += 1
                                                        if CancerFlag and PolypFlag:
                                                            PosPolypCa += 1
                                                        Number_Screening_Colonoscopy[yi] += 1
                                                        _do_colonoscopy('Scre')

                            # flag.perfect
                            if flag.get('perfect', False):
                                PerfectYear = 66
                                if y == PerfectYear:
                                    Polyp_Polyps[:, :] = 0
                                    Polyp_PolypYear[:, :] = 0
                                    Polyp_PolypLocation[:, :] = 0
                                    Polyp_AdvProgression[:, :] = 0
                                    Polyp_EarlyProgression[:, :] = 0
                                    Ca_Cancer[:, :] = 0
                                    Ca_CancerYear[:, :] = 0
                                    Ca_CancerLocation[:, :] = 0
                                    Ca_TimeStage_I[:, :] = 0
                                    Ca_TimeStage_II[:, :] = 0
                                    Ca_TimeStage_III[:, :] = 0
                                    Ca_SympTime[:, :] = 0
                                    Ca_SympStage[:, :] = 0
                                    Ca_DwellTime[:, :] = 0
                                    Detected_Cancer[:, :] = 0
                                    Detected_CancerYear[:, :] = 0
                                    Detected_CancerLocation[:, :] = 0
                                    Detected_MortTime[:, :] = 0
                                    TumorRecord_Stage[:, :] = 0
                                    TumorRecord_Location[:, :] = 0
                                    TumorRecord_Sojourn[:, :] = 0
                                    TumorRecord_DwellTime[:, :] = 0
                                    TumorRecord_Gender[:, :] = 0
                                    TumorRecord_Detection[:, :] = 0
                                    TumorRecord_PatientNumber[:, :] = 0

                            elif flag.get('Kolo1', False):
                                if ScreeningTest[0, 3] == y:
                                    Number_Screening_Colonoscopy[yi] += 1
                                    _do_colonoscopy('Scre')

                            elif flag.get('Kolo2', False):
                                if ScreeningTest[0, 3] == y:
                                    Number_Screening_Colonoscopy[yi] += 1
                                    _do_colonoscopy('Scre')
                                if ScreeningTest[0, 4] == y:
                                    Number_Screening_Colonoscopy[yi] += 1
                                    _do_colonoscopy('Scre')

                            elif flag.get('Kolo3', False):
                                if ScreeningTest[0, 3] == y:
                                    Number_Screening_Colonoscopy[yi] += 1
                                    _do_colonoscopy('Scre')
                                if ScreeningTest[0, 4] == y:
                                    Number_Screening_Colonoscopy[yi] += 1
                                    _do_colonoscopy('Scre')
                                if ScreeningTest[0, 5] == y:
                                    Number_Screening_Colonoscopy[yi] += 1
                                    _do_colonoscopy('Scre')

                            elif flag.get('Po55', False):
                                if y == 56:
                                    if (np.max(Polyp_Polyps[z, :]) > 0 or
                                            np.max(Ca_Cancer[z, :]) > 0 or
                                            Last_Polyp[z] > -100 or Last_Cancer[z] > -100):
                                        Last_TestDone[z] = 1
                                    else:
                                        Last_TestDone[z] = 2
                                    if flag.get('treated', False):
                                        Polyp_Polyps[z, :] = 0
                                        Polyp_PolypYear[z, :] = 0
                                        Polyp_PolypLocation[z, :] = 0
                                        Polyp_AdvProgression[z, :] = 0
                                        Polyp_EarlyProgression[z, :] = 0
                                        Ca_Cancer[z, :] = 0
                                        Ca_CancerYear[z, :] = 0
                                        Ca_CancerLocation[z, :] = 0
                                        Ca_TimeStage_I[z, :] = 0
                                        Ca_TimeStage_II[z, :] = 0
                                        Ca_TimeStage_III[z, :] = 0
                                        Ca_SympTime[z, :] = 0
                                        Ca_SympStage[z, :] = 0
                                        Detected_Cancer[z, :] = 0
                                        Detected_CancerYear[z, :] = 0
                                        Detected_CancerLocation[z, :] = 0
                                        Detected_MortTime[z, :] = 0

                        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        #    summarizing polyps             %
                        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        if _find_last_nonzero(Polyp_Polyps[z, :]) >= 0:
                            MaxPolyps[yi, z] = np.max(Polyp_Polyps[z, :])
                            NumPolyps[yi, z] = _count_nonzero(Polyp_Polyps[z, :])
                            for ff in range(1, 7):  # 1..6
                                AllPolyps[ff - 1, yi] += np.sum(Polyp_Polyps[z, :] == ff)

                #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                #    summarizing cancer             %
                #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # This is outside the q loop but inside z loop
            if _find_last_nonzero(Ca_Cancer[z, :]) >= 0:
                MaxCancer[yi, z] = np.max(Ca_Cancer[z, :])
                NumCancer[yi, z] = _count_nonzero(Ca_Cancer[z, :])

        # we summarize the whole cohort
        YearIncluded[yi, :] = Included
        YearAlive[yi, :] = Alive

        print('Calculating year {}'.format(y))

    # Post-simulation
    for f in range(n):
        if Alive[f]:
            NaturalDeathYear[f] = 100

    Money_AllCost = Money_Treatment + Money_Screening + Money_FollowUp + Money_Other
    Money_AllCostFuture = Money_FutureTreatment + Money_Screening + Money_FollowUp + Money_Other

    # Pack TumorRecord as a dict
    TumorRecord = {
        'Stage': TumorRecord_Stage,
        'Location': TumorRecord_Location,
        'Sojourn': TumorRecord_Sojourn,
        'DwellTime': TumorRecord_DwellTime,
        'Gender': TumorRecord_Gender,
        'Detection': TumorRecord_Detection,
        'PatientNumber': TumorRecord_PatientNumber,
    }

    # Pack Last as a dict
    Last = {
        'Colonoscopy': Last_Colonoscopy,
        'Polyp': Last_Polyp,
        'AdvPolyp': Last_AdvPolyp,
        'Cancer': Last_Cancer,
        'ScreenTest': Last_ScreenTest,
        'Included': Last_Included,
        'TestDone': Last_TestDone,
        'TestYear': Last_TestYear,
        'TestYear2': Last_TestYear2,
    }

    # Pack PaymentType as a dict
    PaymentType = {
        'FOBT': PaymentType_FOBT,
        'I_FOBT': PaymentType_I_FOBT,
        'Sept9_HighSens': PaymentType_Sept9_HighSens,
        'Sept9_HighSpec': PaymentType_Sept9_HighSpec,
        'RS': PaymentType_RS,
        'RSPolyp': PaymentType_RSPolyp,
        'Colonoscopy': PaymentType_Colonoscopy,
        'ColonoscopyPolyp': PaymentType_ColonoscopyPolyp,
        'Colonoscopy_Cancer': PaymentType_Colonoscopy_Cancer,
        'Perforation': PaymentType_Perforation,
        'Serosa': PaymentType_Serosa,
        'Bleeding': PaymentType_Bleeding,
        'BleedingTransf': PaymentType_BleedingTransf,
        'Cancer_ini': PaymentType_Cancer_ini,
        'Cancer_con': PaymentType_Cancer_con,
        'Cancer_fin': PaymentType_Cancer_fin,
        'QCancer_ini': PaymentType_QCancer_ini,
        'QCancer_con': PaymentType_QCancer_con,
        'QCancer_fin': PaymentType_QCancer_fin,
        'Other': PaymentType_Other,
    }

    # Pack Money as a dict
    Money = {
        'AllCost': Money_AllCost,
        'AllCostFuture': Money_AllCostFuture,
        'Treatment': Money_Treatment,
        'FutureTreatment': Money_FutureTreatment,
        'Screening': Money_Screening,
        'FollowUp': Money_FollowUp,
        'Other': Money_Other,
    }

    # Pack Number as a dict
    Number = {
        'Screening_Colonoscopy': Number_Screening_Colonoscopy,
        'Symptoms_Colonoscopy': Number_Symptoms_Colonoscopy,
        'Follow_Up_Colonoscopy': Number_Follow_Up_Colonoscopy,
        'Baseline_Colonoscopy': Number_Baseline_Colonoscopy,
        'RectoSigmo': Number_RectoSigmo,
        'FOBT': Number_FOBT,
        'I_FOBT': Number_I_FOBT,
        'Sept9': Number_Sept9,
        'other': Number_other,
    }

    # Write debug trace CSV if enabled
    if DEBUG_TRACE and _trace_rows:
        import csv
        trace_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'debug_cancer_trace.csv')
        fieldnames = ['patient', 'year', 'quarter', 'time', 'pathway',
                       'polyp_stage', 'polyp_year', 'dwell_time',
                       'location', 'gender']
        with open(trace_path, 'w', newline='') as csvf:
            writer = csv.DictWriter(csvf, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(_trace_rows)
        print(f"DEBUG_TRACE: wrote {len(_trace_rows)} cancer events to {trace_path}")

    return (y, Gender, DeathCause, Last, DeathYear, NaturalDeathYear,
            DirectCancer, DirectCancerR, DirectCancer2, DirectCancer2R,
            ProgressedCancer, ProgressedCancerR, TumorRecord,
            DwellTimeProgression, DwellTimeFastCancer,
            HasCancer, NumPolyps, MaxPolyps, AllPolyps, NumCancer, MaxCancer,
            PaymentType, Money, Number,
            EarlyPolypsRemoved, DiagnosedCancer, AdvancedPolypsRemoved,
            YearIncluded, YearAlive)
