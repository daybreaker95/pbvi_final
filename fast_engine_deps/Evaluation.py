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
import os
import warnings

# ---------------------------------------------------------------------------
# INDEX CONVENTION NOTES:
#
# In the original MATLAB code, array indices are 1-based. In this Python
# translation, all array indices are 0-based. The MATLAB variable y (number
# of years, typically 100) is kept as a count. Loops that ran 1:y in MATLAB
# now run range(y) with 0-based indexing.
#
# MATLAB's  data.TumorRecord.Stage(f, :)  with f in 1:100 becomes
#           data['TumorRecord']['Stage'][f, :]  with f in 0:99.
#
# Cancer stages remain semantic values 7-10.  Locations remain 1-13.
# Gender remains 1=male, 2=female.
#
# The "year adapted" comments in the original MATLAB code refer to
# transformations from 1-based life-years to 0-based age. In the MATLAB
# Evaluation, the comment notes that PRECEDING scripts used 1-100
# (Lebensjahre) and EVALUATION transforms to age 0-99. Since our arrays
# are already 0-based, the slicing is adjusted accordingly.
#
# MATLAB's histc is replaced by np.histogram with appropriate edges.
# MATLAB's quantile(x, p) is replaced by np.quantile(x, p).
# MATLAB's cell arrays become Python lists.
# MATLAB's structs become Python dicts.
# ---------------------------------------------------------------------------


def cumulativeDiscYears(y1, y2, DisCountMask):
    """
    Sub-function: calculate cumulative discounted years.
    MATLAB signature: y = cumulativeDiscYears(y1, y2, DisCountMask)

    y1 and y2 are 0-based indices into DisCountMask (length 101).
    """
    temp = 0.0
    for i in range(y1, y2 + 1):
        if i < len(DisCountMask):
            temp += DisCountMask[i]
    return temp


def CalculateAgreement(DataGraph, bmc, BM, Benchmarks, Struct1, Struct2, Struct3,
                       DispFlag, SubPlotPos, GraphDescription, GraphTitle,
                       tolerance, LineSz, MarkerSz, FontSz, LabelY, Flag):
    """
    Sub-function to calculate agreement between simulation and benchmarks.

    MATLAB signature:
    [BM, bmc, OutputFlags, OutputValues] = CalculateAgreement(DataGraph, bmc, BM,
        Benchmarks, Struct1, Struct2, Struct3, DispFlag, SubPlotPos,
        GraphDescription, GraphTitle, tolerance, LineSz, MarkerSz, FontSz,
        LabelY, Flag)

    Parameters:
        DataGraph   : 1-D array of simulation results
        bmc         : benchmark counter (0-based)
        BM          : benchmark dictionary
        Benchmarks  : dictionary containing benchmark data
        Struct1     : first key into Benchmarks (e.g., 'EarlyPolyp', 'Cancer')
        Struct2     : second key for year data (e.g., 'Ov_y', 'Male_y')
        Struct3     : third key for value data (e.g., 'Ov_perc', 'Male_inc')
        DispFlag    : display flag (plotting skipped in Python)
        SubPlotPos  : subplot position (unused in Python)
        GraphDescription : description prefix for BM entries
        GraphTitle  : graph title (unused in Python)
        tolerance   : tolerance for green/red flagging
        LineSz, MarkerSz, FontSz : plot parameters (unused in Python)
        LabelY      : y-axis label (unused in Python)
        Flag        : 'Polyp' or 'Cancer'

    Returns:
        BM, bmc, OutputFlags, OutputValues
    """
    BM_year = np.array(Benchmarks[Struct1][Struct2])
    BM_value = np.array(Benchmarks[Struct1][Struct3])

    OutputFlags = [None] * len(BM_year)
    OutputValues = np.zeros(len(BM_year))

    # Plotting is skipped in Python (DispFlag handling kept for logic completeness)

    # add benchmarks
    for f in range(len(BM_year)):
        if (BM_year[f] > 5) and (BM_year[f] < 95):
            if Flag == 'Polyp':
                BM['description'][bmc] = GraphDescription + str(int(BM_year[f]))
                BM['benchmark'][bmc] = BM_value[f]
                # MATLAB: mean(DataGraph(BM_year(f)-1 : BM_year(f)+3))  -- year adapted
                # In MATLAB with 1-based indexing into a 1:100 array, BM_year(f) is
                # already an age/index. In Python 0-based, BM_year[f] corresponds
                # to index BM_year[f] directly (since Evaluation uses age 0-99).
                # MATLAB slice BM_year(f)-1 : BM_year(f)+3 (inclusive) = 5 elements
                # Python: int(BM_year[f])-1 : int(BM_year[f])+3+1 = 5 elements
                idx_start = int(BM_year[f]) - 1
                idx_end = int(BM_year[f]) + 3 + 1  # +1 for Python exclusive end
                if idx_start < 0:
                    idx_start = 0
                if idx_end > len(DataGraph):
                    idx_end = len(DataGraph)
                BM['value'][bmc] = np.mean(DataGraph[idx_start:idx_end])

                if (BM['value'][bmc] < (BM['benchmark'][bmc] * (1 + tolerance)) and
                        BM['value'][bmc] > (BM['benchmark'][bmc] * (1 - tolerance))):
                    BM['flag'][bmc] = 'green'
                else:
                    BM['flag'][bmc] = 'red'

                # Initialize nested dict if needed
                if Struct1 not in BM:
                    BM[Struct1] = {}
                if Struct3 not in BM[Struct1]:
                    BM[Struct1][Struct3] = np.zeros(len(BM_year))
                BM[Struct1][Struct3][f] = BM['value'][bmc]
                OutputFlags[f] = BM['flag'][bmc]
                OutputValues[f] = BM['value'][bmc]
                bmc += 1

            elif Flag == 'Cancer':
                if BM_year[f] > 20:  # we ignore benchmarks for age 1-20
                    BM['description'][bmc] = GraphDescription + str(int(BM_year[f]))
                    BM['benchmark'][bmc] = BM_value[f]
                    BM['value'][bmc] = DataGraph[f]  # year adapted

                    if (BM['value'][bmc] >= BM['benchmark'][bmc] * (1 - tolerance) and
                            BM['value'][bmc] <= (BM['benchmark'][bmc] * (1 + tolerance))):
                        BM['flag'][bmc] = 'green'
                    elif abs(BM['value'][bmc] - BM['benchmark'][bmc]) <= 2:
                        # we ignore very small absolute differences
                        BM['flag'][bmc] = 'green'
                    else:
                        BM['flag'][bmc] = 'red'

                    if Struct1 not in BM:
                        BM[Struct1] = {}
                    if Struct3 not in BM[Struct1]:
                        BM[Struct1][Struct3] = np.zeros(len(BM_year))
                    BM[Struct1][Struct3][f] = BM['value'][bmc]
                    OutputFlags[f] = BM['flag'][bmc]
                    OutputValues[f] = BM['value'][bmc]
                    bmc += 1
            else:
                raise ValueError('wrong flag')

    return BM, bmc, OutputFlags, OutputValues


def Evaluation(data, Variables):
    """
    Main Evaluation function.

    MATLAB signature: [data, BM] = Evaluation(data, Variables)

    Parameters:
        data      : dictionary containing simulation results (from NumberCrunching)
        Variables : dictionary containing simulation parameters and benchmarks

    Returns:
        data, BM  : updated data dict and benchmark results dict

    NOTE about TIME (year):
    In all scripts PRECEDING this, time was 1-100 (Lebensjahre).
    In EVALUATION we transform this to age (0-99).
    Since Python arrays are 0-based, index f corresponds to age f.
    """

    DispFlag = Variables['DispFlag']
    ResultsFlag = Variables['ResultsFlag']
    ExcelFlag = Variables['ExcelFlag']

    y = data['y']
    n = data['n']

    # key settings
    FontSz = 7
    MarkerSz = 4
    LineSz = 0.4
    bmc = 0  # 0-based benchmark counter (MATLAB started at 1)

    tolerance = 0.2

    BM = {}
    BM['description'] = [None] * 500
    BM['value'] = [None] * 500
    BM['benchmark'] = [None] * 500
    BM['flag'] = [None] * 500
    BM['Graph'] = {}
    BM['OutputFlags'] = {}
    BM['OutputValues'] = {}
    BM['Cancer'] = {}

    # Benchmarks
    # a few benchmarks remain hardcoded:
    Variables['Benchmarks']['MultiplePolypsYoung'] = np.array([18, 5, 3, 3, 2])
    # MidBenchmark   = [36 16 5  4 3];
    # MidBenchmark   = Variables.Benchmarks.MultiplePolyp;
    Variables['Benchmarks']['MultiplePolypsOld'] = np.array([40, 24, 10, 8, 4])

    if 'Cancer' not in Variables['Benchmarks']:
        Variables['Benchmarks']['Cancer'] = {}
    Variables['Benchmarks']['Cancer']['SymptomaticStageDistribution'] = np.array([15, 35.6, 27.9, 21.5])
    Variables['Benchmarks']['Cancer']['ScreeningStageDistribution'] = np.array([39.5, 34.7, 17.3, 8.5])

    Variables['Benchmarks']['Cancer']['LocationRectumMale'] = np.array([41.2, 34.1, 28.6, 23.8])
    Variables['Benchmarks']['Cancer']['LocationRectumFemale'] = np.array([37.2, 28.3, 23.0, 19.0])
    # year adapted: these are age ranges (MATLAB used 1-based years)
    Variables['Benchmarks']['Cancer']['LocationRectumYear'] = [[51, 55], [61, 65], [71, 75], [81, 85]]

    Variables['Benchmarks']['Cancer']['Fastcancer'] = np.array([0.005, 0.05, 0.08, 0.25, 3, 20])

    ##################################################################
    ###       FIGUREs  (skipped in Python -- no MATLAB plotting)   ###
    ##################################################################

    #####################################
    ###   Early/ Advanced polyps All  ###
    #####################################

    # we calculate the number of patients with polyps 1-4 and express
    # them as percentage of survivors
    NumPolyps = np.zeros(y)
    NumPolyps_2 = np.zeros(y)
    NumPolyps_3 = np.zeros(y)
    NumPolyps_4 = np.zeros(y)
    NumPolyps_5 = np.zeros(y)
    NumPolyps_6 = np.zeros(y)

    for f in range(y):
        included_mask = data['YearIncluded'][f, :] == 1
        NumPolyps[f] = np.sum(data['MaxPolyps'][f, included_mask] > 0)
        NumPolyps_2[f] = np.sum(data['MaxPolyps'][f, included_mask] > 1)
        NumPolyps_3[f] = np.sum(data['MaxPolyps'][f, included_mask] > 2)
        NumPolyps_4[f] = np.sum(data['MaxPolyps'][f, included_mask] > 3)
        NumPolyps_5[f] = np.sum(data['MaxPolyps'][f, included_mask] > 4)
        NumPolyps_6[f] = np.sum(data['MaxPolyps'][f, included_mask] > 5)

    FracPolyps = np.zeros(y)
    FracPolyps_2 = np.zeros(y)
    FracPolyps_3 = np.zeros(y)
    FracPolyps_4 = np.zeros(y)
    FracPolyps_5 = np.zeros(y)
    FracPolyps_6 = np.zeros(y)

    for f in range(y):
        total_included = np.sum(data['YearIncluded'][f, :])
        if total_included > 0:
            FracPolyps[f] = NumPolyps[f] / total_included * 100
            FracPolyps_2[f] = NumPolyps_2[f] / total_included * 100
            FracPolyps_3[f] = NumPolyps_3[f] / total_included * 100
            FracPolyps_4[f] = NumPolyps_4[f] / total_included * 100
            FracPolyps_5[f] = NumPolyps_5[f] / total_included * 100
            FracPolyps_6[f] = NumPolyps_6[f] / total_included * 100

    # the fraction of surviving patients with early polyps
    BM, bmc, OutputFlags, OutputValues = CalculateAgreement(
        FracPolyps, bmc, BM, Variables['Benchmarks'], 'EarlyPolyp', 'Ov_y', 'Ov_perc',
        DispFlag, 1, 'early polyps year ', 'early polyps overall',
        tolerance, LineSz, MarkerSz, FontSz, '% of survivors', 'Polyp')
    BM['Graph']['EarlyAdenoma_Ov'] = FracPolyps.copy()
    BM['OutputFlags']['EarlyAdenoma_Ov'] = OutputFlags
    BM['OutputValues']['EarlyAdenoma_Ov'] = OutputValues

    # the fraction of surviving patients with advanced polyps
    BM, bmc, OutputFlags, OutputValues = CalculateAgreement(
        FracPolyps_5, bmc, BM, Variables['Benchmarks'], 'AdvPolyp', 'Ov_y', 'Ov_perc',
        DispFlag, 2, 'advanced polyps year ', 'advanced polyps overall',
        tolerance, LineSz, MarkerSz, FontSz, '% of survivors', 'Polyp')
    BM['Graph']['AdvAdenoma_Ov'] = FracPolyps_5.copy()
    BM['OutputFlags']['AdvAdenoma_Ov'] = OutputFlags
    BM['OutputValues']['AdvAdenoma_Ov'] = OutputValues

    ##############################
    ###  Cancer Incidence All  ###
    ##############################

    i_arr = np.zeros(y)
    j_arr = np.zeros(y)
    for f in range(y):
        i_arr[f] = np.count_nonzero(data['TumorRecord']['Stage'][f, :])
        j_arr[f] = np.sum(data['YearIncluded'][f, :])

    # we summarize in 5 year intervals
    # MATLAB indices (1-based): 1:4, 5:8, 11:15, 16:20, ... 86:90
    # Python indices (0-based): 0:4, 4:8, 10:15, 15:20, ... 85:90
    SumCa = np.array([
        np.sum(i_arr[0:4]),   np.sum(i_arr[4:8]),   np.sum(i_arr[10:15]),
        np.sum(i_arr[15:20]), np.sum(i_arr[20:25]),  np.sum(i_arr[25:30]),
        np.sum(i_arr[30:35]), np.sum(i_arr[35:40]),  np.sum(i_arr[40:45]),
        np.sum(i_arr[45:50]), np.sum(i_arr[50:55]),  np.sum(i_arr[55:60]),
        np.sum(i_arr[60:65]), np.sum(i_arr[65:70]),  np.sum(i_arr[70:75]),
        np.sum(i_arr[75:80]), np.sum(i_arr[80:85]),  np.sum(i_arr[85:90])
    ])  # year adapted

    SumPat = np.array([
        np.sum(j_arr[0:4]),   np.sum(j_arr[4:8]),   np.sum(j_arr[10:15]),
        np.sum(j_arr[15:20]), np.sum(j_arr[20:25]),  np.sum(j_arr[25:30]),
        np.sum(j_arr[30:35]), np.sum(j_arr[35:40]),  np.sum(j_arr[40:45]),
        np.sum(j_arr[45:50]), np.sum(j_arr[50:55]),  np.sum(j_arr[55:60]),
        np.sum(j_arr[60:65]), np.sum(j_arr[65:70]),  np.sum(j_arr[70:75]),
        np.sum(j_arr[75:80]), np.sum(j_arr[80:85]),  np.sum(j_arr[85:90])
    ])

    # and express as new cancer cases per 100'000 patients
    Incidence = np.zeros(len(SumCa))
    for f in range(len(SumCa)):
        if SumPat[f] > 0:
            Incidence[f] = SumCa[f] / SumPat[f]
    Incidence = Incidence * 100000

    # Overall cancer incidence
    BM, bmc, OutputFlags, OutputValues = CalculateAgreement(
        Incidence, bmc, BM, Variables['Benchmarks'], 'Cancer', 'Ov_y', 'Ov_inc',
        DispFlag, 3, 'Cancer incidence year ', 'cancer incidence overall',
        tolerance, LineSz, MarkerSz, FontSz, "per 100'000 per year", 'Cancer')
    BM['Graph']['Cancer_Ov'] = Incidence.copy()
    BM['OutputFlags']['Cancer_Ov'] = OutputFlags
    BM['OutputValues']['Cancer_Ov'] = OutputValues

    BM['Incidence'] = Incidence.copy()

    ########################################
    ###   Early/ advanced polyps Male/ Female    ###
    ########################################

    # we calculate the presence of polyps (all polyps or Advanced polyps and
    # express as percent of survivors
    EarlyPolyps = [np.zeros(y), np.zeros(y)]  # index 0=male(Gender==1), 1=female(Gender==2)
    AdvPolyps = [np.zeros(y), np.zeros(y)]

    for f1 in range(2):  # 0=male, 1=female
        gender_val = f1 + 1  # 1=male, 2=female
        Gender_mask = data['Gender'] == gender_val
        for f in range(y):
            combined_mask = np.logical_and(Gender_mask, data['YearIncluded'][f, :].astype(bool))
            EarlyPolyps[f1][f] = np.sum(data['MaxPolyps'][f, combined_mask] > 0)
            AdvPolyps[f1][f] = np.sum(data['MaxPolyps'][f, combined_mask] > 4)
            Included = np.sum(data['YearIncluded'][f, Gender_mask])
            if Included > 0:
                EarlyPolyps[f1][f] = EarlyPolyps[f1][f] / Included * 100
                AdvPolyps[f1][f] = AdvPolyps[f1][f] / Included * 100

    # Early polyps male
    BM, bmc, OutputFlags, OutputValues = CalculateAgreement(
        EarlyPolyps[0], bmc, BM, Variables['Benchmarks'], 'EarlyPolyp', 'Male_y', 'Male_perc',
        DispFlag, 4, 'Early polyps male year ', 'early polyps present male',
        tolerance, LineSz, MarkerSz, FontSz, '% of survivors', 'Polyp')
    BM['Graph']['EarlyAdenoma_Male'] = EarlyPolyps[0].copy()
    BM['OutputFlags']['EarlyAdenoma_Male'] = OutputFlags
    BM['OutputValues']['EarlyAdenoma_Male'] = OutputValues

    # Early polyps female
    BM, bmc, OutputFlags, OutputValues = CalculateAgreement(
        EarlyPolyps[1], bmc, BM, Variables['Benchmarks'], 'EarlyPolyp', 'Female_y', 'Female_perc',
        DispFlag, 7, 'Early polyps female year ', 'early polyps present female',
        tolerance, LineSz, MarkerSz, FontSz, '% of survivors', 'Polyp')
    BM['Graph']['EarlyAdenoma_Female'] = EarlyPolyps[1].copy()
    BM['OutputFlags']['EarlyAdenoma_Female'] = OutputFlags
    BM['OutputValues']['EarlyAdenoma_Female'] = OutputValues

    # advanced polyps male
    BM, bmc, OutputFlags, OutputValues = CalculateAgreement(
        AdvPolyps[0], bmc, BM, Variables['Benchmarks'], 'AdvPolyp', 'Male_y', 'Male_perc',
        DispFlag, 5, 'Advanced polyps male year ', 'advanced polyps present male',
        tolerance, LineSz, MarkerSz, FontSz, '% of survivors', 'Polyp')
    BM['Graph']['AdvAdenoma_Male'] = AdvPolyps[0].copy()
    BM['OutputFlags']['AdvAdenoma_Male'] = OutputFlags
    BM['OutputValues']['AdvAdenoma_Male'] = OutputValues

    # advanced polyps female
    BM, bmc, OutputFlags, OutputValues = CalculateAgreement(
        AdvPolyps[1], bmc, BM, Variables['Benchmarks'], 'AdvPolyp', 'Female_y', 'Female_perc',
        DispFlag, 8, 'Advanced polyps female year ', 'advanced polyps present female',
        tolerance, LineSz, MarkerSz, FontSz, '% of survivors', 'Polyp')
    BM['Graph']['AdvAdenoma_Female'] = AdvPolyps[1].copy()
    BM['OutputFlags']['AdvAdenoma_Female'] = OutputFlags
    BM['OutputValues']['AdvAdenoma_Female'] = OutputValues

    #########################################
    ###   Cancer Incidence Male/ Female   ###
    #########################################

    Incidence_gender = [None, None]  # index 0=male, 1=female
    for f1 in range(2):
        gender_val = f1 + 1  # 1=male, 2=female
        i_g = np.zeros(y)
        j_g = np.zeros(y)
        for f2 in range(y):
            i_g[f2] = np.count_nonzero(
                data['TumorRecord']['Stage'][f2, data['TumorRecord']['Gender'][f2, :] == gender_val])
            j_g[f2] = np.sum(data['YearIncluded'][f2, data['Gender'] == gender_val])

        tmp3 = np.array([
            np.sum(i_g[0:4]),   np.sum(i_g[4:8]),   np.sum(i_g[10:15]),
            np.sum(i_g[15:20]), np.sum(i_g[20:25]),  np.sum(i_g[25:30]),
            np.sum(i_g[30:35]), np.sum(i_g[35:40]),  np.sum(i_g[40:45]),
            np.sum(i_g[45:50]), np.sum(i_g[50:55]),  np.sum(i_g[55:60]),
            np.sum(i_g[60:65]), np.sum(i_g[65:70]),  np.sum(i_g[70:75]),
            np.sum(i_g[75:80]), np.sum(i_g[80:85]),  np.sum(i_g[85:90])
        ])  # year adapted

        tmp4 = np.array([
            np.sum(j_g[0:4]),   np.sum(j_g[4:8]),   np.sum(j_g[10:15]),
            np.sum(j_g[15:20]), np.sum(j_g[20:25]),  np.sum(j_g[25:30]),
            np.sum(j_g[30:35]), np.sum(j_g[35:40]),  np.sum(j_g[40:45]),
            np.sum(j_g[45:50]), np.sum(j_g[50:55]),  np.sum(j_g[55:60]),
            np.sum(j_g[60:65]), np.sum(j_g[65:70]),  np.sum(j_g[70:75]),
            np.sum(j_g[75:80]), np.sum(j_g[80:85]),  np.sum(j_g[85:90])
        ])

        tmp5 = np.zeros(len(tmp3))
        for f in range(len(tmp3)):
            if tmp4[f] > 0:
                tmp5[f] = tmp3[f] / tmp4[f]
        Incidence_gender[f1] = tmp5 * 100000

    # male cancer incidence
    BM, bmc, OutputFlags, OutputValues = CalculateAgreement(
        Incidence_gender[0], bmc, BM, Variables['Benchmarks'], 'Cancer', 'Male_y', 'Male_inc',
        DispFlag, 6, 'Cancer incidence year male ', 'cancer incidence male',
        tolerance, LineSz, MarkerSz, FontSz, "per 100'000 per year", 'Cancer')
    BM['Graph']['Cancer_Male'] = Incidence_gender[0].copy()
    BM['OutputFlags']['Cancer_Male'] = OutputFlags
    BM['OutputValues']['Cancer_Male'] = OutputValues

    # female cancer incidence
    BM, bmc, OutputFlags, OutputValues = CalculateAgreement(
        Incidence_gender[1], bmc, BM, Variables['Benchmarks'], 'Cancer', 'Female_y', 'Female_inc',
        DispFlag, 9, 'Cancer incidence year female ', 'cancer incidence female',
        tolerance, LineSz, MarkerSz, FontSz, "per 100'000 per year", 'Cancer')
    BM['Graph']['Cancer_Female'] = Incidence_gender[1].copy()
    BM['OutputFlags']['Cancer_Female'] = OutputFlags
    BM['OutputValues']['Cancer_Female'] = OutputValues

    ###############################
    ###   Early Polyps present  ###
    ###############################
    # Plotting skipped in Python

    ####################################
    ###   Early Polyps distribution  ###
    ####################################

    Polyp_early = np.zeros(6)
    Polyp_adv = np.zeros(6)
    BM_value_early = np.zeros(6)
    BM_value_adv = np.zeros(6)

    BM_value_polyp = np.array(Variables['Benchmarks']['Polyp_Distr'])
    # MATLAB: sum(sum(data.AllPolyps(1:4, 51:76)))  -- year adapted
    # Python: data['AllPolyps'][0:4, 50:76]
    Summe_early = np.sum(data['AllPolyps'][0:4, 50:76])  # year adapted
    Summe_adv = np.sum(data['AllPolyps'][4:6, 50:76])    # year adapted

    sum_bm_early = np.sum(BM_value_polyp[0:4])
    if sum_bm_early > 0:
        BM_value_early[0:4] = BM_value_polyp[0:4] / sum_bm_early * 100
    sum_bm_adv = np.sum(BM_value_polyp[4:6])
    if sum_bm_adv > 0:
        BM_value_adv[4:6] = BM_value_polyp[4:6] / sum_bm_adv * 100

    Color = [None] * 6
    LinePos = np.zeros(6)

    for f in range(4):  # MATLAB 1:4 -> Python 0:4
        BM['description'][bmc] = '% of all early polyps P ' + str(f + 1)
        if Summe_early > 0:
            Polyp_early[f] = np.sum(data['AllPolyps'][f, 50:76]) / Summe_early * 100  # year adapted
        BM['value'][bmc] = Polyp_early[f]
        BM['benchmark'][bmc] = BM_value_early[f]

        if f == 0:
            LinePos[f] = Polyp_early[f] / 2
        else:
            LinePos[f] = np.sum(Polyp_early[0:f]) + Polyp_early[f] / 2

        if (BM['value'][bmc] > BM_value_early[f] * (1 - tolerance) and
                BM['value'][bmc] < BM_value_early[f] * (1 + tolerance)):
            BM['flag'][bmc] = 'green'
            Color[f] = 'g'
        else:
            BM['flag'][bmc] = 'red'
            Color[f] = 'r'

        if 'Polyp_Distr' not in BM:
            BM['Polyp_Distr'] = np.zeros(6)
        BM['Polyp_Distr'][f] = BM['value'][bmc]
        bmc += 1

    for f in range(4, 6):  # MATLAB 5:6 -> Python 4:6
        BM['description'][bmc] = '% of all early polyps P ' + str(f + 1)
        if Summe_adv > 0:
            Polyp_adv[f] = np.sum(data['AllPolyps'][f, 50:76]) / Summe_adv * 100  # year adapted
        BM['value'][bmc] = Polyp_adv[f]
        BM['benchmark'][bmc] = BM_value_adv[f]

        if f == 0:
            LinePos[f] = Polyp_adv[f] / 2
        else:
            LinePos[f] = np.sum(Polyp_adv[0:f]) + Polyp_adv[f] / 2

        if (BM['value'][bmc] > BM_value_adv[f] * (1 - tolerance) and
                BM['value'][bmc] < BM_value_adv[f] * (1 + tolerance)):
            BM['flag'][bmc] = 'green'
            Color[f] = 'g'
        else:
            BM['flag'][bmc] = 'red'
            Color[f] = 'r'

        if 'Polyp_Distr' not in BM:
            BM['Polyp_Distr'] = np.zeros(6)
        BM['Polyp_Distr'][f] = BM['value'][bmc]
        bmc += 1

    # Plotting skipped
    BM['Polyp_early'] = Polyp_early.copy()
    BM['BM_value_early'] = BM_value_early.copy()
    BM['Polyp_adv'] = Polyp_adv.copy()
    BM['BM_value_adv'] = BM_value_adv.copy()
    BM['Pflag'] = Color[:]

    #############################
    ###   Cumulative Cancer   ###
    #############################

    Early_Cancer = np.zeros(100)
    Late_Cancer = np.zeros(100)
    for f in range(100):
        Early_Cancer[f] = np.sum(data['TumorRecord']['Stage'][f, :] == 7) + \
                          np.sum(data['TumorRecord']['Stage'][f, :] == 8)
        Late_Cancer[f] = np.sum(data['TumorRecord']['Stage'][f, :] == 9) + \
                         np.sum(data['TumorRecord']['Stage'][f, :] == 10)

    PatientNumber = data['TumorRecord']['PatientNumber']

    DiagCancer = np.zeros(n)
    CumDiagCancer = np.zeros(100)
    for f in range(100):
        tmp2 = np.nonzero(PatientNumber[f, :])[0]
        for f2 in range(len(tmp2)):
            # PatientNumber stores 1-based patient numbers; convert to 0-based index
            pat_idx = int(PatientNumber[f, tmp2[f2]]) - 1
            DiagCancer[pat_idx] = 1
        CumDiagCancer[f] = np.sum(DiagCancer) / n * 100

    DiagCancer = np.zeros(n)
    DiagYCancer = 500 * np.ones(n)
    MultipleCancer = np.zeros(100)
    MultipleSurvCanc = np.zeros(100)

    for f in range(100):
        tmp2 = np.nonzero(PatientNumber[f, :])[0]
        for f2 in range(len(tmp2)):
            pos = int(PatientNumber[f, tmp2[f2]]) - 1  # 0-based patient index
            if DiagCancer[pos] == 1:
                MultipleCancer[f] += 1
                # MATLAB: (f-DiagYCancer(pos))<=5  (f is 1-based year)
                # Python: (f+1 - DiagYCancer[pos]) <= 5
                if ((f + 1) - DiagYCancer[pos]) <= 5:
                    MultipleSurvCanc[f] += 1
            else:
                DiagCancer[pos] = 1
                DiagYCancer[pos] = f + 1  # store as 1-based year for consistency with MATLAB logic

    DoubleCancer = np.zeros(100)
    for f in range(100):
        DoubleCancer[f] = np.sum(MultipleCancer[0:f + 1])
    DoubleCancer = DoubleCancer / n * 100

    # Recurrence/Metachronous
    tmpL = data['TumorRecord']['Location']
    PatLoc = np.zeros((13, n))
    MetachronCancer = np.zeros(n)
    RecurrenCancer = np.zeros(PatientNumber.shape[1])

    for fn in range(PatientNumber.shape[1]):
        tmp2 = np.nonzero(PatientNumber[:, fn])[0]

        if len(tmp2) > 1:
            RecurrenCancer[fn] = 1

        for f2 in range(len(tmp2)):
            pat = int(PatientNumber[tmp2[f2], fn]) - 1  # 0-based patient index
            loc = int(tmpL[tmp2[f2], fn])  # 1-based location
            if loc > 0 and pat >= 0:
                if PatLoc[loc - 1, pat] == 0:  # loc-1 for 0-based array
                    PatLoc[loc - 1, pat] = 1
                else:
                    MetachronCancer[pat] += 1

    CumulativeCancer = np.zeros(y)
    for f in range(y):
        CumulativeCancer[f] = np.sum(data['HasCancer'][f, 0:n]) / n * 100

    # Plotting skipped

    MultCanc = DoubleCancer.copy()
    Metachronous = np.zeros(3)
    Metachronous[0] = np.sum(RecurrenCancer)
    Metachronous[1] = np.sum(MetachronCancer)
    Metachronous[2] = np.sum(MultipleSurvCanc)
    BM['Cancer']['Metachronous'] = Metachronous.copy()
    BM['Cancer']['MultCanc'] = MultCanc.copy()

    ###############################
    ###  Cancer Survival    4-4 ###
    ###############################

    All = np.zeros(y)
    AllNoCa = np.zeros(y)
    Man = np.zeros(y)
    ManNoCa = np.zeros(y)
    Woman = np.zeros(y)
    WomanNoCa = np.zeros(y)

    for f in range(y):
        All[f] = np.sum(data['YearIncluded'][f, :])
        AllNoCa[f] = np.sum(data['YearAlive'][f, :])
        Man[f] = np.sum(data['YearIncluded'][f, data['Gender'] == 1])
        ManNoCa[f] = np.sum(data['YearAlive'][f, data['Gender'] == 1])
        Woman[f] = np.sum(data['YearIncluded'][f, data['Gender'] == 2])
        WomanNoCa[f] = np.sum(data['YearAlive'][f, data['Gender'] == 2])

    Number = All[0]
    if Number > 0:
        All = All / Number * 100
        AllNoCa = AllNoCa / Number * 100
        Man = Man / Number * 100
        ManNoCa = ManNoCa / Number * 100
        Woman = Woman / Number * 100
        WomanNoCa = WomanNoCa / Number * 100

    # Plotting skipped

    #############################
    ###    Sojourn Time       ###
    #############################

    SojournCancer = np.array([])
    DwellCancer = np.array([])
    DwellFastCancer = np.array([])
    AgeSojourn = np.array([])
    AgeDwellCa = np.array([])
    AgeDwellFastCa = np.array([])

    for f in range(99):  # MATLAB 1:99 -> Python 0:99
        # find last nonzero in row f
        nz_sojourn = np.nonzero(data['TumorRecord']['Sojourn'][f, :])[0]
        if len(nz_sojourn) > 0:
            last_idx = nz_sojourn[-1]
            SojournCancer = np.concatenate([SojournCancer,
                                            data['TumorRecord']['Sojourn'][f, 0:last_idx + 1]])
            AgeSojourn = np.concatenate([AgeSojourn,
                                         np.ones(last_idx + 1) * (f + 1)])  # f+1 to match MATLAB 1-based year

        nz_dwell = np.nonzero(data['DwellTimeProgression'][f, :])[0]
        if len(nz_dwell) > 0:
            last_idx = nz_dwell[-1]
            DwellCancer = np.concatenate([DwellCancer,
                                          data['DwellTimeProgression'][f, 0:last_idx + 1]])
            AgeDwellCa = np.concatenate([AgeDwellCa,
                                          np.ones(len(nz_dwell)) * (f + 1)])

        nz_fast = np.nonzero(data['DwellTimeFastCancer'][f, :])[0]
        if len(nz_fast) > 0:
            last_idx = nz_fast[-1]
            DwellFastCancer = np.concatenate([DwellFastCancer,
                                               data['DwellTimeFastCancer'][f, 0:last_idx + 1]])
            AgeDwellFastCa = np.concatenate([AgeDwellFastCa,
                                              np.ones(len(nz_fast)) * (f + 1)])

    SojournDoc = {}
    if len(SojournCancer) > 0:
        SojournDoc['SojournMedian'] = np.median(SojournCancer)
        SojournDoc['SojournMean'] = np.mean(SojournCancer)
        SojournDoc['SojournLowQuart'] = np.quantile(SojournCancer, 0.25)
        SojournDoc['SojournUppQuart'] = np.quantile(SojournCancer, 0.75)
    else:
        SojournDoc['SojournMedian'] = 0
        SojournDoc['SojournMean'] = 0
        SojournDoc['SojournLowQuart'] = 0
        SojournDoc['SojournUppQuart'] = 0

    # we record the time for overall cancer
    AllTimeCa = np.concatenate([DwellCancer, DwellFastCancer])
    mean_sojourn = SojournDoc['SojournMean']
    AllTimeCa = AllTimeCa + mean_sojourn  # this is an approximation

    AllTimeDoc = {}
    if len(AllTimeCa) > 0:
        AllTimeDoc['AllTimeMedian'] = np.median(AllTimeCa)
        AllTimeDoc['AllTimeMean'] = np.mean(AllTimeCa)
        AllTimeDoc['AllTimeLowQuart'] = np.quantile(AllTimeCa, 0.25)
        AllTimeDoc['AllTimeUppQuart'] = np.quantile(AllTimeCa, 0.75)
    else:
        AllTimeDoc['AllTimeMedian'] = 0
        AllTimeDoc['AllTimeMean'] = 0
        AllTimeDoc['AllTimeLowQuart'] = 0
        AllTimeDoc['AllTimeUppQuart'] = 0

    AgeSojourn = np.round((AgeSojourn + 4) / 10) * 10       # year adapted
    AgeDwellCa = np.round((AgeDwellCa + 4) / 10) * 10       # year adapted
    AgeDwellFastCa = np.round((AgeDwellFastCa + 4) / 10) * 10  # year adapted

    AllCancer_sojourn = []
    AllAge = []
    for f in range(len(SojournCancer)):
        AllCancer_sojourn.append(SojournCancer[f])
        AllCancer_sojourn.append(SojournCancer[f])
        AllAge.append('all')
        AllAge.append(str(int(AgeSojourn[f])))

    # Plotting skipped (boxplot)

    SojournDoc['MedianAllCa'] = np.median(AllCancer_sojourn) if len(AllCancer_sojourn) > 0 else 0
    SojournDoc['MeanAllCa'] = np.mean(AllCancer_sojourn) if len(AllCancer_sojourn) > 0 else 0
    SojournDoc['LowQuartAllCa'] = np.quantile(AllCancer_sojourn, 0.25) if len(AllCancer_sojourn) > 0 else 0
    SojournDoc['UppQuartAllCa'] = np.quantile(AllCancer_sojourn, 0.75) if len(AllCancer_sojourn) > 0 else 0

    #############################
    ###  Adenoma Dwell Time   ###
    #############################

    AllDwellCa = []
    AllAgeDwellCa = []
    AllDwellFastCa = []
    AllAgeDwellFastCa = []

    for f in range(len(DwellCancer)):
        AllDwellCa.append(DwellCancer[f])
        AllDwellCa.append(DwellCancer[f])
        AllAgeDwellCa.append('all')
        AllAgeDwellCa.append(str(int(AgeDwellCa[f])))

    for f in range(len(DwellFastCancer)):
        AllDwellFastCa.append(DwellFastCancer[f])
        AllDwellFastCa.append(DwellFastCancer[f])
        AllAgeDwellFastCa.append('all')
        AllAgeDwellFastCa.append(str(int(AgeDwellFastCa[f])))

    combined_dwell = AllDwellCa + AllDwellFastCa

    DwellTimeAllCa = np.median(combined_dwell) if len(combined_dwell) > 0 else 0
    DwellTimeProgressedCa = np.median(AllDwellCa) if len(AllDwellCa) > 0 else 0
    DwellTimeFastCa = np.median(AllDwellFastCa) if len(AllDwellFastCa) > 0 else 0
    # print(f"DEBUG: DwellTimeAllCa = {DwellTimeAllCa}", flush=True)
    # print(f"DEBUG: DwellTimeProgressedCa = {DwellTimeProgressedCa}", flush=True)
    # print(f"DEBUG: DwellTimeFastCa = {DwellTimeFastCa}", flush=True)

    DwellDoc = {}
    if len(combined_dwell) > 0:
        DwellDoc['MedianAllCa'] = np.median(combined_dwell)
        DwellDoc['MeanAllCa'] = np.mean(combined_dwell)
        DwellDoc['LowQuartAllCa'] = np.quantile(combined_dwell, 0.25)
        DwellDoc['UppQuartAllCa'] = np.quantile(combined_dwell, 0.75)
    else:
        DwellDoc['MedianAllCa'] = 0
        DwellDoc['MeanAllCa'] = 0
        DwellDoc['LowQuartAllCa'] = 0
        DwellDoc['UppQuartAllCa'] = 0

    if len(AllDwellFastCa) > 0:
        DwellDoc['MedianFastCa'] = np.median(AllDwellFastCa)
        DwellDoc['MeanFastCa'] = np.mean(AllDwellFastCa)
        DwellDoc['LowQuartFastCa'] = np.quantile(AllDwellFastCa, 0.25)
        DwellDoc['UppQuartFastCa'] = np.quantile(AllDwellFastCa, 0.75)
    else:
        DwellDoc['MedianFastCa'] = 0
        DwellDoc['MeanFastCa'] = 0
        DwellDoc['LowQuartFastCa'] = 0
        DwellDoc['UppQuartFastCa'] = 0

    if len(AllDwellCa) > 0:
        DwellDoc['MedianProgCa'] = np.median(AllDwellCa)
        DwellDoc['MeanProgCa'] = np.mean(AllDwellCa)
        DwellDoc['LowQuartProgCa'] = np.quantile(AllDwellCa, 0.25)
        DwellDoc['UppQuartProgCa'] = np.quantile(AllDwellCa, 0.75)
    else:
        DwellDoc['MedianProgCa'] = 0
        DwellDoc['MeanProgCa'] = 0
        DwellDoc['LowQuartProgCa'] = 0
        DwellDoc['UppQuartProgCa'] = 0

    DwellString = [
        'median dwell time all ca: {:.2f}'.format(DwellTimeAllCa),
        'median dwell time progressed ca: {:.2f}'.format(DwellTimeProgressedCa),
        'median dwell time fast ca: {:.2f}'.format(DwellTimeFastCa),
        'avg dwell time all ca: ' + str(round(np.mean(combined_dwell) * 10) / 10 if len(combined_dwell) > 0 else 0),
        'avg dwell time progressed ca: ' + str(round(np.mean(AllDwellCa) * 10) / 10 if len(AllDwellCa) > 0 else 0),
        'avg dwell time fast ca: ' + str(round(np.mean(AllDwellFastCa) * 10) / 10 if len(AllDwellFastCa) > 0 else 0),
    ]

    # we calculate again, using only diagnosed cancer
    diag_mask = data['TumorRecord']['Gender'] > 0
    DwellTime_diag = data['TumorRecord']['DwellTime'][diag_mask]
    SojournTime_diag = data['TumorRecord']['Sojourn'][diag_mask]
    OverallTime_diag = DwellTime_diag + SojournTime_diag

    Doc = {}
    if len(DwellTime_diag) > 0:
        Doc['MedianDwellTime'] = np.median(DwellTime_diag)
        Doc['MeanDwellTime'] = np.mean(DwellTime_diag)
        Doc['LowQuartDwellTime'] = np.quantile(DwellTime_diag, 0.25)
        Doc['UpQuartDwellTime'] = np.quantile(DwellTime_diag, 0.75)
    else:
        Doc['MedianDwellTime'] = 0
        Doc['MeanDwellTime'] = 0
        Doc['LowQuartDwellTime'] = 0
        Doc['UpQuartDwellTime'] = 0

    if len(SojournTime_diag) > 0:
        Doc['MedianSojournTime'] = np.median(SojournTime_diag)
        Doc['MeanSojournTime'] = np.mean(SojournTime_diag)
        Doc['LowQuartSojournTime'] = np.quantile(SojournTime_diag, 0.25)
        Doc['UpQuartSojournTime'] = np.quantile(SojournTime_diag, 0.75)
    else:
        Doc['MedianSojournTime'] = 0
        Doc['MeanSojournTime'] = 0
        Doc['LowQuartSojournTime'] = 0
        Doc['UpQuartSojournTime'] = 0

    if len(OverallTime_diag) > 0:
        Doc['MedianOverAllTime'] = np.median(OverallTime_diag)
        Doc['MeanOverAllTime'] = np.mean(OverallTime_diag)
        Doc['LowQuartOverAllTime'] = np.quantile(OverallTime_diag, 0.25)
        Doc['UpQuartOverAllTime'] = np.quantile(OverallTime_diag, 0.75)
    else:
        Doc['MedianOverAllTime'] = 0
        Doc['MeanOverAllTime'] = 0
        Doc['LowQuartOverAllTime'] = 0
        Doc['UpQuartOverAllTime'] = 0

    # Plotting skipped (boxplot of dwell time)

    BM['DwellTime'] = round(DwellTimeAllCa * 10) / 10

    BM['description'][bmc] = 'dwell time diagnosed cancer'
    BM['value'][bmc] = BM['DwellTime']
    BM['flag'][bmc] = 'black'
    BM['benchmark'][bmc] = 0
    bmc += 1

    ########################################################
    ###  Adenoma, cancer in (screening) population       ###
    ########################################################

    # MATLAB: for f=41:50  -> Python: for f in range(40, 50) (year adapted)
    Polyp_40_49 = np.array([])
    for f in range(40, 50):  # year adapted
        tmp = data['NumPolyps'][f, :]
        Polyp_40_49 = np.concatenate([Polyp_40_49, tmp[tmp > 0]])

    Polyp_50_59 = np.array([])
    for f in range(50, 60):  # year adapted
        tmp = data['NumPolyps'][f, :]
        Polyp_50_59 = np.concatenate([Polyp_50_59, tmp[tmp > 0]])

    Polyp_60_69 = np.array([])
    for f in range(60, 70):  # year adapted
        tmp = data['NumPolyps'][f, :]
        Polyp_60_69 = np.concatenate([Polyp_60_69, tmp[tmp > 0]])

    Polyp_70_79 = np.array([])
    for f in range(70, 80):  # year adapted
        tmp = data['NumPolyps'][f, :]
        Polyp_70_79 = np.concatenate([Polyp_70_79, tmp[tmp > 0]])

    Polyp_80_89 = np.array([])
    for f in range(80, 90):  # year adapted
        tmp = data['NumPolyps'][f, :]
        Polyp_80_89 = np.concatenate([Polyp_80_89, tmp[tmp > 0]])

    String = [None] * 16
    String[0] = 'summary number polyps'
    String[1] = ''
    String[2] = '40-49y: {:.2g} ({:.2g})'.format(
        round(np.mean(Polyp_40_49) * 100) / 100 if len(Polyp_40_49) > 0 else 0,
        round(np.std(Polyp_40_49) * 100) / 100 if len(Polyp_40_49) > 0 else 0)
    String[3] = '50-59y: {:.2g} ({:.2g})'.format(
        round(np.mean(Polyp_50_59) * 100) / 100 if len(Polyp_50_59) > 0 else 0,
        round(np.std(Polyp_50_59) * 100) / 100 if len(Polyp_50_59) > 0 else 0)
    String[4] = '60-69y: {:.2g} ({:.2g})'.format(
        round(np.mean(Polyp_60_69) * 100) / 100 if len(Polyp_60_69) > 0 else 0,
        round(np.std(Polyp_60_69) * 100) / 100 if len(Polyp_60_69) > 0 else 0)
    String[5] = '70-79y: {:.2g} ({:.2g})'.format(
        round(np.mean(Polyp_70_79) * 100) / 100 if len(Polyp_70_79) > 0 else 0,
        round(np.std(Polyp_70_79) * 100) / 100 if len(Polyp_70_79) > 0 else 0)
    String[6] = '80-89y: {:.2g} ({:.2g})'.format(
        round(np.mean(Polyp_80_89) * 100) / 100 if len(Polyp_80_89) > 0 else 0,
        round(np.std(Polyp_80_89) * 100) / 100 if len(Polyp_80_89) > 0 else 0)

    # we give a summary of the screening population 50-80 years of age
    # MATLAB: for f=51:81 -> Python: for f in range(50, 81) (year adapted)
    tmp_count = 0
    Polyp_count = 0
    AdvPolyp_count = 0
    Cancer_count = 0
    for f in range(50, 81):  # year adapted
        tmp_count += np.sum(data['YearIncluded'][f, :])
        included_mask = data['YearIncluded'][f, :] == 1
        Polyp_count += np.sum(data['MaxPolyps'][f, included_mask] > 0)
        AdvPolyp_count += np.sum(data['MaxPolyps'][f, included_mask] > 4)
        Cancer_count += np.sum(data['MaxCancer'][f, included_mask] > 6)

    String[7] = ''
    String[8] = 'screening population (50-80y)'
    String[9] = ''
    if tmp_count > 0:
        String[10] = 'adenoma prevalence   : {}%'.format(round(Polyp_count / tmp_count * 1000) / 10)
        String[11] = 'advanced adenoma prev.:{}%'.format(round(AdvPolyp_count / tmp_count * 1000) / 10)
        String[12] = 'carcinoma prevalence:{}%'.format(round(Cancer_count / tmp_count * 1000) / 10)
    else:
        String[10] = 'adenoma prevalence   : 0%'
        String[11] = 'advanced adenoma prev.:0%'
        String[12] = 'carcinoma prevalence:0%'

    # Plotting skipped

    if tmp_count > 0:
        BM['Preval'] = np.array([
            round(Polyp_count / tmp_count * 1000) / 10,
            round(AdvPolyp_count / tmp_count * 1000) / 10,
            round(Cancer_count / tmp_count * 1000) / 10
        ])
    else:
        BM['Preval'] = np.array([0, 0, 0])

    ################################
    ### number polyps age graph  ###
    ################################

    # we summarize the number of polyps
    FivePolyps = np.zeros(y)
    FourPolyps = np.zeros(y)
    ThreePolyps = np.zeros(y)
    TwoPolyps = np.zeros(y)
    OnePolyp = np.zeros(y)
    for f in range(y):
        FivePolyps[f] = np.sum(data['NumPolyps'][f, :] > 4)
        FourPolyps[f] = np.sum(data['NumPolyps'][f, :] > 3)
        ThreePolyps[f] = np.sum(data['NumPolyps'][f, :] > 2)
        TwoPolyps[f] = np.sum(data['NumPolyps'][f, :] > 1)
        OnePolyp[f] = np.sum(data['NumPolyps'][f, :] > 0)

    # these data are for the next plot which uses uncorrected numbers (at least
    # one polyp... we summarize the population of different ages
    NumYoung = 0
    NumMid = 0
    NumOld = 0
    NumAllAges = 0
    # MATLAB: for f=41:55 -> Python: for f in range(40, 55) (year adapted)
    for f in range(40, 55):
        NumYoung += np.sum(data['YearIncluded'][f, :])
    # MATLAB: for f=56:75 -> Python: for f in range(55, 75) (year adapted)
    for f in range(55, 75):
        NumMid += np.sum(data['YearIncluded'][f, :])
    # MATLAB: for f=76:91 -> Python: for f in range(75, 91) (year adapted)
    for f in range(75, 91):
        NumOld += np.sum(data['YearIncluded'][f, :])
    # MATLAB: for f=50:100 -> Python: for f in range(49, 100) (year adapted)
    for f in range(49, 100):
        NumAllAges += np.sum(data['YearIncluded'][f, :])

    YoungPop = np.zeros(5)
    MidPop = np.zeros(5)
    OldPop = np.zeros(5)

    # MATLAB: OnePolyp(41:55) -> Python: OnePolyp[40:55] (year adapted)
    if NumYoung > 0:
        YoungPop[0] = np.sum(OnePolyp[40:55]) / NumYoung
        YoungPop[1] = np.sum(TwoPolyps[40:55]) / NumYoung
        YoungPop[2] = np.sum(ThreePolyps[40:55]) / NumYoung
        YoungPop[3] = np.sum(FourPolyps[40:55]) / NumYoung
        YoungPop[4] = np.sum(FivePolyps[40:55]) / NumYoung
    if NumMid > 0:
        MidPop[0] = np.sum(OnePolyp[55:75]) / NumMid
        MidPop[1] = np.sum(TwoPolyps[55:75]) / NumMid
        MidPop[2] = np.sum(ThreePolyps[55:75]) / NumMid
        MidPop[3] = np.sum(FourPolyps[55:75]) / NumMid
        MidPop[4] = np.sum(FivePolyps[55:75]) / NumMid
    if NumOld > 0:
        OldPop[0] = np.sum(OnePolyp[75:91]) / NumOld
        OldPop[1] = np.sum(TwoPolyps[75:91]) / NumOld
        OldPop[2] = np.sum(ThreePolyps[75:91]) / NumOld
        OldPop[3] = np.sum(FourPolyps[75:91]) / NumOld
        OldPop[4] = np.sum(FivePolyps[75:91]) / NumOld

    YoungPop = YoungPop * 100
    MidPop = MidPop * 100
    OldPop = OldPop * 100
    BM['YoungPop'] = YoungPop.copy()
    BM['MidPop'] = MidPop.copy()
    BM['OldPop'] = OldPop.copy()

    # we correct for multiple polyps
    AllPolyps_frac = OnePolyp[0:100] / 100.0
    OnePolyp_corr = OnePolyp - TwoPolyps
    TwoPolyps_corr = TwoPolyps - ThreePolyps
    ThreePolyps_corr = ThreePolyps - FourPolyps
    FourPolyps_corr = FourPolyps - FivePolyps

    # Plotting skipped

    ##################################################
    ###    Number Polyps Frequency distribution    ###
    ##################################################

    YoungBenchmark = Variables['Benchmarks']['MultiplePolypsYoung']
    MidBenchmark = np.array(Variables['Benchmarks']['MultiplePolyp'])
    OldBenchmark = Variables['Benchmarks']['MultiplePolypsOld']

    BM['OutputValues']['YoungPop'] = YoungPop.copy()
    BM['OutputValues']['MidPop'] = MidPop.copy()
    BM['OutputValues']['OldPop'] = OldPop.copy()

    # Plotting skipped (Young/Mid/Old population plots)

    BM['OutputFlags']['MidPop'] = [None] * 5
    for f in range(5):
        BM['description'][bmc] = 'middle ' + str(f + 1) + ' polyp'
        BM['value'][bmc] = MidPop[f]
        BM['benchmark'][bmc] = MidBenchmark[f]
        if (BM['value'][bmc] > BM['benchmark'][bmc] * (1 - tolerance) and
                BM['value'][bmc] < BM['benchmark'][bmc] * (1 + tolerance)):
            BM['flag'][bmc] = 'green'
        else:
            BM['flag'][bmc] = 'red'
        BM['OutputFlags']['MidPop'][f] = BM['flag'][bmc]
        bmc += 1

    for f in range(5):
        BM['description'][bmc] = 'old ' + str(f + 1) + ' polyp'
        BM['value'][bmc] = OldPop[f]
        BM['benchmark'][bmc] = OldBenchmark[f]
        if (BM['value'][bmc] > BM['benchmark'][bmc] * (1 - tolerance) and
                BM['value'][bmc] < BM['benchmark'][bmc] * (1 + tolerance)):
            BM['flag'][bmc] = 'green'
        else:
            BM['flag'][bmc] = 'red'
        bmc += 1

    #############################
    ###    Written Summary    ###
    #############################

    female_count = np.sum(data['Gender'] == 2)

    SummaryVariable = [None] * 66  # 0-based, size 66 to accommodate index 65

    SummaryVariable[0] = n
    SummaryVariable[1] = round(np.sum(data['DeathYear']) / n * 100) / 100 - 1     # year adapted
    male_count = n - female_count
    if male_count > 0:
        SummaryVariable[2] = round(np.sum(data['DeathYear'][data['Gender'] == 1]) / male_count * 100) / 100 - 1  # year adapted
    else:
        SummaryVariable[2] = 0
    if female_count > 0:
        SummaryVariable[3] = round(np.sum(data['DeathYear'][data['Gender'] == 2]) / female_count * 100) / 100 - 1  # year adapted
    else:
        SummaryVariable[3] = 0
    SummaryVariable[4] = np.sum(data['Number']['Screening_Colonoscopy'])
    SummaryVariable[5] = np.sum(data['Number']['Symptoms_Colonoscopy'])
    SummaryVariable[6] = np.sum(data['Number']['Follow_Up_Colonoscopy'])
    SummaryVariable[7] = np.sum(data['Number']['RectoSigmo'])
    SummaryVariable[8] = np.sum(data['Number']['FOBT'])
    SummaryVariable[9] = np.sum(data['Number']['I_FOBT'])
    SummaryVariable[10] = np.sum(data['Number']['Sept9'])
    SummaryVariable[11] = np.sum(data['Number']['other'])
    SummaryVariable[12] = np.sum(data['DeathCause'] == 2)
    SummaryVariable[13] = np.sum(data['NaturalDeathYear'][data['DeathCause'] == 2]
                                  - data['DeathYear'][data['DeathCause'] == 2])
    SummaryVariable[14] = np.sum(data['DeathCause'] == 3)
    SummaryVariable[15] = np.sum(data['NaturalDeathYear'][data['DeathCause'] == 3]
                                  - data['DeathYear'][data['DeathCause'] == 3])
    SummaryVariable[16] = np.sum(data['Money']['AllCost'][0:100])
    SummaryVariable[17] = DwellTimeAllCa
    SummaryVariable[18] = DwellTimeProgressedCa
    SummaryVariable[19] = DwellTimeFastCa
    SummaryVariable[20] = SojournDoc['SojournMedian']

    SummaryVariable[63] = Variables.get('Comment', '')
    SummaryVariable[64] = Variables.get('Settings_Name', '')

    SummaryVariable[55] = SojournDoc['SojournMedian']
    SummaryVariable[56] = SojournDoc['SojournMean']
    SummaryVariable[57] = SojournDoc['SojournLowQuart']
    SummaryVariable[58] = SojournDoc['SojournUppQuart']

    SummaryVariable[59] = AllTimeDoc['AllTimeMedian']
    SummaryVariable[60] = AllTimeDoc['AllTimeMean']
    SummaryVariable[61] = AllTimeDoc['AllTimeLowQuart']
    SummaryVariable[62] = AllTimeDoc['AllTimeUppQuart']

    SummaryVariable[43] = DwellDoc['MedianAllCa']  # 44-47 AllCa (0-based: 43-46)
    SummaryVariable[44] = DwellDoc['MeanAllCa']
    SummaryVariable[45] = DwellDoc['LowQuartAllCa']
    SummaryVariable[46] = DwellDoc['UppQuartAllCa']

    SummaryVariable[47] = DwellDoc['MedianFastCa']  # 48-51: fast Ca (0-based: 47-50)
    SummaryVariable[48] = DwellDoc['MeanFastCa']
    SummaryVariable[49] = DwellDoc['LowQuartFastCa']
    SummaryVariable[50] = DwellDoc['UppQuartFastCa']

    SummaryVariable[51] = DwellDoc['MedianProgCa']  # 52-55 progressed Ca (0-based: 51-54)
    SummaryVariable[52] = DwellDoc['MeanProgCa']
    SummaryVariable[53] = DwellDoc['LowQuartProgCa']
    SummaryVariable[54] = DwellDoc['UppQuartProgCa']

    # Discounted years
    DisCountMask = np.zeros(101)
    DisCountMask[0:21] = 1
    for ff in range(21, 101):
        DisCountMask[ff] = DisCountMask[ff - 1] * 0.97

    Diff = np.zeros(n)
    for i in range(n):
        y1_idx = int(np.floor(data['DeathYear'][i]))       # 0-based index
        y2_idx = int(np.floor(data['NaturalDeathYear'][i]))  # 0-based index
        Diff[i] = cumulativeDiscYears(y1_idx, y2_idx, DisCountMask)

    # Build summary strings (for display, kept as data)
    StringList = [None] * 14
    StringList[0] = 'population: {} patients'.format(n)
    StringList[1] = 'age: all: {}, male: {}, female: {}'.format(
        round(np.sum(data['DeathYear']) / n * 100) / 100 - 1,
        SummaryVariable[2], SummaryVariable[3])
    StringList[2] = '{} screening colos performed'.format(int(np.sum(data['Number']['Screening_Colonoscopy'])))
    StringList[3] = '{} symptom colos performed'.format(int(np.sum(data['Number']['Symptoms_Colonoscopy'])))
    StringList[4] = '{} follow up colos performed'.format(int(np.sum(data['Number']['Follow_Up_Colonoscopy'])))
    StringList[5] = '{} custom tests performed'.format(
        int(np.sum(data['Number']['RectoSigmo']) + np.sum(data['Number']['FOBT']) +
            np.sum(data['Number']['I_FOBT']) + np.sum(data['Number']['Sept9']) +
            np.sum(data['Number']['other'])))
    StringList[6] = '{} patients died of CRC'.format(int(np.sum(data['DeathCause'] == 2)))
    StringList[7] = '{} years lost to CRC'.format(
        np.sum(data['NaturalDeathYear'][data['DeathCause'] == 2]
               - data['DeathYear'][data['DeathCause'] == 2]))
    StringList[8] = '{} pat. died due to colo'.format(int(np.sum(data['DeathCause'] == 3)))
    StringList[9] = '{} years lost to colo'.format(
        np.sum(data['NaturalDeathYear'][data['DeathCause'] == 3]
               - data['DeathYear'][data['DeathCause'] == 3]))
    StringList[10] = '{} total CRR rel costs'.format(np.sum(data['Money']['AllCost']))
    StringList[11] = 'comment: {}'.format(Variables.get('Comment', ''))
    StringList[12] = 'settings: {}'.format(Variables.get('Settings_Name', ''))

    # Plotting skipped

    #############################
    ###    Fast Cancer        ###
    #############################

    # we summarize the instances of progression of fast cancer and progressed
    # cancer per decade
    ProgressedCancer = np.zeros(10)
    FastCancer_1 = np.zeros(10)
    FastCancer_2 = np.zeros(10)
    FastCancer_3 = np.zeros(10)
    FastCancer_4 = np.zeros(10)
    FastCancer_5 = np.zeros(10)
    FastCancer_x = np.zeros(10)

    for f in range(10):
        Start = f * 10       # 0-based
        Ende = (f + 1) * 10  # exclusive end for Python slicing
        ProgressedCancer[f] = np.sum(data['ProgressedCancer'][Start:Ende])
        FastCancer_1[f] = np.sum(data['DirectCancer'][0, Start:Ende])  # cancer derived from polyp p1
        FastCancer_2[f] = np.sum(data['DirectCancer'][1, Start:Ende])  # cancer derived from polyp p2
        FastCancer_3[f] = np.sum(data['DirectCancer'][2, Start:Ende])  # etc.
        FastCancer_4[f] = np.sum(data['DirectCancer'][3, Start:Ende])
        FastCancer_5[f] = np.sum(data['DirectCancer'][4, Start:Ende])
        FastCancer_x[f] = np.sum(data['DirectCancer2'][Start:Ende])    # cancer derived without precursor

    AllCancer_fc = ProgressedCancer + FastCancer_1 + FastCancer_2 + \
                   FastCancer_3 + FastCancer_4 + FastCancer_5 + FastCancer_x

    # we will later draw lines to visualize the whole cohort
    Summary = np.zeros(6)
    Summary[0] = np.sum(FastCancer_1)
    Summary[1] = Summary[0] + np.sum(FastCancer_2)
    Summary[2] = Summary[1] + np.sum(FastCancer_3)
    Summary[3] = Summary[2] + np.sum(FastCancer_4)
    Summary[4] = Summary[3] + np.sum(FastCancer_5)
    Summary[5] = Summary[4] + np.sum(FastCancer_x)
    total_all_cancer = np.sum(AllCancer_fc)
    if total_all_cancer > 0:
        Summary = Summary / total_all_cancer * 100

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        PlotData = np.array([
            FastCancer_1 / AllCancer_fc,
            FastCancer_2 / AllCancer_fc,
            FastCancer_3 / AllCancer_fc,
            FastCancer_4 / AllCancer_fc,
            FastCancer_5 / AllCancer_fc,
            ProgressedCancer / AllCancer_fc,
            FastCancer_x / AllCancer_fc
        ]) * 100
    PlotData = np.nan_to_num(PlotData, nan=0.0)  # we replace empty elements by zero

    # Plotting skipped

    # we save for later display as a benchmark
    BM['CancerOriginArea'] = PlotData.T.copy()
    BM['CancerOriginSummary'] = Summary.copy()

    value_fc = np.zeros(6)
    for f in range(5):
        denom = np.sum(data['AllPolyps'][f, 0:100])
        if denom > 0:
            value_fc[f] = np.sum(data['DirectCancer'][f, 0:100]) / denom * 100
    denom_6 = np.sum(data['AllPolyps'][5, 0:100])
    if denom_6 > 0:
        value_fc[5] = np.sum(data['ProgressedCancer'][0:100]) / denom_6 * 100

    BenchMark_fc = np.array(Variables['Benchmarks']['Cancer']['Fastcancer'])
    FastCancerValue = value_fc.copy()
    FastCancerBenchMark = BenchMark_fc.copy()

    # we correct and now talk about relative danger of each polyp
    sum_bm_fc = np.sum(BenchMark_fc)
    if sum_bm_fc > 0:
        BenchMark_fc = BenchMark_fc / sum_bm_fc * 100

    # we correct to relative danger
    sum_val_fc = np.sum(value_fc)
    if sum_val_fc > 0:
        value_fc = value_fc / sum_val_fc * 100

    # we save for later display as a benchmark
    BM['CancerOriginValue'] = value_fc.copy()

    ypos = 0
    BM['CancerOriginFlag'] = [None] * 6
    for f in range(6):
        BM['description'][bmc] = '%P' + str(f + 1) + ' transforming'
        BM['value'][bmc] = value_fc[f]
        BM['benchmark'][bmc] = BenchMark_fc[f]
        if (BM['value'][bmc] > BM['benchmark'][bmc] * (1 - tolerance) and
                BM['value'][bmc] < BM['benchmark'][bmc] * (1 + tolerance)):
            BM['flag'][bmc] = 'green'
        else:
            BM['flag'][bmc] = 'red'
        ypos += value_fc[f]
        # we save for later display as a benchmark
        BM['CancerOriginFlag'][f] = BM['flag'][bmc]
        bmc += 1

    ###############################
    ###    Stage Distribution   ###
    ###############################

    for x in range(1, 4):  # MATLAB 1:3
        if x == 1:
            headline = 'stage distribution screening'
            tmp_stage = data['TumorRecord']['Stage'].copy()
            tmp_stage[data['TumorRecord']['Detection'] != 1] = 0
            benchmark = Variables['Benchmarks']['Cancer']['ScreeningStageDistribution']
        elif x == 2:
            headline = 'stage distribution symptomatic cancer'
            tmp_stage = data['TumorRecord']['Stage'].copy()
            tmp_stage[data['TumorRecord']['Detection'] != 2] = 0
            benchmark = Variables['Benchmarks']['Cancer']['SymptomaticStageDistribution']
        elif x == 3:
            headline = 'stage distribution follow up'
            tmp_stage = data['TumorRecord']['Stage'].copy()
            tmp_stage[data['TumorRecord']['Detection'] != 3] = 0
            benchmark = Variables['Benchmarks']['Cancer']['ScreeningStageDistribution']

        # MATLAB indices: 1:50, 51:60, etc. -> Python: 0:50, 50:60, etc. (year adapted)
        population = np.zeros((9, 4))
        population[0, :] = [np.sum(tmp_stage[0:50, :] == 7),  np.sum(tmp_stage[0:50, :] == 8),
                            np.sum(tmp_stage[0:50, :] == 9),  np.sum(tmp_stage[0:50, :] == 10)]
        population[1, :] = [np.sum(tmp_stage[50:60, :] == 7), np.sum(tmp_stage[50:60, :] == 8),
                            np.sum(tmp_stage[50:60, :] == 9), np.sum(tmp_stage[50:60, :] == 10)]
        population[2, :] = [np.sum(tmp_stage[60:70, :] == 7), np.sum(tmp_stage[60:70, :] == 8),
                            np.sum(tmp_stage[60:70, :] == 9), np.sum(tmp_stage[60:70, :] == 10)]
        population[3, :] = [np.sum(tmp_stage[70:80, :] == 7), np.sum(tmp_stage[70:80, :] == 8),
                            np.sum(tmp_stage[70:80, :] == 9), np.sum(tmp_stage[70:80, :] == 10)]
        population[4, :] = [np.sum(tmp_stage[80:90, :] == 7), np.sum(tmp_stage[80:90, :] == 8),
                            np.sum(tmp_stage[80:90, :] == 9), np.sum(tmp_stage[80:90, :] == 10)]
        population[5, :] = [np.sum(tmp_stage[90:100, :] == 7), np.sum(tmp_stage[90:100, :] == 8),
                            np.sum(tmp_stage[90:100, :] == 9), np.sum(tmp_stage[90:100, :] == 10)]
        population[6, :] = [np.sum(tmp_stage[0:100, :] == 7), np.sum(tmp_stage[0:100, :] == 8),
                            np.sum(tmp_stage[0:100, :] == 9), np.sum(tmp_stage[0:100, :] == 10)]

        if x == 1:
            # MATLAB: SummaryVariable{22} = population(7, 1) -> Python 0-based: [21] = population[6, 0]
            SummaryVariable[21] = population[6, 0]
            SummaryVariable[22] = population[6, 1]
            SummaryVariable[23] = population[6, 2]
            SummaryVariable[24] = population[6, 3]
        elif x == 2:
            SummaryVariable[25] = population[6, 0]
            SummaryVariable[26] = population[6, 1]
            SummaryVariable[27] = population[6, 2]
            SummaryVariable[28] = population[6, 3]

        for f_row in range(7):
            row_sum = np.sum(population[f_row, :])
            if row_sum > 0:
                population[f_row, :] = population[f_row, :] / row_sum * 100

        population[7, :] = [0, 0, 0, 0]
        population[8, :] = benchmark

        # Plotting skipped

        if x == 2:
            ypos = 0
            for f in range(4):
                BM['description'][bmc] = '% stage ' + str(f + 1)
                BM['value'][bmc] = population[6, f]
                BM['benchmark'][bmc] = benchmark[f]
                if (BM['value'][bmc] > BM['benchmark'][bmc] * (1 - tolerance) and
                        BM['value'][bmc] < BM['benchmark'][bmc] * (1 + tolerance)):
                    BM['flag'][bmc] = 'green'
                else:
                    BM['flag'][bmc] = 'red'
                ypos += BM['value'][bmc]
                bmc += 1

    stage_I = np.sum(data['TumorRecord']['Stage'] == 7)
    stage_II = np.sum(data['TumorRecord']['Stage'] == 8)
    stage_III = np.sum(data['TumorRecord']['Stage'] == 9)
    stage_IV = np.sum(data['TumorRecord']['Stage'] == 10)

    Summe = np.sum(data['TumorRecord']['Stage'] > 0)
    if Summe > 0:
        SummaryVariable[29] = stage_I / Summe * 100
        SummaryVariable[30] = stage_II / Summe * 100
        SummaryVariable[31] = stage_III / Summe * 100
        SummaryVariable[32] = stage_IV / Summe * 100
    else:
        SummaryVariable[29] = 0
        SummaryVariable[30] = 0
        SummaryVariable[31] = 0
        SummaryVariable[32] = 0

    SummaryVariable[33] = stage_I
    SummaryVariable[34] = stage_II
    SummaryVariable[35] = stage_III
    SummaryVariable[36] = Summe

    SummaryVariable[37] = np.sum(data['TumorRecord']['Detection'] == 1)
    SummaryVariable[38] = np.sum(data['TumorRecord']['Detection'] == 2)
    SummaryVariable[39] = np.sum(data['TumorRecord']['Detection'] == 3)
    SummaryVariable[40] = np.sum(data['TumorRecord']['Detection'] == 4)

    #############################
    ###    Cause of Death     ###
    #############################

    edges = np.array([0, 9.1, 19.1, 29.1, 39.1, 49.1, 59.1, 69.1, 79.1, 89.1, 150])  # year adapted
    NaturalDeath, _ = np.histogram(data['DeathYear'][data['DeathCause'] == 1], bins=edges)
    CancerDeath, _ = np.histogram(data['DeathYear'][data['DeathCause'] == 2], bins=edges)
    ColonoscDeath, _ = np.histogram(data['DeathYear'][data['DeathCause'] == 3], bins=edges)

    # Plotting skipped

    #############################
    ###    Location           ###
    #############################

    tmp_all_loc = data['TumorRecord']['Stage'].copy()
    tmp_male_mask = data['TumorRecord']['Gender'] == 1
    tmp_female_mask = data['TumorRecord']['Gender'] == 2

    tmp_Rectum = data['TumorRecord']['Stage'].copy()
    tmp_Rectum[data['TumorRecord']['Location'] < 13] = 0
    tmp_Right = data['TumorRecord']['Stage'].copy()
    tmp_Right[data['TumorRecord']['Location'] > 3] = 0
    tmp_Rest = data['TumorRecord']['Stage'].copy()
    tmp_Rest[data['TumorRecord']['Location'] == 13] = 0
    tmp_Rest[data['TumorRecord']['Location'] < 4] = 0

    Sum_Stage_all = np.zeros(4)
    Sum_Stage_Rectum = np.zeros(4)
    Sum_Stage_Right = np.zeros(4)
    Sum_Stage_Rest = np.zeros(4)
    for f in range(4):
        Sum_Stage_all[f] = np.sum(tmp_all_loc == f + 7)
        Sum_Stage_Rectum[f] = np.sum(tmp_Rectum == f + 7)
        Sum_Stage_Right[f] = np.sum(tmp_Right == f + 7)
        Sum_Stage_Rest[f] = np.sum(tmp_Rest == f + 7)

    tmp_Rectum_male = (tmp_Rectum > 0).astype(float)
    tmp_Rectum_male[tmp_female_mask] = 0
    tmp_Rest_male = (tmp_Rest > 0).astype(float)
    tmp_Rest_male[tmp_female_mask] = 0  # Note: MATLAB code also set tmp_Rectum_male here (line 1174)
    tmp_all_male = (tmp_all_loc > 0).astype(float)
    tmp_all_male[tmp_female_mask] = 0

    tmp_Rectum_female = (tmp_Rectum > 0).astype(float)
    tmp_Rectum_female[tmp_male_mask] = 0
    tmp_Rest_female = (tmp_Rest > 0).astype(float)
    tmp_Rest_female[tmp_male_mask] = 0  # Note: MATLAB code also set tmp_Rectum_female here (line 1181)
    tmp_all_female = (tmp_all_loc > 0).astype(float)
    tmp_all_female[tmp_male_mask] = 0

    LocationRectum = [np.zeros(100), np.zeros(100)]  # [0]=male, [1]=female
    LocationRest = [np.zeros(100), np.zeros(100)]
    LocationAll = [np.zeros(100), np.zeros(100)]

    for f in range(100):
        LocationRectum[0][f] = np.sum(tmp_Rectum_male[f, :])
        LocationRest[0][f] = np.sum(tmp_Rest_male[f, :])
        LocationRectum[1][f] = np.sum(tmp_Rectum_female[f, :])
        LocationRest[1][f] = np.sum(tmp_Rest_female[f, :])
        LocationAll[0][f] = np.sum(tmp_all_male[f, :])
        LocationAll[1][f] = np.sum(tmp_all_female[f, :])

    # for calculating the percentage of rectal cancer

    ### benchmarks
    LocBenchmarkMale = Variables['Benchmarks']['Cancer']['LocationRectumMale']
    LocBenchmarkFemale = Variables['Benchmarks']['Cancer']['LocationRectumFemale']
    LocX = Variables['Benchmarks']['Cancer']['LocationRectumYear']

    #########################################################
    ### carcinoma rectum both genders                     ###
    #########################################################

    # we average the male and female benchmarks
    # here we only collect the data for display during adjustment of adenomas
    BM['LocationRectumAllGender'] = (LocationRectum[0][0:100] + LocationRectum[1][0:100]) / 2.0
    BM['LocationRest'] = (LocationRest[0][0:100] + LocationRest[1][0:100]) / 2.0
    BM['LocBenchmark'] = (LocBenchmarkMale + LocBenchmarkFemale) / 2.0
    BM['LocX'] = LocX

    BM['LocationRectumFlag'] = [None] * len(LocX)
    BM['LocationRectum'] = np.zeros(len(LocX))

    for f in range(len(LocX)):
        # MATLAB: mean(BM.LocX{f}(1):BM.LocX{f}(2))
        x_val = np.mean(np.arange(LocX[f][0], LocX[f][1] + 1))

        # MATLAB: sum(BM.LocationRectumAllGender((BM.LocX{f}(1)-2):(BM.LocX{f}(2)+2)))
        # MATLAB indices are 1-based. LocX values are ages (already matching 0-based Python since
        # they were defined as [51,55] etc. which in MATLAB referred to 1-based year indices).
        # To match MATLAB: (LocX{f}(1)-2):(LocX{f}(2)+2) inclusive
        # Python: [LocX[f][0]-2 : LocX[f][1]+2+1]  (since Python exclusive end)
        # But we need to be careful: in MATLAB the array is 1-based indexed 1:100.
        # LocX values are 51,55 etc. In Python the array is 0-based indexed 0:99.
        # The MATLAB code indexes directly with LocX values, so LocX{f}(1)-2 = 49 in MATLAB (1-based),
        # which is index 48 in Python (0-based).
        # Therefore: Python index = MATLAB_index - 1 = (LocX[f][0]-2) - 1 = LocX[f][0] - 3
        lo = LocX[f][0] - 3  # 0-based start
        hi = LocX[f][1] + 2  # 0-based end (exclusive, since MATLAB +2 inclusive -> Python +2+1-1=+2)
        if lo < 0:
            lo = 0
        if hi > 100:
            hi = 100

        rect_sum = np.sum(BM['LocationRectumAllGender'][lo:hi])
        rest_sum = np.sum(BM['LocationRest'][lo:hi])
        total = rect_sum + rest_sum
        if total > 0:
            value_loc = rect_sum / total * 100
        else:
            value_loc = 0

        if (value_loc > BM['LocBenchmark'][f] * (1 - tolerance) and
                value_loc < BM['LocBenchmark'][f] * (1 + tolerance)):
            tmpflag = 'green'
        else:
            tmpflag = 'red'

        if f == 1 or f == 2:  # MATLAB: f==2 or f==3 (1-based)
            BM['description'][bmc] = '% rectum Ca year ' + str(LocX[f][0]) + ' to ' + str(LocX[f][1])
            BM['flag'][bmc] = tmpflag
            BM['benchmark'][bmc] = BM['LocBenchmark'][f]
            BM['value'][bmc] = value_loc
            BM['LocationRectumFlag'][f] = tmpflag
            bmc += 1
        else:
            BM['LocationRectumFlag'][f] = 'black'
        BM['LocationRectum'][f] = value_loc

    #########################################################
    ### carcinoma rectum male                             ###
    #########################################################

    # Plotting skipped
    if 'Cancer' not in BM:
        BM['Cancer'] = {}
    BM['Cancer']['LocationRectumMale'] = np.zeros(len(LocX))
    BM['Cancer']['LocationRectumMaleYear'] = [None] * len(LocX)

    for f in range(len(LocX)):
        x_val = np.mean(np.arange(LocX[f][0], LocX[f][1] + 1))
        lo = LocX[f][0] - 3  # 0-based
        hi = LocX[f][1] + 2
        if lo < 0:
            lo = 0
        if hi > 100:
            hi = 100

        rect_sum = np.sum(LocationRectum[0][lo:hi])
        rest_sum = np.sum(LocationRest[0][lo:hi])
        total = rect_sum + rest_sum
        if total > 0:
            value_loc = rect_sum / total * 100
        else:
            value_loc = 0

        if (value_loc > LocBenchmarkMale[f] * (1 - tolerance) and
                value_loc < LocBenchmarkMale[f] * (1 + tolerance)):
            tmpflag = 'green'
        else:
            tmpflag = 'red'

        if f == 1 or f == 2:  # MATLAB: f==2 or f==3 (1-based)
            BM['description'][bmc] = '% rectum Ca year male ' + str(LocX[f][0]) + ' to ' + str(LocX[f][1])
            BM['flag'][bmc] = tmpflag
            BM['benchmark'][bmc] = LocBenchmarkMale[f]
            BM['value'][bmc] = value_loc
            BM['Cancer']['LocationRectumMale'][f] = BM['value'][bmc]
            BM['Cancer']['LocationRectumMaleYear'][f] = LocX[f]
            bmc += 1

    #########################################################
    ### carcinoma rectum female                           ###
    #########################################################

    # Plotting skipped
    BM['Cancer']['LocationRectumFemale'] = np.zeros(len(LocX))
    BM['Cancer']['LocationRectumFemaleYear'] = [None] * len(LocX)

    for f in range(len(LocX)):
        x_val = np.mean(np.arange(LocX[f][0], LocX[f][1] + 1))
        lo = LocX[f][0] - 3  # 0-based
        hi = LocX[f][1] + 2
        if lo < 0:
            lo = 0
        if hi > 100:
            hi = 100

        rect_sum = np.sum(LocationRectum[1][lo:hi])
        rest_sum = np.sum(LocationRest[1][lo:hi])
        total = rect_sum + rest_sum
        if total > 0:
            value_loc = rect_sum / total * 100
        else:
            value_loc = 0

        if (value_loc > LocBenchmarkFemale[f] * (1 - tolerance) and
                value_loc < LocBenchmarkFemale[f] * (1 + tolerance)):
            tmpflag = 'green'
        else:
            tmpflag = 'red'

        if f == 1 or f == 2:  # MATLAB: f==2 or f==3 (1-based)
            BM['description'][bmc] = '% rectum Ca year female ' + str(LocX[f][0]) + ' to ' + str(LocX[f][1])
            BM['flag'][bmc] = tmpflag
            BM['benchmark'][bmc] = LocBenchmarkFemale[f]
            BM['value'][bmc] = value_loc
            BM['Cancer']['LocationRectumFemale'][f] = BM['value'][bmc]
            BM['Cancer']['LocationRectumFemaleYear'][f] = LocX[f]
            bmc += 1

    #########################################################
    ### stage distribution location                       ###
    #########################################################

    Summe_loc = np.sum(Sum_Stage_all) / 100.0
    if Summe_loc > 0:
        PlotData_loc = np.array([
            Sum_Stage_all / Summe_loc,
            Sum_Stage_Rectum / Summe_loc,
            Sum_Stage_Right / Summe_loc,
            Sum_Stage_Rest / Summe_loc
        ])
    else:
        PlotData_loc = np.zeros((4, 4))

    # Plotting skipped

    #########################################################
    ### relative danger polyps                            ###
    #########################################################

    value_rel = FastCancerValue / np.sum(FastCancerValue) * 100 if np.sum(FastCancerValue) > 0 else np.zeros(6)
    BenchMark_rel = FastCancerBenchMark / np.sum(FastCancerBenchMark) * 100 if np.sum(FastCancerBenchMark) > 0 else np.zeros(6)

    String1 = [None] * 7
    String2 = [None] * 7
    String3 = [None] * 7
    String4 = [None] * 7
    String1[0] = 'Relative danger adenomas'
    AdenomaLabel = ['Ad 3mm', 'Ad 3mm', 'Ad 3mm', 'Ad 3mm', 'Adv P5', 'Adv P6']

    for f in range(6):
        BM['description'][bmc] = AdenomaLabel[f] + str(f + 1) + ' relative danger'
        BM['value'][bmc] = value_rel[f]
        BM['benchmark'][bmc] = BenchMark_rel[f]
        String1[f + 1] = AdenomaLabel[f]
        String2[f + 1] = str(round(BM['value'][bmc] * 1000) / 1000)
        String3[f + 1] = str(round(BM['benchmark'][bmc] * 1000) / 1000)
        if (BM['value'][bmc] > BM['benchmark'][bmc] * (1 - tolerance) and
                BM['value'][bmc] < BM['benchmark'][bmc] * (1 + tolerance)):
            BM['flag'][bmc] = 'green'
        else:
            BM['flag'][bmc] = 'red'
        String4[f + 1] = BM['flag'][bmc]
        bmc += 1

    # Plotting skipped

    ##############################################
    ###   Cancer Mortality All/ Male/ Female   ###
    ##############################################

    DeathYear_floor = np.floor(data['DeathYear']).astype(int)
    Mortality = [None, None, None]  # index 0=male, 1=female, 2=overall

    for f1 in range(1, 4):  # MATLAB 1:3
        i_mort = np.zeros(y)
        j_mort = np.zeros(y)
        if f1 != 3:
            tmp_mort = DeathYear_floor.copy()
            tmp_mort[data['Gender'] != f1] = 0
            tmp_mort[data['DeathCause'] != 2] = 0
            for f2 in range(y):
                # MATLAB: length(find(tmp == f2))  -- f2 is 1-based year
                # Python: f2+1 because DeathYear values are 1-based years
                i_mort[f2] = np.sum(tmp_mort == (f2 + 1))
                j_mort[f2] = np.sum(data['YearIncluded'][f2, data['Gender'] == f1])
        else:
            for f2 in range(y):
                tmp_mort = DeathYear_floor.copy()
                tmp_mort[data['DeathCause'] != 2] = 0
                i_mort[f2] = np.sum(tmp_mort == (f2 + 1))
                j_mort[f2] = np.sum(data['YearIncluded'][f2, :])

        tmp3_m = np.array([
            np.sum(i_mort[0:4]),   np.sum(i_mort[4:8]),   np.sum(i_mort[10:15]),
            np.sum(i_mort[15:20]), np.sum(i_mort[20:25]),  np.sum(i_mort[25:30]),
            np.sum(i_mort[30:35]), np.sum(i_mort[35:40]),  np.sum(i_mort[40:45]),
            np.sum(i_mort[45:50]), np.sum(i_mort[50:55]),  np.sum(i_mort[55:60]),
            np.sum(i_mort[60:65]), np.sum(i_mort[65:70]),  np.sum(i_mort[70:75]),
            np.sum(i_mort[75:80]), np.sum(i_mort[80:85]),  np.sum(i_mort[85:90])
        ])  # year adapted

        tmp4_m = np.array([
            np.sum(j_mort[0:4]),   np.sum(j_mort[4:8]),   np.sum(j_mort[10:15]),
            np.sum(j_mort[15:20]), np.sum(j_mort[20:25]),  np.sum(j_mort[25:30]),
            np.sum(j_mort[30:35]), np.sum(j_mort[35:40]),  np.sum(j_mort[40:45]),
            np.sum(j_mort[45:50]), np.sum(j_mort[50:55]),  np.sum(j_mort[55:60]),
            np.sum(j_mort[60:65]), np.sum(j_mort[65:70]),  np.sum(j_mort[70:75]),
            np.sum(j_mort[75:80]), np.sum(j_mort[80:85]),  np.sum(j_mort[85:90])
        ])

        tmp5_m = np.zeros(len(tmp3_m))
        for f in range(len(tmp3_m)):
            if tmp4_m[f] > 0:
                tmp5_m[f] = tmp3_m[f] / tmp4_m[f]
        Mortality[f1 - 1] = tmp5_m * 100000

    # cancer mortality male
    BM, bmc, OutputFlags, OutputValues = CalculateAgreement(
        Mortality[0], bmc, BM, Variables['Benchmarks'], 'Cancer', 'Ov_y_mort', 'Male_mort',
        DispFlag, 1, 'Cancer mortality male year ', 'Cancer mortality per year male',
        tolerance, LineSz, MarkerSz, FontSz, 'per 100 000 per year', 'Cancer')

    # cancer mortality female
    BM, bmc, OutputFlags, OutputValues = CalculateAgreement(
        Mortality[1], bmc, BM, Variables['Benchmarks'], 'Cancer', 'Ov_y_mort', 'Female_mort',
        DispFlag, 2, 'Cancer mortality female year ', 'Cancer mortality per year female',
        tolerance, LineSz, MarkerSz, FontSz, 'per 100 000 per year', 'Cancer')

    # cancer mortality overall
    BM, bmc, OutputFlags, OutputValues = CalculateAgreement(
        Mortality[2], bmc, BM, Variables['Benchmarks'], 'Cancer', 'Ov_y_mort', 'Ov_mort',
        DispFlag, 3, 'Cancer mortality overall year ', 'Cancer mortality per year overall',
        tolerance, LineSz, MarkerSz, FontSz, 'per 100 000 per year', 'Cancer')

    #############################
    ###    Direct Cancer      ###
    #############################

    tmp_all_dc = np.sum(data['DirectCancer'], axis=0) + data['DirectCancer2'] + data['ProgressedCancer']
    tmp_right_dc = data['DirectCancerR'] + data['DirectCancer2R'] + data['ProgressedCancerR']

    SumAll_dc = np.sum(tmp_all_dc)
    DirectAll = np.sum(data['DirectCancer2'])
    SumRight_dc = np.sum(tmp_right_dc)
    DirectRight = np.sum(data['DirectCancer2R'])

    SummaryVariable[41] = round(DirectAll / SumAll_dc * 1000) / 10 if SumAll_dc > 0 else 0
    SummaryVariable[42] = round(DirectRight / SumRight_dc * 1000) / 10 if SumRight_dc > 0 else 0

    # Plotting skipped

    if 'Graph' not in BM:
        BM['Graph'] = {}
    if 'DirectCa' not in BM['Graph']:
        BM['Graph']['DirectCa'] = {}
    BM['Graph']['DirectCa']['All'] = DirectAll / SumAll_dc * 100 if SumAll_dc > 0 else 0
    BM['Graph']['DirectCa']['Right'] = DirectRight / SumRight_dc * 100 if SumRight_dc > 0 else 0

    BM['description'][bmc] = 'fraction of all carcinoma without polyp precursor all'
    BM['value'][bmc] = SummaryVariable[41]
    BM['flag'][bmc] = 'black'
    BM['benchmark'][bmc] = 0
    bmc += 1

    BM['description'][bmc] = 'fraction of all carcinoma without polyp precursor right'
    BM['value'][bmc] = SummaryVariable[42]
    BM['flag'][bmc] = 'black'
    BM['benchmark'][bmc] = 0
    bmc += 1

    ##############################
    ###    Live Years Lost     ###
    ##############################

    # we need to calculate life years lost for each year of the
    # simulation for subsequent discounting
    n_ca_deaths = int(np.sum(data['DeathCause'] == 2))
    n_colo_deaths = int(np.sum(data['DeathCause'] == 3))
    LY_Ca_Temp = np.zeros((max(n_ca_deaths, 1), 101))
    LY_Colo_Temp = np.zeros((max(n_colo_deaths, 1), 101))
    Ca_Counter = 0
    Colo_Counter = 0

    for f in range(n):
        if data['DeathCause'][f] == 2:
            tmp1 = np.zeros(101)
            tmp2 = np.zeros(101)
            nat_dy = data['NaturalDeathYear'][f]
            dy = data['DeathYear'][f]
            floor_nat = int(np.floor(nat_dy))
            floor_dy = int(np.floor(dy))

            # MATLAB: tmp1(1:floor(data.NaturalDeathYear(f))) = 1
            # Python 0-based: tmp1[0:floor_nat] = 1
            if floor_nat > 0:
                tmp1[0:min(floor_nat, 101)] = 1
            if (nat_dy - floor_nat) > 0 and floor_nat < 101:
                tmp1[floor_nat] = nat_dy - floor_nat

            if floor_dy > 0:
                tmp2[0:min(floor_dy, 101)] = 1
            if (dy - floor_dy) > 0 and floor_dy < 101:
                tmp2[floor_dy] = dy - floor_dy

            LY_Ca_Temp[Ca_Counter, :] = tmp1 - tmp2
            Ca_Counter += 1

        elif data['DeathCause'][f] == 3:
            tmp1 = np.zeros(101)
            tmp2 = np.zeros(101)
            nat_dy = data['NaturalDeathYear'][f]
            dy = data['DeathYear'][f]
            floor_nat = int(np.floor(nat_dy))
            floor_dy = int(np.floor(dy))

            if floor_nat > 0:
                tmp1[0:min(floor_nat, 101)] = 1
            if (nat_dy - floor_nat) > 0 and floor_nat < 101:
                tmp1[floor_nat] = nat_dy - floor_nat

            if floor_dy > 0:
                tmp2[0:min(floor_dy, 101)] = 1
            if (dy - floor_dy) > 0 and floor_dy < 101:
                tmp2[floor_dy] = dy - floor_dy

            LY_Colo_Temp[Colo_Counter, :] = tmp1 - tmp2
            Colo_Counter += 1

    # we save results to the Results variable
    Results = {}
    Results['YearsLostCa'] = np.sum(LY_Ca_Temp, axis=0)
    Results['YearsLostColo'] = np.sum(LY_Colo_Temp, axis=0)

    ###########################################################################
    ###                           SAVING DATA                               ###
    ###########################################################################

    if Variables.get('StarterFlag') == 'on':
        answer = 'Yes'
        ResultsName = Variables['Settings_Name']
        ResultsPath = Variables['ResultsPath']
    else:
        answer = 'Yes'

    ResultsFullfile = os.path.join(Variables.get('ResultsPath', ''), Variables.get('Settings_Name', ''))

    # PDF saving is skipped (MATLAB figure saving)

    ### Excel (skipped in Python -- use CSV or other formats instead)
    # The original MATLAB code wrote to Excel files using xlswrite.
    # In Python, this could be done with openpyxl or pandas if needed.
    # For now, we build the data structures but skip the actual Excel writing.

    SummaryLegend = [
        'Number Patients', 'Average Age', 'Average Age male', 'Average Age female',
        'Screening Colonoscopies', 'Symptom Colonoscopies',
        'Follow up Colonoscopies', 'Number Rectosigmo', 'Number FOBT', 'Numer I-FOBT',
        'Number Septin9', 'Number other',
        'Colon cancer deaths', 'Years lost to colon cancer',
        'Patients died of colonoscopy', 'Years lost due to colonoscopy',
        'Total costs',  # 17 (0-based: 16)
        'Dwell time all cancer (median)',  # 18
        'Dwell time all progressed cancer (median)',  # 19
        'Dwell time all fast cancer (median)',  # 20
        'Sojourn time (median)',  # 21
        'screening stage I', 'screening stage II', 'screening stage III', 'screening stage IV',  # 22-25
        'symptoms stage I', 'symptoms stage II', 'symptoms stage III', 'symptoms stage IV',  # 26-29
        'all stage I', 'all stage II', 'all stage III', 'all stage IV',  # 30-33
        'number stage I', 'number stage II', 'number stage III', 'Number ALL Ca',  # 34-37
        'detected screening', 'detected symptoms', 'detected surveillance', 'detected baseline',  # 38-41
        'fraction direct all', 'fraction direct right',  # 42-43
        'dwell time all ca median', 'dwell time all ca mean',
        'dwell time all ca lower quartile', 'dwell time all ca upper quartile',  # 44-47
        'dwell time fast ca. median', 'dwell time fast ca. mean',
        'dwell time fast ca. lower quartile', 'dwell time fast ca. upper quartile',  # 48-51
        'progressed ca dwell time time median', 'progressed ca dwell time mean',
        'progressed ca dwell time lower quartile', 'progressed ca dwell time upper quartile',  # 52-55
        'sojourn time median', 'sojourn time mean',
        'sojourn time lower quartile', 'sojourn time upper quartile',  # 56-59
        'overall time median', 'overall time mean',
        'overall time lower quartile', 'overall time upper quartile',  # 60-63
        'comment', 'settings name'  # 64-65
    ]

    if ResultsFlag:
        FileName = ResultsFullfile + '_Results.npz'
        try:
            Results['Var_Legend'] = SummaryLegend
            Results['Variable'] = SummaryVariable
            Results['BM_Description'] = BM['description']
            Results['BM_Value'] = BM['value']
            Results['Benchmark'] = BM['benchmark']

            Results['NumberPatients'] = np.zeros(100)
            for f in range(100):
                Results['NumberPatients'][f] = np.sum(data['YearIncluded'][f, :])

            Results['Early_Cancer'] = Early_Cancer[0:100].copy()
            Results['Late_Cancer'] = Late_Cancer[0:100].copy()

            Results['Treatment'] = np.round(data['Money']['Treatment'][0:100] / n * 100) / 100
            Results['TreatmentFuture'] = np.round(data['Money']['FutureTreatment'][0:100] / n * 100) / 100
            Results['Screening'] = np.round(data['Money']['Screening'][0:100] / n * 100) / 100
            Results['FollowUp'] = np.round(data['Money']['FollowUp'][0:100] / n * 100) / 100
            Results['Other'] = np.round(data['Money']['Other'][0:100] / n * 100) / 100
            Results['InputCost'] = data['InputCost']
            Results['InputCostStage'] = data['InputCostStage']
            Results['PaymentType'] = data['PaymentType']

            # Ensure the directory exists
            results_dir = os.path.dirname(FileName)
            if results_dir and not os.path.exists(results_dir):
                os.makedirs(results_dir)

            np.savez(FileName, **Results)
            print(f"Results saved to: {FileName}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"ERROR: Could not save results file '{FileName}': {e}")
            warnings.warn('Could not save results file, try entering a correct pathway '
                          'to the save data path in main window.')
    else:
        print("ResultsFlag is disabled -- skipping results file save. "
              "Enable 'Enable Results' checkbox in main window to save results.")

    return data, BM
