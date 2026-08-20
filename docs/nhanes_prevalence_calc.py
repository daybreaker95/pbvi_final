"""nhanes_prevalence_calc.py
=============================
Reproduces the BMI>=25 / diabetes / alcohol prevalence figures used in
tests/jeon_elbow_analysis.py, computed directly from raw NHANES August
2021-August 2023 microdata (not a published summary table) so that all
three factors come from exactly one survey cycle, one sample, one set of
survey weights.

Data files (download once, not included in this repo -- see URLs below):
  https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DEMO_L.xpt
  https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BMX_L.xpt
  https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DIQ_L.xpt
  https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/ALQ_L.xpt

Family history of cancer has NO corresponding NHANES variable in any
cycle (confirmed by inspecting MCQ_L, the Medical Conditions file, for
this cycle -- it covers personal disease history only), so that factor's
prevalence stays sourced from Jeon et al. 2018's own validation-cohort
controls (Supplementary Table S3) instead.

Run: python nhanes_prevalence_calc.py  (with the 4 XPT files in this dir)
"""
import pandas as pd
import numpy as np

demo = pd.read_sas('DEMO_L.xpt', format='xport')[['SEQN', 'RIAGENDR', 'RIDAGEYR', 'WTMEC2YR']]
bmx = pd.read_sas('BMX_L.xpt', format='xport')[['SEQN', 'BMXBMI']]
diq = pd.read_sas('DIQ_L.xpt', format='xport')[['SEQN', 'DIQ010']]
alq = pd.read_sas('ALQ_L.xpt', format='xport')[['SEQN', 'ALQ121', 'ALQ130']]

df = demo.merge(bmx, on='SEQN', how='left').merge(diq, on='SEQN', how='left').merge(alq, on='SEQN', how='left')
df = df[df['RIDAGEYR'] >= 18].copy()
df = df[df['WTMEC2YR'] > 0].copy()

df['bmi_ge25'] = df['BMXBMI'] >= 25
df['diabetes_yes'] = df['DIQ010'] == 1  # 1=yes, 2=no, 3=borderline -- only 1 counted

# ALQ121 (past-12mo frequency category) -> approx days/week, per the NHANES
# codebook's own category definitions.
freq_to_days_per_week = {
    0: 0.0, 1: 7.0, 2: 6.0, 3: 3.5, 4: 2.0, 5: 1.0,
    6: 2.5 * 12 / 52, 7: 12 / 52, 8: 9 / 52, 9: 4.5 / 52, 10: 1.5 / 52,
}


def alc_gday(row):
    freq, drinks = row['ALQ121'], row['ALQ130']
    if pd.isna(freq) or freq in (77, 99):
        return np.nan
    if freq == 0:
        return 0.0
    if pd.isna(drinks) or drinks in (777, 999):
        return np.nan
    return freq_to_days_per_week.get(int(freq), np.nan) / 7.0 * drinks * 14.0  # 14g/standard US drink


df['g_day'] = df.apply(alc_gday, axis=1)
df['alc_tier'] = pd.cut(df['g_day'], bins=[-0.01, 0.999, 28, 1e9], labels=['abstinent', 'moderate', 'heavy'])


def wpct(mask, denom_mask):
    w = df.loc[denom_mask, 'WTMEC2YR']
    return 100 * (w * mask.loc[denom_mask]).sum() / w.sum()


for sex, label in [(1, 'male'), (2, 'female')]:
    sub = df['RIAGENDR'] == sex
    print(f'=== {label} (n={sub.sum()}) ===')
    bmi_valid = sub & df['BMXBMI'].notna()
    print(f'  BMI>=25: {wpct(df["bmi_ge25"], bmi_valid):.1f}%')
    dia_valid = sub & df['DIQ010'].isin([1, 2, 3])
    print(f'  diabetes: {wpct(df["diabetes_yes"], dia_valid):.1f}%')
    alc_valid = sub & df['alc_tier'].notna()
    for tier in ['abstinent', 'moderate', 'heavy']:
        print(f'  alcohol {tier}: {wpct(df["alc_tier"]==tier, alc_valid):.1f}%')
