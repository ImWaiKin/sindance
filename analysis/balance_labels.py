"""Composite balance score and the two balance-status groups (reference standard).

Method (as reported in the revised manuscript):
  1. For each participant and metric (Trace Length, C90 Area, Mean Sway Velocity, SD of Sway Velocity),
     a condition-weighted mean is taken across the four 30-s trials. Condition weights, set by the
     authors: eyes open 1.0, eyes open unstable 1.3, eyes closed 1.5, eyes closed unstable 2.0.
  2. Each metric is converted to a robust z score using the SAMPLE median and MAD:
     z = (x - median) / (1.4826 * MAD), then winsorised at the SAMPLE 5th and 95th percentiles.
  3. S = 0.30 z(Trace Length) + 0.25 z(C90 Area) + 0.25 z(Mean Velocity) + 0.20 z(SD Velocity)
     (metric weights set by the authors; not taken from published values).
  4. S is min-max rescaled to 0-1 within the SAMPLE.
  5. Score <= 0.33 -> 'Low' (lower composite sway); > 0.33 -> 'Elevated' (higher composite sway).
     0.33 divides the rescaled score into equal thirds; an earlier three-level grouping
     (0.33 / 0.67) was collapsed to two groups because of the small sample.

Because steps 2-4 use statistics of this sample, the score expresses relative standing within the
sample, is not a clinical threshold, and cannot be applied to a new individual without fixed
reference values. With fixed 30-s trials, Trace Length = Mean Velocity x 30 s, so these two
metrics are the same quantity (55% of the weight combined).

Placeholder files for participants without platform data (named '*-duplicated*') are excluded.
Usage: python balance_labels.py   -> writes ../data/balance_labels.csv
"""
import os
import pandas as pd
from common import HUR, DATA

CONDITION_WEIGHTS = {'Eyes Open': 1.0, 'Eyes Open Unstable': 1.3, 'Eyes Closed': 1.5, 'Eyes Closed Unstable': 2.0}
METRIC_WEIGHTS = {'Trace Length["]': 0.30, 'C90 Area["^2]': 0.25, 'Velocity["/s]': 0.25, 'STD Velocity["/s]': 0.20}
CUT = 0.33


def composite(perf=None, metric_weights=METRIC_WEIGHTS):
    """Return DataFrame [Subject_ID, composite] (0-1, higher = more sway)."""
    if perf is None:
        perf = pd.read_csv(HUR)
    perf = perf[~perf['subject_id'].astype(str).str.contains('duplicated')].copy()
    perf['w'] = perf['Performance Name'].map(CONDITION_WEIGHTS).fillna(1.0)
    rows = []
    for sid, g in perf.groupby('subject_id', sort=False):
        rows.append({'Subject_ID': sid, **{m: (g[m] * g['w']).sum() / g['w'].sum() for m in metric_weights}})
    agg = pd.DataFrame(rows)
    s = 0.0
    for m, w in metric_weights.items():
        v = agg[m]; med = v.median(); mad = (v - med).abs().median()
        z = (v - med) / (1.4826 * mad) if mad > 0 else v * 0
        z = z.clip(z.quantile(0.05), z.quantile(0.95))
        s = s + w * z
    agg['composite'] = (s - s.min()) / (s.max() - s.min())
    return agg[['Subject_ID', 'composite']]


def labels(cut=CUT):
    c = composite()
    c['Balance_Category'] = (c['composite'] > cut).map({True: 'Elevated', False: 'Low'})
    return c


if __name__ == '__main__':
    lab = labels()
    out = os.path.join(DATA, 'balance_labels.csv')
    lab.to_csv(out, index=False)
    print(lab['Balance_Category'].value_counts().to_string())
    print('written', out)
