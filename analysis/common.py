"""Shared paths, data loading and statistics for the R1 analyses (JMIR Serious Games ms#107960).

Participant data are not in this repository. Place them in ../data/ (or set SINDANCE_DATA):
    hur_performance.csv   HUR platform metrics, one row per participant x condition
                          (output of consolidate_hur_data.py)
    enhanced_features.csv 77 dance features per participant (output of the feature-extraction scripts)
    clinical.csv          sociodemographic, physical performance and MoCA data per participant
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('SINDANCE_DATA', os.path.join(os.path.dirname(HERE), 'data'))
OUT = os.path.join(HERE, 'results')
os.makedirs(OUT, exist_ok=True)

FEATURES = os.path.join(DATA, 'enhanced_features.csv')
CLINICAL = os.path.join(DATA, 'clinical.csv')
HUR = os.path.join(DATA, 'hur_performance.csv')

EDU_ORDER = {'No formal education': 0, 'Primary School': 1, 'Secondary School': 2,
             'ITE/Polytechnic/Junior College': 3, 'University and above': 4}
DEMO = ['Age', 'Gender_M', 'Education_ord']
PHYS = ['FRT', 'TUG', 'FTSTS', 'Handgrip R', 'Handgrip L']


def load_features():
    f = pd.read_csv(FEATURES)
    f['k'] = f['Subject_ID'].str.extract(r'(\d+)').astype(int)
    return f.sort_values('k').drop(columns='k').reset_index(drop=True)


def load_clinical():
    c = pd.read_csv(CLINICAL).rename(columns={'SubjectID': 'Subject_ID'})
    c['Gender_M'] = (c['Gender'] == 'M').astype(int)
    c['Education_ord'] = c['Education level'].map(EDU_ORDER)
    assert c['Education_ord'].notna().all(), 'unknown education category'
    return c


def engineer_balance_features(X_df):
    """Row-wise engineered features used by the balance model (no information shared across participants)."""
    df = X_df.copy()
    groups = {
        'sway':  [c for c in df.columns if 'sway' in c.lower()],
        'vel':   [c for c in df.columns if 'velocity' in c.lower() or 'speed' in c.lower()],
        'asym':  [c for c in df.columns if 'asym' in c.lower() or 'sym' in c.lower()],
        'coord': [c for c in df.columns if 'coord' in c.lower() or 'coupling' in c.lower()],
        'stab':  [c for c in df.columns if 'stab' in c.lower() or 'balance' in c.lower()],
    }
    for g, cols in groups.items():
        if len(cols) >= 2:
            a = df[cols].values
            df['grp_' + g + '_mean'] = a.mean(1); df['grp_' + g + '_std'] = a.std(1)
            df['grp_' + g + '_max'] = a.max(1); df['grp_' + g + '_range'] = a.max(1) - a.min(1)
    key = (groups['sway'][:4] + groups['vel'][:4] + groups['asym'][:4])
    key = [c for c in key if c in df.columns][:8]
    for i, c1 in enumerate(key):
        for c2 in key[i + 1:]:
            d = df[c2].replace(0, np.nan)
            df['ratio_' + c1[:8] + '_' + c2[:8]] = (df[c1] / d).fillna(0).clip(-10, 10)
    return df


# ---------------------------------------------------------------- statistics
def cp_ci(k, n, a=0.05):
    """Exact (Clopper-Pearson) binomial CI."""
    lo = stats.beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


def fmt_prop(k, n):
    lo, hi = cp_ci(k, n)
    return f"{k}/{n} = {100*k/n:.1f}% (95% CI {100*lo:.1f}-{100*hi:.1f})"


def boot_ci(fn, *arrays, B=2000, seed=0, stratify=None):
    """Percentile bootstrap CI of fn(*arrays)."""
    rng = np.random.RandomState(seed); n = len(arrays[0]); vals = []
    for _ in range(B):
        if stratify is None:
            idx = rng.randint(0, n, n)
        else:
            idx = np.concatenate([rng.choice(np.where(stratify == g)[0], (stratify == g).sum())
                                  for g in np.unique(stratify)])
        try:
            vals.append(fn(*[a[idx] for a in arrays]))
        except ValueError:
            pass
    return np.percentile(vals, [2.5, 97.5])


def write(name, text):
    p = os.path.join(OUT, name)
    with open(p, 'w') as fh:
        fh.write(text)
    print(text)
    return p
