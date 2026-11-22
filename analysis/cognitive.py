"""Cognitive (MoCA) re-analysis: baseline/incremental comparisons on identical LOOCV splits,
null model, pooled metrics with CIs, paired differences, permutation test."""
import sys, time, json, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy import stats
from common import *
warnings.filterwarnings('ignore')
N_PERM = 200 if '--quick' not in sys.argv else 20


def make(algo, n_feat):
    if algo == 'ridge':
        steps = [StandardScaler()]
        if n_feat > 15:
            steps.append(SelectKBest(f_regression, k=15))
        return make_pipeline(*steps, RidgeCV(alphas=np.logspace(-3, 3, 50)))
    if algo == 'rf':
        return RandomForestRegressor(n_estimators=500, min_samples_leaf=3, max_features=0.33, random_state=42, n_jobs=1)


def loocv(X, y, algo):
    n = len(y)
    if algo == 'null':
        return np.array([y[np.arange(n) != i].mean() for i in range(n)])
    def one(i):
        tr = np.arange(n) != i
        return make(algo, X.shape[1]).fit(X[tr], y[tr]).predict(X[[i]])[0]
    return np.array(Parallel(n_jobs=-1)(delayed(one)(i) for i in range(n)))


def metrics(y, p):
    e = np.abs(y - p); n = len(y)
    r2lo, r2hi = boot_ci(r2_score, y, p); mlo, mhi = boot_ci(mean_absolute_error, y, p)
    d = dict(R2=r2_score(y, p), R2_ci=(r2lo, r2hi), MAE=mean_absolute_error(y, p), MAE_ci=(mlo, mhi))
    for k in (1, 2, 3):
        d[f'w{k}'] = int((e <= k).sum())
    d['n'] = n
    return d


def row(name, d):
    n = d['n']
    return (f"  {name:<34} R2 {d['R2']:+.3f} ({d['R2_ci'][0]:+.3f} to {d['R2_ci'][1]:+.3f})  "
            f"MAE {d['MAE']:.2f} ({d['MAE_ci'][0]:.2f}-{d['MAE_ci'][1]:.2f})  "
            + "  ".join(f"+/-{k}: {d[f'w{k}']}/{n} ({100*d[f'w{k}']/n:.1f}%, {100*cp_ci(d[f'w{k}'],n)[0]:.0f}-{100*cp_ci(d[f'w{k}'],n)[1]:.0f})" for k in (1, 2, 3)))


def paired(y, pa, pb, B=2000, seed=0):
    """b minus a: dMAE (negative = b better), dR2 (positive = b better)."""
    rng = np.random.RandomState(seed); n = len(y); dm, dr = [], []
    for _ in range(B):
        i = rng.randint(0, n, n)
        dm.append(mean_absolute_error(y[i], pb[i]) - mean_absolute_error(y[i], pa[i]))
        dr.append(r2_score(y[i], pb[i]) - r2_score(y[i], pa[i]))
    ea, eb = np.abs(y - pa), np.abs(y - pb)
    w = stats.wilcoxon(ea, eb).pvalue
    return (mean_absolute_error(y, pb) - mean_absolute_error(y, pa), np.percentile(dm, [2.5, 97.5]),
            r2_score(y, pb) - r2_score(y, pa), np.percentile(dr, [2.5, 97.5]), w)


def main():
    t0 = time.time(); out = []; J = {}
    feats = load_features(); clin = load_clinical()
    df = feats.merge(clin, on='Subject_ID'); y = df['MoCA'].values.astype(float)
    dance = [c for c in feats.columns if c != 'Subject_ID']
    demo = ['Age', 'Gender_M', 'Education_ord']
    sets = {'demographics': demo, 'demographics+physical': demo + PHYS, 'dance only': dance,
            'demographics+dance': demo + dance, 'demographics+physical+dance': demo + PHYS + dance}
    out.append(f"COGNITIVE RE-ANALYSIS  n={len(y)}; MoCA range {y.min():.0f}-{y.max():.0f}, mean {y.mean():.2f} (SD {y.std(ddof=1):.2f})\n"
               f"Education coded ordinally 0-4 (No formal < Primary < Secondary < ITE/Poly/JC < University).\n"
               f"NOTE: the original notebooks LabelEncoded education alphabetically (ITE=0, No formal=1, Primary=2, Secondary=3, University=4),\n"
               f"      i.e. not ordinal, and also used Weight, Height (median-imputed), falls, activity, marital, employment, ethnicity.\n")

    # --- how the published Model 1 numbers arose
    yq = pd.qcut(pd.Series(y), q=4, labels=False, duplicates='drop')
    _, Xte, _, yte = train_test_split(np.arange(len(y)), y, test_size=0.15, random_state=42, stratify=yq)
    out.append(f"[MODEL 1 AS PUBLISHED] notebook used ONE 85/15 train/test split (not LOOCV): test n = {len(yte)}.\n"
               f"  9/13 = {100*9/13:.1f}%, 3/13 = {100*3/13:.1f}% -> matches the reported 69.2% and 23.1%; R2/MAE were from these 13 people.\n")

    P = {}
    P['null'] = loocv(None, y, 'null')
    for algo in ['ridge', 'rf']:
        out.append(f"\n[LOOCV, identical 81 splits, algorithm = {algo.upper()} for every predictor set; all scaling/selection inside folds]")
        J[algo] = {}
        d = metrics(y, P['null']); out.append(row('null (training-fold mean)', d)); J[algo]['null'] = d
        for nm, cols in sets.items():
            X = df[cols].values.astype(float)
            P[(algo, nm)] = loocv(X, y, algo)
            d = metrics(y, P[(algo, nm)]); J[algo][nm] = d; out.append(row(nm, d))
        out.append("  Paired comparisons (bootstrap 95% CI; Wilcoxon on absolute errors):")
        for a, b in [('demographics', 'demographics+dance'), ('demographics', 'demographics+physical'),
                     ('demographics+physical', 'demographics+physical+dance'), ('demographics+dance', 'demographics+physical+dance')]:
            dm, dmci, dr, drci, w = paired(y, P[(algo, a)], P[(algo, b)])
            out.append(f"    {b}  vs  {a}: dMAE {dm:+.2f} ({dmci[0]:+.2f} to {dmci[1]:+.2f}), dR2 {dr:+.3f} ({drci[0]:+.3f} to {drci[1]:+.3f}), Wilcoxon P={w:.3f}")

    # permutation test for the dance-only model (does dance carry any signal on its own?)
    X = df[dance].values.astype(float); rng = np.random.RandomState(3); obs = r2_score(y, P[('ridge', 'dance only')])
    null = [r2_score(yp, loocv(X, yp, 'ridge')) for yp in (rng.permutation(y) for _ in range(N_PERM))]
    pp = (1 + np.sum(np.array(null) >= obs)) / (1 + N_PERM)
    out.append(f"\n[PERMUTATION] ridge, dance-only, {N_PERM} permutations: observed R2 {obs:.3f}, null 95th pct {np.percentile(null,95):.3f}, P={pp:.3f}")
    J['perm_dance_only'] = dict(obs=obs, p=pp)

    # features selected inside the LOOCV folds (ridge sets with >15 predictors) -> Multimedia Appendix 2
    rows = []
    for nm in ['dance only', 'demographics+dance', 'demographics+physical+dance']:
        cols = sets[nm]; X = df[cols].values.astype(float); cnt = np.zeros(len(cols), int)
        for i in range(len(y)):
            tr = np.arange(len(y)) != i
            sel = SelectKBest(f_regression, k=15).fit(StandardScaler().fit_transform(X[tr]), y[tr])
            cnt += sel.get_support()
        for c, k in zip(cols, cnt):
            if k:
                rows.append(dict(predictor_set=nm, feature=c, folds_selected=int(k), of_folds=len(y)))
    sel = pd.DataFrame(rows).sort_values(['predictor_set', 'folds_selected'], ascending=[True, False])
    sel.to_csv(os.path.join(OUT, 'cognitive_selected_features.csv'), index=False)
    out.append("\n[FEATURES SELECTED IN >= 41 OF 81 FOLDS] (ridge; top 15 by F-score within each training fold)")
    for nm, g in sel.groupby('predictor_set', sort=False):
        out.append(f"  {nm}: " + ", ".join(f"{r.feature} ({r.folds_selected})" for r in g.itertuples() if r.folds_selected >= 41))
    figure_pred_vs_actual(y, P)
    out.append(f"\nruntime {time.time()-t0:.0f}s")
    write('cognitive_results.txt', "\n".join(out) + "\n")
    json.dump(J, open(os.path.join(OUT, 'cognitive_results.json'), 'w'), default=lambda o: list(o) if isinstance(o, tuple) else float(o), indent=1)


def figure_pred_vs_actual(y, P):
    """LOOCV predicted vs observed MoCA for the main ridge predictor sets (Multimedia Appendix 2)."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    panels = [('demographics', 'Sociodemographic'), ('demographics+physical', 'Sociodemographic + physical'),
              ('demographics+dance', 'Sociodemographic + dance'), ('demographics+physical+dance', 'All predictors')]
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
    for ax, (key, title) in zip(axes.ravel(), panels):
        p = P[('ridge', key)]
        ax.scatter(y, p, s=18, alpha=.7)
        ax.plot([10, 33], [10, 33], ':', color='grey')
        ax.set_title(f'{title}\nR² {r2_score(y, p):.2f}, MAE {mean_absolute_error(y, p):.2f}', fontsize=10)
        ax.set_xlim(10, 33); ax.set_ylim(10, 33)
    for ax in axes[1]: ax.set_xlabel('Observed MoCA')
    for ax in axes[:, 0]: ax.set_ylabel('Predicted MoCA (LOOCV)')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'cognitive_pred_vs_actual.png'), dpi=300); plt.close(fig)


if __name__ == '__main__':
    main()
