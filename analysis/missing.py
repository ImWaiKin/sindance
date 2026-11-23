"""Missing-data analysis: Little's MCAR test, missing vs complete comparison,
multiple imputation (m=20) sensitivity analysis for the balance model."""
import sys, json, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from common import *
import balance as B
import balance_labels as BL
warnings.filterwarnings('ignore')
M = 20 if '--quick' not in sys.argv else 3


def em_mvn(X, iters=500, tol=1e-8):
    n, p = X.shape; mu = np.nanmean(X, 0); S = np.diag(np.nanvar(X, 0))
    for _ in range(iters):
        T1 = np.zeros(p); T2 = np.zeros((p, p))
        for i in range(n):
            o = ~np.isnan(X[i]); m = ~o; x = X[i].copy(); C = np.zeros((p, p))
            if m.any():
                Soo_inv = np.linalg.inv(S[np.ix_(o, o)])
                x[m] = mu[m] + S[np.ix_(m, o)] @ Soo_inv @ (X[i, o] - mu[o])
                C[np.ix_(m, m)] = S[np.ix_(m, m)] - S[np.ix_(m, o)] @ Soo_inv @ S[np.ix_(o, m)]
            T1 += x; T2 += np.outer(x, x) + C
        mu_new = T1 / n; S_new = T2 / n - np.outer(mu_new, mu_new)
        if np.max(np.abs(mu_new - mu)) < tol and np.max(np.abs(S_new - S)) < tol:
            mu, S = mu_new, S_new; break
        mu, S = mu_new, S_new
    return mu, S


def little_mcar(df):
    X = df.values.astype(float); n, p = X.shape
    Xs = (X - np.nanmean(X, 0)) / np.nanstd(X, 0)  # scale for numerical stability (test is invariant)
    mu, S = em_mvn(Xs)
    pats = pd.Series([tuple(r) for r in np.isnan(Xs)])
    d2 = 0.0; df_ = 0
    for pat, idx in pats.groupby(pats).groups.items():
        o = ~np.array(pat); nj = len(idx)
        diff = np.nanmean(Xs[list(idx)][:, o], 0) - mu[o]
        d2 += nj * diff @ np.linalg.inv(S[np.ix_(o, o)]) @ diff; df_ += o.sum()
    df_ -= p
    return d2, df_, stats.chi2.sf(d2, df_), pats.value_counts()


def rubin(est, var):
    est, var = np.asarray(est), np.asarray(var); m = len(est)
    qb = est.mean(); W = var.mean(); Bv = est.var(ddof=1); T = W + (1 + 1 / m) * Bv
    r = (1 + 1 / m) * Bv / W if W > 0 else np.inf
    dfr = (m - 1) * (1 + 1 / r) ** 2 if r > 0 else np.inf
    t = stats.t.ppf(.975, dfr); return qb, qb - t * np.sqrt(T), qb + t * np.sqrt(T)


def main():
    out = []
    clin = load_clinical()
    comp = BL.composite().rename(columns={'composite': 'composite_risk_score'})  # 77 participants with platform data
    d = clin.merge(comp, on='Subject_ID', how='left')
    vars_ = ['Age', 'Gender_M', 'Education_ord', 'Weight', 'Height', 'FRT', 'TUG', 'FTSTS', 'Handgrip R', 'Handgrip L', 'MoCA', 'composite_risk_score']
    miss = d[vars_].isna().sum(); out.append("MISSING DATA (n=81)\n" + miss[miss > 0].to_string() + "\n")
    out.append(f"Complete cases on all variables: {int(d[vars_].notna().all(1).sum())}/81\n")
    ids = d.loc[d['composite_risk_score'].isna(), 'Subject_ID'].tolist()
    out.append(f"Participants without HUR platform data: {ids}\n")

    # Primary test, as reported in the manuscript: age, education, sex, FRT, TUG, FTSTS, left and right
    # handgrip strength, MoCA and balance group (1 = higher composite sway, 0 = lower; missing without platform data)
    d['balance_group'] = (d['composite_risk_score'] > .33).astype(float).where(d['composite_risk_score'].notna())
    mcar_vars = ['Age', 'Education_ord', 'Gender_M', 'FRT', 'TUG', 'FTSTS', 'Handgrip L', 'Handgrip R', 'MoCA', 'balance_group']
    d2, df_, p, pats = little_mcar(d[mcar_vars])
    out.append(f"[LITTLE'S MCAR TEST - MANUSCRIPT] variables: {', '.join(mcar_vars)}\n  chi2 = {d2:.2f}, df = {df_}, P = {p:.3f}\n"
               f"  Missingness patterns (counts): {len(pats)} patterns; {pats.values.tolist()}\n")
    # Sensitivity: all recorded variables, including weight and height (not used in any model) and the continuous composite
    d2b, dfb, pb, patsb = little_mcar(d[vars_])
    out.append(f"  Sensitivity, all variables ({', '.join(vars_)}): chi2 = {d2b:.2f}, df = {dfb}, P = {pb:.3f}; "
               f"{len(patsb)} missingness patterns\n")

    out.append("[MISSING vs COMPLETE HUR DATA] (Mann-Whitney U; n=4 vs 77, descriptive only - very low power)")
    g = d['composite_risk_score'].isna()
    for v in ['Age', 'FRT', 'TUG', 'FTSTS', 'Handgrip R', 'MoCA']:
        a, b = d.loc[g, v], d.loc[~g, v]
        out.append(f"  {v:<11} missing median {a.median():.1f} vs complete median {b.median():.1f}; P={stats.mannwhitneyu(a, b).pvalue:.3f}")
    out.append(f"  Female: missing {int(d.loc[g,'Gender_M'].eq(0).sum())}/4 vs complete {int(d.loc[~g,'Gender_M'].eq(0).sum())}/77\n")

    # --- multiple imputation of the balance composite -> balance label, re-run classifier on all 81
    feats = load_features(); fe = engineer_balance_features(feats.drop(columns='Subject_ID'))
    Xd = fe.values.astype(float); order = feats['Subject_ID']
    dd = d.set_index('Subject_ID').loc[order].reset_index()
    imp_vars = ['Age', 'Gender_M', 'Education_ord', 'FRT', 'TUG', 'FTSTS', 'Handgrip R', 'Handgrip L', 'MoCA', 'Weight', 'Height', 'composite_risk_score']
    accs, avars, aucs, aucvars, nelev = [], [], [], [], []
    for m in range(M):
        imp = IterativeImputer(estimator=BayesianRidge(), sample_posterior=True, max_iter=20, random_state=m)
        Z = pd.DataFrame(imp.fit_transform(dd[imp_vars]), columns=imp_vars)
        yy = (Z['composite_risk_score'].clip(0, 1) <= .33).astype(int).values  # 1 = Low, 0 = Elevated
        pr = B.loocv(Xd, yy, aug='smote'); _, r = B.summarise('', yy, pr)
        n = len(yy); accs.append(r['acc']); avars.append(r['acc'] * (1 - r['acc']) / n)
        a = r['auc']; n1 = int((yy == 0).sum()); n2 = n - n1; q1 = a / (2 - a); q2 = 2 * a * a / (1 + a)
        aucs.append(a); aucvars.append((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n2 - 1) * (q2 - a * a)) / (n1 * n2)); nelev.append(n1)
    acc = rubin(accs, avars); auc = rubin(aucs, aucvars)
    out.append(f"[MULTIPLE IMPUTATION] m={M} chained-equation imputations (Bayesian ridge, posterior draws) of the 4 missing composite scores,\n"
               f"  then 0.33 cut-point, then the full LOOCV SMOTE ensemble on n=81; pooled with Rubin's rules.\n"
               f"  Elevated per imputation: {min(nelev)}-{max(nelev)} of 81\n"
               f"  Pooled accuracy {acc[0]:.3f} (95% CI {acc[1]:.3f}-{acc[2]:.3f}); pooled AUC {auc[0]:.3f} (95% CI {auc[1]:.3f}-{auc[2]:.3f})\n"
               f"  Weight/Height ({int(miss.get('Weight',0))}/{int(miss.get('Height',0))} missing) are not predictors in the re-specified cognitive models; MoCA and all\n"
               f"  cognitive-model predictors are complete, so the cognitive analyses (n=81) involve no imputation.\n")
    write('missing_data_results.txt', "\n".join(out) + "\n")


if __name__ == '__main__':
    main()
