"""Balance classification re-analysis: reproduction, no-augmentation baseline,
nested CV, permutation test, calibration, repeated 5-fold, label sensitivity."""
import sys, time, json, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, brier_score_loss, r2_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.isotonic import IsotonicRegression
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from scipy import stats
from common import *
import balance_labels as BL
warnings.filterwarnings('ignore')

QUICK = '--quick' in sys.argv
N_PERM = 50 if QUICK else 500
RS = 42; NF = 30; NJ = -1


def mixup(Xtr, ytr, n_mix=30, alpha=0.2, seed=42):
    rng = np.random.RandomState(seed); mino = np.argmin(np.bincount(ytr))
    idx = np.where(ytr == mino)[0]; ax = []
    for _ in range(n_mix):
        i1, i2 = rng.choice(idx, 2, replace=True); lam = rng.beta(alpha, alpha)
        ax.append(lam * Xtr[i1] + (1 - lam) * Xtr[i2])
    return np.vstack([Xtr, ax]), np.concatenate([ytr, np.full(n_mix, mino)])


def prep(Xtr, Xte, ytr, aug):
    sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    vm = Xtr.std(0) > 0; Xtr, Xte = Xtr[:, vm], Xte[:, vm]
    sel = SelectKBest(f_classif, k=min(NF, Xtr.shape[1])).fit(Xtr, ytr)
    Xtr, Xte = sel.transform(Xtr), sel.transform(Xte)
    if aug == 'smote':
        Xtr, ytr = SMOTE(random_state=RS).fit_resample(Xtr, ytr)
    elif aug == 'mixup':
        Xtr, ytr = mixup(Xtr, ytr)
    return Xtr, Xte, ytr


def fit_predict(Xtr, Xte, ytr, aug='smote', C=50, which='ens'):
    """Returns P(class 0 = Elevated) for Xte, as in the notebook's ensemble."""
    Xtr, Xte, ytr = prep(Xtr, Xte, ytr, aug)
    brf = BalancedRandomForestClassifier(n_estimators=200, random_state=RS, n_jobs=1).fit(Xtr, ytr)
    pb = brf.predict_proba(Xte)[:, 0]
    if which == 'brf':
        return pb
    svm = SVC(kernel='rbf', C=C, gamma='scale', class_weight='balanced', random_state=RS,
              probability=True).fit(Xtr, ytr)
    return 0.5 * pb + 0.5 * svm.predict_proba(Xte)[:, 0]


def loocv(X, y, **kw):
    n = len(y)
    p = Parallel(n_jobs=NJ)(delayed(fit_predict)(X[np.arange(n) != i], X[[i]], y[np.arange(n) != i], **kw)
                            for i in range(n))
    return np.array([q[0] for q in p])


def summarise(name, y, p_elev):
    """y: 0=Elevated,1=Low (alphabetical, as notebook). prediction = argmax -> Elevated if p>=0.5."""
    pred = np.where(p_elev >= 0.5, 0, 1)  # argmax ties -> class 0 in notebook; ties essentially never occur
    n = len(y); k = int((pred == y).sum())
    tp = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 0) & (y == 1)).sum())
    yb = (y == 0).astype(int); auc = roc_auc_score(yb, p_elev)
    alo, ahi = boot_ci(lambda a, b: roc_auc_score(a, b), yb, p_elev, stratify=yb)
    txt = (f"{name}\n  Accuracy    {fmt_prop(k, n)}\n"
           f"  Sensitivity {fmt_prop(tp, tp + fn)}  (Elevated = positive)\n"
           f"  Specificity {fmt_prop(tn, tn + fp)}\n"
           f"  AUC {auc:.3f} (95% CI {alo:.3f}-{ahi:.3f}, stratified bootstrap)\n"
           f"  Confusion [rows=true Elevated/Low; cols=pred Elevated/Low]: [[{tp},{fn}],[{fp},{tn}]]\n")
    return txt, dict(acc=k / n, k=k, n=n, sens=tp / (tp + fn), spec=tn / (tn + fp), auc=auc, auc_ci=[alo, ahi])


def mcnemar_exact(y, p1, p2):
    c1 = (np.where(p1 >= .5, 0, 1) == y); c2 = (np.where(p2 >= .5, 0, 1) == y)
    b = int((c1 & ~c2).sum()); c = int((~c1 & c2).sum())
    return b, c, stats.binomtest(b, b + c, .5).pvalue if b + c else 1.0


def nested_one(i, X, y, configs):
    tr = np.arange(len(y)) != i; Xtr, ytr = X[tr], y[tr]
    skf = StratifiedKFold(5, shuffle=True, random_state=1000 + i)
    scores = []
    for cfg in configs:
        acc = []
        for a, b in skf.split(Xtr, ytr):
            p = fit_predict(Xtr[a], Xtr[b], ytr[a], **cfg)
            acc.append(accuracy_score(ytr[b], np.where(p >= .5, 0, 1)))
        scores.append(np.mean(acc))
    best = configs[int(np.argmax(scores))]
    return fit_predict(Xtr, X[[i]], ytr, **best)[0], best


def main():
    t0 = time.time(); out = []; J = {}
    feats = load_features()
    lab = BL.labels()[['Subject_ID', 'Balance_Category']]
    df = feats.merge(lab, on='Subject_ID')
    fcols = [c for c in df.columns if c not in ('Subject_ID', 'Balance_Category')]
    Xe = engineer_balance_features(df[fcols]); X = Xe.values.astype(float)
    y = (df['Balance_Category'] == 'Low').astype(int).values  # Elevated=0, Low=1 (LabelEncoder order)
    out.append(f"BALANCE RE-ANALYSIS  n={len(y)} (Elevated={int((y==0).sum())}, Low={int((y==1).sum())}); "
               f"{len(fcols)} base -> {X.shape[1]} engineered features\n"
               f"Majority-class (all 'Low') accuracy: {fmt_prop(int((y==1).sum()), len(y))}\n")

    # 1. reproduction + no-augmentation + mixup (fixed pipelines, LOOCV)
    P = {}
    for aug in ['smote', 'none', 'mixup']:
        P[aug] = loocv(X, y, aug=aug)
        t, J['loocv_' + aug] = summarise(f"[LOOCV] BRF+SVM ensemble, augmentation={aug}", y, P[aug]); out.append(t)
    P['brf_smote'] = loocv(X, y, aug='smote', which='brf')
    t, J['loocv_brf_smote'] = summarise("[LOOCV] BRF only + SMOTE (the model the original ROC/AUC cell used)", y, P['brf_smote']); out.append(t)
    b, c, pv = mcnemar_exact(y, P['smote'], P['none'])
    out.append(f"McNemar SMOTE vs no-augmentation: discordant {b} vs {c}, exact P={pv:.3f}\n")

    # 2. nested CV: augmentation (and SVM C) chosen inside each outer fold
    configs = [dict(aug=a, C=C) for a in ['none', 'smote', 'mixup'] for C in [1, 10, 50]]
    res = Parallel(n_jobs=NJ)(delayed(nested_one)(i, X, y, configs) for i in range(len(y)))
    pn = np.array([r[0] for r in res]); chosen = pd.Series([f"{r[1]['aug']}/C={r[1]['C']}" for r in res]).value_counts()
    t, J['nested'] = summarise("[NESTED LOOCV] augmentation in {none,SMOTE,Mixup} x SVM C in {1,10,50} selected by inner 5-fold CV", y, pn)
    out.append(t + "  Configurations chosen across the 77 outer folds:\n" +
               "\n".join(f"    {k}: {v}" for k, v in chosen.items()) + "\n")

    # 3. permutation test on the fixed SMOTE pipeline
    rng = np.random.RandomState(7); null = []
    for b_ in range(N_PERM):
        yp = rng.permutation(y); pp = loocv(X, yp, aug='smote')
        null.append(accuracy_score(yp, np.where(pp >= .5, 0, 1)))
    null = np.array(null); obs = J['loocv_smote']['acc']
    pperm = (1 + (null >= obs).sum()) / (1 + N_PERM)
    out.append(f"[PERMUTATION] {N_PERM} label permutations, full LOOCV SMOTE pipeline re-run each time\n"
               f"  observed accuracy {obs:.3f}; null mean {null.mean():.3f} (95th pct {np.percentile(null,95):.3f}); P = {pperm:.3f}\n")
    J['perm'] = dict(B=N_PERM, p=pperm, null_mean=float(null.mean()), null_95=float(np.percentile(null, 95)))

    # 4. calibration of held-out (LOOCV) probabilities
    yb = (y == 0).astype(int); ps = np.clip(P['smote'], 1e-3, 1 - 1e-3); lg = np.log(ps / (1 - ps))
    slope_m = LogisticRegression(C=1e6).fit(lg[:, None], yb); slope = slope_m.coef_[0, 0]
    brier = brier_score_loss(yb, ps); brier0 = brier_score_loss(yb, np.full_like(ps, yb.mean()))
    slo, shi = boot_ci(lambda a, b: LogisticRegression(C=1e6).fit(np.log(b / (1 - b))[:, None], a).coef_[0, 0], yb, ps)
    bins = pd.qcut(ps, 4, duplicates='drop'); rel = pd.DataFrame({'p': ps, 'y': yb, 'bin': bins}).groupby('bin').agg(n=('y', 'size'), mean_pred=('p', 'mean'), obs=('y', 'mean'))
    out.append(f"[CALIBRATION, LOOCV held-out probabilities, SMOTE ensemble, no recalibration]\n"
               f"  Brier {brier:.3f} (prevalence-only reference {brier0:.3f}); mean predicted {ps.mean():.3f} vs observed {yb.mean():.3f}\n"
               f"  Calibration slope {slope:.2f} (95% CI {slo:.2f}-{shi:.2f}); ideal = 1\n"
               f"  Reliability by quartile of predicted risk:\n{rel.round(3).to_string()}\n")
    J['calib'] = dict(brier=brier, brier_ref=brier0, slope=slope, slope_ci=[slo, shi])

    # 4b. what the original two-stage isotonic cell actually used
    Xp, Xt, yp_, yt = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    Xf, Xc, yf, yc = train_test_split(Xp, yp_, test_size=.2, random_state=42, stratify=yp_)
    out.append(f"  Original isotonic-calibration cell: fit n={len(yf)}, calibration n={len(yc)} "
               f"(Elevated {int((yc==0).sum())}), test n={len(yt)} -- a single split, not LOOCV.\n")

    # 5. repeated stratified 5-fold
    def kfold_acc(aug, seed):
        accs = []
        for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            accs.append(accuracy_score(y[b], np.where(fit_predict(X[a], X[b], y[a], aug=aug) >= .5, 0, 1)))
        return np.mean(accs), np.std(accs)
    for aug in ['smote', 'none']:
        r42 = kfold_acc(aug, 42)
        reps = Parallel(n_jobs=NJ)(delayed(kfold_acc)(aug, s) for s in range(20))
        m = np.array([r[0] for r in reps])
        out.append(f"[5-FOLD] aug={aug}: original split (seed 42) {r42[0]:.3f} +/- {r42[1]:.3f} across folds; "
                   f"20 repeats mean {m.mean():.3f} (range {m.min():.3f}-{m.max():.3f})\n")
        J['kfold_' + aug] = dict(seed42=r42, rep_mean=m.mean(), rep_min=m.min(), rep_max=m.max())

    # 6. label construction & sensitivity analyses
    comp = BL.composite().set_index('Subject_ID').loc[df['Subject_ID'], 'composite'].values
    perf = pd.read_csv(HUR)
    ratio = perf['Trace Length["]'] / perf['Velocity["/s]']
    r_tv = np.corrcoef(perf['Trace Length["]'], perf['Velocity["/s]'])[0, 1]
    out.append(f"[LABELS] Elevated (higher composite sway) n={int((y==0).sum())}; no participant between "
               f"{comp[comp<=BL.CUT].max():.3f} and {comp[comp>BL.CUT].min():.3f}.\n"
               f"  Trace Length / Mean Velocity ratio: median {ratio.median():.2f} (IQR {ratio.quantile(.25):.2f}-{ratio.quantile(.75):.2f}); "
               f"r = {r_tv:.4f} -> the two metrics are the same quantity (30-s trial), jointly 55% of the weight.\n"
               f"  Composite score: median {np.median(comp):.3f}, IQR {np.percentile(comp,25):.3f}-{np.percentile(comp,75):.3f}\n")
    w3 = {'Trace Length["]': .30 / .75, 'C90 Area["^2]': .25 / .75, 'STD Velocity["/s]': .20 / .75}
    comp_nov = BL.composite(metric_weights=w3).set_index('Subject_ID').loc[df['Subject_ID'], 'composite'].values
    out.append(f"  Composite without the redundant Mean Velocity term: Spearman rho with original = {stats.spearmanr(comp, comp_nov)[0]:.3f}\n")

    clin = load_clinical().set_index('Subject_ID').loc[df['Subject_ID']]
    for v in ['FRT', 'TUG', 'FTSTS', 'Handgrip R', 'MoCA', 'Age']:
        r, p = stats.spearmanr(comp, clin[v]); out.append(f"  Spearman composite vs {v}: rho={r:.2f}, P={p:.3f}\n")

    sens_rows = []
    cuts = [('0.25', .25), ('0.33 (original)', .33), ('0.40', .40), ('0.50', .50), ('median', float(np.median(comp)))]
    for nm, cut in cuts:
        yy = (comp <= cut).astype(int)
        if min(np.bincount(yy)) < 8:
            continue
        for aug in ['none', 'smote']:
            pp = loocv(X, yy, aug=aug); _, r = summarise('', yy, pp)
            sens_rows.append(dict(cut=nm, n_elevated=int((yy == 0).sum()), aug=aug, majority=max(np.bincount(yy)) / len(yy),
                                  acc=r['acc'], auc=r['auc'], auc_lo=r['auc_ci'][0], auc_hi=r['auc_ci'][1]))
    sens = pd.DataFrame(sens_rows)
    out.append("[CUT-POINT SENSITIVITY] (LOOCV, same pipeline, duplicates excluded from normalisation)\n" + sens.round(3).to_string(index=False) + "\n")

    # continuous composite as outcome
    def reg_fold(i):
        tr = np.arange(len(comp)) != i
        sc = StandardScaler().fit(X[tr]); A, B = sc.transform(X[tr]), sc.transform(X[[i]])
        vm = A.std(0) > 0; A, B = A[:, vm], B[:, vm]
        s = SelectKBest(f_regression, k=NF).fit(A, comp[tr])
        return RidgeCV(alphas=np.logspace(-3, 3, 50)).fit(s.transform(A), comp[tr]).predict(s.transform(B))[0]
    pr = np.array(Parallel(n_jobs=NJ)(delayed(reg_fold)(i) for i in range(len(comp))))
    r2 = r2_score(comp, pr); rho = stats.spearmanr(comp, pr)[0]
    r2lo, r2hi = boot_ci(r2_score, comp, pr)
    out.append(f"[CONTINUOUS COMPOSITE] LOOCV ridge (top-30 features in-fold): R2 {r2:.3f} (95% CI {r2lo:.3f}-{r2hi:.3f}); "
               f"Spearman(pred, obs) {rho:.3f}\n")
    J['continuous'] = dict(r2=r2, r2_ci=[r2lo, r2hi], rho=rho)

    figures(y, P['smote'], pn)
    out.append(f"runtime {time.time()-t0:.0f}s\n")
    write('balance_results.txt', "\n".join(out))
    json.dump(J, open(os.path.join(OUT, 'balance_results.json'), 'w'), default=float, indent=1)
    # per-participant predictions are not saved (participant-level data)


def figures(y, p_fixed, p_nested):
    """Figure 5 (ROC of the reported ensemble, LOOCV) and a reliability plot of held-out probabilities."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    yb = (y == 0).astype(int)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for p, lab, ls in [(p_fixed, 'Ensemble + SMOTE (LOOCV)', '-'), (p_nested, 'Nested cross-validation', '--')]:
        f, t, _ = roc_curve(yb, p); a = roc_auc_score(yb, p)
        lo, hi = boot_ci(lambda u, v: roc_auc_score(u, v), yb, p, stratify=yb)
        ax.plot(f, t, ls, lw=2, label=f'{lab}: AUC {a:.2f} (95% CI {lo:.2f}-{hi:.2f})')
    ax.plot([0, 1], [0, 1], ':', color='grey', label='Chance')
    ax.set_xlabel('1 - specificity'); ax.set_ylabel('Sensitivity (higher composite sway)')
    ax.set_title('ROC, balance classification (n=77)'); ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'figure5_roc_balance.png'), dpi=300); plt.close(fig)
    ps = np.clip(p_fixed, 1e-3, 1 - 1e-3)
    q = pd.qcut(ps, 4, labels=False, duplicates='drop')
    m = [(ps[q == k].mean(), yb[q == k].mean(), (q == k).sum()) for k in np.unique(q)]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], ':', color='grey')
    ax.plot([a for a, _, _ in m], [b for _, b, _ in m], 'o-', lw=2)
    for a, b_, n in m:
        lo, hi = cp_ci(int(round(b_ * n)), n); ax.plot([a, a], [lo, hi], color='C0', alpha=.5)
    ax.set_xlabel('Mean predicted probability (quartile)'); ax.set_ylabel('Observed proportion, higher sway (95% CI)')
    ax.set_title('Calibration of held-out probabilities'); fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'calibration_balance.png'), dpi=300); plt.close(fig)


if __name__ == '__main__':
    main()
