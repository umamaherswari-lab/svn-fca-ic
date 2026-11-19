"""Evaluation helpers shared by both notebooks: baselines, one cross-validation fold, statistics."""
import warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import wittgenstein as lw
from svnfca import fit_tertile_edges, fit_knots, crisp_context, mine_rules, RuleBase, run_svn, metrics
from svnfca2 import SVN2, bal_acc_abstain_wrong

warnings.filterwarnings("ignore")
MAIN = "SVN-FCA-IC"
_STORE = None


def top_k(A, ya, k=5):
    """Indices of the k features with the largest ANOVA F value (training fold only)."""
    if A.shape[1] <= k:
        return np.arange(A.shape[1])
    F, _ = f_classif(A, ya == "M")
    return np.sort(np.argsort(-np.nan_to_num(F))[:k])


def base_models(n_trees=300):
    sc = lambda m: make_pipeline(StandardScaler(), m)
    return {
        "LR": sc(LogisticRegression(max_iter=5000, random_state=42)),
        "DT (depth 5)": DecisionTreeClassifier(max_depth=5, random_state=42),
        "RF": RandomForestClassifier(n_estimators=n_trees, random_state=42),
        "SVM-RBF": sc(SVC(probability=True, random_state=42)),
        "kNN": sc(KNeighborsClassifier(5)),
        "GNB": GaussianNB(),
        "LR (balanced)": sc(LogisticRegression(max_iter=5000, random_state=42, class_weight="balanced")),
        "DT (depth 5, balanced)": DecisionTreeClassifier(max_depth=5, random_state=42, class_weight="balanced"),
        "RF (balanced)": RandomForestClassifier(n_estimators=n_trees, random_state=42, class_weight="balanced"),
        "SVM-RBF (balanced)": sc(SVC(probability=True, random_state=42, class_weight="balanced")),
    }


def ripper(A, ya, B):
    cn = [f"f{i}" for i in range(A.shape[1])]
    dfa = pd.DataFrame(A, columns=cn); dfa["y"] = (ya == "M")
    r = lw.RIPPER(random_state=0)
    r.fit(dfa, class_feat="y", pos_class=True)
    dfb = pd.DataFrame(B, columns=cn)
    pred = np.where(np.array(r.predict(dfb)), "M", "B").astype(object)
    return pred, np.asarray(r.predict_proba(dfb))[:, 1], len(r.ruleset_)


def _row(yb, pred, score, name, fi, rep, n_rules, feats, extra=None):
    d = metrics(yb, pred)
    d.update(model=name, fold=fi, rep=rep, n_rules=n_rules, feats=feats,
             auc=roc_auc_score(yb == "M", score), bal_all=bal_acc_abstain_wrong(yb, np.asarray(pred, dtype=object)))
    if extra:
        d.update(extra)
    if _STORE is not None:
        _STORE[(name, feats)] = (np.asarray(pred, dtype=object), np.asarray(score, float))
    return d


def one_fold(Xa, yv, tr, va, fi, rep, fixed_cols=None, ablations=False, refs_all=True, v1=True, n_trees=300):
    rows = []
    A_all, B_all, ya, yb = Xa[tr], Xa[va], yv[tr], yv[va]
    cols = np.asarray(fixed_cols) if fixed_cols is not None else top_k(A_all, ya)
    A, B = A_all[:, cols], B_all[:, cols]
    ft = str(len(cols))
    full = SVN2().fit(A, ya, seed=fi)
    p, mg, s = full.predict(B, "svn")
    rows.append(_row(yb, p, s, MAIN, fi, rep, full.n_rules, ft,
                     dict(delta=full.delta, kk=full.k, rules_used=float(np.mean(full.last_used)), n_rules_M=full.n_rules_M, margin_mean=float(np.mean(np.abs(mg))))))
    oof = pd.DataFrame(dict(fold=fi, rep=rep, y=yb, pred=p, margin=mg, score=s))
    cl = SVN2(closed_only=True).fit(A, ya, full.delta, full.k)
    p, _, s = cl.predict(B); rows.append(_row(yb, p, s, "Closed-intent rules", fi, rep, cl.n_rules, ft))
    if ablations:
        for name, mode in [("Crisp matching", "crisp"), ("Truth only (I = 0)", "T"), ("Entropy indeterminacy", "entropy")]:
            p, _, s = full.predict(B, mode); rows.append(_row(yb, p, s, name, fi, rep, full.n_rules, ft))
        for name, mdl in [("Absolute purity filter", SVN2(relative=False)), ("Sum of rule evidence", SVN2(agg="sum")), ("Percentile knots", SVN2(knots="percentile"))]:
            mdl.fit(A, ya, seed=fi); p, _, s = mdl.predict(B)
            rows.append(_row(yb, p, s, name, fi, rep, mdl.n_rules, ft, dict(delta=mdl.delta, kk=mdl.k)))
    if v1:  # original model: fixed purity 0.85, summed evidence, percentile knots
        e, k = fit_tertile_edges(A), fit_knots(A)
        C, Cb = crisp_context(A, e), crisp_context(B, e)
        rb = RuleBase(mine_rules(C, ya, 10, .85, 5), C.shape[1])
        p, mg1, b, m = run_svn(rb, B, Cb, k, "svn")
        tot = b + m
        rows.append(_row(yb, p, np.where(tot > 0, (m - b) / np.where(tot > 0, tot, 1), 0.0), "Fixed 0.85 purity (first version)", fi, rep, len(rb), ft))
    bm = base_models(n_trees)
    for name, mdl in bm.items():
        m_ = clone(mdl).fit(A, ya)
        rows.append(_row(yb, m_.predict(B), m_.predict_proba(B)[:, list(m_.classes_).index("M")], name, fi, rep, 0, ft))
    p, s, nr = ripper(A, ya, B); rows.append(_row(yb, p, s, "RIPPER", fi, rep, nr, ft))
    if refs_all and A_all.shape[1] > len(cols):
        for name in ["LR", "RF", "SVM-RBF", "LR (balanced)", "RF (balanced)", "SVM-RBF (balanced)"]:
            m_ = clone(bm[name]).fit(A_all, ya)
            rows.append(_row(yb, m_.predict(B_all), m_.predict_proba(B_all)[:, list(m_.classes_).index("M")], name, fi, rep, 0, "all"))
    return rows, oof


def run_cv(Xa, yv, n_repeats=10, seed=2024, **kw):
    rows, oofs = [], []
    for fi, (tr, va) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=seed).split(Xa, yv)):
        r, o = one_fold(Xa, yv, tr, va, fi, fi // 5, **kw)
        rows += r; oofs.append(o)
    return pd.DataFrame(rows), pd.concat(oofs, ignore_index=True)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); out = np.empty_like(p); run = 0.0
    for i, j in enumerate(o):
        run = max(run, (len(p) - i) * p[j]); out[j] = min(1.0, run)
    return out


def corrected_t(d, k_ratio=0.25):
    """Corrected resampled t-test (Nadeau & Bengio, 2003) on paired fold differences d."""
    d = np.asarray(d, float); k = len(d); v = d.var(ddof=1)
    if v <= 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / k + k_ratio) * v)
    return t, 2 * stats.t.sf(abs(t), k - 1)


def summarise(cv, ref=MAIN):
    cols = ["accuracy", "balanced_accuracy", "precision_M", "recall_M", "f1_M", "auc", "bal_all"]
    g = cv.groupby(["feats", "model"])
    out = g[cols + ["n_rules", "coverage"]].mean().join(g[["accuracy", "balanced_accuracy", "auc"]].std().add_suffix("_std")).reset_index()
    r = cv[cv.model == ref].sort_values("fold")
    for metric in ["accuracy", "balanced_accuracy", "auc"]:
        ps = []
        for (f, mname), gg in cv.groupby(["feats", "model"]):
            if mname == ref and f == r.feats.iloc[0]:
                ps.append(np.nan); continue
            d = r[metric].values - gg.sort_values("fold")[metric].values
            ps.append(corrected_t(d)[1])
        out["p_" + metric] = ps
        mk = out["p_" + metric].notna().values
        out.loc[mk, "p_" + metric + "_holm"] = holm(out.loc[mk, "p_" + metric].values)
    return out

