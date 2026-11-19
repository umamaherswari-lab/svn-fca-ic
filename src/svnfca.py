"""Clean re-implementation of the SVN-FCA-IC pipeline found in the ChatGPT notebook.
All preprocessing parameters are learned on training data only."""
import numpy as np, pandas as pd
from sklearn.datasets import load_breast_cancer

CATS = ["low", "medium", "high"]


def load():
    d = load_breast_cancer()
    X = pd.DataFrame(d.data, columns=d.feature_names)
    y = pd.Series(np.where(d.target == 0, "M", "B"), name="diagnosis")
    return X, y


# ---------- crisp (tertile) context ----------
def fit_tertile_edges(Xtr):
    return np.quantile(Xtr, [1 / 3, 2 / 3], axis=0).T  # (p,2)


def crisp_context(X, edges):
    n, p = X.shape
    C = np.zeros((n, 3 * p), bool)
    for j in range(p):
        q1, q2 = edges[j]
        cat = np.where(X[:, j] <= q1, 0, np.where(X[:, j] <= q2, 1, 2))
        C[np.arange(n), 3 * j + cat] = True
    return C


# ---------- neutrosophic (SVNS) layer ----------
def fit_knots(Xtr):
    return np.quantile(Xtr, [0.2, 0.5, 0.8], axis=0).T  # (p,3)


def memberships(X, knots):
    n, p = X.shape
    T = np.zeros((n, p, 3))
    for j in range(p):
        a, b, c = knots[j]
        x = X[:, j]
        T[:, j, 0] = np.clip((b - x) / (b - a), 0, 1)
        left = np.clip((x - a) / (b - a), 0, 1)
        right = np.clip((c - x) / (c - b), 0, 1)
        T[:, j, 1] = np.minimum(left, right)
        T[:, j, 2] = np.clip((x - b) / (c - b), 0, 1)
    return T


def indeterminacy(T, kind="max"):
    if kind == "max":
        return 1.0 - T.max(axis=2)
    if kind == "entropy":
        with np.errstate(divide="ignore", invalid="ignore"):
            h = -np.where(T > 0, T * np.log(T), 0.0).sum(axis=2)
        return h / np.log(3)
    if kind == "none":
        return np.zeros(T.shape[:2])
    raise ValueError(kind)


def evidence_matrix(X, knots, ikind="max"):
    T = memberships(X, knots)
    I = indeterminacy(T, ikind)
    E = T * (1.0 - I[:, :, None])
    n, p, _ = E.shape
    return E.reshape(n, 3 * p), T, I


# ---------- rule mining (Apriori-style, identical rule set to brute force) ----------
def mine_rules(C, y, min_extent=10, min_purity=0.85, max_len=5, closed_only=False):
    n, m = C.shape
    yM = (np.asarray(y) == "M")
    feat = np.arange(m) // 3
    out = []
    level = [((j,), C[:, j]) for j in range(m) if C[:, j].sum() >= min_extent]
    k = 1
    while level and k <= max_len:
        for items, ext in level:
            size = int(ext.sum())
            nM = int((ext & yM).sum())
            nB = size - nM
            maj = "M" if nM > nB else "B"
            pur = max(nM, nB) / size
            if pur >= min_purity:
                closed = int(C[ext].all(axis=0).sum()) == len(items)
                if closed or not closed_only:
                    out.append(dict(items=items, cls=maj, extent=size,
                                    support=size / n, purity=pur, closed=closed,
                                    ext=ext.copy()))
        if k == max_len:
            break
        nxt = []
        for items, ext in level:
            last = items[-1]
            for j in range(last + 1, m):
                if feat[j] <= feat[last]:
                    continue
                e2 = ext & C[:, j]
                if e2.sum() >= min_extent:
                    nxt.append((items + (j,), e2))
        level = nxt
        k += 1
    return out


class RuleBase:
    def __init__(self, rules, m):
        self.rules = rules
        self.m = m
        r = len(rules)
        self.R = np.zeros((r, m))
        for i, ru in enumerate(rules):
            self.R[i, list(ru["items"])] = 1
        self.len = self.R.sum(1)
        self.isM = np.array([ru["cls"] == "M" for ru in rules], bool)
        self.pur = np.array([ru["purity"] for ru in rules])

    def __len__(self):
        return len(self.rules)

    def soft_scores(self, E):
        """geometric mean of per-condition evidence; 0 if any condition has no evidence."""
        pos = E > 1e-12
        LE = np.where(pos, np.log(np.where(pos, E, 1.0)), 0.0)
        Z = (~pos).astype(float)
        nz = Z @ self.R.T
        sl = LE @ self.R.T
        return np.where(nz == 0, np.exp(sl / self.len), 0.0)

    def crisp_scores(self, C):
        return ((C.astype(float) @ self.R.T) == self.len).astype(float)

    def aggregate(self, S):
        W = S * self.pur
        b = W[:, ~self.isM].sum(1)
        mm = W[:, self.isM].sum(1)
        return b, mm


def decide(b, m):
    tot = b + m
    pred = np.where(b > m, "B", "M").astype(object)
    pred[(tot == 0) | np.isclose(b, m)] = "abstain"
    margin = np.where(tot > 0, (b - m) / np.where(tot > 0, tot, 1), 0.0)
    return pred, margin


def run_svn(rb, Xte, Cte, knots, mode="svn", ikind="max"):
    if mode == "crisp":
        S = rb.crisp_scores(Cte)
    else:
        E, _, _ = evidence_matrix(Xte, knots, "none" if mode == "T" else ikind)
        S = rb.soft_scores(E)
    b, m = rb.aggregate(S)
    pred, margin = decide(b, m)
    return pred, margin, b, m


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred, dtype=object)
    acc_mask = y_pred != "abstain"
    cov = acc_mask.mean()
    yt, yp = y_true[acc_mask], y_pred[acc_mask]
    if len(yt) == 0:
        return dict(coverage=cov)
    tp = int(((yt == "M") & (yp == "M")).sum())
    fn = int(((yt == "M") & (yp == "B")).sum())
    fp = int(((yt == "B") & (yp == "M")).sum())
    tn = int(((yt == "B") & (yp == "B")).sum())
    rec = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(coverage=cov, accuracy=(tp + tn) / len(yt), balanced_accuracy=(rec + spec) / 2,
                precision_M=prec, recall_M=rec, specificity=spec, f1_M=f1,
                tp=tp, fn=fn, fp=fp, tn=tn, n_accepted=len(yt))

