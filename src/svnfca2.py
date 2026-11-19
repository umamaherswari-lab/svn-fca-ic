"""SVN-FCA-IC, version 2: class-prior-aware rule filter, top-k class evidence and
tertile-consistent neutrosophic memberships. Every learned quantity (cut points, knots, rules,
filter parameters) is estimated from the training data only; filter parameters are chosen by
inner cross-validation on the training data."""
import math
import numpy as np
from sklearn.model_selection import StratifiedKFold
from svnfca import crisp_context, fit_tertile_edges

EPS = 1e-12


# ---------- neutrosophic memberships consistent with the tertile context ----------
def fit_knots_consistent(Xtr):
    """Knots (a, b, c) such that the crossing points of the memberships are the tertile cut points
    and the peak of 'medium' is the median: (a+b)/2 = q33, b = q50, (b+c)/2 = q67."""
    q = np.quantile(Xtr, [1 / 3, .5, 2 / 3], axis=0).T
    return np.stack([2 * q[:, 0] - q[:, 1], q[:, 1], 2 * q[:, 2] - q[:, 1]], axis=1)


def fit_knots_percentile(Xtr):
    return np.quantile(Xtr, [.2, .5, .8], axis=0).T


def memberships(X, knots):
    n, p = X.shape
    T = np.zeros((n, p, 3))
    for j in range(p):
        a, b, c = knots[j]
        wl, wr = max(b - a, EPS), max(c - b, EPS)
        x = X[:, j]
        T[:, j, 0] = np.clip((b - x) / wl, 0, 1)
        T[:, j, 1] = np.minimum(np.clip((x - a) / wl, 0, 1), np.clip((c - x) / wr, 0, 1))
        T[:, j, 2] = np.clip((x - b) / wr, 0, 1)
    return T


def evidence(X, knots, kind="max"):
    """kind: 'max' (I = 1 - max T), 'none' (I = 0, truth only) or 'entropy' (normalised entropy of the memberships)."""
    T = memberships(X, knots)
    if kind == "max":
        I = 1.0 - T.max(axis=2)
    elif kind == "entropy":
        with np.errstate(divide="ignore", invalid="ignore"):
            I = -np.where(T > 0, T * np.log(T), 0.0).sum(axis=2) / np.log(3)
    else:
        I = np.zeros(T.shape[:2])
    E = T * (1.0 - I[:, :, None])
    return E.reshape(len(X), -1)


# ---------- candidate antecedents ----------
def mine_candidates(C, min_extent, max_len=5):
    """All attribute sets with at most one category per feature, at least min_extent objects, length <= max_len."""
    n, m = C.shape
    feat = np.arange(m) // 3
    out, level, k = [], [((j,), C[:, j]) for j in range(m) if C[:, j].sum() >= min_extent], 1
    while level and k <= max_len:
        out += level
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
        level, k = nxt, k + 1
    return out


class Candidates:
    def __init__(self, C, yM, min_extent, max_len=5):
        cand = mine_candidates(C, min_extent, max_len)
        self.m = C.shape[1]
        self.items = [c[0] for c in cand]
        self.R = np.zeros((len(cand), self.m))
        for i, it in enumerate(self.items):
            self.R[i, list(it)] = 1
        self.len = self.R.sum(1)
        ext = np.array([c[1] for c in cand]) if cand else np.zeros((0, len(C)), bool)
        self.size = ext.sum(1).astype(float)
        self.nM = (ext & yM[None, :]).sum(1).astype(float)
        self.piM = float(yM.mean())
        self._C, self._ext, self._closed = C, ext, None

    @property
    def closed(self):
        if self._closed is None:
            self._closed = np.array([int(self._C[e].all(axis=0).sum()) == len(it) for it, e in zip(self.items, self._ext)], bool) if len(self.items) else np.zeros(0, bool)
        return self._closed

    def select(self, delta, relative=True):
        """Return (mask, isM, weight). Laplace-smoothed class purity; relative = excess over the class prior."""
        pM = (self.nM + 1) / (self.size + 2)
        pB = 1 - pM
        if relative:
            rM = (pM - self.piM) / max(1 - self.piM, EPS)
            rB = (pB - (1 - self.piM)) / max(self.piM, EPS)
        else:
            rM, rB = pM, pB
        isM = rM >= rB
        q = np.where(isM, rM, rB)
        return q >= delta, isM, q

    def soft(self, E):
        pos = E > 1e-12
        LE = np.where(pos, np.log(np.where(pos, E, 1.0)), 0.0)
        nz = (~pos).astype(float) @ self.R.T
        return np.where(nz == 0, np.exp(LE @ self.R.T / self.len), 0.0)

    def crisp(self, C):
        return ((C.astype(float) @ self.R.T) == self.len).astype(float)


def aggregate(S, mask, isM, w, k, agg):
    """Class evidence. 'topk': mean of the k strongest weighted rule scores; 'sum': sum of all."""
    W = S * w[None, :]
    out, used = [], []
    for cls_mask in (mask & ~isM, mask & isM):
        Wc = W[:, cls_mask]
        if Wc.shape[1] == 0:
            out.append(np.zeros(len(S))); used.append(np.zeros(len(S)))
        elif agg == "sum":
            out.append(Wc.sum(1)); used.append((Wc > 0).sum(1))
        else:
            kk = min(k, Wc.shape[1])
            top = -np.sort(-Wc, axis=1)[:, :kk]
            out.append(top.sum(1) / k); used.append(np.minimum((Wc > 0).sum(1), k))
    return out[0], out[1], used[0], used[1]  # evidence B, evidence M, rules used B, rules used M


def decide(b, m):
    tot = b + m
    pred = np.where(b > m, "B", "M").astype(object)
    pred[(tot == 0) | np.isclose(b, m)] = "abstain"
    margin = np.where(tot > 0, (b - m) / np.where(tot > 0, tot, 1), 0.0)
    return pred, margin


def bal_acc_abstain_wrong(y, pred):
    yM = np.asarray(y) == "M"
    rM = ((pred == "M") & yM).sum() / max(yM.sum(), 1)
    rB = ((pred == "B") & ~yM).sum() / max((~yM).sum(), 1)
    return (rM + rB) / 2


def min_extent_for(n, frac=0.02):
    return max(5, math.ceil(frac * n))


DELTAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
KS = [1, 3, 5, 10]
THETAS = [0.60, 0.70, 0.80, 0.85, 0.90, 0.95]


def select_params(A, ya, relative=True, agg="topk", n_inner=5, seed=0, knots="consistent", min_frac=0.02, max_len=5):
    grid_d = DELTAS if relative else THETAS
    grid_k = KS if agg == "topk" else [1]
    sc = np.zeros((len(grid_d), len(grid_k)))
    fk = fit_knots_consistent if knots == "consistent" else fit_knots_percentile
    for tr, va in StratifiedKFold(n_inner, shuffle=True, random_state=seed).split(A, ya):
        e, kn = fit_tertile_edges(A[tr]), fk(A[tr])
        C = crisp_context(A[tr], e)
        cd = Candidates(C, ya[tr] == "M", min_extent_for(len(tr), min_frac), max_len)
        if len(cd.items) == 0:
            continue
        S = cd.soft(evidence(A[va], kn, "max"))
        for di, d in enumerate(grid_d):
            mask, isM, w = cd.select(d, relative)
            for ki, k in enumerate(grid_k):
                b, m, _, _ = aggregate(S, mask, isM, w, k, agg)
                pred, _ = decide(b, m)
                sc[di, ki] += bal_acc_abstain_wrong(ya[va], pred) / n_inner
    best = sc.max()
    ok = np.argwhere(sc >= best - 0.005)
    di, ki = max(ok.tolist(), key=lambda t: (t[0], -t[1]))  # parsimony: stricter filter, then smaller k
    return grid_d[di], grid_k[ki], sc


class SVN2:
    def __init__(self, relative=True, agg="topk", knots="consistent", closed_only=False, min_frac=0.02, max_len=5):
        self.relative, self.agg, self.knots, self.closed_only, self.min_frac, self.max_len = relative, agg, knots, closed_only, min_frac, max_len

    def fit(self, A, ya, delta=None, k=None, seed=0):
        if delta is None:
            delta, k, self.grid = select_params(A, ya, self.relative, self.agg, seed=seed, knots=self.knots, min_frac=self.min_frac, max_len=self.max_len)
        self.delta, self.k = delta, k
        self.edges = fit_tertile_edges(A)
        self.kn = (fit_knots_consistent if self.knots == "consistent" else fit_knots_percentile)(A)
        C = crisp_context(A, self.edges)
        self.cd = Candidates(C, ya == "M", min_extent_for(len(A), self.min_frac), self.max_len)
        mask, self.isM, self.w = self.cd.select(delta, self.relative)
        if self.closed_only:
            mask &= self.cd.closed
        self.mask = mask
        self.n_rules = int(mask.sum())
        self.n_rules_M = int((mask & self.isM).sum())
        return self

    def evidence_scores(self, B, mode="svn"):
        if len(self.cd.items) == 0:
            z = np.zeros(len(B)); return z, z, z, z
        if mode == "crisp":
            S = self.cd.crisp(crisp_context(B, self.edges))
        else:
            S = self.cd.soft(evidence(B, self.kn, {"svn": "max", "T": "none", "entropy": "entropy"}[mode]))
        return aggregate(S, self.mask, self.isM, self.w, self.k, self.agg)

    def predict(self, B, mode="svn"):
        b, m, ub, um = self.evidence_scores(B, mode)
        self.last_used = np.where(b > m, ub, um)
        pred, margin = decide(b, m)
        tot = b + m
        score = np.where(tot > 0, (m - b) / np.where(tot > 0, tot, 1), 0.0)
        return pred, margin, score

