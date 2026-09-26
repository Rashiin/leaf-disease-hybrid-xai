"""Reviewer controls, step 2 (Section 3.5, Table 5): analysis of the cached outputs of
run_controls_fit.py.  Usage:  python src/run_controls_analysis.py

References: solo ablation, leave-one-group-out (LOGO), group Shapley.
Fusion objectives: Dirichlet/Macro-F1 (paper), convex NLL, convex Brier.
Attributions: raw alpha; effective (true label, micro); effective (predicted class, micro);
effective (true label, class-balanced); effective (predicted class, class-balanced).
"""
import glob, json, sys, itertools, math
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr, wilcoxon, binomtest
from sklearn.metrics import f1_score

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULT_DIR, load_all
OUT = RESULT_DIR / 'reviewer_controls'
G = ('color', 'texture', 'shape')
train, val, test, _ = load_all()
y = pd.concat([train, val, test], ignore_index=True)['label_id'].values

def f1(yt, yp): return f1_score(yt, yp, average='macro') * 100
def fuse(a, Ps):
    s = sum(ai * P for ai, P in zip(a, Ps)); return s / (s.sum(1, keepdims=True) + 1e-12)

def w_dirichlet(Pv, yv):
    rng = np.random.default_rng(42); best, bs = None, -1
    for _ in range(1000):
        c = rng.dirichlet([1, 1, 1]); s = f1(yv, fuse(c, Pv).argmax(1))
        if s > bs: best, bs = c, s
    return best

def w_smooth(Pv, yv, kind):
    onehot = np.eye(Pv[0].shape[1])[yv]
    def loss(z):
        a = np.exp(z - z.max()); a /= a.sum(); P = sum(ai * p for ai, p in zip(a, Pv))
        if kind == 'nll': return -np.log(P[np.arange(len(yv)), yv] + 1e-12).mean()
        return ((P - onehot) ** 2).sum(1).mean()
    best = None
    for z0 in ([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 1, 1]):
        r = minimize(loss, np.array(z0, float), method='Nelder-Mead', options=dict(xatol=1e-6, fatol=1e-9, maxiter=4000))
        if best is None or r.fun < best.fun: best = r
    a = np.exp(best.x - best.x.max()); return a / a.sum()

def w_loglinear(Pv, yv):
    L = [np.log(p + 1e-9) for p in Pv]
    def loss(z):
        w = np.exp(z); Z = sum(wi * l for wi, l in zip(w, L)); Z = Z - Z.max(1, keepdims=True)
        return -(Z[np.arange(len(yv)), yv] - np.log(np.exp(Z).sum(1))).mean()
    best = None
    for z0 in ([0, 0, 0], [0, -1, -1], [-1, 0, 0]):
        r = minimize(loss, np.array(z0, float), method='Nelder-Mead', options=dict(xatol=1e-6, fatol=1e-10, maxiter=4000))
        if best is None or r.fun < best.fun: best = r
    return np.exp(best.x)

def attributions(a, Pt, yt, yhat):
    idx = np.arange(len(yt)); C = Pt[0].shape[1]
    out = {'raw': np.array(a)}
    for lab, ref in (('true', yt), ('pred', yhat)):
        mic = np.array([ai * P[idx, ref].mean() for ai, P in zip(a, Pt)])
        mac = np.array([ai * np.mean([P[idx, ref][ref == c].mean() for c in range(C) if (ref == c).any()])
                        for ai, P in zip(a, Pt)])
        out[f'eff_{lab}_micro'] = mic / mic.sum(); out[f'eff_{lab}_macro'] = mac / mac.sum()
    return out

def order_ok(v, ref): return bool(np.array_equal(np.argsort(-np.asarray(v)), np.argsort(-np.asarray(ref))))

rows = []
for fn in sorted(glob.glob(str(OUT / 'cache' / 'cache_*.npz'))):
    s = fn.split('cache_')[1][:-4]; d = np.load(fn)
    iva, ite = d['iva'], d['ite']; yv, yt = y[iva], y[ite]
    Pv = [d[f'Pval_{g}'] for g in G]; Pt = [d[f'Pte_{g}'] for g in G]
    v = {(): 0.0}
    for r in (1, 2, 3):
        for c in itertools.combinations(G, r):
            v[c] = f1(yt, d['pred_' + '+'.join(c)])
    solo = np.array([v[(g,)] for g in G])
    full = v[G]
    logo = np.array([full - v[tuple(h for h in G if h != g)] for g in G])
    shap = []
    for g in G:
        others = [h for h in G if h != g]; phi = 0
        for r in range(3):
            for S in itertools.combinations(others, r):
                wgt = math.factorial(r) * math.factorial(2 - r) / 6
                with_g = tuple(h for h in G if h in S or h == g)
                phi += wgt * (v[with_g] - v[tuple(h for h in G if h in S)])
        shap.append(phi)
    refs = {'solo': solo, 'logo': logo, 'shapley': np.array(shap)}
    sharp = [P.max(1).mean() for P in Pt]
    for obj, fnw in (('macroF1_dirichlet', lambda: w_dirichlet(Pv, yv)),
                     ('nll', lambda: w_smooth(Pv, yv, 'nll')), ('brier', lambda: w_smooth(Pv, yv, 'brier'))):
        a = fnw(); yhat = fuse(a, Pt).argmax(1)
        att = attributions(a, Pt, yt, yhat)
        for an, av in att.items():
            rec = dict(split=s, objective=obj, attribution=an, fused_f1=f1(yt, yhat),
                       **{f'val_{g}': x for g, x in zip(G, av)})
            for rn, rv in refs.items():
                rec[f'ok_{rn}'] = order_ok(av, rv); rec[f'rho_{rn}'] = spearmanr(av, rv)[0]
            rows.append(rec)
    wl = w_loglinear(Pv, yv); Zt = sum(wi * np.log(p + 1e-9) for wi, p in zip(wl, Pt))
    rec = dict(split=s, objective='loglinear_nll', attribution='raw', fused_f1=f1(yt, Zt.argmax(1)),
               **{f'val_{g}': x for g, x in zip(G, wl / wl.sum())}, **{f'w_{g}': x for g, x in zip(G, wl)})
    for rn, rv in refs.items():
        rec[f'ok_{rn}'] = order_ok(wl, rv); rec[f'rho_{rn}'] = spearmanr(wl, rv)[0]
    rows.append(rec)
    json.dump(dict(split=s, solo=solo.tolist(), logo=logo.tolist(), shapley=shap, full=full, sharp=sharp,
                   subsets={'+'.join(k) if k else 'empty': x for k, x in v.items()}),
              open(OUT / f'refs_{s}.json', 'w'))
R = pd.DataFrame(rows); R.to_csv(OUT / 'controls_long.csv', index=False)

# ---- summaries over the re-splits (paper split reported separately)
RS = R[R.split != 'paper']
print('re-splits:', RS.split.nunique())
summ = []
for (obj, an), g in RS.groupby(['objective', 'attribution'], sort=False):
    summ.append(dict(objective=obj, attribution=an, n=len(g),
                     **{f'ok_{r}': int(g[f'ok_{r}'].sum()) for r in ('solo', 'logo', 'shapley')},
                     **{f'rho_{r}': f"{g[f'rho_{r}'].mean():.2f}±{g[f'rho_{r}'].std():.2f}" for r in ('solo', 'logo', 'shapley')},
                     alpha=' / '.join(f"{g[f'val_{x}'].mean():.3f}±{g[f'val_{x}'].std():.3f}" for x in G),
                     fusedF1=f"{g.fused_f1.mean():.2f}±{g.fused_f1.std():.2f}"))
S = pd.DataFrame(summ); S.to_csv(OUT / 'controls_summary.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(S.to_string(index=False))

# ---- paired tests: each effective variant vs raw, same objective
print('\nPaired tests vs raw (re-splits): Wilcoxon on Spearman, exact McNemar on ordering')
tests = []
for obj in RS.objective.unique():
    base = RS[(RS.objective == obj) & (RS.attribution == 'raw')].set_index('split')
    cands = [(obj, an) for an in RS[RS.objective == obj].attribution.unique() if an != 'raw']
    if obj == 'loglinear_nll': cands = [('macroF1_dirichlet', 'eff_true_micro'), ('macroF1_dirichlet', 'eff_pred_micro')]
    for eobj, an in cands:
        e = RS[(RS.objective == eobj) & (RS.attribution == an)].set_index('split').loc[base.index]
        for r in ('solo', 'logo', 'shapley'):
            dr = e[f'rho_{r}'] - base[f'rho_{r}']
            pw = wilcoxon(e[f'rho_{r}'], base[f'rho_{r}'], alternative='greater', zero_method='zsplit').pvalue if (dr != 0).any() else 1.0
            b = int((e[f'ok_{r}'] & ~base[f'ok_{r}']).sum()); c = int((~e[f'ok_{r}'] & base[f'ok_{r}']).sum())
            pm = binomtest(b, b + c, 0.5, alternative='greater').pvalue if b + c else 1.0
            tests.append(dict(raw_of=obj, attribution=eobj[:5]+':'+an, reference=r, eff_wins=b, raw_wins=c,
                              p_mcnemar=round(pm, 4), p_wilcoxon=round(pw, 4)))
T = pd.DataFrame(tests); T.to_csv(OUT / 'controls_tests.csv', index=False)
print(T.to_string(index=False))

# ---- pooled across objectives: sign test on ordering (all splits x objectives)
print('\nReference orderings per split:')
for fn in sorted(glob.glob(str(OUT / 'refs_*.json'))):
    j = json.load(open(fn))
    print(j['split'], 'solo', np.round(j['solo'], 2), 'logo', np.round(j['logo'], 2), 'shap', np.round(j['shapley'], 2), 'sharp', np.round(j['sharp'], 3))
