"""
Reviewer controls, step 1 (Section 3.5 of the manuscript): fit the three Platt-scaled
expert SVMs and the SVMs on all seven feature-group subsets, for the paper split and the
ten stratified re-splits (seeds 0-9), and cache their outputs.

Usage:  python src/run_controls_fit.py paper 0 1 2 3 4 5 6 7 8 9
(about 2-3 minutes per split on one CPU core; run two processes in parallel if you like)
"""
import sys, time, itertools
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import GROUPS, RESULT_DIR, feature_matrix, load_all, make_svm
CACHE = RESULT_DIR / 'reviewer_controls' / 'cache'
from sklearn.model_selection import train_test_split

train, val, test, columns = load_all()
pooled = pd.concat([train, val, test], ignore_index=True)
y = pooled['label_id'].values
n_tr, n_va = len(train), len(val)
paths = pooled['path'].values
X = {g: feature_matrix(pooled, columns, (g,)) for g in GROUPS}

def split_idx(seed):
    if seed == 'paper':
        idx = np.arange(len(y)); return idx[:n_tr], idx[n_tr:n_tr+n_va], idx[n_tr+n_va:]
    idx = np.arange(len(y))
    a, b = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
    v, t = train_test_split(b, test_size=0.50, stratify=y[b], random_state=seed)
    return a, v, t

seeds = sys.argv[1:]
for s in seeds:
    seed = s if s == 'paper' else int(s)
    t0 = time.time()
    itr, iva, ite = split_idx(seed)
    out = dict(itr=itr, iva=iva, ite=ite)
    for g in GROUPS:
        m = make_svm(probability=True).fit(X[g][itr], y[itr])
        out[f'Pval_{g}'] = m.predict_proba(X[g][iva]); out[f'Pte_{g}'] = m.predict_proba(X[g][ite])
        out[f'pred_{g}'] = m.predict(X[g][ite])
    for r in (2, 3):
        for combo in itertools.combinations(GROUPS, r):
            Xc = np.hstack([X[g] for g in combo])
            m = make_svm(probability=False).fit(Xc[itr], y[itr])
            out['pred_' + '+'.join(combo)] = m.predict(Xc[ite])
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE / f'cache_{s}.npz', **out)
    print(s, 'done', round(time.time() - t0), 's', flush=True)
