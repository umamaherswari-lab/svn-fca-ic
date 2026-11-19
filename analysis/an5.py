import sys; sys.path.insert(0,'../src')
import numpy as np, pandas as pd, warnings, time; warnings.filterwarnings('ignore')
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import RepeatedStratifiedKFold
from svnfca2 import SVN2
d=load_breast_cancer(); X=d.data[:,:5]; y=np.where(d.target==0,'M','B')
rows=[]; t0=time.time()
for fi,(a,b) in enumerate(RepeatedStratifiedKFold(n_splits=5,n_repeats=10,random_state=2024).split(X,y)):
    m=SVN2().fit(X[a],y[a],seed=fi); L=[len(m.cd.items[i]) for i in np.where(m.mask)[0]]
    rows.append(dict(fold=fi,svn_rules=m.n_rules,svn_len=np.mean(L),delta=m.delta,k=m.k))
    if fi==2: print('3 folds',round(time.time()-t0),'s',flush=True)
R=pd.DataFrame(rows); print(R.mean().round(3)); R.to_csv('../results/tuned_baselines/interp_svn_cv.csv',index=False)
