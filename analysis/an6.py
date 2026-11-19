import sys; sys.path.insert(0,'../src')
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from svnfca2 import SVN2
d=load_breast_cancer(); X=d.data[:,:5]; y=np.where(d.target==0,'M','B')
rows=[]
for fi,(a,b) in enumerate(RepeatedStratifiedKFold(n_splits=5,n_repeats=10,random_state=2024).split(X,y)):
    m=SVN2().fit(X[a],y[a],seed=fi)
    cl=SVN2(closed_only=True).fit(X[a],y[a],m.delta,m.k)
    L=[len(cl.cd.items[i]) for i in np.where(cl.mask)[0]]
    t=DecisionTreeClassifier(max_depth=5,random_state=42).fit(X[a],y[a]); tr=t.tree_
    leaves=np.where(tr.children_left==-1)[0]
    dep=np.zeros(tr.node_count,int)
    for n in range(tr.node_count):
        if tr.children_left[n]!=-1: dep[tr.children_left[n]]=dep[tr.children_right[n]]=dep[n]+1
    # path length of the test cases
    ap=t.decision_path(X[b]).sum(axis=1).A1-1
    rows.append(dict(closed_rules=cl.n_rules,closed_len=np.mean(L),k=m.k,leaf_depth=dep[leaves].mean(),path_test=ap.mean()))
R=pd.DataFrame(rows); print(R.mean().round(3))
