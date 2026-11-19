import sys; sys.path.insert(0,'../src')
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings('ignore')
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
import wittgenstein as lw
from svnfca import fit_tertile_edges, crisp_context
d=load_breast_cancer(); X=d.data[:,:5]; y=np.where(d.target==0,'M','B')
# full-context concept count on the 455 hold-out training cases
tr,va=train_test_split(np.arange(len(y)),test_size=.2,random_state=42,stratify=y)
A=X[tr]; e=fit_tertile_edges(A); C=np.asarray(crisp_context(A,e)).astype(bool)
print('context',C.shape)
masks=[int(''.join('1' if b else '0' for b in row),2) for row in C]
full=(1<<C.shape[1])-1
ints={full}
for m in masks: ints.add(m)
frontier=set(ints)
while frontier:
    new=set()
    for a in frontier:
        for m in masks:
            c=a&m
            if c not in ints: new.add(c)
    ints|=new; frontier=new
print('concepts in full training context (15 attributes):',len(ints))
# RIPPER / DT on the 50 CV folds
rows=[]
for fi,(a,b) in enumerate(RepeatedStratifiedKFold(n_splits=5,n_repeats=10,random_state=2024).split(X,y)):
    cn=[f'f{i}' for i in range(5)]
    df=pd.DataFrame(X[a],columns=cn); df['y']=(y[a]=='M')
    r=lw.RIPPER(random_state=0); r.fit(df,class_feat='y',pos_class=True)
    rs=r.ruleset_; nconds=[len(rule.conds) for rule in rs.rules] if hasattr(rs,'rules') else [len(x.conds) for x in rs]
    t=DecisionTreeClassifier(max_depth=5,random_state=42).fit(X[a],y[a])
    rows.append(dict(fold=fi,ripper_rules=len(nconds),ripper_len=np.mean(nconds) if nconds else np.nan,dt_leaves=t.get_n_leaves()))
R=pd.DataFrame(rows); print(R.mean().round(2)); R.to_csv('../results/tuned_baselines/interp_cv.csv',index=False)
