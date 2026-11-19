import pandas as pd,numpy as np,json,sys
sys.path.insert(0,'../src')
from svn_eval import corrected_t,holm
from scipy.stats import friedmanchisquare,wilcoxon,rankdata
a=pd.read_csv('../results/nine_datasets/cv_folds_all_datasets.csv'); t=pd.read_csv('../results/tuned_baselines/cv_folds_tuned.csv')
a5=a[a.feats!='all']; svn=a5[a5.model=='SVN-FCA-IC'].set_index(['dataset','fold'])
TM=['LR (tuned)','DT (tuned)','RF (tuned)','SVM-RBF (tuned)','kNN (tuned)']
R={}
# per dataset x model paired tests on balanced accuracy and accuracy
rows=[]
for ds in t.dataset.unique():
    for m in TM:
        x=t[(t.dataset==ds)&(t.model==m)].set_index('fold'); s=svn.loc[ds].loc[x.index]
        r=dict(dataset=ds,model=m)
        for met in ['accuracy','balanced_accuracy','auc']:
            d=(s[met]-x[met]).values; tt,p=corrected_t(d)
            r[met+'_svn']=s[met].mean(); r[met+'_tuned']=x[met].mean(); r[met+'_diff']=d.mean(); r[met+'_p']=p
        rows.append(r)
P=pd.DataFrame(rows)
# WDBC: Holm over the 5 comparisons per metric
w=P[P.dataset.str.startswith('WDBC')].copy()
for met in ['accuracy','balanced_accuracy','auc']: w[met+'_holm']=holm(w[met+'_p'].values)
P['ba_holm_all45']=holm(P.balanced_accuracy_p.values)
P.to_csv('../results/tuned_baselines/paired_tuned.csv',index=False); w.to_csv('../results/tuned_baselines/wdbc_tuned_tests.csv',index=False)
print(w[['model','accuracy_svn','accuracy_tuned','accuracy_p','accuracy_holm','balanced_accuracy_svn','balanced_accuracy_tuned','balanced_accuracy_holm','auc_svn','auc_tuned','auc_holm']].round(4).to_string())
print(P[['dataset','model','balanced_accuracy_svn','balanced_accuracy_tuned','balanced_accuracy_diff','balanced_accuracy_p','ba_holm_all45']].round(3).to_string())
# cross-dataset
s=pd.read_csv('../results/nine_datasets/summary_all_datasets.csv'); s5=s[s.feats!='all']
ba=s5.pivot_table(index='dataset',columns='model',values='balanced_accuracy')[['SVN-FCA-IC','RIPPER']]
tb=t.groupby(['dataset','model']).balanced_accuracy.mean().unstack()[TM]
M=ba.join(tb); print(M.round(3).to_string())
ranks=M.apply(lambda r: pd.Series(rankdata(-r.values),index=r.index),axis=1)
st,p=friedmanchisquare(*[M[c].values for c in M.columns])
pw={c:float(wilcoxon(M['SVN-FCA-IC'].values,M[c].values).pvalue) for c in M.columns if c!='SVN-FCA-IC'}
hp=dict(zip(pw,holm(list(pw.values()))))
avg=ranks.mean().sort_values()
res=dict(friedman=float(st),p=float(p),avg_rank=avg.to_dict(),wil=pw,holm=hp,n=len(M))
print(json.dumps(res,indent=1)); json.dump(res,open('../results/tuned_baselines/cross_tuned.json','w'),indent=1)
M.to_csv('../results/tuned_baselines/ba_matrix_tuned.csv')
# wins of SVN over best tuned
best=tb.max(axis=1); print((M['SVN-FCA-IC']-best).round(3).to_dict())
