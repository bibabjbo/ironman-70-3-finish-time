"""Usage: python scripts/run.py DATA_DIR {basic,catboost,rf}"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

root = Path(__file__).resolve().parents[1]
source = Path(sys.argv[1])
mode = sys.argv[2]
assert mode in {'basic','catboost','rf'}
cfg = json.loads((root/'config/run.json').read_text())
def read(name, columns):
    df = pd.read_csv(source/(name+'.csv'), usecols=lambda c: c.lower() in columns,
                     keep_default_na=False, encoding='utf-8-sig')
    df.columns = df.columns.str.lower()
    return df

times = ['swimtime','biketime','finishtime']
df = read('Results',['resultid','athleteid','raceid'] + times)
for name, key, cols in [('Athletes','athleteid',['gender','agegroup','ageband','athletetype','countryiso']),
                         ('Races','raceid',['eventyear','eventlocation'])]:
    lookup = read(name,[key]+cols)
    if not df[key].isin(lookup[key]).all():
        raise ValueError('Unmatched '+key)
    df = df.merge(lookup,on=key,validate='many_to_one',sort=False)
assert df.resultid.is_unique
for col in times+['ageband']:
    df[col] = pd.to_numeric(df[col],errors='coerce')
ok = df.gender.isin(['M','F']) & df.athletetype.eq('Age-Group') & df.agegroup.isin(cfg['agegroups'])
ok &= (np.isfinite(df[times]) & df[times].gt(0)).all(axis=1)
data = df.loc[ok].sort_values('resultid').copy()
assert data.agegroup.str.split('-').str[0].astype(int).eq(data.ageband).all()
for col in ['countryiso','eventlocation','gender']:
    data[col] = data[col].str.strip().replace('','__MISSING__')
del df
if mode=='basic':
    data.groupby('eventyear').agg(records=('resultid','size'),races=('raceid','nunique')).to_csv(root/'metrics/cohort.csv')
    hashes = {name: hashlib.sha256((source/name).read_bytes()).hexdigest() for name in ['Athletes.csv','Races.csv','Results.csv']}
    (root/'config/input_sha256.json').write_text(json.dumps(hashes,indent=2))
metrics=[]
for split, spec in cfg['evaluations'].items():
    fit = data[data.eventyear.between(2004,spec['fit_end'])]
    test = data[data.eventyear.between(*spec['test'])]
    assert not set(fit.raceid).intersection(test.raceid)
    cats=['gender','countryiso','eventlocation']
    for cp, splits in cfg['checkpoints'].items():
        features=cfg['features_pre']+splits
        X=fit[features]
        Xt=test[features]
        y=fit.finishtime
        preds={}
        if mode=='basic':
            known=fit[splits].sum(axis=1) if splits else 0
            rem=fit.assign(remainder=y-known)
            known_test=test[splits].sum(axis=1).to_numpy() if splits else 0
            base=np.full(len(test),rem.remainder.median())
            lookup_rows=[pd.DataFrame([{'level':'all','gender':'','agegroup':'','size':len(rem),'median':rem.remainder.median()}])]
            for keys in [['gender'],['gender','agegroup']]:
                stats=rem.groupby(keys).remainder.agg(['size','median'])
                lookup_rows.append(stats.reset_index().assign(level='+'.join(keys)))
                stats=stats.loc[stats['size']>=cfg['baseline_min_group'],'median']
                lookup=test[keys].merge(stats.rename('v'),left_on=keys,right_index=True,how='left',sort=False)
                assert lookup.index.equals(test.index)
                base=np.where(lookup.v.notna(),lookup.v,base)
            pd.concat(lookup_rows,ignore_index=True).to_csv(root/'models'/f'{split}_{cp}_baseline.csv',index=False)
            preds['B0']=np.full(len(test),y.median())
            preds['B2']=known_test+base
            if cp=='C0':preds['B1']=preds['B2']
        if mode in ['basic','rf']:
            # Scaling the year prevents poor numerical conditioning in sparse least squares.
            nums=[c for c in features if c not in cats]
            from sklearn.preprocessing import StandardScaler
            enc=ColumnTransformer([('category',OneHotEncoder(handle_unknown='ignore'),cats),
                                   ('numeric',StandardScaler(),nums)],sparse_threshold=1)
            train=enc.fit_transform(X)
            held=enc.transform(Xt)
            model=LinearRegression(tol=1e-9) if mode=='basic' else RandomForestRegressor(**cfg['rf'])
            model.fit(train,y)
            preds['LR' if mode=='basic' else 'RF']=model.predict(held)
        if mode=='catboost':
            from catboost import CatBoostRegressor
            model=CatBoostRegressor(**cfg['catboost'])
            model.fit(X,y,cat_features=cats)
            preds['CatBoost_ref']=model.predict(Xt)
            model.save_model(str(root/'models'/f'{split}_{cp}.cbm'))
        out=test[['resultid','raceid','eventyear']].copy()
        out['split']=split
        out['checkpoint']=cp
        for name,pm in preds.items():
            assert len(pm)==len(test) and np.isfinite(pm).all()
            out[name]=pm
        out.to_csv(root/'predictions'/f'{split}_{cp}_{mode}.csv.gz',index=False,float_format='%.8f',compression={'method':'gzip','compresslevel':1})
        for name,pm in preds.items():
            err=(pm-test.finishtime.to_numpy())/60
            metrics.append(dict(split=split,checkpoint=cp,model=name,n_fit=len(fit),n_test=len(test),
                                races=test.raceid.nunique(),mae=np.mean(abs(err)),rmse=np.mean(err**2)**.5,bias=np.mean(err)))
        pd.DataFrame(metrics).to_csv(root/'metrics'/f'{mode}.csv',index=False)
        print(mode,split,cp,len(fit),len(test),flush=True)
