"""Record row membership and disjoint race groups in the two temporal evaluations."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

root=Path(__file__).resolve().parents[1]
source=Path(sys.argv[1])
cfg=json.loads((root/'config/run.json').read_text())
d=pd.read_csv(source/'Results.csv',usecols=['ResultID','RaceID','AthleteID','SwimTime','BikeTime','FinishTime'])
a=pd.read_csv(source/'Athletes.csv',usecols=['AthleteID','Gender','AgeGroup','AthleteType'])
r=pd.read_csv(source/'Races.csv',usecols=['RaceID','EventYear'])
d=d.merge(a,on='AthleteID',validate='many_to_one').merge(r,on='RaceID',validate='many_to_one')
ts=d[['FinishTime','SwimTime','BikeTime']]
d=d[d.Gender.isin(['M','F']) & d.AgeGroup.isin(cfg['agegroups']) & d.AthleteType.eq('Age-Group') & (np.isfinite(ts)&ts.gt(0)).all(axis=1)].sort_values('ResultID')
d['valid_total']=d.FinishTime.gt(d.SwimTime+d.BikeTime)
for split,spec in cfg['evaluations'].items():
    fit=d.EventYear.between(2004,spec['fit_end'])
    test=d.EventYear.between(*spec['test'])
    assert d.loc[fit,'valid_total'].all()
    d[split]=np.select([fit,test & d.valid_total,test & ~d.valid_total],['fit','test','excluded_total'],'unused')
    assert not set(d.loc[fit,'RaceID']).intersection(d.loc[test,'RaceID'])
manifest=d[['ResultID','RaceID','EventYear','I','II']].copy()
manifest.columns=['resultid','raceid','eventyear','split_I','split_II']
pq.write_table(pa.Table.from_pandas(manifest,preserve_index=False),root/'cohort_manifest.parquet',compression='zstd',use_dictionary=['raceid','eventyear','split_I','split_II'],column_encoding={'resultid':'DELTA_BINARY_PACKED'})
print(manifest.split_I.value_counts(),manifest.split_II.value_counts())
