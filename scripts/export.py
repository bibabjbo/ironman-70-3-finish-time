"""Merge checkpoint outputs into one long-format predictions file and verify every MAE."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

root=Path(__file__).resolve().parents[1]
raw=pd.read_csv(Path(sys.argv[1])/'Results.csv',usecols=['ResultID','FinishTime','SwimTime','BikeTime']).set_index('ResultID')
factual=raw.FinishTime
valid=raw.FinishTime.gt(raw.SwimTime+raw.BikeTime)
final_rows=[]
demo=[]
tables=pd.concat([pd.read_csv(root/'metrics'/f'{tag}.csv') for tag in ['basic','rf','catboost']],ignore_index=True)
writer=None
total=0
for split in ['I','II']:
    common=None
    for cp in ['C0','C1','C2']:
        for tag in ['basic','catboost','rf']:
            frame=pd.read_csv(root/'predictions'/f'{split}_{cp}_{tag}.csv.gz')
            ids=frame.resultid
            assert ids.is_unique and (frame.split==split).all() and (frame.checkpoint==cp).all()
            if common is None: common=ids
            assert ids.equals(common)
            actual=factual.loc[ids].to_numpy(dtype=float)
            for model in frame.columns.difference(['resultid','raceid','eventyear','split','checkpoint']):
                pred=frame[model].to_numpy(dtype=float)
                row=tables.query('split==@split and checkpoint==@cp and model==@model').iloc[0]
                assert row.n_test==len(frame) and np.isclose(np.mean(abs(pred-actual))/60,row.mae,atol=1e-8,rtol=0)
                keep=valid.loc[ids].to_numpy()
                out=frame.loc[keep,['resultid','raceid','eventyear','split','checkpoint']].copy()
                err=(pred[keep]-actual[keep])/60
                final_rows.append(dict(split=split,checkpoint=cp,model=model,n_fit=int(row.n_fit),n_test=len(out),races=out.raceid.nunique(),mae_minutes=abs(err).mean(),rmse_minutes=np.sqrt(np.mean(err**2)),bias_minutes=err.mean(),p90_abs_minutes=np.quantile(abs(err),.9)))
                out['model']=model
                out['seed']=42 if model in ['RF','CatBoost_ref'] else 0
                out['pred_finish_seconds']=pred[keep]
                out['actual_finish_seconds']=actual[keep]
                # Seed 0 denotes a deterministic estimator.
                table=pa.Table.from_pandas(out,preserve_index=False)
                if writer is None:
                    writer=pq.ParquetWriter(root/'predictions.parquet',table.schema,compression='zstd',
                         use_dictionary=['split','checkpoint','model','seed','raceid','eventyear','actual_finish_seconds'],
                         column_encoding={'resultid':'DELTA_BINARY_PACKED'})
                writer.write_table(table)
                total+=len(out)
                if split=='II':
                    demo.append(out.head(10))
writer.close()
assert pq.ParquetFile(root/'predictions.parquet').metadata.num_rows==total
pd.DataFrame(final_rows).to_csv(root/'metrics/all_models.csv',index=False)
tables.to_csv(root/'metrics/before_consistency_check.csv',index=False)
pd.concat(demo,ignore_index=True).to_csv(root/'metrics/example.csv',index=False)
print('Verified',total,'predictions')
