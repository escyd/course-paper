"""Post-process v2 study without changing saved simulations or historical results."""
from pathlib import Path
import json,importlib.util
import numpy as np
import pandas as pd
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
OLD=ROOT/'tables/stylized_facts/20261017T120604Z_96dfd328'
NEW=ROOT/'tables/stylized_facts_v2/20261017T153832Z_86aa3981'
def main():
    source=ROOT/'tables/methodology_audit/20261017T122418Z/uncertainty.py'
    spec=importlib.util.spec_from_file_location('ci',source);ci=importlib.util.module_from_spec(spec);spec.loader.exec_module(ci)
    out=NEW/('uncertainty_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir(exist_ok=False)
    ci.OUT=out;ci.STUDY=NEW;ci.main()
    a=pd.read_csv(OLD/'per_run_statistics.csv');b=pd.read_csv(NEW/'per_run_statistics.csv')
    metrics=['excess_kurtosis','tail_3sigma_frequency','acf_returns_1','acf_abs_returns_1','acf_squared_returns_1'];rows=[]
    for steps in [500,5000,10000]:
        aa=a[(a.steps==steps)&(a.burn_in==100)&(a.status=='ok')];bb=b[(b.steps==steps)&(b.burn_in==100)&(b.status=='ok')]
        paired=aa.merge(bb,on='seed',suffixes=('_old','_new'))
        rng=np.random.default_rng(20261017+steps);idx=rng.integers(0,len(paired),size=(10000,len(paired)))
        for metric in metrics:
            diff=(paired[metric+'_new']-paired[metric+'_old']).to_numpy();lo,hi=np.quantile(np.median(diff[idx],axis=1),[.025,.975])
            rows.append(dict(steps=steps,metric=metric,n_old=len(aa),n_new=len(bb),old_median=aa[metric].median(),new_median=bb[metric].median(),n_paired=len(paired),paired_median_change=np.median(diff),paired_change_ci_low=lo,paired_change_ci_high=hi))
    pd.DataFrame(rows).to_csv(out/'version_comparison.csv',index=False)
    print('COMPARISON',out,flush=True)
if __name__=='__main__':main()
