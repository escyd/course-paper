"""Readable final-grid CSV overview and figures from verified statistical outputs."""
import argparse,json
from pathlib import Path
import pandas as pd
import numpy as np
from experiment_runner import _digest

def main(directory):
    directory=Path(directory);verified=max(directory.glob('verified_analysis_*/verification.json'),key=lambda p:p.stat().st_mtime_ns)
    verification=json.loads(verified.read_text());report=Path(verification['main_report']);out=verified.parent
    counts=pd.read_csv(report/'cohort_counts.csv');desc=pd.read_csv(report/'descriptive_statistics.csv');rates=pd.read_csv(report/'rate_statistics.csv');sources=pd.read_csv(out/'recovery_sources_by_scenario.csv')
    for cohort in ['all_attempts','two_sided_before_shock']:
        overview=counts[counts.cohort==cohort].copy()
        for metric,part in desc[desc.cohort==cohort].groupby('metric'):
            part=part[['scenario_id','n','mean','median','mean_ci_low','mean_ci_high','median_ci_low','median_ci_high']].rename(columns={c:metric+'_'+c for c in ['n','mean','median','mean_ci_low','mean_ci_high','median_ci_low','median_ci_high']})
            overview=overview.merge(part,on='scenario_id',validate='one_to_one')
        for metric,part in rates[rates.cohort==cohort].groupby('metric'):
            part=part[['scenario_id','numerator','denominator','estimate','ci_low','ci_high']].rename(columns={c:metric+'_'+c for c in ['numerator','denominator','estimate','ci_low','ci_high']})
            overview=overview.merge(part,on='scenario_id',validate='one_to_one')
        overview.to_csv(out/('scenario_overview_'+cohort+'.csv'),index=False)
    data=pd.read_csv(out/'scenario_overview_all_attempts.csv');fx=sorted(data.fx_cost.unique())
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with plt.rc_context({'font.size':9}):
        fig,axes=plt.subplots(2,5,figsize=(16,7),layout='constrained')
        for row,(metric,title,is_rate) in enumerate([('avg_gap_after_mean','Mean adjusted gap, first 50 post-shock steps',False),('liquidity_aware_recovery_rate_estimate','Liquidity-aware recovery rate',True)]):
            lo,hi=(0,1) if is_rate else (data[metric].min(),data[metric].max())
            for col,cost in enumerate(fx):
                ax=axes[row,col];part=data[data.fx_cost==cost].pivot(index='beta',columns='tau',values=metric).sort_index();values=part.to_numpy()
                im=ax.imshow(values,origin='lower',vmin=lo,vmax=hi,cmap='viridis',aspect='auto')
                ax.set(xticks=range(len(part.columns)),xticklabels=[f'{x:g}' for x in part.columns],yticks=range(len(part.index)),yticklabels=[f'{x:g}' for x in part.index],xlabel='tau',ylabel='beta' if col==0 else '',title=f'FX cost {cost:g}')
                for i in range(values.shape[0]):
                    for j in range(values.shape[1]):
                        val=values[i,j];label=f'{val:.0%}' if is_rate else f'{val:.2f}';ax.text(j,i,label,ha='center',va='center',fontsize=8,color='white' if (val-lo)/max(hi-lo,1e-12)<.55 else 'black')
            fig.colorbar(im,ax=axes[row,:].tolist(),shrink=.8,label=title)
        fig.suptitle('Full sample retained; 500 seeds per cell; conditional-on-recovery times reported separately')
        fig.savefig(out/'final_grid_overview.png',dpi=220);fig.savefig(out/'final_grid_overview.pdf');plt.close(fig)
    # Publish a separate manifest for these derived overview artifacts.
    names=['scenario_overview_all_attempts.csv','scenario_overview_two_sided_before_shock.csv','final_grid_overview.png','final_grid_overview.pdf']
    (out/'overview_complete.json').write_text(json.dumps(dict(files={name:_digest(out/name) for name in names},renderer_sha256=_digest(Path(__file__))),indent=2))
    print('Overview:',out,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory');main(p.parse_args().directory)
