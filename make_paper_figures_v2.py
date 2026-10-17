"""Publication artifacts from verified saved outputs only; never runs simulations."""
import json,hashlib,datetime,uuid
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.stats import norm

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'tables/final_experiments_v2/20261017T170952Z_aaa6373d'
VER=BASE/'verified_analysis_54e52fb7';EFF=BASE/'parameter_effects_ce4683ad'
STUDY=ROOT/'tables/empty_both_validation/20261017T163358Z/study/20261017T163639Z_0c804ebf'
SENS=STUDY/'liquidity_analysis_72e22f57'
OUT=ROOT/'paper_figures_v2'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
INPUTS={};CAPTIONS={};COUNTS={}
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def track(p):
 p=Path(p);INPUTS[str(p.relative_to(ROOT))]=digest(p);return p

def read(p):return pd.read_csv(track(p),float_precision='round_trip')
def manifest(folder,name='complete.json'):
 m=json.loads(track(folder/name).read_text())
 for n,h in m['files'].items():assert digest(track(folder/n))==h,(folder,n)

def save(fig,name,caption):
 fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');fig.savefig(OUT/(name+'.png'),dpi=300,bbox_inches='tight');plt.close(fig);CAPTIONS[name]=caption

def label(ax,x,y,title=None):
 ax.set_xlabel(x);ax.set_ylabel(y)
 if title:ax.set_title(title,loc='left')
 ax.spines[['top','right']].set_visible(False)

def heat(data,column,name,title,rate=False):
 a=data.pivot(index='beta',columns='tau',values=column).sort_index();v=a.to_numpy();fig,ax=plt.subplots(figsize=(6.8,3.8),layout='constrained');cm=plt.get_cmap('viridis');nm=Normalize(0 if rate else v.min(),1 if rate else v.max())
 im=ax.imshow(v,origin='lower',aspect='auto',cmap=cm,norm=nm)
 for i in range(v.shape[0]):
  for j in range(v.shape[1]):
   rgb=np.array(cm(nm(v[i,j]))[:3]);lum=np.dot(np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4),[.2126,.7152,.0722]);color='black' if lum>.179 else 'white'
   ax.text(j,i,f'{v[i,j]:.1%}' if rate else f'{v[i,j]:.3f}',ha='center',va='center',color=color,fontsize=10)
 ax.set(xticks=range(5),xticklabels=[f'{x:g}' for x in a.columns],yticks=range(4),yticklabels=[f'{x:g}' for x in a.index],xlabel=r'Transport cost $\tau$',ylabel=r'Spillover $\beta$',title=title)
 fig.colorbar(im,ax=ax,label='Recovery probability' if rate else 'Adjusted price gap (CUR_A)')
 save(fig,name,'Equal-weight mean over the five FX costs at each tau/beta cell. All 500 seeds retained per scenario; 2500 scenario-seed outcomes per cell, with 500 shared seed clusters. '+('Unrecovered paths remain in the denominator.' if rate else 'Gap averaged over recorded steps 200–249.'))

def table_figure(df,name,caption,width=7):
 fig,ax=plt.subplots(figsize=(width,.48+.34*(len(df)+1)),layout='constrained');ax.axis('off');tab=ax.table(cellText=df.astype(str).values,colLabels=df.columns,loc='center',cellLoc='left',colLoc='left');tab.auto_set_font_size(False);tab.set_fontsize(8);tab.auto_set_column_width(range(len(df.columns)))
 for (i,j),cell in tab.get_celld().items():
  cell.set_linewidth(0);cell.set_height(1/(len(df)+1));cell.set_facecolor('#edf1f4' if i==0 else 'white')
  if i==0:cell.set_text_props(weight='bold')
 save(fig,name,caption)

def main():
 assert json.loads(track(VER/'verification.json').read_text())['simulations']==50000
 for d,m in [(VER,'complete.json'),(VER,'overview_complete.json'),(EFF,'complete.json'),(EFF,'supplement_complete.json')]:manifest(d,m)
 assert json.loads(track(EFF/'postprocessing_qa.json').read_text())['status']=='passed'
 old=json.loads((ROOT/'tables/final_run_audit/20261017T170950Z/prior_results_sha256.json').read_text())
 # All selected single-market analysis files were part of the preserved, audited inputs.
 for p in SENS.glob('*.csv'):
  assert digest(p)==old[str(p.relative_to(ROOT))];track(p)
 single=json.loads(track(ROOT/'tables/empty_both_validation/20261017T163358Z/verification.json').read_text());assert single['completed']==50 and single['full_price_series_reproduced']==50
 for seed in range(42,92):manifest(STUDY/f'steps_10000/seed_{seed}')
 OUT.mkdir(parents=True,exist_ok=False)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.6,'lines.linewidth':1.4,'savefig.facecolor':'white'})
 raw=read(BASE/'raw_results.csv');assert len(raw)==50000 and not raw.duplicated(['scenario_id','seed']).any()
 overview=read(VER/'scenario_overview_all_attempts.csv');effects=read(EFF/'parameter_effects_equal_grid_weights.csv');global_metrics=read(EFF/'grid_average_metrics.csv')
 agg=overview.groupby(['tau','beta'],as_index=False).agg(avg_gap_after=('avg_gap_after_mean','mean'),liquidity_rate=('liquidity_aware_recovery_rate_estimate','mean'))
 agg.to_csv(OUT/'heatmap_values_fx_averaged.csv',index=False)
 heat(agg,'avg_gap_after','fig01_adjusted_gap_heatmap','Mean adjusted price gap: equal weights over FX costs')
 heat(agg,'liquidity_rate','fig02_liquidity_recovery_heatmap','Liquidity-aware recovery: equal weights over FX costs',True)
 # Seed-level aggregation before uncertainty calculations; never pretend that
 # the 100 scenarios with a common seed are independent observations.
 paired=raw.copy();paired['price']=paired.recovery_time_status.eq('recovered').astype(float);paired['liquidity']=paired.liquidity_recovery_time_status.eq('recovered').astype(float)
 rng=np.random.default_rng(20261017);rows=[]
 fig,axes=plt.subplots(1,2,figsize=(7,3.4),layout='constrained')
 for metric,color,marker,title in [('price','#0072B2','o','Price recovery'),('liquidity','#D55E00','s','Liquidity-aware recovery')]:
  for tau,part in paired.groupby('tau'):
   values=part.groupby('seed')[metric].mean().to_numpy();draw=values[rng.integers(0,500,(9999,500))].mean(axis=1);lo,hi=np.quantile(draw,[.025,.975]);mean=values.mean();rows.append(dict(tau=tau,metric=metric,estimate=mean,ci_low=lo,ci_high=hi,n_seed_clusters=500))
   axes[0].errorbar(tau,mean,yerr=[[mean-lo],[hi-mean]],color=color,marker=marker,capsize=3,label=title if tau==0 else None)
 axes[0].set_ylim(-.04,1.08);label(axes[0],r'$\tau$ (equal weights over $\beta$ and FX cost)','Recovery probability','(a) Recovery rates');axes[0].legend(loc='center right',frameon=False)
 for metric,col,color in [('Price','recovery_time','#0072B2'),('Liquidity-aware','liquidity_recovery_time','#D55E00')]:
  time=raw[col].to_numpy();surv=np.array([np.mean(np.isnan(time)|(time>k)) for k in range(300)])
  axes[1].step(range(300),surv,where='post',color=color,label=metric);axes[1].plot(299,surv[-1],marker='|',markersize=13,color=color)
  n=int(raw[col+'_status'].eq('censored').sum());axes[1].annotate(f'{n:,} censored',xy=(299,surv[-1]),xytext=(-5,8),textcoords='offset points',ha='right',color=color,fontsize=8)
 axes[1].set_ylim(-.04,1.08);label(axes[1],'Steps since shock','Proportion not yet recovered','(b) Administrative censoring at 299');axes[1].legend(loc='center right',frameon=False)
 pd.DataFrame(rows).to_csv(OUT/'recovery_rates_by_tau.csv',index=False)
 save(fig,'fig03_recovery_and_censoring','Rates average equally over beta and FX cost. Pointwise 95% percentile bootstrap intervals resample 500 whole seed clusters, B=9999. Survival curves average all 100 scenarios equally; marks at 299 denote administrative censoring, not events. Price recovery does not require two-sided liquidity.')
 print('Rendered heatmaps and recovery comparison',flush=True)
 # Four endpoints per parameter, family-adjusted intervals already verified.
 panels=[('avg_gap_after','Adjusted gap (CUR_A)',1),('liquidity_aware_recovery_rate','Liquidity-aware recovery (pp)',100),('pre_shock_liquidity_failure_rate','Pre-shock liquidity failure (pp)',100),('liquidity_restricted_wait_299','Restricted liquidity waiting time (steps)',1)]
 symbols={'tau':r'$\tau$','beta':r'$\beta$','fx_cost':'FX cost'}
 for parameter in symbols:
  fig,axs=plt.subplots(2,2,figsize=(7,5.1),layout='constrained')
  for ax,(metric,title,scale) in zip(axs.flat,panels):
   q=effects[(effects.parameter==parameter)&(effects.metric==metric)].sort_values('low_level');y=np.arange(len(q));means=q.mean_difference_high_minus_low.to_numpy()*scale;lo=q.simultaneous_ci_low.to_numpy()*scale;hi=q.simultaneous_ci_high.to_numpy()*scale
   ax.hlines(y,lo,hi,color='#0072B2');ax.plot(means,y,'o',color='#0072B2',markersize=4);ax.axvline(0,color='.45',ls='--',lw=.7);ax.set_yticks(y,[f'{r.low_level:g} → {r.high_level:g}' for r in q.itertuples()]);ax.invert_yaxis();label(ax,'High minus low','',title)
  fig.suptitle('Paired effects of '+symbols[parameter]+'; equal weights over other parameters',fontsize=11)
  save(fig,'fig04_effects_'+parameter,'Adjacent-level paired effects on all 500 seeds, averaged equally over the other grid parameters. Approximate simultaneous 95% Bonferroni Student-t intervals across 2192 mean effects. No automatic hypothesis decisions; constant-data intervals may degenerate. Restricted waiting time integrates survival through 299 and retains censored paths; it is not an imputed recovery event time.')
 effects.to_csv(OUT/'table_parameter_effects_numeric.csv',index=False)
 # Design-central scenario and first seed in each pre-shock state, declared
 # before looking at any selected trajectory values.
 chosen=raw[(raw.tau==.3)&(raw.beta==.2)&(raw.fx_cost==.003)];selection=[]
 fig,axs=plt.subplots(2,2,figsize=(7,5.6),layout='constrained')
 for col,(state,part) in enumerate(chosen.groupby('pre_shock_two_sided',sort=False)):
  r=part.sort_values('seed').iloc[0];idx=int(raw.index[raw.scenario_id.eq(r.scenario_id)&raw.seed.eq(r.seed)][0]);series_path=BASE/'runs'/f'{idx:07d}_timeseries.csv';marker=json.loads(track(BASE/'runs'/f'{idx:07d}.json').read_text());assert digest(series_path)==marker['timeseries_sha256'];d=read(series_path);x=np.arange(150,500)-200;mask=(~(d.liquid_a&d.liquid_b)).to_numpy()[150:];threshold=sum(d.price_gap.to_numpy().tolist()[150:200])/50+.5
  selection.append(dict(scenario_id=r.scenario_id,seed=int(r.seed),tau=.3,beta=.2,fx_cost=.003,pre_shock_two_sided=bool(state),rule='minimum seed in each pre-shock state of design-central scenario; not selected by trajectory appearance',timeseries_sha256=digest(series_path)))
  ax=axs[0,col];ax.plot(x,d.prices_a.iloc[150:],color='#0072B2',label='Market A');ax.plot(x,(d.prices_b/d.fx_rates).iloc[150:],color='#D55E00',ls='--',label='Market B / FX');ax.set_title(f'Seed {int(r.seed)}; '+('two-sided' if state else 'illiquid')+' before shock');label(ax,'','Price (CUR_A)');ax.legend(frameon=False,loc='best')
  ax=axs[1,col];ax.plot(x,d.price_gap.iloc[150:],color='#333333');ax.axhline(threshold,color='#009E73',ls='--',label='Recovery band ceiling')
  for metric,color,mark in [('recovery_time','#0072B2','o'),('liquidity_recovery_time','#D55E00','s')]:
   value=r[metric]
   if pd.notna(value):ax.plot(value,d.price_gap.iloc[200+int(value)],mark,color=color,label=('Price' if metric=='recovery_time' else 'Liquidity-aware')+' recovery')
   else:ax.text(.98,.90 if metric=='recovery_time' else .81,('Price' if metric=='recovery_time' else 'Liquidity-aware')+': censored at 299',transform=ax.transAxes,ha='right',fontsize=7,color=color)
  label(ax,'Steps since shock','Adjusted gap (CUR_A)');ax.legend(frameon=False,fontsize=6.5,loc='lower right')
  for ax in axs[:,col]:
   ax.axvline(0,color='.3',lw=.7);ax.fill_between(x,0,1,where=mask,transform=ax.get_xaxis_transform(),color='.8',alpha=.4,zorder=-1,step='mid')
 (OUT/'trajectory_selection.json').write_text(json.dumps(selection,indent=2));save(fig,'fig05_saved_price_trajectories','Fixed central scenario tau=0.3, beta=0.2, FX cost=0.003. Minimum seed in each pre-shock two-sided/illiquid stratum, not chosen by appearance; illustrative strata, not typical-path estimators. Prices converted to CUR_A; shaded recorded steps lack two-sided A/B liquidity. Shock at 0. Censored outcomes labeled; price-band crossing is not liquidity restoration.')
 # Audited single-market series, not the 100-scenario two-market grid.
 ci=read(SENS/'median_ci.csv');comp=read(SENS/'comparison_per_seed.csv');ac=read(SENS/'acf_by_segment.csv');liq=read(SENS/'liquidity_per_seed.csv')
 ci.to_csv(OUT/'table_stylized_facts_median_ci.csv',index=False);read(SENS/'test_rejections.csv').to_csv(OUT/'table_stylized_test_rejections.csv',index=False)
 fig,axs=plt.subplots(1,3,figsize=(7,2.8),layout='constrained')
 selected=[('excess_kurtosis','Excess kurtosis',0),('tail_3sigma_frequency',r'$P(|r-\bar r|>3s)$',2*norm.sf(3)),('acf_returns_1','Return ACF(1)',0)]
 for ax,(metric,title,reference) in zip(axs,selected):
  q=ci[ci.metric==metric].set_index('subset').loc[['full','longest_two_sided']];m=q['median'].to_numpy();ax.errorbar([0,1],m,yerr=[m-q.ci_low.to_numpy(),q.ci_high.to_numpy()-m],fmt='o',color='#0072B2',capsize=3);ax.axhline(reference,color='.5',ls='--',lw=.7);ax.set_xticks([0,1],['Full','Two-sided\nsegment']);label(ax,'','',title)
 save(fig,'fig06_single_market_tail_summary','Separate audited baseline: 50 seeds 42–91, 10000 prices, burn-in 100, original composition 5 Random + 5 Fundamentalist + 5 Chartist + 5 Universalist + 1 MarketMaker; no shock. Medians of per-run statistics, pointwise 95% seed-bootstrap CIs from prior verified analysis (B=10000). Two-sided statistics use longest continuous liquid segment per run; unequal durations and endogenous selection limit interpretation. Gaussian/zero references dashed. No returns concatenated across runs or gaps.')
 # ACF bands across seed-specific ACFs, no concatenation. Bootstrap medians.
 fig,axs=plt.subplots(1,3,figsize=(7,3),layout='constrained');acfsummary=[];rng=np.random.default_rng(20261018)
 for ax,transform,title in zip(axs,['returns','abs_returns','squared_returns'],['Returns',r'Absolute returns $|r|$',r'Squared returns $r^2$']):
  for subset,color,style in [('full','#0072B2','-'),('longest_two_sided','#D55E00','--')]:
   q=ac[(ac.subset==subset)&(ac['transform']==transform)&(ac.lag.between(1,40))];med=[];low=[];high=[]
   for lag,part in q.groupby('lag'):
    assert not part.seed.duplicated().any();z=part.acf.dropna().to_numpy();draw=np.median(z[rng.integers(0,len(z),(1999,len(z)))],axis=1);lo,hi=np.quantile(draw,[.025,.975]);med.append(np.median(z));low.append(lo);high.append(hi);acfsummary.append(dict(subset=subset,transform=transform,lag=int(lag),n_seeds=len(z),median=med[-1],ci_low=lo,ci_high=hi))
   assert len(med)==40;ax.plot(range(1,41),med,color=color,ls=style,label='Full' if subset=='full' else 'Two-sided');ax.fill_between(range(1,41),low,high,color=color,alpha=.13)
  ax.axhline(0,color='.6',lw=.6);label(ax,'Lag','ACF' if ax==axs[0] else '',title)
 axs[2].legend(frameon=False);pd.DataFrame(acfsummary).to_csv(OUT/'single_market_acf_median_ci.csv',index=False)
 save(fig,'fig07_single_market_volatility_clustering','Same single-market sample and burn-in as fig06. Median seed-specific ACFs with pointwise 95% bootstrap CIs (B=1999); not simultaneous lag-wise bands. Longest contiguous liquid segment used for sensitivity analysis; no joining over missing liquidity or across seeds. Negative return ACF(1) is shown explicitly.')
 # Minimum seed Q-Q example; never pool observations from independent runs.
 seed=42;prices=read(STUDY/f'steps_10000/seed_{seed}/prices.csv');r=np.diff(np.log(prices.price.to_numpy()[100:]));z=(r-r.mean())/r.std(ddof=1);zsort=np.sort(z);qq=norm.ppf((np.arange(len(z))+.5)/len(z));fig,axs=plt.subplots(1,2,figsize=(7,3.2),layout='constrained')
 axs[0].hist(z,bins=100,density=True,histtype='step',color='#0072B2',label='Seed 42');x=np.linspace(-6,6,401);axs[0].plot(x,norm.pdf(x),color='#D55E00',ls='--',label='Gaussian');axs[0].set_yscale('log');label(axs[0],'Standardized log return','Density','(a) Distribution, log density');axs[0].legend(frameon=False)
 axs[1].plot(qq,zsort,'.',ms=1.7,color='#0072B2',rasterized=True);axs[1].plot([qq.min(),qq.max()],[qq.min(),qq.max()],ls='--',color='#D55E00');label(axs[1],'Gaussian quantile','Empirical quantile','(b) Q–Q plot')
 save(fig,'fig08_single_market_distribution_qq','Illustrative lowest seed (42), selected before inspecting statistics; 10000-price single-market run, burn-in 100, full series including fallback observations. Returns centered and divided by sample standard deviation. All empirical quantiles displayed; Gaussian reference does not imply calibrated Gaussian returns. Ensemble evidence and liquid-segment sensitivity are in fig06–07.')
 print('Rendered paired effects, trajectories, and single-market diagnostics',flush=True)
 # Compact tables, each delivered as editable CSV and typeset PDF/PNG.
 table=global_metrics[['metric','mean','mean_ci_low','mean_ci_high']].copy();table.to_csv(OUT/'table01_grid_summary_numeric.csv',index=False)
 names={'avg_gap_after':'Post-shock adjusted gap','final_gap':'Final adjusted gap','price_recovery_rate':'Price recovery (%)','liquidity_aware_recovery_rate':'Liquidity-aware recovery (%)','pre_shock_liquidity_failure_rate':'Pre-shock liquidity failure (%)','price_restricted_wait_299':'Restricted price waiting (steps)','liquidity_restricted_wait_299':'Restricted liquidity waiting (steps)'};display=[]
 for r in table.itertuples():
  scale=100 if r.metric.endswith('_rate') else 1;display.append({'Outcome':names[r.metric],'Estimate':f'{r.mean*scale:.3f}','95% CI':f'[{r.mean_ci_low*scale:.3f}, {r.mean_ci_high*scale:.3f}]'})
 cens=read(EFF/'grid_average_censoring_rates.csv');cens.to_csv(OUT/'table01_censoring_numeric.csv',index=False)
 for r in cens.itertuples():display.append({'Outcome':('Price' if r.metric.startswith('price') else 'Liquidity-aware')+' censoring (%)','Estimate':f'{100*r.estimate:.3f}','95% CI':f'[{100*r.ci_low:.3f}, {100*r.ci_high:.3f}]'})
 table_figure(pd.DataFrame(display),'table01_grid_summary','Equal-weight grid averages; uncertainty resamples the 500 independent whole-seed clusters. All attempts retained, 50000/50000 completed valid. Restricted waiting is a survival integral to 299, not an unconditional uncensored recovery time.')
 compact=read(EFF/'effect_summary_for_review.csv');compact.to_csv(OUT/'table02_parameter_effects_numeric.csv',index=False);disp=[]
 for r in compact.itertuples():
  disp.append({'Parameter':r.parameter,'Change':f'{r.low_level:g} → {r.high_level:g}','Gap effect [95% CI]':f'{r.gap_mean_difference_high_minus_low:.3f} [{r.gap_simultaneous_ci_low:.3f}, {r.gap_simultaneous_ci_high:.3f}]','Liquidity recovery, pp [95% CI]':f'{r.liquidity_recovery_pp_mean_difference_high_minus_low:.2f} [{r.liquidity_recovery_pp_simultaneous_ci_low:.2f}, {r.liquidity_recovery_pp_simultaneous_ci_high:.2f}]'})
 table_figure(pd.DataFrame(disp),'table02_parameter_effects','Same equal-grid paired estimand and approximate simultaneous Bonferroni 95% intervals as fig04. The full numeric table also includes pre-shock liquidity loss.')
 brief=[]
 for r in ci[ci.metric.isin(['excess_kurtosis','tail_3sigma_frequency','acf_returns_1','acf_abs_returns_1','acf_squared_returns_1'])].itertuples():brief.append({'Subset':'Full' if r.subset=='full' else 'Two-sided segment','Statistic':r.metric.replace('_',' '),'Median [95% CI]':f'{r.median:.4f} [{r.ci_low:.4f}, {r.ci_high:.4f}]','n':r.n_valid})
 table_figure(pd.DataFrame(brief),'table03_single_market','Audited single-market series, not the final two-market grid. Pointwise median bootstrap intervals; selected continuous liquid-segment analysis is descriptive. See test-rejection CSV for Ljung–Box and ARCH-LM with previously defined BH families.')
 (OUT/'captions.md').write_text('\n\n'.join(f'**{name}**\n\n{caption}' for name,caption in CAPTIONS.items()))
 metadata=dict(final_source=str(BASE.relative_to(ROOT)),single_market_source=str(STUDY.relative_to(ROOT)),single_market_verification='separate prior audited 50-seed study; not generated by final two-market experiment',no_simulations_run=True,model_unchanged=True,style='7 inch figures, 8–10 pt text, embedded TrueType in PDF, PNG 300 dpi, colorblind-friendly line colors',aggregation='Equal weights over all remaining grid levels. Shared seeds clustered, no concatenation of independent time series.',trajectory_selection=selection,source_sha256=INPUTS,renderer_sha256=digest(Path(__file__)))
 (OUT/'provenance.json').write_text(json.dumps(metadata,indent=2))
 assert all(digest(ROOT/p)==sha for p,sha in INPUTS.items())
 (OUT/'complete.json').write_text(json.dumps(dict(files={p.name:digest(p) for p in OUT.iterdir() if p.is_file()},figure_pairs=len(CAPTIONS),input_files_verified=len(INPUTS)),indent=2))
 print('OUTPUT',OUT,flush=True);print('PDF/PNG pairs',len(CAPTIONS),'verified source files',len(INPUTS),flush=True)
if __name__=='__main__':main()
