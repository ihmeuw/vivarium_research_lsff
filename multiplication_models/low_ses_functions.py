import pandas as pd, numpy as np
from db_queries import get_ids, get_outputs, get_location_metadata, get_population, get_covariate_estimates
from get_draws.api import get_draws
import scipy.stats
import scipy.integrate as integrate
import matplotlib.pyplot as plt

def pull_exposure(rei_id, sex_ids, age_group_ids, location_ids):
    exposure = get_draws(gbd_id_type='rei_id',
                              gbd_id=rei_id,
                              sex_id=sex_ids,
                              age_group_id=age_group_ids,
                              location_id=location_ids,
                              year_id=2019,
                              gbd_round_id=6,
                              source='exposure',
                              decomp_step='step4')
    return exposure

def adjust_exposure(exposure_data, exposure_category, burden_multiplier, index_cols):
    adjusted_exposure = exposure_data.loc[exposure_data.parameter==exposure_category]
    adjusted_exposure = adjusted_exposure.set_index(index_cols)
    adjusted_exposure = adjusted_exposure.drop(columns=[c for c in adjusted_exposure.columns if 'draw' not in c])
    adjusted_exposure = (adjusted_exposure * burden_multiplier).sort_index()
    return adjusted_exposure
    
def pull_affected_dalys(cause_ids, age_group_ids, sex_ids, location_ids):
    ylls = get_draws(gbd_id_type='cause_id',
                              gbd_id=cause_ids,
                              sex_id=sex_ids,
                              age_group_id=age_group_ids,
                              location_id=location_ids,
                              year_id=2019,
                              gbd_round_id=6,
                              source='codcorrect',
                              measure_id=4, #ylls
                              decomp_step='step5')
    ylds = get_draws(gbd_id_type='cause_id',
                              gbd_id=cause_ids,
                              sex_id=sex_ids,
                              age_group_id=age_group_ids,
                              location_id=location_ids,
                              year_id=2019,
                              gbd_round_id=6,
                              source='como',
                              measure_id=3, #ylds
                              decomp_step='step5')
    pop = get_population(sex_id=sex_ids,
                              age_group_id=age_group_ids,
                              location_id=location_ids,
                              year_id=2019,
                              gbd_round_id=6,
                             decomp_step='step4')
    yll_rate = ylls.merge(pop, on=['location_id','sex_id','age_group_id'])
    for i in list(range(0,1000)):
        yll_rate[f'draw_{i}'] = yll_rate[f'draw_{i}'] / yll_rate['population']
    ylls_prepped = yll_rate.set_index(['location_id','age_group_id','sex_id','cause_id'])
    ylls_prepped = ylls_prepped.drop(columns=[c for c in ylls_prepped if 'draw' not in c])
    ylds_prepped = ylds.set_index(['location_id','age_group_id','sex_id','cause_id'])
    ylds_prepped = ylds_prepped.drop(columns=[c for c in ylds_prepped if 'draw' not in c])
    dalys = ylls_prepped + ylds_prepped
    return dalys
    
def adjust_dalys(daly_data, multiplier_data):
    adjusted_dalys = daly_data.reset_index().merge(multiplier_data)
    for i in list(range(0,1000)):
        adjusted_dalys[f'draw_{i}'] = adjusted_dalys[f'draw_{i}'] * adjusted_dalys['multiplier']
    adjusted_dalys = adjusted_dalys.set_index(['location_id','age_group_id','sex_id','cause_id']).drop(columns='multiplier').sort_index()
    return adjusted_dalys
    
def pull_relative_risks(rei_id, age_group_ids, sex_ids):
    relative_risks = get_draws(gbd_id_type='rei_id',
                          gbd_id=rei_id,
                          age_group_id=age_group_ids,
                          sex_id=sex_ids,
                          year_id=2019,
                          source='rr',
                          gbd_round_id=6,
                          decomp_step='step4')
    relative_risks = relative_risks.loc[relative_risks.parameter=='cat1']
    relative_risks = relative_risks.set_index(['age_group_id','sex_id','cause_id'])
    relative_risks = relative_risks.drop(columns=[c for c in relative_risks.columns if 'draw' not in c]).sort_index()
    return relative_risks
    
def calculate_adjusted_paf(adjusted_exposure_data, relative_risks):
    pafs = pd.DataFrame()
    for location_id in adjusted_exposure_data.reset_index().location_id.unique():
        adjusted_exposure = adjusted_exposure_data.reset_index().loc[adjusted_exposure_data.reset_index().location_id==location_id]
        adjusted_exposure = adjusted_exposure.set_index(['age_group_id','sex_id']).drop(columns='location_id')
        paf = (adjusted_exposure * relative_risks + (1 - adjusted_exposure) - 1) / (adjusted_exposure * relative_risks + (1 - adjusted_exposure))
        paf['location_id'] = location_id
        pafs = pd.concat([pafs,paf.reset_index()],ignore_index=True)
    return pafs.set_index(['location_id','age_group_id','sex_id','cause_id']).dropna()

def calculate_attributable_dalys(dalys, pafs):
    attributable_dalys = (dalys * pafs).replace(np.nan,0)
    return attributable_dalys
    
def calculate_overall_attributable_daly_counts(attributable_daly_rates, age_group_ids, location_ids, sex_ids):
    pop = get_population(sex_id=sex_ids,
                              age_group_id=age_group_ids,
                              location_id=location_ids,
                              year_id=2019,
                              gbd_round_id=6,
                             decomp_step='step4')
    data = attributable_daly_rates.reset_index().merge(pop, on=['location_id','age_group_id','sex_id'])
    for i in list(range(0,1000)):
        data[f'draw_{i}'] = data[f'draw_{i}'] * data['population']
    data = data.groupby(['location_id']).sum()
    data = data.drop(columns=[c for c in data.columns if 'draw' not in c])
    return data
    
def add_in_adjusted_paf_of_one(attributable_dalys, location_ids, sex_ids, age_group_ids, cause_id, burden_multiplier):
    ylds = get_draws(gbd_id_type='cause_id',
                              gbd_id=cause_id,
                              sex_id=sex_ids,
                              age_group_id=age_group_ids,
                              location_id=location_ids,
                              year_id=2019,
                              gbd_round_id=6,
                              source='como',
                              measure_id=3, #ylds
                              decomp_step='step5')
    pop = get_population(sex_id=sex_ids,
                              age_group_id=age_group_ids,
                              location_id=location_ids,
                              year_id=2019,
                              gbd_round_id=6,
                              decomp_step='step4')
    data = ylds.merge(pop, on=['location_id','sex_id','age_group_id'])
    for i in list(range(0,1000)):
        data[f'draw_{i}'] = data[f'draw_{i}'] * data['population']
    data = data.groupby(['location_id']).sum()
    data = data.drop(columns=[c for c in data.columns if 'draw' not in c])
    data = data * burden_multiplier
    return data + attributable_dalys

def make_ses_comparison_dot_plots(data, nutrient, measure, coverage_levels, subtitle, output_filename, wra=False):
    """This function takes a dataframe,
    nutrient (as a string),
    and measure (as a string, either: 'rates', 'counts', or 'pifs').
    """

    f, ax = plt.subplots(figsize=(7, 4), dpi=120)
    colors = ['tab:red', 'tab:orange', 'tab:green']

    location_spacer = 0.15
    coverage_spacer = 0.025
    df = data.drop(columns='measure', errors='ignore').apply(pd.DataFrame.describe, percentiles=[0.025, 0.975], axis=1).reset_index()

    for n in list(range(0, len(coverage_levels))):
        rate = (df.loc[df.year == 2025]
            .loc[df.coverage_level == coverage_levels[n]])
        for i in list(range(0, len(rate))):
            plt.plot([location_spacer * i + coverage_spacer * n, location_spacer * i + coverage_spacer * n],
                     [rate['2.5%'].values[i], rate['97.5%'].values[i]], c='black')
            plt.scatter([location_spacer * i + coverage_spacer * n], rate['2.5%'].values[i], s=50, marker='_',
                        c='black')
            plt.scatter([location_spacer * i + coverage_spacer * n], rate['97.5%'].values[i], s=50, marker='_',
                        c='black')

        x_vals = []
        for x in list(range(0, len(rate))):
            x_vals.append(location_spacer * x + coverage_spacer * n)
        plt.scatter(x_vals, rate['mean'], s=50,
                    label=f'{int(coverage_levels[n] * 100)} percent coverage', color=colors[n])

    plt.hlines(0, 0 - coverage_spacer * 2,
               location_spacer * (len(rate)) - coverage_spacer * 2,
               linestyle='dashed', color='grey', alpha=0.5)

    plt.plot()

    if wra == True:
        subpop = 'Women of Reproductive Age'
    else:
        subpop = 'Children Under Five'

    if measure == 'rates':
        plt.title(f'DALYs Averted per 100,000 Person-Years due to\n{nutrient} Fortication Among {subpop}\n{subtitle}')
        plt.ylabel('DALYs Averted per 100,000')
    elif measure == 'counts':
        plt.title(f'DALYs Averted due to\n{nutrient} Fortication Among {subpop}\n{subtitle}')
        plt.ylabel('DALYs')
    elif measure == 'pifs':
        plt.title(f'Population Impact Fraction of {nutrient} Fortication\non DALYs Among {subpop}\n{subtitle}')
        plt.ylabel('Population Impact Fraction (Percent)')

    plt.legend(bbox_to_anchor=[1.5, 1])

    x_ticks = []
    for x in list(range(0, len(rate))):
        x_ticks.append(location_spacer * x + coverage_spacer)
    ax.set_xticks(x_ticks)
    plt.xticks(rotation=90)
    l = get_ids('location')
    l_names = df.loc[df.coverage_level == coverage_levels[0]].loc[df.year == 2025]
    l_names = l_names.reset_index().merge(l, on='location_id')
    l_names['label'] = l_names.location_name + ' ' + l_names.subgroup
    l_names = list(l_names.label.values)
    ax.set_xticklabels(l_names)
    plt.savefig(f'results_plots/low_ses/{output_filename}.png', bbox_inches='tight')
