import pandas as pd, numpy as np
from db_queries import get_ids, get_outputs, get_location_metadata, get_population, get_covariate_estimates
from get_draws.api import get_draws
import scipy.stats
import scipy.integrate as integrate
import matplotlib.pyplot as plt

def generate_normal_rr_deficiency_nofort_draws(mean, std, location_ids):
    import pandas as pd, numpy as np
    """This function takes a distribution for the relative risk
    for lack of fortification of a particular nutrient and generates
    1,000 draws based on that distribution. The data is the duplicated
    so that it is the same for each location ID so that it can be easily
    used later in the calculations."""
    data = pd.DataFrame()
    np.random.seed(7)
    data['rr'] = np.random.normal(mean, std, size=1000)
    draws = []
    for i in list(range(0, 1000)):
        draws.append(f'draw_{i}')
    data['draws'] = draws
    data = pd.DataFrame.pivot_table(data, values='rr', columns='draws').reset_index().drop(columns=['index'])
    df = pd.DataFrame(np.repeat(data.values, len(location_ids), axis=0))
    df.columns = data.columns
    df['location_id'] = location_ids
    df = df.set_index('location_id')
    return df

def apply_iron_hemoglobin_age_related_effective_coverage_restrictions(data,
                                                                sex_ids,
                                                                age_group_ids,
                                                                effective_fractions):
    """This function takes an dataframe of population coverage and generates a dataframe of *effective* coverage
    rates by age group, using the effective coverage assumptions for vitamin A by age (no effect of fortification under
    six months of age)"""
    final = pd.DataFrame()
    for n in list(range(0, len(sex_ids))):
        out_data = pd.DataFrame()
        for i in list(range(0, len(age_group_ids))):
            temp = (data * effective_fractions[i]).reset_index()
            temp['age_group_id'] = age_group_ids[i]
            out_data = pd.concat([out_data, temp], ignore_index=True)
        out_data['sex_id'] = sex_ids[n]
        final = pd.concat([final, out_data], ignore_index=True)
    final = (final.set_index(
        ['location_id','vehicle', 'age_group_id', 'sex_id', 'year'] + [c for c in final.columns if c == 'coverage_level'])
             .sort_index())
    return final


def calculate_iron_hemoglobin_time_lag_effective_fraction(df, years):
    """This function calculates the proportion of individuals covered by vitamin a fortification who
    are recieving an effect from the fortification based on the time lag assumptions (5 month delay
    from the start of new coverage until vitamin a fortification has an effect on vitamin a deficiency).
    This function also assumes that everyone who is covered at baseline has been covered for at least five
    months and therefore 100% of covered individuals are effectively covered at baseline."""
    final = pd.DataFrame()
    data = df.reset_index()
    for i in list(range(0, len(years))):
        current = (data.loc[data.year == years[i]]
                   .set_index([c for c in data.columns if 'draw' not in c and c != 'year'])
                   .drop(columns='year'))
        if i == 0:
            for draw in list(range(0, 1000)):
                current[f'draw_{draw}'] = 1
        else:
            prior = (data.loc[data.year == years[i - 1]]
                     .set_index([c for c in data.columns if 'draw' not in c and c != 'year'])
                     .drop(columns='year'))
            current = 1 - ((current - prior) * 0.75 / current)
        current['year'] = years[i]
        final = pd.concat([final, current])
    final = final.reset_index().set_index([c for c in data.columns if 'draw' not in c]).sort_index()
    return final


def get_effective_iron_hemoglobin_coverage(df, sex_ids, age_group_ids, effective_fractions, years):
    """This function takes a total population coverage dataframe and applies age and time lag
    effective coverage restrictions for population levels of effective vitamin a fortification
    coverage by sex, age group, and year"""
    effective_coverage_by_age = apply_iron_hemoglobin_age_related_effective_coverage_restrictions(df,
                                                                                            sex_ids,
                                                                                            age_group_ids,
                                                                                            effective_fractions)
    #effective_fraction_by_time_lag = calculate_iron_hemoglobin_time_lag_effective_fraction(df, years)
    effective_coverage = effective_coverage_by_age #* effective_fraction_by_time_lag
    print('NOTE: not currently applying time lag effect.')
    effective_coverage = (effective_coverage.reset_index()
                          .set_index(['location_id', 'sex_id', 'age_group_id', 'vehicle', 'year'] +
                                     [c for c in effective_coverage.reset_index().columns if c == 'coverage_level'])
                          .sort_index())

    return effective_coverage
    
def generate_hemoglobin_values(delta_effective_coverage, 
                               mean_difference_hemoglobin_fort, 
                               location_ids, age_group_ids, sex_ids):
    hgb_mean = get_draws('modelable_entity_id',
                    10487,
                    source='epi',
                    location_id=location_ids,
                    age_group_id=age_group_ids,
                    sex_id=sex_ids,
                    year_id=2019,
                    gbd_round_id=6,
                    decomp_step='step4',
                    status='best')
    hgb_mean_prepped = hgb_mean.set_index(['location_id','sex_id','age_group_id'])
    hgb_mean_prepped = hgb_mean_prepped.drop(columns=[c for c in hgb_mean_prepped.columns if 'draw' not in c])
    counterfactual_hgb_mean = hgb_mean_prepped + delta_effective_coverage * mean_difference_hemoglobin_fort
    counterfactual_hgb_mean_prepped = counterfactual_hgb_mean.reset_index().rename(columns={'year':'year_id'})
    hgb_mean_prepped['coverage_level'] = 'baseline'
    hgb_mean_prepped = hgb_mean_prepped.reset_index()
    hgb_mean_vehicle = pd.DataFrame()
    for vehicle in delta_effective_coverage.reset_index().vehicle.unique():
        hgb_mean_prepped_temp = hgb_mean_prepped.copy()
        hgb_mean_prepped_temp['vehicle'] = vehicle
        hgb_mean_vehicle = pd.concat([hgb_mean_vehicle, hgb_mean_prepped_temp], ignore_index=True)
    mean_hgb_overall = (pd.concat([hgb_mean_vehicle,
                                  counterfactual_hgb_mean_prepped.reset_index()],
                                ignore_index=True)
                        .drop(columns='index')
                        .set_index(['location_id','vehicle','sex_id','age_group_id','year_id','coverage_level']))
    return mean_hgb_overall
    
def load_anemia_prev_and_calculate_ylds(file):
    data = pd.read_csv(file)
    data = (data.loc[data.draw.str.contains('draw')]
                   .filter(['age_group_id','sex_id','location_id','draw','vehicle',
                            'mild','moderate','severe','anemic','year_id','coverage_level'])
           .rename(columns={'year_id':'year'}))
    data['mild_ylds'] = data['mild'] * 0.004
    data['moderate_ylds'] = data['moderate'] * 0.052
    data['severe_ylds'] = data['severe'] * 0.149
    data['anemic_ylds'] = data['mild_ylds'] + data['moderate_ylds'] + data['severe_ylds']
    return data
    
def population_weight_values(df, age_group_ids, sex_ids, location_ids):
    pop = get_population(location_id=location_ids,
                        sex_id=sex_ids,
                        age_group_id=age_group_ids,
                        gbd_round_id=6,
                         year_id=2019,
                         decomp_step='step4')
    df = df.merge(pop, on=['location_id','sex_id','age_group_id'])
    for col in ['mild','moderate','severe','anemic','mild_ylds','moderate_ylds','severe_ylds','anemic_ylds']:
        df[f'{col}'] = df[f'{col}'] * df['population']
    counts = df.groupby(['location_id','vehicle','year','draw','coverage_level']).sum()
    rates = counts.copy()
    for col in ['mild','moderate','severe','anemic','mild_ylds','moderate_ylds','severe_ylds','anemic_ylds']:
        rates[f'{col}'] = rates[f'{col}'] / rates['population'] * 100_000
    counts = counts.filter(['mild','moderate','severe','anemic','mild_ylds','moderate_ylds','severe_ylds','anemic_ylds'])
    rates = rates.filter(['mild','moderate','severe','anemic','mild_ylds','moderate_ylds','severe_ylds','anemic_ylds'])
    return counts, rates
    
def summarize_data(df):
    data = df.stack().reset_index().rename(columns={0:'value','level_5':'severity'})
    data['measure'] = np.where(data.severity.str.contains('ylds'), 'ylds', 'prevalence')
    data['severity'] = data.severity.str.split('_', expand=True)[0]
    data = pd.pivot_table(data, index=['location_id','vehicle','year','coverage_level','severity','measure'],
                         values='value', columns='draw').reset_index()
    data['coverage_level'] = data.coverage_level.astype(float)
    return data
    
def duplicate_over_simulation_years(df, years):
    data_years = pd.DataFrame()
    for year in years:
        temp = df.copy()
        temp['year'] = year
        data_years = pd.concat([data_years, temp], ignore_index=True)
    return data_years