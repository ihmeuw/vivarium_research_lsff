import numpy as np, pandas as pd, scipy.stats



def format_coverage_by_year(baseline, counterfactual, year_range, year_start):
    """
    Formats drawspace coverage dataframes (baseline and counterfactual) over the modelled years
    ----
    INPUT:
    - baseline, a pd.DataFrame()
        - index = location_ids
        - columns = draws
        - values = baseline coverage proportion
    - counterfactual, a pd.DataFrame()
        - index = (location_id, coverage_level)
        - columns = draws
        - values = counterfactual coverage proportion for each coverage level & location
    - year_range: the years for which to estimate
    - year_start: the first year of scaled-up coverage in the counterfactual scenario
    ----
    NOTE: this function assumes an immediate (one-year) scale-up
    ----
    OUTPUT:
    - alpha: a pd.DataFrame()
        - index = (location_ids, year_ids)
        - columns = draws
        - values = baseline coverage proportion
    - alpha_star: a pd.DataFrame()
        - index = (location_id, year_id, coverage_level)
        - columns = draws
        - values = counterfactual coverage proportion for each coverage level, year, & location
    """
    # PULL IDS
    location_ids = counterfactual.reset_index().location_id.unique()
    coverage_levels = counterfactual.reset_index().coverage_level.unique()
    
    # BASELINE SCENARIO
    # for all years of simulation, we hold baseline coverage constant
    alpha = baseline.reset_index().merge(
        pd.DataFrame([(loc,year) for loc in location_ids for year in year_range], columns=['location_id','year_id']),
        on='location_id', how = 'outer'
    ).set_index(['location_id', 'year_id']).sort_index()
    
    # COUNTERFACTUAL SCENARIO
    before_start = [year for year in year_range if year < year_start]
    after_start = [year for year in year_range if year >= year_start]
    
    # for years before intervention starts, counterfactual scenario looks like baseline
    counterfactual_before_start = baseline.reset_index().merge(
        pd.DataFrame([(loc,year,lvl) for loc in location_ids for year in before_start for lvl in coverage_levels],
                     columns=['location_id','year_id','coverage_level']),
        on='location_id', how = 'outer'
    )
    
    # for years on/after intervention starts, we assume an immediate scale-up
    counterfactual_after_start = counterfactual.reset_index().merge(
        pd.DataFrame([(loc,year) for loc in location_ids for year in after_start], columns=['location_id','year_id']),
        on='location_id', how = 'outer'
    )
    
    # combine counterfactural coverage for all years
    alpha_star = pd.concat([counterfactual_before_start,counterfactual_after_start]).reset_index().set_index(['location_id', 'year_id', 'coverage_level']).sort_index()
    
    return alpha, alpha_star




def generate_coverage_tables(data, coverage_levels, seed = 11):
    """
    Generates baseline and counterfactual population coverage tables in drawspace
    ----
    INPUT:
    - data: a pd.DataFrame() with columns:
        - location_id
        - value_description
        - value_mean
        - value_025_percentile
        - value_975_percentile
    - a list of coverage levels to model (% of fortifiable for which to achieve coverage)
    - random seed
    ----
    OUTPUT:
    - a baseline pd.DataFrame()
        - index = location_ids
        - columns = draws
    - a counterfactual pd.DataFrame()
        - index = (location_id, coverage_level)
        - columns = draws
    """
    # generate draws from a truncated normal distribution   
    fortified = data.loc[data.value_description == 'percent of population eating fortified vehicle']
    baseline = generate_coverage_parameter_draws(fortified,
                                                  seed = seed,
                                                  lower_bound = 0,
                                                  upper_bound = 100)
    
    fortifiable = data.loc[data.value_description == 'percent of population eating industrially produced vehicle']
    fortifiable = generate_coverage_parameter_draws(fortifiable, 
                                                    seed = seed, 
                                                    lower_bound = 0, 
                                                    upper_bound = 100)
    
    # scale counterfactual fortified coverage by coverage level targets
    counterfactual = pd.DataFrame()
    for lvl in coverage_levels:
        tmp = fortifiable * lvl
        tmp['coverage_level'] = lvl
        counterfactual = pd.concat([counterfactual, tmp])
    counterfactual = counterfactual.reset_index().set_index(['location_id','coverage_level']).sort_index()
    
    return baseline, counterfactual


def generate_coverage_parameter_draws(data, seed, lower_bound = 0, upper_bound = 100):
    """
    helper fn
    """
    assert(data.value_description.nunique()==1), "input data should contain exactly one type of value_description"
    draws = [f'draw_{i}' for i in range(1_000)]
    
    np.random.seed(seed)
    data['value_std'] = (data.value_975_percentile - data.value_mean) / 1.96
    data['a'] = (lower_bound - data.value_mean) / data.value_std
    data['b'] = (upper_bound - data.value_mean) / data.value_std

    out = pd.DataFrame([[data.iloc[i].location_id] +
                  scipy.stats.truncnorm.rvs(data.iloc[i].a,
                                            data.iloc[i].b,
                                            data.iloc[i].value_mean,
                                            data.iloc[i].value_std, size = 1_000).tolist()
                  for i in range(len(data))],
                columns = ['location_id'] + draws)
    
    return out.set_index(['location_id'])/100


def create_marginal_uncertainty(data):
    """
    Replace any rows of data with mean = 100, CIs = 0 with CIs=epislon>0
    This is a transformation for a potential data issue and should be removed when resolved
    ----
    INPUT:
    - pd.DataFrame() with columns ['value_mean','value_025_percentile','value_975_percentile']
    ---
    OUTPUT:
    - input dataframe, with all rows with (100,100,100) replaced with marginal confidence intervals|
    """
    # the following is a transformation for a potential data issue and should be removed when resolved
    data['value_mean'] = data['value_mean'].replace(100, 100 - 0.00001 * 2)
    data['value_025_percentile'] = data['value_025_percentile'].replace(100, 100 - 0.00001 * 3)
    data['value_975_percentile'] = data['value_975_percentile'].replace(100, 100 - 0.00001)
    
    # the following is a transformation for a potential data issue and should be removed when resolved
    data['value_mean'] = data['value_mean'].replace(0, 0 + 0.00001 * 2)
    data['value_025_percentile'] = data['value_025_percentile'].replace(0, 0 + 0.00001)
    data['value_975_percentile'] = data['value_975_percentile'].replace(0, 0 + 0.00001 * 3)
    
    return data


def pull_coverage_data(input_data_path, nutrient, vehicle, location_ids, sub_pop):
    """
    Load pct of fortified and fortifiable data for a given vehicle/nutrient/loc_id set
    ----
    INPUTS:
    - filepath to the vivarium_data_analysis repo. eg, '/ihme/homes/beatrixh/vivarium_data_analysis'
    - nutrient of interest
    - vehicle of interest
    - sup_pop of interest, either 'wra' or 'u5'
    ----
    OUTPUT:
    a pd.DataFrame() with:
    - location_id, location_name, sub_populuation, value_descrip, nutrient, value_mean, value_025_percentile, value_975_percentile
    """
    data = pd.read_csv(input_data_path)
    #TODO: fix this to deal more cleanly with all possible cases!
    if sub_pop=='u5':
        data = data.loc[data.location_id.isin(location_ids)].loc[data.sub_population!='women of reproductive age']
    elif sub_pop=='wra':
        data = data.loc[data.location_id.isin(location_ids)].loc[data.sub_population!='under-5']
    else:
        raise Exception("Subpop must be either 'wra' or 'u5'")
        
    #TODO: ADD A CHECK FOR UNIQUENESS OF ROWS
    return data.loc[(data.vehicle == vehicle) & (data.nutrient.isin([nutrient, 'na']))].drop_duplicates()



def lognormal_draws(mu, sigma, seed):
    """
    INPUT:
    - mean of distirbution
    - std dev of distribution
    - random seed
    -----
    @returns: 1000 draws from a lognormal distribution with requested params
    """
    np.random.seed(seed)
    return np.random.lognormal(mu, sigma, size=1000)


def format_rrs(rrs, location_ids):
    """
    INPUT:
    - an array of 1000 rrs, representing the rr distribution
    - a list of location_ids
    -----
    @ returns: a DataFrame
        - long by loc_id (as index)
        - wide by draws
        - same rr for each draw for each loc_id
    """
    draws = [f'draw_{i}' for i in range(1_000)]
    
    df = pd.DataFrame([rrs for i in location_ids], index = location_ids, columns = draws)
    df.index.name = 'location_id'
    
    return df