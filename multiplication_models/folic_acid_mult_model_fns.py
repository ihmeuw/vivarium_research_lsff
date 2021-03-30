import pandas as pd, numpy as np
from db_queries import get_ids, get_outputs, get_population
from get_draws.api import get_draws

def calc_dalys_averted(dalys, pif):
    a = dalys.reset_index().set_index(['location_id','age_group_id','sex_id','cause_id'])
    b = pif.reset_index().set_index(['location_id','age_group_id','sex_id','year_id','coverage_level'])
    
    return a * b

def pull_dalys(cause_ids, nonfatal_cause_ids, location_ids, ages, sexes, index_cols):
    """
    This function pulls dalys for specified cause IDs from GBD
    -----
    INPUT (all in List() format):
    - cause
    ids for YLL models
    - nonfatal_cause_ids for YLD models
    - location_ids for which to pull dalys
    - ages (age_group_ids) for which to pull dalys
    - sexes (sex_ids) for which to pull dalys
    - index_cols with which to format output
    -----
    @returns a drawspace dataframe of DALYS attributable to each fatal or nonfatal cause_id:
        - columns = draws
        - index = multiindex(loc_id, sex_id, age_group_id, cause_id)
    """
    if len(cause_ids) + len(nonfatal_cause_ids) == 0:
        raise Exception("Must select at least one fatal or nonfatal cause_id")
        
    #init empty dfs
    ylds, ylls = pd.DataFrame(), pd.DataFrame()
    
    if len(nonfatal_cause_ids)>0:
        ylds = get_draws(
            gbd_id_type='cause_id',
            gbd_id=cause_ids,
            source='como',
            measure_id=3,
            metric_id=3,  # only available as rate
            location_id=location_ids,
            year_id=2019,
            age_group_id=ages,
            sex_id=sexes,
            gbd_round_id=6,
            status='best',
            decomp_step='step5',
        ).set_index(index_cols + ['cause_id'])
        ylds = ylds.drop(columns=[c for c in ylds.columns if 'draw' not in c])

        #convert rate to count
        pop = get_population(
            location_id=location_ids,
            year_id=2019,
            age_group_id=ages,
            sex_id=sexes,
            gbd_round_id=6,
            decomp_step='step4').set_index(index_cols)
        for i in list(range(0, 1000)):
            ylds[f'draw_{i}'] = ylds[f'draw_{i}'] * pop['population']
    else:
        print("No nonfatal ids selected; returning ylls only")
    
    if len(cause_ids)>0:
        ylls = get_draws(
            gbd_id_type='cause_id',
            gbd_id=cause_ids,
            source='codcorrect',
            measure_id=4,
            metric_id=1,
            location_id=location_ids,
            year_id=2019,
            age_group_id=ages,
            sex_id=sexes,
            gbd_round_id=6,
            status='latest',
            decomp_step='step5',
        ).set_index(index_cols + ['cause_id']).replace(np.nan, 0)
        ylls = ylls.drop(columns=[c for c in ylls.columns if 'draw' not in c])
    else:
        print("No fatal ids selected; returning ylds only")
    
    return ylls + ylds

def pif_o_r(paf_o_r, alpha, alpha_star):
    """
    Following Nathaniel's Vitamin A multiplicative model writeup
    Calculates, in draw space, the potential impact fraction (PIF)
    of fortification (i.e., risk exposure prevalence reduction)
    on some outcome, o.
    INPUT:
    - paf_o_r: - the PAF of r on o
               - a DataFrame with index = location_ids, columns=draws
    - alpha:  - the baseline proportion of the population receiving fortification
              - thus (1 - alpha) = the prop of the pop experiencing the risk 'r' or 'no fort' for 'o' or 'NTDs'
              - a DataFrame with index = location_ids, columns=draws 
    - alpha_star:  - the counterfactual proportion of the population receiving fortification
                   - a DataFrame with index = location_ids, columns=draws 
    """
    return paf_o_r * ((alpha_star - alpha) / (1 - alpha))

def paf_o_r(rr_o_r, alpha):
    """
    Following Nathaniel's Vitamin A multiplicative model writeup
    Calculates, in draw space, the PAF of some outcome 'o' (such as NTD incidence)
    given some dichomotous exposure 'r' (such as a lack of folic acid fortification)
    ----
    INPUT:
    - rr_o_r: - the risk of o given r.
              - a DataFrame with index = location_ids, columns=draws
    - alpha:  - the baseline proportion of the population receiving fortification
              - thus (1 - alpha) = the prop of the pop experiencing the risk 'r' or 'no fort' for 'o' or 'NTDs'
              - a DataFrame with index = location_ids, columns=draws
    ----
    @returns: the PAF r on o in drawspace. a DataFrame with index = loc_ids, columns = draws
    """
    return ((rr_o_r - 1) * (1 - alpha)) / ((rr_o_r - 1) * (1 - alpha) + 1)


def percolate_new_coverage(gets_intervn, alpha, alpha_star):
    
    index_cols = ['location_id','year_id','age_group_id','sex_id']
    a = gets_intervn * alpha_star
    a = a.reset_index().set_index(index_cols + ['coverage_level'])
    
    b = (1-gets_intervn) * alpha
    b = b.reset_index().set_index(index_cols)
    
    return a + b

def pull_u5_age_groups_formatted():
    """
    @returns a pd.DataFrame() with
    - age_group_id
    - age_group_name
    - age_start float

    for neonatal + 1 to 4 year old age groups  
    """
    age_start_map = {
    'Early Neonatal': 0,
    'Late Neonatal': 7/365,
    'Post Neonatal': 28/365,
    '1 to 4': 1
    }

    age_end_map = {
        'Early Neonatal': 7/365,
        'Late Neonatal': 28/365,
        'Post Neonatal': 365/365,
        '1 to 4': 5
    }

    # pull age 
    age_groups = get_ids("age_group")
    age_groups = age_groups[age_groups.age_group_id.isin([2, 3, 4, 5])]
    age_groups['age_start'] = age_groups.age_group_name.map(age_start_map)
    age_groups['age_end'] = age_groups.age_group_name.map(age_end_map)

    return age_groups

def get_age_1_4_age_splits(location_ids, sexes):
    """
    @returns a pd.DataFrame() with
    - specified location_ids
    - age_group_ids for 1, 2, 3, and 4 year olds
    - sex_ids
    - "age_name" with an age group label
    - "prop_1_4" with the proportion of age_group X out of age_group_id 5 
    """
    # pull population data
    location_ids = list(location_ids)
    age_split_pop_count = get_population(
        location_id=location_ids,
        year_id=2019,
        age_group_id=[49,50,51,52],
        single_year_age=True,
        sex_id=sexes,
        gbd_round_id=6,
        decomp_step='step4')

    # calculate proportions
    age_split_pop_count['denom'] = age_split_pop_count.groupby('location_id').transform('sum').population
    age_split_pop_count['prop_1_4'] = age_split_pop_count.population / age_split_pop_count.denom

    # add formatting
    age_group_names = {
        49 : 'age1',
        50 : 'age2',
        51 : 'age3',
        52 : 'age4'
    }
    age_split_pop_count['age_name'] = age_split_pop_count.age_group_id.map(age_group_names)
    
    usecols = ['location_id','age_group_id','sex_id','age_name','prop_1_4']
    return age_split_pop_count[usecols]



def prop_gets_intervention_effect(location_ids, year_start, estimation_years = range(2022,2026)):
    """
    PURPOSE:
    Calculate, for each year, the proportion of each age/sex/loc_id group receiving benefit
    from folic acid fortification, assuming folic acid fortification in the specified population
    ------
    INPUT:
    - year intervention starts (assumes Jan 1)
    - estimation years (list)
    ------
    OUTPUT:
    - DataFrame with index cols:
        - year_id
        - age_group_id
        - location_id
        - sex_id
    - with columns = draws, containing the proportion that gets intervention effect
    NOTE: CURRENTLY DRAWS ARE ALL THE SAME, HAVE NO UNCERTAINTY
    ------
    ASSUMPTIONS:
    Children of mothers who receive the intervention (food fortified with folic acid)
    for >= 3 months preceeding conception receive the benefit.
    This means that during year ZERO, no one benefits from the intervention.
    In year ONE, proportions of the following will receive benefit:
        - ealy neonatal ([0,7) days),
        - late neonatal ([7,28) days) and
        - post neonatal ([28,365) days)
    In year TWO, all neonatal groups and prorortions of age_group_id = 5 (1 to 4 year olds) will receive benefit
    ...
    In year SIX, all age groups under 5 would receive benefit
    
    
    We calculate these proportions assuming that babies are born at a constant rate.
    """
    sexes = [1,2]
    ages = [2,3,4,5]
    
    df = pd.DataFrame([(loc,sex,age,year)
              for loc in location_ids
              for sex in sexes
              for age in ages
             for year in estimation_years],
             columns = ['location_id','sex_id','age_group_id','year_id'])
    
    # MERGE ON AGE DATA FOR CALCULATIONS ---------------------------------------
    
    # merge on age range data
    age_groups = pull_u5_age_groups_formatted()
    df = df.merge(age_groups, on = ['age_group_id'], how = 'left')

    # merge on population-weighted age splits
    age_split_pop_count = get_age_1_4_age_splits(location_ids, sexes)
    age_split_pop_count['age_group_id'] = 5 #this is the id we want to merge onto
    
    # cast to wide
    age_split_pop_count = pd.pivot_table(data = age_split_pop_count, index = ['location_id','sex_id','age_group_id'], columns = 'age_name', values = 'prop_1_4')
    # merge
    df = df.merge(age_split_pop_count.reset_index(), on = age_split_pop_count.index.names, how = 'left')


    ## CALCULATE PROPORTION AFFECTED ------------------------------------------

    # if you were born in or before the year fortification started, get zero benefit wrt rr(NTDs)
    df.loc[(df.year_id - df.age_start) <= year_start,'prop_gets_intervention_effect'] = 0

    # if the oldest children in your cohort were born at least one year 
    # after fortification started, everyone in your age_group/year gets benefit
    df.loc[(df.year_id - df.age_end >= year_start + 1),'prop_gets_intervention_effect'] = 1


    # a proportion of each of the <1 year old age cohorts receives benefits
    # starting one year after the intervention starts
    # we assume children born at a constant rate to calculate this proportion
    # == whole age cohort
    #    - (the fraction of a year before those born on jan 1 age into the cohort)
    #    - 0.5 * (the fraction of a year during which those born before jan 1 still make up part of age cohort)
    df.loc[(df.year_id - year_start == 1) & (df.age_start < 1),
             'prop_gets_intervention_effect'] = 1 - df.age_start - 0.5 * (df.age_end - df.age_start)

    # started two years ago: those who age into "one year old" this year get benefit
    df.loc[((df.year_id - year_start) == 2) & (df.age_group_id==5),
            'prop_gets_intervention_effect'] = 0.5 * df.age1

    # started three years ago: those who are age 1 or
    # age into "two years old" this year get benefit
    df.loc[((df.year_id - year_start) == 3) & (df.age_group_id==5),
            'prop_gets_intervention_effect'] = df.age1 + 0.5 * df.age2

    # started four years ago: those who are age 1-2 or
    # age into "three years old" this year get benefit
    df.loc[((df.year_id - year_start) == 4) & (df.age_group_id==5),
            'prop_gets_intervention_effect'] = df.age1 + df.age2 + 0.5 * df.age3

    # started five years ago: those who are age 1-3 or
    # age into "four years old" this year get benefit
    df.loc[((df.year_id - year_start) == 5) & (df.age_group_id==5),
            'prop_gets_intervention_effect'] = df.age1 + df.age2 + df.age3 + 0.5 * df.age4
    
    assert(len(df[df.prop_gets_intervention_effect.isna()])==0), "Some cases missed; fn error"

    df = df[['year_id','location_id','sex_id','age_group_id','prop_gets_intervention_effect']]
    
    #TODO: add uncertainty
    #Here we assume babies in a given age group born at constant rate
    #Could replace with:
        #Given a baby is born in [t_0,t_1), what is the probability is it born
        #in [t_i, t_j) \subset [t_0,t_1)?
    draws = [f'draw_{i}' for i in range(1_000)]
    for i in draws:
        df[i] = df.prop_gets_intervention_effect
    
    return df.set_index(['location_id','sex_id','age_group_id','year_id'])[draws]