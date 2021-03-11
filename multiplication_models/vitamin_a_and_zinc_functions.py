import pandas as pd, numpy as np
from db_queries import get_ids, get_outputs, get_location_metadata, get_population, get_covariate_estimates
from get_draws.api import get_draws
import scipy.stats
import scipy.integrate as integrate
import matplotlib.pyplot as plt

def apply_age_related_effective_coverage_restrictions(data,
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
        ['location_id', 'age_group_id', 'sex_id', 'year'] + [c for c in final.columns if c == 'coverage_level'])
             .sort_index())
    return final
    

def calculate_vitamin_a_time_lag_effective_fraction(df, years):
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
            current = 1 - ((current - prior) * 5 / 12 / current)
        current['year'] = years[i]
        final = pd.concat([final, current])
    final = final.reset_index().set_index([c for c in data.columns if 'draw' not in c]).sort_index()
    return final
    
def get_effective_vitamin_a_coverage(df, sex_ids, age_group_ids, effective_fractions, years):
    """This function takes a total population coverage dataframe and applies age and time lag
    effective coverage restrictions for population levels of effective vitamin a fortification
    coverage by sex, age group, and year"""
    effective_coverage_by_age = apply_age_related_effective_coverage_restrictions(df,
                                                                                            sex_ids,
                                                                                            age_group_ids,
                                                                                            effective_fractions)
    effective_fraction_by_time_lag = calculate_vitamin_a_time_lag_effective_fraction(df, years)
    effective_coverage = effective_coverage_by_age * effective_fraction_by_time_lag
    effective_coverage = (effective_coverage.reset_index()
                          .set_index(['location_id', 'sex_id', 'age_group_id', 'year'] +
                                     [c for c in effective_coverage.reset_index().columns if c == 'coverage_level'])
                          .sort_index())

    return effective_coverage
    
def pull_deficiency_attributable_dalys(rei_id,
                                              location_ids,
                                              ages,
                                              sexes,
                                              index_cols):
    """This function pulls deficiency attributable DALYs
    from GBD 2019 using a specified risk ID

    This can be used for vitamin A and zinc"""

    data = get_draws(
            gbd_id_type=['rei_id', 'cause_id'],
            gbd_id=[rei_id, 294],
            source='burdenator',
            measure_id=2,  # dalys
            metric_id=1,  # number
            location_id=location_ids,
            year_id=2019,
            age_group_id=ages,
            sex_id=sexes,
            gbd_round_id=6,
            status='best',
            decomp_step='step5'
        ).set_index(index_cols)
    data = data.drop(columns=[c for c in data.columns if 'draw' not in c]).sort_index()
    return data

def calculate_paf_deficiency_nofort(rr_deficiency_nofort, effective_baseline_coverage):
    """This function calculates the population attributable fraction of UNfortified food
    on the fortification outcome of interest (outcome defined in the fortification
    effect size, which is generally nutrient deficiency)

    NOTE: this function does not consider age/time lags of fortification effects
    (assumes that every individual covered by fortification is effectively covered)"""

    paf_deficiency_nofort = ((rr_deficiency_nofort - 1) * (1 - effective_baseline_coverage)) / (
            (rr_deficiency_nofort - 1) * (1 - effective_baseline_coverage) + 1)
    return paf_deficiency_nofort


def calculate_pif_deficiency_nofort(paf_deficiency_nofort,
                                    effective_baseline_coverage,
                                    effective_alternative_coverage):
    """This function calculates the population impact fraction for UNfortified
    foods and nutrient deficiency based on the location-specific coverage
    levels of fortified foods; specifically, p (1 - proportion of population
    that eats fortified vehicle) and p_start (1 - proportion of population that
    eats industrially produced vehicle).

    NOTE: this function does not consider age/time lags of fortification effects
    (assumes that every individual covered by fortification is effectively covered)"""
    pif_deficiency_nofort = (paf_deficiency_nofort *
                             (effective_alternative_coverage
                              - effective_baseline_coverage)
                             / (1 - effective_baseline_coverage))
    return pif_deficiency_nofort

def calculate_final_pifs_and_daly_reductions(pif_deficiency_nofort,
                                             deficiency_dalys,
                                             coverage_levels, years):
    """This function calcualtes the PIF for fortification on DALYs as well as the
    overall reduction in the number of DALYs at the location, year, draw, and coverage
    level specific level"""

    dalys_reduction = pd.DataFrame()
    for coverage_level in coverage_levels:
        pif_deficiency_nofort_level = (pif_deficiency_nofort.reset_index()
                                       .loc[pif_deficiency_nofort.reset_index().coverage_level == coverage_level]
                                       .drop(columns='coverage_level')
                                       .set_index(['location_id', 'sex_id', 'age_group_id', 'year']))
        daly_reduction_level = pif_deficiency_nofort_level * deficiency_dalys
        daly_reduction_level['coverage_level'] = coverage_level
        dalys_reduction = pd.concat([dalys_reduction, daly_reduction_level])

    daly_reduction = (dalys_reduction.reset_index()
                      .set_index([c for c in dalys_reduction.reset_index().columns if 'draw' not in c])
                      .replace(np.nan, 0)).groupby(['location_id','year','coverage_level']).sum()
    overall_pifs = daly_reduction / deficiency_dalys.groupby('location_id').sum() * 100

    return overall_pifs, daly_reduction

def calculate_rates(data, location_ids, age_group_ids, sex_ids):
    pop = (get_population(location_id=location_ids,
                             age_group_id=age_group_ids,
                             sex_id=sex_ids,
                             year_id=2019,
                             gbd_round_id=6,
                             decomp_step='step4')
           .groupby('location_id').sum()
           .filter(['population']))
    rates = data.reset_index().merge(pop.reset_index(), on='location_id')
    for i in list(range(0,1000)):
        rates[f'draw_{i}'] = rates[f'draw_{i}'] / rates['population'] * 100_000
    rates = rates.drop(columns='population')
    rates = rates.set_index([c for c in rates.columns if 'draw' not in c])
    return rates