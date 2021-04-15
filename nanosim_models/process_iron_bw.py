import pandas as pd, numpy as np
import os

from db_queries import get_ids, get_population
from vivarium_helpers.id_helper import *

import data_processing

def load_iron_bw_results(directory: str) -> dict:
    """
    Loads iron-LBW output file into a dataframe,
    and returns a dictionary whose keys are the location names and values are
    are the corresponding dataframes.
    """
    dfs = {}
    for entry in os.scandir(directory):
        filename_root, extension = os.path.splitext(entry.name)
        if extension == '.csv' and filename_root.startswith('iron_bw_results__'):
            location = filename_root.split('__')[1]
#             print(filename_root, type(filename_root), extension, entry.path, location)
            dfs[location] = pd.read_csv(entry.path)
    return dfs

def filter_by_draw_count(locations_outputs:dict, num_draws:int)->dict:
    """Filter output dataframes to only those with the specified number of draws.
    (Necessary because 'results' folder contains data from some preliminary runs
    with fewer than 250 draws.)
    """
    return {location: output_df for location, output_df in locations_outputs.items()
            if len(output_df.filter(like='draw').columns) == num_draws}

def merge_location_outputs(locations_outputs: dict, copy=True) -> pd.DataFrame:
    """
    Concatenate the output DataFrames for all locations stored in locations_outputs into a single DataFrame.
    """
    return pd.concat(locations_outputs.values(), sort=False)

# Master function to read in all data:

def load_and_merge_location_outputs(*directories, num_draws=250):
    locations_outputs = {}
    for directory in directories:
        locations_outputs.update(filter_by_draw_count(load_iron_bw_results(directory), num_draws))
    return merge_location_outputs(locations_outputs)

def filter_measures_and_coverage(data):
    """Filter to coverage levels [0.2, 0.5, 0.8] and measure == 'averted_dalys'
    Leaving out coverage level corresponding to 100% population coverage
    and measure == 'categorical_pif' because these are unneeded for results.
    Leaving out measure=='pif' because we'll compute an aggregate PIF using averted DALYs.
    """
    return data.query("coverage_level in [0.2, 0.5, 0.8] and measure in ['averted_dalys']")

def filter_data(data, filters):
    query_string = " and ".join(filters)
    return data.query(query_string)

def aggregate_over_age_groups(data):
    """Add draws over different age groups."""
    draw_cols = data.filter(like='draw').columns.to_list()
    index_cols = data.columns.difference(['age_group_id', *draw_cols]).to_list()
    return data.groupby(index_cols)[draw_cols].sum()

def get_dalys_averted_u5(data):
    """ 1. Filter to coverage levels [0.2, 0.5, 0.8] and measure == 'averted_dalys'.
    2. Add averted DALYs for age groups 2 and 3 to get total for Under 5.
    (Note that DALYs exist in age groups 4 and 5 for preterm birth (but not other affected causes),
    but I'm not counting them because RRs are only for age groups 2 and 3, and iron only affects
    birthweight, not gestational age.)
    3. Rename measure 'averted_dalys' to 'counts_averted' to match Ali's code.
    """
    dalys_averted = (
        data
        .query("measure=='averted_dalys'")
        .replace('averted_dalys', 'counts_averted') # To match Ali's code
        .pipe(aggregate_over_age_groups)
    )
    return dalys_averted
    
def compute_aggregate_pif(dalys_averted):
    """Use aggregated averted DALYs to compute overall PIF (in units of %) for age groups 2 and 3."""
#     draw_cols = data.filter(like='draw').columns.to_list()
    location_ids = dalys_averted.index.unique('location_id')
    hdfstore = '/share/scratch/users/ndbs/vivarium_lsff/gbd_data/lbwsg_data.hdf'
    daly_key = '/gbd_2019/burden/dalys/bmgf_25_countries_all_causes_u5'
    lbwsg_dalys = (
        pd.read_hdf(hdfstore, daly_key)
        .query('age_group_id in [2,3] and metric_id==1')
        .query("location_id in @location_ids") # Filter to locations in data
        .groupby(['location_id'])
        .sum() # Aggregate over age_group and sex
        [dalys_averted.columns] # Filter to draws that exist in data
    )
    pif = (dalys_averted / lbwsg_dalys) * 100 # convert to percent
    pif.index = pif.index.set_levels(['pif'], level='measure') # measure was still counts_averted, so rename
    return pif

def pull_u5_population_for_locations(location_ids):
    pop_u5 = get_population(
        age_group_id=1,
        location_id=location_ids,
        year_id=2019,
        gbd_round_id=6,
        status='best',
        decomp_step='step5',
        with_ui=True
    )
    return pop_u5

def get_u5_population_draws_for_locations(location_ids, draws, random_state=None):
    pop_u5 = pull_u5_population_for_locations(location_ids)
    values = data_processing.generate_normal_draws(
        pop_u5['population'], pop_u5['lower'], pop_u5['upper'],
        shape=(len(draws), len(pop_u5)),
        random_state=random_state
    )
    pop_u5_draws = pd.DataFrame(values.T, index=pop_u5['location_id'], columns=draws)
    return pop_u5_draws

def compute_averted_daly_rate(dalys_averted):
    location_ids = dalys_averted.index.unique('location_id').to_list()
    # Set the random state so we'll get consistent results
    pop_u5 = get_u5_population_draws_for_locations(location_ids, dalys_averted.columns, random_state=246)
    dalys_averted_rate = (dalys_averted / pop_u5) * 100_000 # Convert to per 100K person-years
    dalys_averted_rate.index = dalys_averted_rate.index.set_levels(
        ['rates_averted'], level='measure') # measure was still counts_averted, so rename
    return dalys_averted_rate

# Master function to perform all steps

def read_output_and_compute_final_results(*directories, num_draws=250, filters=None):
    # .csv's in 'results3/' used 40_000 simulants per location, while .csv's in 'results2/' and 'results/'
    # used 80_000 simulants per location.
    # Read .csv's in opposite order they were generated in order to overwrite results from smaller populations
    # if results for larger populations exist.
    data = load_and_merge_location_outputs(*directories, num_draws=num_draws)
    if filters is not None:
        data = filter_data(data, filters)
    dalys_averted = get_dalys_averted_u5(data)
    pif = compute_aggregate_pif(dalys_averted)
    averted_daly_rate = compute_averted_daly_rate(dalys_averted)
    return pd.concat([dalys_averted, pif, averted_daly_rate])