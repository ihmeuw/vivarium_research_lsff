"""
This file will probably be a lightweight version of some of the things from the
LBWSG component in Vivarium Public Health:

https://github.com/ihmeuw/vivarium_public_health/blob/master/src/vivarium_public_health/risks/implementations/low_birth_weight_and_short_gestation.py
"""

import pandas as pd, numpy as np
import re
from typing import Tuple#, Dict, Iterable
from pandas.api.types import CategoricalDtype
from scipy.interpolate import interp2d, griddata, RectBivariateSpline

import demography
# from demography import get_age_group_data, get_sex_id_map

# import gbd_mapping as gbd
import importlib
if importlib.util.find_spec('gbd_mapping') is not None:
    gbd_mapping = importlib.import_module('gbd_mapping')
if importlib.util.find_spec('db_queries') is not None:
    get_ids = importlib.import_module('db_queries').get_ids
if importlib.util.find_spec('get_draws.api') is not None:
    get_draws = importlib.import_module('get_draws.api').get_draws

LBWSG_REI_ID = 339 # GBD's "risk/etiology/impairment" id for Low birthweight and short gestation
GBD_2019_ROUND_ID = 6

# The support of the LBWSG distribution is nonconvex, but adding this one category makes it convex,
# which makes life easier when shifting the birthweights or gestational ages.
# I think the category number was just arbitrarily chosen from those that weren't already taken.
# Note: In GBD 2019, this category is `cat124`, meid=20224.
MISSING_CATEGORY_GBD_2017 = {'cat212': 'Birth prevalence - [37, 38) wks, [1000, 1500) g'}

TMREL_CATEGORIES = ('cat53', 'cat54', 'cat55', 'cat56')

# Category to indicate that birthweight and gestational age are outside the domain of the risk distribution
# OUTSIDE_BOUNDS_CATEGORY = 'cat_outside_bounds'

# The dictionary below was created with the following code:
# CATEGORY_TO_MEID_GBD_2019 = (
#     lbwsg_exposure_nigeria_birth_male
#      .dropna()
#      .set_index('parameter')
#      ['modelable_entity_id']
#      .astype(int)
#      .sort_index(key=lambda s: s.str.strip('cat').astype(int))
#      .to_dict()
#     )
# where `lbwsg_exposure_nigeria_birth_male` was exposure data from `get_draws`.

# TODO: Perhaps store the category descriptions here as well
# Modelable entity IDs for LBWSG categories
CATEGORY_TO_MEID_GBD_2019 = {
 'cat2': 10755,
 'cat8': 10761,
 'cat10': 10763,
 'cat11': 10764,
 'cat14': 10767,
 'cat15': 10768,
 'cat17': 10770,
 'cat19': 10772,
 'cat20': 10773,
 'cat21': 10774,
 'cat22': 10775,
 'cat23': 10776,
 'cat24': 10777,
 'cat25': 10778,
 'cat26': 10779,
 'cat27': 10780,
 'cat28': 10781,
 'cat29': 10782,
 'cat30': 10783,
 'cat31': 10784,
 'cat32': 10785,
 'cat33': 10786,
 'cat34': 10787,
 'cat35': 10788,
 'cat36': 10789,
 'cat37': 10790,
 'cat38': 10791,
 'cat39': 10792,
 'cat40': 10793,
 'cat41': 10794,
 'cat42': 10795,
 'cat43': 10796,
 'cat44': 10797,
 'cat45': 10798,
 'cat46': 10799,
 'cat47': 10800,
 'cat48': 10801,
 'cat49': 10802,
 'cat50': 10803,
 'cat51': 10804,
 'cat52': 10805,
 'cat53': 10806,
 'cat54': 10807,
 'cat55': 10808,
 'cat56': 10809,
 'cat80': 20203,
 'cat81': 20204,
 'cat82': 20205,
 'cat88': 20209,
 'cat89': 20210,
 'cat90': 20211,
 'cat95': 20214,
 'cat96': 20215,
 'cat106': 20221,
 'cat116': 20227,
 'cat117': 20228,
 'cat123': 20232,
 'cat124': 20224
}

####################################################
# READING AND PROCESSING GBD DATA FOR LBWSG #
####################################################

def get_lbwsg_category_order():
    """Returns LBWSG categories, sorted numerically."""
    return sorted(CATEGORY_TO_MEID_GBD_2019.keys(), key=lambda s: int(s.strip('cat')))

def get_lbwsg_category_dtype():
    """Gets the data type for LBWSG categories -
    an ordered Catgorical with order determined by get_lbwsg_category_order().
    """
    return CategoricalDtype(categories=get_lbwsg_category_order(), ordered=True)

def read_lbwsg_data_by_draw_from_gbd_2017_artifact(artifact_path, measure, draw, rename=None):
    """
    Reads one draw of LBWSG data from an artifact.
    
    measure should be one of:
    'exposure'
    'relative_risk'
    'population_attributable_fraction'
    rename should be a string or None (default)
    """
    key = f'risk_factor/low_birth_weight_and_short_gestation/{measure}'
    with pd.HDFStore(artifact_path, mode='r') as store:
        index = store.get(f'{key}/index')
        draw = store.get(f'{key}/draw_{draw}')
    if rename is not None:
        draw.rename(rename)
    data = pd.concat([index, draw], axis=1)
    return data

def read_lbwsg_data_from_gbd_2017_artifact(artifact_path, measure, *filter_terms, draws=None):
    """
    Reads multiple draws from the artifact.
    """
    key = f'risk_factor/low_birth_weight_and_short_gestation/{measure}'
    query_string = ' and '.join(filter_terms)
    # NOTE: If draws is a numpy array, the line `if draws=='all':` threw a warning:
    #  "FutureWarning: elementwise comparison failed; returning scalar instead,
    #   but in the future will perform elementwise comparison"
    # So I changed default from 'all' to None.
    if draws is None:
        draws = range(1000)
    
    with pd.HDFStore(artifact_path, mode='r') as store:
        index_cols = store.get(f'{key}/index')
        if query_string != '':
            index_cols = index_cols.query(query_string)
        draw_data_dfs = [index_cols]
        for draw in draws:
            draw_data = store.get(f'{key}/draw_{draw}') # draw_data is a pd.Series
            draw_data = draw_data[index_cols.index] # filter to query on index columns
            draw_data_dfs.append(draw_data)

#     print(index_cols.columns)
    return pd.concat(draw_data_dfs, axis=1, copy=False).set_index(index_cols.columns.to_list())

def pull_lbwsg_exposure_from_gbd_2019(location_ids, year_ids=2019, save_to_hdf=None, hdf_key=None):
    """Calls get_draws to pull LBWSG exposure data from GBD 2019."""
    # Make sure type(location_ids) is list
    location_ids = [location_ids] if isinstance(location_ids, int) else list(location_ids)
    lbwsg_exposure = get_draws(
        'rei_id',
        gbd_id=LBWSG_REI_ID,
        source='exposure',
        location_id=location_ids,
        year_id=year_ids,
#         sex_id=sex_ids, # Default is [1,2], but 3 also exists
        gbd_round_id=GBD_2019_ROUND_ID,
        status='best',
        decomp_step='step4',
    )
    if save_to_hdf is not None:
        if hdf_key is None:
            description = f"location_ids_{'_'.join(location_ids)}"
            hdf_key = f"/gbd_2019/exposure/{description}"
        lbwsg_exposure.to_hdf(save_to_hdf, hdf_key)
    return lbwsg_exposure

def pull_lbwsg_relative_risks_from_gbd_2019(cause_ids=None, year_ids=2019, save_to_hdf=None, hdf_key=None):
    """Calls get_draws to pull LBWSG relative risk data from GBD 2019."""
    global_location_id = 1 # RR's are the same for all locations - they all propagate up to location_id==1
    if cause_ids is None:
        cause_ids = []
    elif isinstance(cause_ids, int):
        cause_ids = [cause_ids]
    lbwsg_rr = get_draws(
            gbd_id_type=['rei_id']+['cause_id']*len(cause_ids), # Types must match gbd_id's
            gbd_id=[LBWSG_REI_ID, *cause_ids],
            source='rr',
            location_id=global_location_id,
            year_id=year_ids,
            gbd_round_id=GBD_2019_ROUND_ID,
            status='best',
            decomp_step='step4',
        )
    if save_to_hdf is not None:
        if hdf_key is None:
            description = 'all' #if len(cause_ids)==0 else f"cause_ids_{'_'.join(cause_ids)}"
            hdf_key = f"/gbd_2019/relative_risk/{description}"
        lbwsg_rr.to_hdf(save_to_hdf, hdf_key)
    return lbwsg_rr

def rescale_prevalence(exposure):
    """Rescales prevalences to add to 1 in LBWSG exposure data pulled from GBD 2019 by get_draws."""
    # Drop residual 'cat125' parameter with meid==NaN, and convert meid col from float to int
    exposure = exposure.dropna().astype({'modelable_entity_id': int})
    # Define some categories of columns
    draw_cols = exposure.filter(regex=r'^draw_\d{1,3}$').columns.to_list()
    category_cols = ['modelable_entity_id', 'parameter']
    index_cols = exposure.columns.difference(draw_cols)
    sum_index = index_cols.difference(category_cols)
    # Add prevalences over categories (indexed by meid and/or parameter) to get denominator for rescaling
    prevalence_sum = exposure.groupby(sum_index.to_list())[draw_cols].sum()
    # Divide prevalences by total to rescale them to add to 1, and reset index to put df back in original form
    exposure = exposure.set_index(index_cols.to_list()) / prevalence_sum
    exposure.reset_index(inplace=True)
    return exposure

def preprocess_gbd_data(df, draws=None, filter_terms=None, mean_draws_name=None):
    """df can be exposure or rr data, or burden (DALYs/YLLs/YLDs) attributable to LBWSG.
    Note that location_id for rr data will always be 1 (Global), so it won't
    match location_id for exposure data.
    """
    # Filter data if requested
    if filter_terms is not None:
        query_string = " and ".join(filter_terms)
        df = df.query(query_string)

    # Determine what data we got and process accordingly
    if 'parameter' in df and 'cause_id' in df and 'mortality' in df:
        measure = 'relative_risk' # df is RR data
        if len(cause_ids:=df['cause_id'].unique())>1: # Note: walrus operator := requires Python 3.8 or higher
            # Filter to a single cause id - all affected causes have the same RR's
            df = df.loc[df['cause_id']==cause_ids[0]]
    elif 'parameter' in df and 'measure_id' in df and (df['measure_id'] == 5).all(): # measure_id 5 is Prevalence
        measure = 'prevalence' # df is exposure data
        df = rescale_prevalence(df) # Fix prevalence because GBD 2019 data was messed up
    elif 'cause_id' in df and 'measure_id' in df and df['measure_id'].isin([2,3,4]).all(): #[2,3,4] = [DALYs,YLDs,YLLs]
        measure = 'burden'
    else:
        raise ValueError("Unexpected GBD LBWSG data format...")

    # Add 'sex' column
    sex_id_to_sex = demography.get_sex_id_to_sex_map()
    df = df.join(sex_id_to_sex, on='sex_id')
    df['sex'] = df['sex'].cat.remove_unused_categories()

    if measure in ['prevalence', 'relative_risk']:
        # Rename 'parameter' column and convert to Categorical
        df = df.rename(columns={'parameter': 'lbwsg_category'})
        df['lbwsg_category'] = df['lbwsg_category'].astype(get_lbwsg_category_dtype())
        index_cols = ['location_id', 'year_id', 'sex', 'age_group_id', 'lbwsg_category']
    elif measure == 'burden':
        # Set index to everything except draws, 'sex_id', and 'rei_id'
        index_cols = ['location_id', 'year_id', 'sex', 'age_group_id', 'cause_id', 'measure_id', 'metric_id']
    
    # Find draw columns or filter to requested draws
    if draws is None:
        draw_cols = df.filter(regex=r'^draw_\d{1,3}$').columns.to_list()
    else:
        draw_cols = [f"draw_{i}" for i in draws]

    # Set index columns and rename draw columns
    df = df.set_index(index_cols)[draw_cols]
    df.columns = df.columns.str.replace('draw_', '').astype(int).rename('draw')
#     df.columns.rename('draw', inplace=True)
    
    # Take mean over draws if this was requested by specifying a name for the mean
    if mean_draws_name is not None:
        if measure in ['prevalence', 'burden']: # Take arithmetic mean of prevalence or DALYs/YLLs/YLDs
            df = df.mean(axis=1).rename(mean_draws_name).to_frame().rename_axis(columns='draw')
        elif measure == 'relative_risk': # Take geometric mean of RR's
            df = np.exp(np.log(df).mean(axis=1)).rename(mean_draws_name).to_frame().rename_axis(columns='draw')
#             df = df.mean(axis=1).rename(mean_draws_name).to_frame().rename_axis(columns='draw')
    # Reshape draws to long form
    series = df.stack('draw').rename(measure)
    return series

def preprocess_artifact_data(df):
    pass

def convert_draws_to_long_form(data, name='value', copy=True):
    """
    Converts GBD data stored with one column per draw to "long form" with a 'draw' column wihch specifies the draw.
    """
    if copy:
        data = data.copy()
#     draw_cols = data.filter(like='draw').columns
#     index_columns = data.columns.difference(draw_cols)
#     data.set_index(index_columns.to_list())
    # Check whether columns are named draw_### in case we already took a mean over all draws
    if len(data.filter(regex=r'^draw_\d{1,3}$').columns) == data.shape[1]:
        data.columns = data.columns.str.replace('draw_', '').astype(int)
    data.columns.rename('draw', inplace=True)
    data = data.stack()
    data.rename(name, inplace=True)
    return data.reset_index()

##########################################################
# DATA ABOUT LBWSG RISK CATEGORIES #
##########################################################

def get_intervals_from_name(name: str) -> Tuple[pd.Interval, pd.Interval]:
    """Converts a LBWSG category name to a pair of intervals.

    The first interval corresponds to gestational age in weeks, the
    second to birth weight in grams.
    """
    numbers_only = [int(n) for n in re.findall(r'\d+', name)] # The regex \d+ matches 1 or more digits
    return (pd.Interval(numbers_only[0], numbers_only[1], closed='left'),
            pd.Interval(numbers_only[2], numbers_only[3], closed='left'))

def get_category_descriptions(source='gbd_mapping'):
    # The "description" is the modelable entity name for the category
    if source=='get_ids':
        descriptions = get_ids('modelable_entity')
    else:
        if source=='gbd_mapping':
            descriptions = gbd_mapping.risk_factors.low_birth_weight_and_short_gestation.categories.to_dict()
        # Assume source is a dictionary of categories to descriptions (i.e. modelable entity names)
        descriptions = (pd.Series(descriptions, name='modelable_entity_name')
                        .rename_axis('lbwsg_category').reset_index())

    cats = (pd.Series(CATEGORY_TO_MEID_GBD_2019, name='modelable_entity_id')
            .rename_axis('lbwsg_category').reset_index()
            .merge(descriptions) # merge on 'modelable_entity_id' if source=='get_ids', on 'lbwsg_category' if source=='gbd_mapping'
           )
    cats['lbwsg_category'] = cats['lbwsg_category'].astype(get_lbwsg_category_dtype())
    return cats

def get_category_data(source='gbd_mapping'):
    # Get the interval descriptions (modelable entity names) for the categories
    cat_df = get_category_descriptions(source)

    # Extract the endpoints of the gestational age and birthweight intervals into 4 separate columns
    extraction_regex = r'Birth prevalence - \[(?P<ga_start>\d+), (?P<ga_end>\d+)\) wks, \[(?P<bw_start>\d+), (?P<bw_end>\d+)\) g'
    cat_df = cat_df.join(cat_df['modelable_entity_name'].str.extract(extraction_regex).astype(int,copy=False))

    @np.vectorize
    def interval_width_midpoint(left, right):
        interval = pd.Interval(left=left, right=right, closed='left')
        return interval, interval.length, interval.mid

    # Create 2 new columns of pandas.Interval objects for the gestational age and birthweight intervals,
    # and 2 more new columns for the interval widths
    cat_df['ga'], cat_df['ga_width'], cat_df['ga_midpoint'] = interval_width_midpoint(cat_df.ga_start, cat_df.ga_end)
    cat_df['bw'], cat_df['bw_width'], cat_df['bw_midpoint'] = interval_width_midpoint(cat_df.bw_start, cat_df.bw_end)
    return cat_df

def get_category_neighbors(cat_df=None):
    """Returns a dataframe indexed by LBWSG category, with each row showing the 4 neighboring categories.
    Also contains a boolean column indicating whether the category is on the boundary of the support
    of the exposure distribution.
    If a category is on the boundary, its nonexistent neighbor categories are filled with NaN.
    The returned dataframe `cat_neighbors` can be joined with the category dataframe `cat_df` as follows:
    cat_df.join(cat_neighbors, on='lbwsg_category')
    """
    if cat_df is None:
        cat_df = get_category_data()
    # Choose shift values that are smaller than the width of any category to ensure shifts land in the interior
    ga_shift = 0.5 # < 1
    bw_shift = 100 # < 500
    # Get coordinates past the boundary of each category in each of the four directions
    ga_up = pd.Index(zip(cat_df['ga_end']+ga_shift, cat_df['bw_midpoint']))
    ga_down = pd.Index(zip(cat_df['ga_start']-ga_shift, cat_df['bw_midpoint']))
    bw_up = pd.Index(zip(cat_df['ga_midpoint'], cat_df['bw_end']+bw_shift))
    bw_down = pd.Index(zip(cat_df['ga_midpoint'], cat_df['bw_start']-bw_shift))
    
    # Get a map from intervals to categories
    cats_by_interval = cat_df.set_index(['ga','bw'])['lbwsg_category']

    # Use .reindex to look up the category for the coordinates in each of the four directions
    cat_neighbors = cat_df.set_index('lbwsg_category')[[]]
    cat_neighbors['ga_up'] = cats_by_interval.reindex(ga_up).array
    cat_neighbors['ga_down'] = cats_by_interval.reindex(ga_down).array
    cat_neighbors['bw_up'] = cats_by_interval.reindex(bw_up).array
    cat_neighbors['bw_down'] = cats_by_interval.reindex(bw_down).array
    # Record whether thee category is on the boundary of the support of the distribution
    cat_neighbors['on_boundary'] = cat_neighbors.isna().any(axis=1)
    
    return cat_neighbors

def get_category_midpoint_grid_axes(cat_df=None, include_boundary_points=True):
    """Returns sorted 1d-arrays of GA coordinates and BW coordinates for the range of category midpoints,
    optionally including the minimum and maximum values of GA and BW.
    """
    if cat_df is None:
        cat_df = get_category_data()
    ga_range = np.unique(cat_df['ga_midpoint'])
    bw_range = np.unique(cat_df['bw_midpoint'])
    if include_boundary_points:
        ga_range = np.append(ga_range, [cat_df['ga_start'].min(), cat_df['ga_end'].max()])
        bw_range = np.append(bw_range, [cat_df['bw_start'].min(), cat_df['bw_end'].max()])
    ga_range.sort()
    bw_range.sort()
    return ga_range, bw_range

def get_relative_risk_set_by_category(rr_data, draw=0, age_group_id=2, sex='Female', location_id=1, year_id=2019, take_mean=False):
    """rr_data is assumed to be GBD's LBWSG RR data returned by get_draws.
    `draw` should be a single draw if take_mean=False; otherwise `draw` should be an iterable of draws
    over which to compute the mean relative risk.
    This function could be useful for plotting RR's for the LBWSG categories, e.g. as heat maps.
    """
    if take_mean:
        draws = draw # Draws should be a list of draws in this case
        draw = mean_draws_name = 'mean' # This will be the single value of the `draw` level in preprocessed rr's index
    else:
        draws = [draw]
        mean_draws_name = None
#     if mean_draws_name is None:
#         draws = [draw]
#     else: 
#         draws = draw
#         draw = mean_draw_name # This will be the name of the mean
    rr_data = preprocess_gbd_data(rr_data, draws=draws, mean_draws_name=mean_draws_name)
    rr_data = rr_data.xs(
        (location_id, year_id, sex, age_group_id, draw),
        level=('location_id', 'year_id', 'sex','age_group_id','draw')
    ).sort_index()
    return rr_data

##########################################################
# CLASS FOR LBWSG RISK DISTRIBUTION #
##########################################################

# TODO: Move this function to prob_utils.py, and come up with a better name for it
def sample_from_propensity(propensity, categories, category_cdf):
    """Sample categories using the propensities.
    propensity is a number between 0 and 1
    categories is a list of categories
    category_cdf is a mapping of categories to cumulative probabilities.
    """
    condlist = [propensity <= category_cdf[cat] for cat in categories]
    return np.select(condlist, choicelist=categories)

def sample_from_propensity_as_arrays(propensity, categories, category_cdf):
    """Sample categories using the propensities.
    propensity is a number (or list/array of numbers) between 0 and 1
    categories is a list of categories
    category_cdf is a 2d array of shape (len(propensity), len(categories)).
    """
    category_index = (np.asarray(propensity).reshape((-1,1)) > np.asarray(category_cdf)).sum(axis=1)
    return np.asarray(categories)[category_index]

class LBWSGDistribution:
    """
    Class to assign and adjust birthweights and gestational ages of a simulated population.
    """
    def __init__(self, exposure_data):
        """"exposure_data is assumed to be preprocessed using above functions."""
        # TODO: Enable passing raw data and then sending to preprocess from the constructor.
        self.exposure_dist = exposure_data#.to_frame().unstack('lbwsg_category')
        cat_df = get_category_data()
        cat_data_cols = ['ga_start', 'ga_end', 'bw_start', 'bw_end', 'ga_width', 'bw_width']
        self.interval_data_by_category = cat_df.set_index('lbwsg_category')[cat_data_cols]
        self.categories_by_interval = cat_df.set_index(['ga','bw'])['lbwsg_category']

    def get_propensity_names(self):
        """Get the names of the propensities used by this object."""
        return ['lbwsg_category_propensity', 'ga_propensity', 'bw_propensity']

    def assign_exposure(self, pop, category_col='lbwsg_category', bw_col='birthweight', ga_col='gestational_age'):
        """
        Assign birthweights and gestational ages to each simulant in the population based on
        this object's distribution.
        """
        # Based on simulant's age and sex, assign a random LBWSG category from GBD distribution
        self.assign_category_from_propensity(pop, category_col)
        # Use propensities for ga and bw to assign a ga and bw within each category
        self.assign_ga_bw_from_propensities_within_cat(pop, category_col, bw_col, ga_col)

    def assign_category_from_propensity(self, pop, category_col='lbwsg_category', inplace=True):
        # TODO: allow specifying the category column name, and add an option to not modify pop in place
        """Assigns LBWSG categories to the population based on simulant propensities."""
        pop_exposure_cdf = self.get_exposure_cdf_for_population(pop)
        lbwsg_cat = sample_from_propensity(pop['lbwsg_category_propensity'], pop_exposure_cdf.columns, pop_exposure_cdf)
        lbwsg_cat = pd.Series(lbwsg_cat, index=pop.index, name=category_col, dtype=get_lbwsg_category_dtype())
        if inplace:
            pop[category_col] = lbwsg_cat
            return pop
        else:
            return lbwsg_cat

    def get_exposure_cdf_for_population(self, pop):
        """Returns the cumulative distribution function for each simulant in a population."""
        exposure_cdf = self.get_exposure_cdf()
        # index_cols = exposure_cdf.index.names # Should be ['age_group_id', 'draw', 'sex'], but order not guaranteed
        extra_index_cols = ['age_group_id', 'sex']
        pop_exposure_cdf = pop[extra_index_cols].join(exposure_cdf, on=exposure_cdf.index.names)
        pop_exposure_cdf.drop(columns=extra_index_cols, inplace=True)
        pop_exposure_cdf.rename_axis(columns=exposure_cdf.columns.name, inplace=True, copy=False)
        return pop_exposure_cdf

    def get_exposure_cdf(self):
        """Returns the cumulative distribution function corresponding to the prevalences
        in this object's exposure distribution. Categories are ordered numerically.
        """
        # Calling unstack() on the Series drops the name and makes the columns a simple Index.
        # Using .to_frame() instead would keep column name 'prevalence' as level 0 in MultiIndex columns, i.e.:
        # exposure_dist = self.exposure_dist.to_frame().unstack('lbwsg_category')
        # exposure_cdf = exposure_dist.loc[:,'prevalence'].cumsum(axis=1)
        exposure_cdf = self.exposure_dist.unstack('lbwsg_category').cumsum(axis=1)
       # QUESTION: Is there any situation where we will need 'location_id' or 'year_id'?
        exposure_cdf = exposure_cdf.droplevel(['location_id','year_id'])
        return exposure_cdf

    def assign_ga_bw_from_propensities_within_cat(self, pop, category_col='lbwsg_category',
                                                  bw_col='birthweight', ga_col='gestational_age'):
        """Assigns birthweights and gestational ages using propensities.
        If the propensities are uniformly distributed in [0,1], the birthweights and gestational ages
        will be uniformly distributed within each LBWSG category.
        """
        category_data = self.get_category_data_for_population(pop, category_col)
        pop[ga_col] = category_data['ga_start'] + pop['ga_propensity'] * category_data['ga_width']
        pop[bw_col] = category_data['bw_start'] + pop['bw_propensity'] * category_data['bw_width']

    def get_category_data_for_population(self, pop, category_column):
        interval_data = (pop[[category_column]]
                         .join(self.interval_data_by_category, on=category_column)
                         .drop(columns=category_column))
        return interval_data

    def apply_birthweight_shift(self, pop, shift, bw_col='birthweight', ga_col='gestational_age',
                                 cat_col='lbwsg_category', shifted_col_prefix='shifted', inplace=True):
        """Applies the specified birthweight shift to the population. Modifies the population table in place
        unless inplace=False, in which case a new population table is returned.
        """
        if not inplace:
            pop = pop[[ga_col, bw_col, cat_col]].copy()
        shifted_bw_col = f'{shifted_col_prefix}_{bw_col}'
        shifted_cat_col = f'{shifted_col_prefix}_{cat_col}'
        # Apply the shift in the new birthweight column
        pop[shifted_bw_col] = pop[bw_col] + shift
        # Assign the new category and mark where (ga,bw) is out of bounds
        self.assign_category_for_bw_ga(pop, shifted_bw_col, ga_col, shifted_cat_col, inplace=True) #fill_outside_bounds=OUTSIDE_BOUNDS_CATEGORY,
        pop['valid_shift'] = pop[shifted_cat_col].notna() #!= OUTSIDE_BOUNDS_CATEGORY
        # Reset out-of-bounds birthweights and categories back to their original values
        pop.loc[~pop['valid_shift'], shifted_bw_col] = pop.loc[~pop['valid_shift'], bw_col].array
        pop.loc[~pop['valid_shift'], shifted_cat_col] = self.assign_category_for_bw_ga(
            pop.loc[~pop['valid_shift']], shifted_bw_col, ga_col, shifted_cat_col, inplace=False)
        pop[f'{cat_col}_changed'] = pop[shifted_cat_col] != pop[cat_col]
        if not inplace:
            pop.drop(columns=[ga_col, bw_col, cat_col], inplace=True)
            return pop

    def assign_category_for_bw_ga(self, pop, bw_col, ga_col, cat_col, inplace=True): # fill_outside_bounds=None,
        """Assigns the correct LBWSG category to each simulant in the population,
        given birthweights and gestational ages.
        Modifies the population table in place if inplace=True (default), otherwise returns a pandas Series
        containing each simulant's LBWSG category, indexed by pop.index.
        If `fill_outside_bounds` is None (default), an indexing error (KeyError) will be raised (by IntervalIndex) if any
        (birthweight, gestational age) pair does not correspond to a valid LBWSG category.
        If `fill_outside_bounds` is not None, its value will be used to fill the category value for invalid
        (birthweight, gestational age) pairs.
        """
        # Need to convert the ga and bw columns to a pandas Index to work with .reindex or .get_indexer below
        ga_bw_for_pop = pd.MultiIndex.from_frame(pop[[ga_col, bw_col]])
#         # Default is to raise an indexing error if bw and gw are outside bounds
#         if fill_outside_bounds is None:
#             # Must convert cats to a pandas array to avoid trying to match differing indexes
#             cats = self.categories_by_interval.loc[ga_bw_for_pop].array
#             # TODO: See if doing the following instead will result in the same behavior as below for invalid bw, ga:
#             # cats = self.categories_by_interval.reindex(ga_bw_for_pop).array
#         # Otherwise, the category for out-of-bounds (ga,bw) pairs will be assigned the value `fill_outside_bounds`
#         else:
#             # Get integer index of category, to check for out-of-bounds (ga,bw) pairs (iidx==-1 if (ga,bw) not in index)
#             iidx = self.categories_by_interval.index.get_indexer(ga_bw_for_pop)
#             cats = np.where(iidx != -1, self.categories_by_interval.iloc[iidx], fill_outside_bounds)
        # We have to cast cats to a pandas array to avoid trying to match differing indexes
        cats = self.categories_by_interval.reindex(ga_bw_for_pop).array
        if inplace:
            pop[cat_col] = cats
        else:
            return pd.Series(cats, index=pop.index, name=cat_col)

##########################################################
# CLASSES FOR LBWSG RISK EFFECTS (RELATIVE RISKS) #
##########################################################

class LBWSGRiskEffect:
    def __init__(self, rr_data, paf_data=None):
        """"rr_data is assumed to be preprocessed using above functions."""
        # TODO: Enable passing raw data and then sending to preprocess from the constructor.
        self.rr_data = rr_data
        self.paf_data = paf_data

    def assign_relative_risk(self, pop, cat_colname='lbwsg_category', rr_colname='lbwsg_relative_risk', inplace=True):
        """Assign relative risks to a population based on their lbwsg_category."""
        rrs_by_category = self.get_relative_risks_by_category()
        # Filter population to relevant columns to avoid potential column name collisions
        extra_index_cols = ['sex', 'age_group_id', cat_colname] # columns to match besides draw, which is in pop.index
        pop_data = pop[extra_index_cols]
        # Rename the category column so it matches that in the RR data
        if cat_colname != 'lbwsg_category':
            pop_data = pop_data.rename(columns={cat_colname: 'lbwsg_category'})
        # Get relative risk for each row of pop
        pop_rrs = pop_data.join(rrs_by_category, on=rrs_by_category.index.names)['relative_risk'].rename(rr_colname)
        if inplace:
            pop[rr_colname] = pop_rrs
            return pop
        else:
            return pop_rrs

    def get_relative_risks_by_category(self):
        """Get relative risks indexed by age_group_id, sex, draw, and lbwsg_category."""
       # QUESTION: Is there any situation where we will need 'location_id' or 'year_id'?
        return self.rr_data.droplevel(['location_id','year_id'])

    def compute_paf(self, exposure, save_paf=True):
        """"Computes the Population Attributable Fraction (PAF) using this object's relative risks
        and the given exposure data. Exposure is assumed to be preprocessed using above functions.
        """
         # Drop global location to broadcast over exposure's locations
        rr = self.rr_data.droplevel('location_id')
        # Inner join indices to avoid NaN's when we multiply
        df = rr.to_frame().join(exposure)
        weighted_rr = df['relative_risk'] * df['prevalence']
        # Sum the prevalence-weighted RR's over LBWSG categories to get the mean RR
        groupby_levels = weighted_rr.index.names.difference(['lbwsg_category'])
        mean_rr = weighted_rr.groupby(groupby_levels).sum()
        paf = ((mean_rr - 1) / mean_rr).rename('paf')
        if save_paf:
            self.paf_data = paf
        return paf

    def compute_paf_for_population(self, pop, rr_colname='lbwsg_relative_risk'):
        """"Computes the Population Attributable Fraction (PAF) for the given population,
        indexed by draw, age_group_id, and sex,
        using the relative risks in specified column of the population table.
        This is a convenience method, as it does not rely on any properties of this RiskEffect instance.
        """
        pop_mean_rr = pop.groupby(['draw', 'age_group_id', 'sex'])[rr_colname].mean()
        return ((pop_mean_rr - 1) / pop_mean_rr).rename('population_paf')

class LBWSGRiskEffectInterp2d:
    def __init__(self, rr_data, paf_data=None):
        """"rr_data is assumed to be preprocessed using above functions."""
        self.rr_data = rr_data
        self.paf_data = paf_data
#         self.log_rr = np.log(self.rr_data.unstack('lbwsg_category').droplevel(['location_id', 'year_id']))
        self.log_rr_splines = self.generate_logspace_splines()

    def get_log_relative_risks(self):
        # QUESTION: Is there ever a situation where we'd want 'location_id' or 'year_id'??
        return np.log(self.rr_data.unstack('lbwsg_category').droplevel(['location_id', 'year_id']))

    def generate_logspace_splines(self):
        log_rr = self.get_log_relative_risks()
        interval_data_by_cat = get_category_data().set_index('lbwsg_category')
        x = interval_data_by_cat['bw_midpoint']
        y = interval_data_by_cat['ga_midpoint']
        assert x.index.equals(log_rr.columns) and y.index.equals(log_rr.columns)

        def make_spline(z):
            # z will be a row of log_rr
            # Reindex z to make sure categories are aligned with x... shouldn't be necessary if above assert statement passes
            # Setting bounds_error=False, fill_value=None will extrapolate by nearest neighbor for out of bounds
#             print(x)
#             print(y)
#             print(z)
            return interp2d(x,y,z.reindex(x.index, copy=False), kind='linear', bounds_error=False, fill_value=None)

        log_rr_splines = log_rr.apply(make_spline, axis='columns').rename('log_rr_spline')
        return log_rr_splines

    def get_splines_for_population(self, pop):
        extra_index_cols = ['age_group_id', 'sex']
        pop_splines = (
            pop[extra_index_cols]
            .join(self.log_rr_splines, on=self.log_rr_splines.index.names)
            .drop(columns=extra_index_cols)
        )
        return pop_splines

    def assign_relative_risk(self, pop, bw_colname='birthweight', ga_colname='gestational_age', rr_colname='lbwsg_relative_risk', inplace=True):
        # NOTE: cat_colname is ignored - I included it as a hack to have the same function signature for both classes...
        pop_log_rr_splines = self.get_splines_for_population(pop)

        def evaluate_spline(row):
            # 'log_rr_spline' = pop_log_rr_splines.name...
            # if pop_log_rr_splines were a Series, but it's a DataFrame
            return row['log_rr_spline'](row[bw_colname], row[ga_colname])[0] # spline returns an array, so access 1st element

        # [['birthweight','gestational_age']]
        log_rr = (
            pop[[bw_colname, ga_colname]]
            .join(pop_log_rr_splines)
            .apply(evaluate_spline, axis=1)
#             .astype(float)
        )
        # Make sure log RR's are nonnegative so RR's will be >=1,
        # and that they aren't too large, which will cause overflow.
        # In theory we shouldn't have to do this, but for some reason the splines were giving some negative values,
        # and also giving values up to about 4200(!!), even though the log RR's should be < 7.4
        log_rr = np.clip(log_rr, a_min=0, a_max=7.5)

        rr = np.exp(log_rr).rename(rr_colname)
#         rr = log_rr.rename(rr_colname)
        if inplace:
            pop[rr_colname] = rr
            return pop
        else:
            return rr

class LBWSGRiskEffectRBVSpline(LBWSGRiskEffect):
    def __init__(self, rr_data, paf_data=None):
        super().__init__(rr_data, paf_data)
        self.log_rr_interpolators = self.generate_logspace_interpolators()

    def get_log_relative_risks(self):
        # QUESTION: Is there ever a situation where we'd want 'location_id' or 'year_id'??
        return np.log(self.rr_data.unstack('lbwsg_category').droplevel(['location_id', 'year_id']))

    def generate_logspace_interpolators(self):
        log_rr = self.get_log_relative_risks() # Each row is one draw, age group, and sex, columns are categories
        cat_df = get_category_data()

        # Get the category midpoints, which will be our initial (x,y) values for the interpolation
        interval_data_by_cat = cat_df.set_index('lbwsg_category')
        ga_mid = interval_data_by_cat['ga_midpoint']
        bw_mid = interval_data_by_cat['bw_midpoint']
        # Make sure z values are correctly aligned with x and y values (should hold because categories are ordered)
        assert ga_mid.index.equals(log_rr.columns) and bw_mid.index.equals(log_rr.columns)

        # Get the range of midpoints (plus boundary points) to form an intermediate grid on which
        # to bootstrap the interpolation using nearest neighbor
        ga_range, bw_range = get_category_midpoint_grid_axes(cat_df)

        def make_interpolator(log_rr_for_draw_age_sex):
            # log_rr_for_draw_age_sex will be a single row of log_rr, containing log(RR) for each category
            # First interpolate log(RR)'s to the intermediate midpoint grid using nearest neighbor interpolation
            logrr_nn_griddata = griddata(
                (ga_mid, bw_mid), log_rr_for_draw_age_sex, (ga_range[:,None], bw_range[None,:]),
                method='nearest', rescale=True
            )
            # Now use the grid interpolation to create an interpolator for the whole GAxBW rectangle
            return RectBivariateSpline(ga_range, bw_range, logrr_nn_griddata, kx=1, ky=1)

        log_rr_interpolators = log_rr.apply(make_interpolator, axis='columns').rename('log_rr_interpolator')
        return log_rr_interpolators

    def get_interpolators_for_population(self, pop):
        extra_index_cols = ['age_group_id', 'sex']
        pop_interpolators = (
            pop[extra_index_cols]
            .join(self.log_rr_interpolators, on=self.log_rr_interpolators.index.names)
            #.drop(columns=extra_index_cols)
            ['log_rr_interpolator']
        )
        return pop_interpolators

    # TODO: Passing cat_colname is sort of a hack -- ideally this would be unncessary, as the category could be
    # looked up from bw_colname and ga_colname (though perhaps it's faster to use the precomputed category if it exists),
    # but that would take a bit more code to implement. Ideally, I think the common functionality of converting
    # (bw,ga) <--> category for population tables in both LBWSGDistribution and LBWSGRiskEffect should be abstracted
    # into a separate class, e.g. LBWSGCategoryConverter.
    def assign_relative_risk(self, pop, bw_colname='birthweight', ga_colname='gestational_age', cat_colname='lbwsg_category', rr_colname='lbwsg_relative_risk', logrr_colname='log_lbwsg_relative_risk', inplace=True):
        pop_log_rr_interpolators = self.get_interpolators_for_population(pop)

        def interpolate(row):
            # row will correspond to a row of the population table, joined with the correct interpolator for each simulant
            # 'log_rr_interpolator' = pop_log_rr_interpolators.name...
            # interpolator returns a 0-D, 1-D or 2-D array with 1 element, so access 1st element
            log_rr = row['log_rr_interpolator'](row[ga_colname], row[bw_colname]).flat[0]
            if row[cat_colname] in TMREL_CATEGORIES:
                log_rr = 0 # Explicitly set RR's in TMREL categories to 1, since interpolation can make them higher
            return log_rr

        log_rr = (
            pop[[bw_colname, ga_colname, cat_colname]] # ideally this wouldn't need cat_colname...
            .join(pop_log_rr_interpolators)
            .apply(interpolate, axis=1)
            .rename(logrr_colname)
        )
        # Make sure interpolated log RR's are in same range as GBD data
        assert log_rr.min()>=0 and log_rr.max()<7.4, f"log(RR)'s out of range! {log_rr.min()=} {log_rr.max()=}"

        rr = np.exp(log_rr).rename(rr_colname)
        # If we want to reset the rr's but not the log_rr's:
        # (Not sure if it will work to index rr by pop...)
#         rr.loc[pop[cat_colname].isin(TMREL_CATEGORIES)] = 1
        if inplace:
            if logrr_colname is not None:
                pop[logrr_colname] = log_rr
            pop[rr_colname] = rr
            return pop
        elif logrr_colname is not None:
            return pd.concat([log_rr, rr], axis=1)
        else:
            return rr

    def assign_categorical_relative_risk(self, pop, cat_colname='lbwsg_category', rr_colname='lbwsg_relative_risk_for_category', inplace=True):
        super().assign_relative_risk(pop, cat_colname, rr_colname, inplace)
        
    def compute_categorical_paf(self, exposure, save_paf=False):
        return super().compute_paf(exposure, save_paf)
    
    def compute_paf(self, exposure, save_paf=True):
        return NotImplementedError("PAF computation via 2D-integration is not yet implemented")