"""Module to rank countries by decreasing population of stunted children in GBD 2019."""
import pandas as pd
import sys, os.path
sys.path.append(os.path.abspath("../.."))
from pre_processing.id_helper import *
# get_ids is no longer included in the above import * statement. Woo hoo!
from db_queries import get_ids, get_population
from get_draws.api import get_draws

def clean_orig_country_data(filepath):
    """Clean the original country data from BGMF in Paulina's spreadsheet saved as a .csv.
    
    - Filters to the relevant columns [country, survey year, percent stunting]
    - Adds a column recording the country rank in order of decending stunting prevalence
    - Makes all column names valid Python variables in lower case
    
    Returns a pandas DataFrame.
    """
    orig_countries = pd.read_csv(filepath)
    orig_countries = orig_countries.iloc[:,:3]
    orig_countries.drop(index=0, inplace=True)
    orig_countries.index.rename('rank_2013', inplace=True)
    orig_countries.columns = orig_countries.columns.str.replace('%', 'percent')
    orig_countries.columns = orig_countries.columns.str.replace(' ', '_')
    orig_countries.columns = orig_countries.columns.str.lower()
    orig_countries = orig_countries.reset_index()
    return orig_countries

def add_location_ids_to_orig_countries(orig_countries, location_ids):
    """Adds location id's and GBD location names to the cleaned country data returned by `clean_orig_country_data()`.
    location_ids should be an iterable of unique location id's that is guaranteed to contain correct id's 
    for all the locations in orig_countries. When called below, location_ids will be the location id's in the
    stunting dataframe, i.e. the locations we know we have data for.
    """
    # Load location id table and filter to id and name columns
    locids = get_ids('location')[['location_id', 'location_name']]
    # Get id's for country names
    orig_countries = orig_countries.merge(locids, left_on='country', right_on='location_name', how='left')
    # The names Tanzania and Cote d'Ivoire aren't in the location database,
    # so search for a matching name to get the correct id's and names.
    # (I pre-verified that these searches give exactly one result)
    orig_countries.loc[orig_countries.country=='Tanzania', ['location_id', 'location_name']]=(
        search_id_table(locids, 'Tanzania').iloc[0].array # fails without .array, perhaps because df index doesn't match Series name
    )
    orig_countries.loc[orig_countries.country=="Cote d'Ivoire", ['location_id', 'location_name']]=(
        search_id_table(locids, 'Ivoire').iloc[0].array # fails without .array, perhaps because df index doesn't match Series name
    )
    # Filter out duplicate country names (e.g. Sudan) by inner joining with correct id's
    location_ids=pd.Series(location_ids, name='location_id')
    orig_countries = orig_countries.merge(location_ids, how='inner')
     # Reset dtype to int because NaN's upcasted to float
    orig_countries['location_id'] = orig_countries['location_id'].astype(int)
    return orig_countries

def get_locations_for_stunting_prevalence(category='admin0'):
    """Returns location id table filtered to the specified location category.
    Currently the only supported category is 'admin0' (the default), which finds locations 
    with location_type=="admin0" (returing 299 id's).
    """
    locations = get_ids('location')
    if category=='admin0':
        locations = locations.query('location_type=="admin0"')
    else:
        raise ValueError(
            f"I'm sorry, Dave, I can't do that. I don't know how to find locations in the specified category {category}."
        )
    return locations

def pull_stunting_prevalence_for_locations(location_ids):
    """Calls `get_draws()` to pull Child stunting exposure for the location id's in the locations_ids iterable."""
    stunting = get_draws(
        'rei_id',
        gbd_id=list_ids('rei', 'Child stunting'),
        source='exposure',
        location_id=list(location_ids),
        year_id=2019,
        age_group_id=list_ids('age_group', 'Under 5'),
        sex_id=list_ids('sex', 'Both'),
        gbd_round_id=list_ids('gbd_round', '2019'),
        status='best',
        decomp_step='step4',
    )
    return stunting

def compute_mean_stunting_prevalence_by_location(stunting_df):
    """Calculates mean stunting prevalence over 1000 draws for all locations in stunting_df,
    where stunting is defined as "height-for-age Z-score < -2 standard deviations".
    """
    # To illustrate regex syntax, this matches columns named 'draw_###', where ### consists of 1-3 digits
    draw_cols = stunting_df.filter(regex=r'^draw_\d{1,3}$').columns # .filter(like='draw') would also suffice
    stunting_cat_cols = ['modelable_entity_id','parameter'] # either of these columns identifies the stunting category
    index_cols = stunting_df.columns.difference(draw_cols).difference(stunting_cat_cols) # columns to group by
    unique_location_ids = stunting_df.location_id.unique() # record this to check shape later

    # Fiind meid's for stunting exposure < 2 SD's below mean (paramter column is cat1 or cat2)
    stunted_meids = list_ids('modelable_entity',
                             'Severe Stunting, < -3 SD (post-ensemble)', # cat1
                             'Stunting Between -3 SD and -2 SD (post-ensemble)', # cat2
                             )
    # Filter stunting dataframe to rows found above (filters out cat3 and cat4)
    stunting_df = stunting_df.query(f'modelable_entity_id in {stunted_meids}')

    # Sum prevalences of cat1 and cat2 in each draw to get total stunting prevalence for each location
    stunting_df = stunting_df.groupby(index_cols.to_list())[draw_cols].sum() # fails if pd.Index not converted to list
    assert stunting_df.shape == (len(unique_location_ids), len(draw_cols))

    # Take mean over draws to get a single prevalence estimate for each location
    stunting_df = stunting_df.mean(axis=1) # stunting_df is now a pd.Series
    stunting_df.rename('stunting_prevalence', inplace=True) # changes Series.name

    return stunting_df.reset_index() # converts back to DataFrame, with column named 'stunting_prevalence'

def add_location_names_and_populations(stunting_df):
    """Adds location names and under-5 populations for each location in `stunting_df`."""
    locations = ids_to_names('location', *stunting_df.location_id.unique()).reset_index()
    stunting_df = stunting_df.merge(locations)
    population = get_population(
        age_group_id=list_ids('age_group', 'Under 5'),
        location_id=ids_in(locations),
        year_id=2019,
        gbd_round_id=list_ids('gbd_round', '2019'),
        decomp_step='step4',
        with_ui=False,
    )
    stunting_df = stunting_df.merge(population.drop(columns='run_id'))
    stunting_df.rename(columns={'population':'population_under_5'}, inplace=True)
    return stunting_df

def format_stunting_dataframe(stunting_df):
    """Removes extraneous id columns and reorders remaining columns."""
    return stunting_df[['location_name', 'location_id', 'stunting_prevalence', 'population_under_5']]

def compute_number_stunted_and_sort_descending(stunting_df, copy=False):
    """Adds a column for the number of children stunted in each country,
    sorts by this column in descending order,
    and changes the index to the rank of the location in descending order.
    Modifies stunting_df in place unless copy is set to True.
    """
    if copy:
        stunting_df = stunting_df.copy()
    stunting_df['number_stunted'] = stunting_df['stunting_prevalence'] * stunting_df['population_under_5']
    stunting_df.sort_values('number_stunted', ascending=False, inplace=True)
    stunting_df.index = range(1,len(stunting_df)+1)
    stunting_df.index.rename('rank_2019_all', inplace=True)
    return stunting_df

def compute_cumulative_number_stunted_and_percent_of_global_population(stunting_df, copy=False):
    """Modifies stunting_df in place unless copy is set to True."""
    if copy:
        stunting_df = stunting_df.copy()
    global_stunted_population = stunting_df['number_stunted'].sum()
    stunting_df['cumulative_number_stunted'] = stunting_df['number_stunted'].cumsum()
    stunting_df['cum_percent_global_stunted_pop'] = stunting_df['cumulative_number_stunted']/global_stunted_population
    stunting_df['cum_percent_global_stunted_pop'] = (100*stunting_df['cum_percent_global_stunted_pop']).round(1)
    return stunting_df

# def _get_indicator_colname(percent):
#     """Returns column name for the indicator column for specified stunting percent cutoff."""
#     return f'stunting_above_{percent:.0f}_percent'

def _get_indicator_colname_and_rank_2019_colname(stunting_cutoff):
    """Returns column name for the indicator column for specified stunting percent cutoff,
    and the column name for the corresponding ranking.
    """
#     indicator_colname = _get_indicator_colname(stunting_cutoff)
    indicator_colname = f'stunting_above_{stunting_cutoff:.0f}_percent'
    # If stunting cutoff is 0, include all countries; otherwise, get the rank column for the cutoff
    suffix = 'all' if stunting_cutoff == 0 else f'among_{indicator_colname}'
    rank_2019_colname = f'rank_2019_{suffix}'
    return indicator_colname, rank_2019_colname

def add_rank_and_cumulative_percent_for_cutoffs(stunting_df, *stunting_percent_cutoffs, copy=False):
    """"""
    if copy:
        stunting_df = stunting_df.copy()
    for cutoff in stunting_percent_cutoffs:
        if cutoff == 0:
            continue # The relevant columns are already in the dataframe
#         indicator_colname = _get_indicator_colname(cutoff)
        indicator_colname, rank_2019_colname = _get_indicator_colname_and_rank_2019_colname(cutoff)
        # Add indicator column and rank
        stunting_df[indicator_colname] = (
            stunting_df['stunting_prevalence'] >= cutoff/100
        )
#         stunting_df[f'rank_2019_among_{indicator_colname}'] = (
        stunting_df[rank_2019_colname] = (
            stunting_df[indicator_colname].cumsum()
        )
        # Add cumulative percent stunted among those with indicator==True
        global_stunted_population = stunting_df['number_stunted'].sum()
        cum_number_stunted = (stunting_df['number_stunted'] * stunting_df[indicator_colname]).cumsum()
        cum_percent_stunted = (100*cum_number_stunted/global_stunted_population).round(1)
        stunting_df[f'cum_percent_global_stunted_pop_among_{indicator_colname}'] = cum_percent_stunted
    return stunting_df

def merge_stunting_with_orig_countries(stunting_df, orig_countries_df, stunting_cutoffs=None, num_countries=None):
    """Performs an outer merge of stunting_df with orig_countries_df.
    Assumes location id's have been added to original countries dataframe.
    Columns for stunting_cutoffs must be in stunting_df.
    The number of countries is limited to num_countries for the most restrictive stunting cutoff passed,
    or to the original number of countries, whichever is greater. The total number of rows may be greater
    than both the original number and num_countries (e.g. if one of the stunting cutoffs is 100%, then
    all 195 countries will be returned, regardless of the value of num_countries).
    """
    merged = stunting_df.reset_index().merge(orig_countries_df, how='outer')
    orig_num = merged.rank_2013.notna().sum()
    if num_countries is None:
        num_countries = orig_num
#     max_cutoff = None if stunting_cutoffs is None else max(stunting_cutoffs)
#     suffix = 'all' if max_cutoff is None else f'among_{_get_indicator_colname(max_cutoff)}'
#     rank_2019_col = f'rank_2019_{suffix}'
#     print(rank_2019_col, num_countries, orig_num)
    max_cutoff = 0 if stunting_cutoffs is None else max(stunting_cutoffs)
    _, rank_2019_col = _get_indicator_colname_and_rank_2019_colname(max_cutoff)
    merged = merged.query(f'{rank_2019_col} <= {num_countries} or rank_2013 <= {orig_num}')
    return merged

def _get_query_for_most_stunted(stunting_cutoff, rank_colname, num_countries):
    """Returns query string to find the top num_countries according to rank_colname,
    with stunting prevalance above stunting_cutoff.
    """
    return f'stunting_prevalence >= {stunting_cutoff/100} and {rank_colname} <= {num_countries}'

def get_locations_ranked_by_stunting_for_cutoffs(stunting_df_or_merged_df, stunting_cutoffs, num_countries_list):
    """Returns a dictionary of dataframes, where the key describes the dataframe."""
    rankings = {}
    for cutoff in stunting_cutoffs:
#         indicator_colname = _get_indicator_colname(cutoff)
#         # If stunting cutoff is 0, include all countries; otherwise, get the rank column for the cutoff
#         suffix = 'all' if cutoff == 0 else f'among_{indicator_colname}'
#         rank_2019_col = f'rank_2019_{suffix}'
        indicator_colname, rank_2019_col = _get_indicator_colname_and_rank_2019_colname(cutoff)
        for num_countries in num_countries_list:
            ranking_name = f'top_{num_countries}_countries_with_{indicator_colname}_in_2019'
#             query_string = f'stunting_prevalence >= {cutoff/100} and {rank_2019_col} <= {num_countries}'
            query_string = _get_query_for_most_stunted(cutoff, rank_2019_col, num_countries)
            rankings[ranking_name] = stunting_df_or_merged_df.query(query_string)
    return rankings

def find_differences(merged_df, stunting_cutoffs, num_countries_list, compare_with_orig_num=True):
    """Returns a dictionary of dataframes, where the key describes the dataframe."""
    orig_num = int(merged_df.rank_2013.max())
    differences = {}
    for cutoff in stunting_cutoffs:
#         indicator_colname = _get_indicator_colname(cutoff)
#         # If stunting cutoff is 0, include all countries; otherwise, get the rank column for the cutoff
#         suffix = 'all' if cutoff == 0 else f'among_{indicator_colname}'
#         rank_2019_col = f'rank_2019_{suffix}'
        indicator_colname, rank_2019_col = _get_indicator_colname_and_rank_2019_colname(cutoff)
        for num_countries in num_countries_list:
            old_nums = [num_countries]
            if compare_with_orig_num and num_countries != orig_num:
                old_nums.append(orig_num)
            for old_num in old_nums:
                # Find what's in new and not in old
                diff_name = f'top_{num_countries}_with_{indicator_colname}_in_2019_minus_top_{old_num}_in_2013'
    #             query_string = (f'(stunting_prevalence >= {cutoff/100} and {rank_2019_col} <= {num_countries})' # In new
    #                             ' and rank_2013 != rank_2013') # Not in old iff rank_2013 is NaN (by definition, NaN != NaN)
                above_cutoff_and_in_top = _get_query_for_most_stunted(cutoff, rank_2019_col, num_countries)
                query_string = f'({above_cutoff_and_in_top}) and (rank_2013 != rank_2013 or rank_2013 > {old_num})'
                differences[diff_name] = merged_df.query(query_string)
                # Find what's in old and not in new
                diff_name = f'top_{old_num}_in_2013_minus_top_{num_countries}_with_{indicator_colname}_in_2019'
    #             query_string = (f'(stunting_prevalence < {cutoff/100} or {rank_2019_col} > {num_countries})' # Not in new
    #                             ' and rank_2013 == rank_2013') # In old iff rank_2013 is NOT NaN
                query_string = f'~({above_cutoff_and_in_top}) and rank_2013 <= {old_num}'
                differences[diff_name] = merged_df.query(query_string)
    return differences

def save_csv_files(orig_countries_with_ids_df, ranked_stunting_df, merged_df, top_countries, differences):
    """"""
    pass

def parse_args_and_read_data(args):
    """Helper function for main() to parse arguments and read input data."""
    # Use simple logic and default values until I figure out how to parse arguments to specify different cutoffs...

    stunting_filepath = args[0] if len(args)>0 else None
    orig_countries_filepath = args[1] if len(args)>1 else None

    if stunting_filepath is not None:
        stunting = pd.read_hdf(stunting_filepath)
    else:
        locations = get_locations_for_stunting_prevalence() # currently this pulls more location ids than necessary
        stunting = pull_stunting_prevalence_for_locations(ids_in(locations))

    if orig_countries_filepath is not None:
        orig_countries = pd.read_csv(orig_countries_filepath) # filepath for cleaned country data without location id's
    else:
        orig_countries = clean_orig_country_data('bgmf_countries.csv') # filepath for original uncleaned data

    stunting_cutoffs = [20,18]
    num_countries_list = [25,len(orig_countries)]
#     num_counries_dict = {25:[25,len(orig_countries)], len(orig_countries):[len(orig_countries)]}
    compare_with_orig_num = True

    return orig_countries, stunting, stunting_cutoffs, num_countries_list, compare_with_orig_num

def main(args=None):
    """"""
    # Following the suggestion here: https://realpython.com/python-command-line-arguments/#mutating-sysargv
    if args is None:
        args = sys.argv[1:]
    
    orig_countries, stunting, stunting_cutoffs, num_countries_list, compare_with_orig_num = parse_args_and_read_data(args)
    
    orig_countries = add_location_ids_to_orig_countries(orig_countries, stunting.location_id.unique())

    stunting = (stunting
                .pipe(compute_mean_stunting_prevalence_by_location)
                .pipe(add_location_names_and_populations)
                .pipe(format_stunting_dataframe)
                .pipe(compute_number_stunted_and_sort_descending)
                .pipe(compute_cumulative_number_stunted_and_percent_of_global_population)
                .pipe(add_rank_and_cumulative_percent_for_cutoffs, *stunting_cutoffs)
               )
    
    # The merged dataframe can be used to compute new rankings and differences between new and old
    merged = merge_stunting_with_orig_countries(stunting, orig_countries, stunting_cutoffs)

    # Final results: Find top ranked countries and differences with original countries
    stunting_cutoffs = [0,*stunting_cutoffs] # Add 0 to the list of cutoffs to see ranking if we don't include a stunting cutoff
    top_countries = get_locations_ranked_by_stunting_for_cutoffs(merged, stunting_cutoffs, num_countries_list)
    differences = find_differences(merged, stunting_cutoffs, num_countries_list, compare_with_orig_num)
    
    save_csv_files(orig_countries, stunting, merged, top_countries, differences)
    
if __name__=='__main__':
    main()
