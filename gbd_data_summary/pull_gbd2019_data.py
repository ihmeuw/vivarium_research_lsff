import pandas as pd, numpy as np
import sys, os.path
# from pandas.api.types import CategoricalDtype
from collections import namedtuple

from vivarium_helpers.id_helper import *
# import rank_countries_by_stunting as rbs

from db_queries import get_ids, get_population, get_outputs, get_best_model_versions
from get_draws.api import get_draws

def get_locations(key, data_directory='data'):
    """Reads the location file for the specified description key."""
    location_files = {
        'all': 'all_countries_with_ids.csv',
        'original': 'bgmf_countries_with_ids.csv',
        'top25': 'bmgf_top_25_countries_20201203.csv',
    }
    filepath = f'{data_directory}/{location_files[key]}'
    return pd.read_csv(filepath)

def get_or_append_global_location(locations=None):
    """Returns a dataframe with the name and id of the 'Global' location,
    or appends this data to the passed locations dataframe if not None.
    """
    global_loc = pd.DataFrame({'location_name':'Global', 'location_id':1}, index=[0])
    return global_loc if locations is None else locations.append(global_loc, ignore_index=True)

def split_global_from_other_locations(df):
    """Splits df into two subdataframes and returns the pair of dataframes:
    The first is where df.location_id is NOT the 'Global' id (1);
    The second is where df.location_id IS the 'Global' id (1).
    """
    SubDataFramesByLocation = namedtuple('SubDataFramesByLocation', 'other_locations, global_location')
    return SubDataFramesByLocation(df.query("location_id !=1"), df.query("location_id==1"))

def find_best_model_versions(search_string, entity='modelable_entity', decomp_step='step4', **kwargs_for_contains):
    """Searches for entity id's with names matching search_string using pandas.Series.str.contains,
    and calls get_best_model_versions with appropriate arguments to determine if decomp_step is correct.
    """
    best_model_versions = get_best_model_versions(
        entity,
        ids = find_ids('modelable_entity', search_string, **kwargs_for_contains),
        gbd_round_id = list_ids('gbd_round', '2019'),
        status = 'best',
        decomp_step = decomp_step,
    )
    return best_model_versions

#####################################
# CODE FOR DEC 17, 2020 DELIVERABLE #
#####################################

def pull_dalys_attributable_to_risk_for_locations(location_ids, *risk_factor_names):
    """Calls get_draws to pull all-cause DALYs attributable to the specified risk for the specified locations.
    The call does not specify the age_group_id, so it will pull DALYs for all age groups that contriibute DALYs.
    Estimates from age group aggregates are not available from the burdenator.
    """
    risk_ids = list_ids('rei', *risk_factor_names)
    if isinstance(risk_ids, int): risk_ids = [risk_ids]
    burden = get_draws(
        gbd_id_type=['rei_id']*len(risk_ids) + ['cause_id'], # Types must match gbd_id's
        gbd_id=[*risk_ids, list_ids('cause', 'All causes')],
        source='burdenator',
        measure_id=find_ids('measure', 'DALYs'),
        metric_id=list_ids('metric', 'Number'), # Only available metrics are Number and Percent
        location_id=list(location_ids),
        year_id=2019,
        sex_id=list_ids('sex', 'Male', 'Female'), # Sex aggregates not available
        gbd_round_id=list_ids('gbd_round', '2019'),
        status='best',
        decomp_step='step5',
    )
    return burden

def pull_dalys_due_to_cause_for_locations(location_ids, *cause_names):
    dalys = get_draws(
        gbd_id_type='cause_id',
        gbd_id=list_ids('cause', *cause_names),
        source='dalynator',
        measure_id=find_ids('measure', 'DALYs'),
        metric_id=list_ids('metric', 'Number'),
        location_id=list(location_ids),
        year_id=2019,
        sex_id=list_ids('sex', 'Male', 'Female'),
        age_group_id=list_ids('age_group', 'All Ages'), #22,#
        gbd_round_id=list_ids('gbd_round', '2019'),
        status='best',
        decomp_step='step5',
    )
    return dalys

def concatenate_risk_and_cause_burdens(risk_burdens, cause_burdens):
    """Concatenates the risk and cause DALY dataframes returned by get_draws,
    adding a name column for the risk or cause, and dos some renaming and dropping of unncessary columns. 
    """
    risk_burdens = add_entity_names(risk_burdens, 'rei')
    risk_burdens = risk_burdens.drop(columns='cause_id').rename(
        columns={'rei_id':'gbd_id', 'rei_name': 'gbd_entity_name'})
    cause_burdens = add_entity_names(cause_burdens, 'cause')
    cause_burdens = cause_burdens.rename(
        columns={'cause_id':'gbd_id',  'cause_name': 'gbd_entity_name'})
    risk_burdens['gbd_id_type'] = 'rei'
    cause_burdens['gbd_id_type'] = 'cause'
    all_data = pd.concat([risk_burdens, cause_burdens], ignore_index=True, copy=False)
    all_data = replace_ids_with_names(all_data, 'measure', 'metric')
    return all_data

def aggregate_draws_over_columns(df, marginalized_cols):
    """Aggregates (by summing) over the specified columns in the passed dataframe, draw by draw."""
    draw_cols = df.filter(regex=r'^draw_\d{1,3}$').columns.to_list()
    index_cols = df.columns.difference([*draw_cols, *marginalized_cols]).to_list()
    return df.groupby(index_cols, observed=True)[draw_cols].sum() # observed=True needed for Categorical data

def calculate_proportion_global_burden(burden_for_locations, global_burden):
    """Calculates percent of the global burden of a risk or cause for each of the locations
    in `burden_for_locations`.
    """
    marginalized_cols = ['age_group_id', 'sex_id']
    burden_for_locations = aggregate_draws_over_columns(burden_for_locations, marginalized_cols)
    global_burden = aggregate_draws_over_columns(global_burden, marginalized_cols)
    # Reset the location_id level in denominator to broadcast over global instead of trying to match location
    proportion = burden_for_locations / global_burden.reset_index('location_id', drop=True)
    # I thought it was confusing to have 'Number' as the metric for a proportion, but on the other hand,
    # leaving the metric_id alone tells what data was used in the computation
#     if 'metric_id' in proportion.index.names:
#         proportion.index.set_levels([list_ids('metric', 'Rate')], level='metric_id', inplace=True)
    return proportion

def get_iron_data(risk_burdens):
    """Selects and formats the iron deficiency data from the risk_burdens dataframe."""
    iron_deficiency_id = list_ids('rei', 'Iron deficiency')
    iron_burden = risk_burdens.query('rei_id == @iron_deficiency_id')
    iron_burden = add_entity_names(iron_burden, 'rei')
    iron_burden = replace_ids_with_names(iron_burden, 'measure', 'metric')
#     iron_burden = drop_id_columns(iron_burden, 'rei', 'location', keep=True) # Drop all id columns except rei and location
    return iron_burden

def get_iron_dalys_by_subpopulation(iron_burden, subpopulations='under5_wra'):
    """
    """
    pops_to_masks = {
        # Female and '10 to 14' to '50 to 54'
        'WRA (10-54)': (iron_burden.sex_id==2) & (iron_burden.age_group_id>=7) & (iron_burden.age_group_id<=15),
        'Females 15-54': (iron_burden.sex_id==2) & (iron_burden.age_group_id>=8) & (iron_burden.age_group_id<=15),
        'Under 5': iron_burden.age_group_id<=5, # id 5 is '1 to 4' age group
        '5-9': iron_burden.age_group_id==6, # id 6 is '5 to 9' age group
        'Males 10-14': (iron_burden.sex_id==1) & (iron_burden.age_group_id==7),
        'Females 10-14': (iron_burden.sex_id==2) & (iron_burden.age_group_id==7),
    }
    
    if subpopulations == 'under5_wra':
        subpopulations = ['Under 5', 'WRA (10-54)']
    elif subpopulations == 'under5_5to9_wra':
        subpopulations = ['Under 5', '5-9', 'WRA (10-54)']
    elif subpopulations == 'under5_5to9_m10to14_wra':
        subpopulations = ['Under 5', '5-9', 'Males 10-14', 'WRA (10-54)']
    elif subpopulations == 'under5_5to9_mf10to14_f15to54':
        subpopulations = ['Under 5', '5-9', 'Males 10-14', 'Females 10-14', 'Females 15-54']

    pops_to_masks = {pop: mask for pop, mask in pops_to_masks.items() if pop in subpopulations}

    # Check that subpopulations are mutually exclusive
    masks = list(pops_to_masks.values())
    assert all((~(m1 & m2)).all() for m1,m2 in zip(masks[:-1], masks[1:]))

#     iron_burden = iron_burden.assign(subpopulation = np.select([wra, under5], ['WRA', 'Under 5'], default='Other'))
#     iron_burden = iron_burden.assign(subpopulation = np.select(
#         [wra, under5, five_to_nine], ['WRA', 'Under 5', '5 to 9'], default='Other'))
    iron_burden = iron_burden.assign(
        subpopulation = pd.Categorical(
            np.select(list(pops_to_masks.values()), list(pops_to_masks.keys()), default='Other'),
            categories=['Under 5', '5-9', 'Males 10-14', 'Females 10-14', 'Females 15-54', 'WRA (10-54)', 'Other'],
            ordered=True
        )
    )
    return aggregate_draws_over_columns(iron_burden, ['age_group_id', 'sex_id'])

def summarize_draws_across_locations(df):
    """Aggregates draws over locations in df, then calls .describe() to compute statistics for all draws.
    The dataframe df is assumed to have all relevant data stored in columns, not in its index.
    """
    df = aggregate_draws_over_columns(df, ['location_id'])
    return df.T.describe(percentiles=[0.025, 0.975]).T

def format_summarized_data(summary, number_format='', multiplier=1):
    """Format the mean, lower, and upper values from the summarized data."""
    if number_format == 'percent':
        number_format = '.2%'
    elif number_format == 'count':
        number_format = ',.0f'
    
    units = f'_per_{multiplier}' if multiplier != 1 else ''

    def print_number(x):
#         return eval(f'f"{{x*{multiplier}:{number_format}}}"')
        return f"{x*multiplier:{number_format}}"

    summary = summary.rename(columns={'2.5%':'lower', '97.5%':'upper'})
    cols = ['mean', 'lower', 'upper']
    summary = summary[cols]
    for col in cols:
        summary[f'{col}{units}_formatted'] = summary[col].apply(print_number)
    summary['mean_lower_upper'] = summary.apply(
        lambda row: f"{row[f'mean{units}_formatted']} ({row[f'lower{units}_formatted']}, {row[f'upper{units}_formatted']})",
        axis=1
    )
    return summary

def summarize_percent_global_burdens(risk_dalys=None, cause_dalys=None, location_key=None, save_filepath=None):
    """Returns a dataframe summarizing the percent global burdens of risks and causes, optionally saving the file.
    If either risk_dalys or cause_dalys is None, then location_key must not be None.
    """
    # Process arguments
    # TODO: Probably edit this to pass location ids (like below) instead of a location key
    if location_key is not None:
        locations = get_locations(location_key)
        locations = get_or_append_global_location(locations)

    if risk_dalys is None:
        risks = ['Vitamin A deficiency', 'Zinc deficiency', 'Iron deficiency']
        risk_dalys = pull_dalys_attributable_to_risk_for_locations(locations.location_id, *risks)
    elif location_key is not None:
        risk_dalys = risk_dalys.loc[risk_dalys.location_id.isin(locations.location_id)]

    if cause_dalys is None:
        cause_dalys = pull_dalys_due_to_cause_for_locations(locations.location_id, 'Neural tube defects')
    elif location_key is not None:
        cause_dalys = cause_dalys.loc[cause_dalys.location_id.isin(locations.location_id)]

    # Concatenate risks and causes, calculate proportion global burden, summarize over draws, format, and save
    all_dalys = concatenate_risk_and_cause_burdens(risk_dalys, cause_dalys)

    proportion_global_burden = calculate_proportion_global_burden(
        *split_global_from_other_locations(all_dalys)
    )

    burden_summary = (proportion_global_burden
                      .reset_index()
                      .pipe(summarize_draws_across_locations)
                      .pipe(format_summarized_data, number_format='percent')
                      .sort_values(['gbd_id_type', 'gbd_id'], ascending=[False, True])
                     )
    if save_filepath is not None:
        burden_summary.to_csv(save_filepath)
    return burden_summary

def summarize_iron_burden_by_subpopulation(risk_dalys=None, location_ids=None, save_filepaths=None):
    """Summarizes DALY burden due to Iron Deficiency by subpopulation for the specified locations,
    and calculates the proportion of global DALY burden by subpopulation.
    """
    # Process arguments (I think it might be better to pass location ids instead of a location key as above)
    if location_ids is not None:
        all_location_ids = list(location_ids)
        global_id = get_or_append_global_location().at[0,'location_id']
        if global_id not in all_location_ids:
            all_location_ids.append(global_id)

    if risk_dalys is None:
        risk_dalys = pull_dalys_attributable_to_risk_for_locations(all_location_ids, 'Iron deficiency')
    elif location_ids is not None:
        risk_dalys = risk_dalys.loc[risk_dalys.location_id.isin(all_location_ids)]

    if save_filepaths is None:
        save_filepaths = [None, None, None]

    # Calculate iron deficiency DALYs by subpopulation, summarize across draws, format and save results
    iron_dalys_by_subpop = (risk_dalys
                            .pipe(get_iron_data)
                            .pipe(get_iron_dalys_by_subpopulation, subpopulations='under5_wra')
                            .pipe(split_global_from_other_locations)
                           ) # Result is namedtuple with fields 'other_locations' and 'global_location'

    iron_subpop_dalys_other_summary = (iron_dalys_by_subpop.other_locations
                                        .reset_index()
                                        .pipe(summarize_draws_across_locations)
                                        .pipe(format_summarized_data, number_format='count')
                                       )
    if save_filepaths[0] is not None:
        iron_subpop_dalys_other_summary.to_csv(save_filepaths[0])

    iron_subpop_dalys_global_summary = (iron_dalys_by_subpop.global_location
                                        .reset_index()
                                        .pipe(summarize_draws_across_locations) # (Only location is global, but that's fine)
                                        .pipe(format_summarized_data, number_format='count')
                                       )
    if save_filepaths[1] is not None:
        iron_subpop_dalys_global_summary.to_csv(save_filepaths[1])

    # Calculate percent of global DALYS for each subpopulation, summarize across draws, format and save results
    proportion_global_iron_dalys_by_subpop = calculate_proportion_global_burden(
        iron_dalys_by_subpop.other_locations.reset_index(),
        iron_dalys_by_subpop.global_location.reset_index()
    )

    percent_global_iron_dalys_by_subpop_summary = (proportion_global_iron_dalys_by_subpop
                                               .reset_index()
                                               .pipe(summarize_draws_across_locations)
                                               .pipe(format_summarized_data, number_format='percent')
                                              )
    if save_filepaths[2] is not None:
        percent_global_iron_dalys_by_subpop_summary.to_csv(save_filepaths[2])

    return iron_subpop_dalys_other_summary, iron_subpop_dalys_global_summary, percent_global_iron_dalys_by_subpop_summary

#####################################
# CODE FOR JAN 22, 2021 DELIVERABLE #
#####################################

def pull_binary_risk_summary_input_data_for_locations(location_ids, risk_name, hdf_filepath=None):
    """Calls `get_draws()` to pull input data for data summary of a binary risk.
    
    Vitamin A deficiency:
    Pulls vitamin A deficiency prevalence for the 'Under 5' age group
    for the location id's in the locations_ids iterable.
    Note that prevalence data exists for all age groups, but the VAD risk only applies to Under 5,
    specifically ages 6mo - 5years.
    In particular, VAD relative risks only exist for age groups 4 and 5 ('Post Neonatal' and '1 to 4').
    The relative difference between the (weighted) average prvalence in age groups 4 and 5 and the 'Under 5'
    prevalence is less than 1% (prevalence in 4 and 5 is slightly larger, as expected since the population
    is slightly smaller).
    Returned dataframe is 2 rows per location (one for each exposure category).
    
    Zinc deficiency:
    Pulls Zinc deficiency prevalence for the '1 to 4' age group
    for the location id's in the locations_ids iterable.
    This is the only age group for which prevalence data exists.
    Returned dataframe is 2 rows per location (one for each exposure category).
    """
    if risk_name=='Vitamin A deficiency':
        age_group = 'Under 5'
    elif risk_name=='Zinc deficiency':
        age_group='1 to 4'
    else:
        raise ValueError(f'Unsupported risk name: {risk_name}')
    
    risk_name_formatted = risk_name.lower().replace(' ', '_')
    age_group_formatted = age_group.lower().replace(' ', '_')
    
    location_ids=list(location_ids)
    risk_id=list_ids('rei', risk_name)
    age_group_id=list_ids('age_group', age_group)
    sex_id=list_ids('sex', 'Male', 'Female', 'Both')
    gbd_round_id=list_ids('gbd_round', '2019')
    
#     metadata = 
    
    # Pulls prevalence for specified age group - 2 rows per location (cat1, cat2) per sex id
    prevalence = get_draws(
        gbd_id_type='rei_id',
        gbd_id=risk_id,
        source='exposure',
        location_id=location_ids,
        year_id=2019,
        age_group_id=age_group_id,
        sex_id=sex_id,
        gbd_round_id=gbd_round_id,
        status='best',
        decomp_step='step4',
    )
    if hdf_filepath is not None:
        prevalence.to_hdf(hdf_filepath, f'{risk_name_formatted}/prevalence_among_{age_group_formatted}')
    
    # Pulls population for specified age group - 1 row per location per sex id
    population = get_population(
        age_group_id=age_group_id,
        sex_id=sex_id,
        location_id=location_ids,
        year_id=2019,
        gbd_round_id=gbd_round_id,
        decomp_step='step4',
        with_ui=True,
    )
    if hdf_filepath is not None:
        population.to_hdf(hdf_filepath, f'{risk_name_formatted}/population_{age_group_formatted}')
    
    # Pulls all-cause DALY burden (number and percent) for all age groups
    # Vitamin A: 92 rows per location (23 age groups x 2 sexes x 2 metrics)
    # Zinc: 4 rows per location (1 age group x 2 sexes x 2 metrics)
    daly_burden = get_draws(
        gbd_id_type=['rei_id', 'cause_id'], # Types must match gbd_id's
        gbd_id=[risk_id, list_ids('cause', 'All causes')],
        source='burdenator',
        measure_id=find_ids('measure', 'DALYs'),
        metric_id=None, # Only available metrics are Number and Percent
        location_id=list(location_ids),
        year_id=2019,
        sex_id=None, # Sex aggregates not available
        gbd_round_id=gbd_round_id,
        status='best',
        decomp_step='step5',
    )
    if hdf_filepath is not None:
        daly_burden.to_hdf(hdf_filepath, f'{risk_name_formatted}/daly_burden')

    BinaryRiskSummaryData = namedtuple('BinaryRiskSummaryInputData', 'prevalence, population, daly_burden')
    return BinaryRiskSummaryInputData(prevalence, population, daly_burden)

def preprocess_binary_risk_summary_input_data(risk_data, risk_name):
    """Processes raw Vitamin A deficiency or Zinc deficiency data from GBD pulled by
    `pull_binary_risk_summary_input_data_for_locations` for inputting into
    `get_binary_outcome_data_summary`.
    """
    prevalence, population, daly_burden = risk_data

    if risk_name=='Vitamin A deficiency':
        # Need to sum DALYs over all Under-5 age groups
        age_group='Under 5'
        age_group_ids=list_ids('age_group',
                               'Early Neonatal', 'Late Neonatal','Post Neonatal', '1 to 4')
    elif risk_name=='Zinc deficiency':
        # Only one age group exists for Zinc deficiency data
        age_group='1 to 4'
        age_group_ids=[list_ids('age_group', age_group)]
    else:
        raise ValueError(f"Unsupported risk: {risk_name}")

    sex_id=list_ids('sex', 'Both')
    number_id=list_ids('metric', 'Number')
    percent_id=list_ids('metric', 'Percent')
    
    # Filter prevalence to the 'exposed' category (get rid of, 'cat2', the TMREL)
    prevalence = prevalence.query("sex_id == @sex_id and parameter == 'cat1'")
    # Filter population to the correct sex id(s)
    population = population.query("sex_id == @sex_id")
    # Sum DALYs over sexes and age groups if necessary, and record the age group name
    daly_count = (daly_burden
                  .query("metric_id == @number_id and age_group_id in @age_group_ids")
                  .pipe(aggregate_draws_over_columns, ['sex_id', 'age_group_id'])
                  .reset_index()
                  .assign(age_group=age_group)
                 )
    # I think this is wrong because it doesn't do a weighted average over sexes
#     daly_percent = (zinc_data.daly_burden
#                     .query("metric_id == @percent_id")
#                     .pipe(aggregate_draws_over_columns, ['sex_id'])
#                     .reset_index()
#                    )
    BinaryRiskSummaryPreprocessedData = namedtuple(
        'BinaryRiskSummaryPreprocessedData', 'prevalence, population, daly_count, daly_rate, daly_percent')
    return BinaryRiskSummaryPreprocessedData(prevalence, population, daly_count, daly_rate=None, daly_percent=None)

def pull_neural_tube_defects_summary_input_data_for_locations(location_ids, hdf_filepath=None):
    """Calls `get_draws()` to pull input data for the data summary for Neural tube defects.
    Pull Neural tube defects incidence at birth (i.e. birth prevalence)
    for the location id's in the locations_ids iterable.
    'Birth' is the only age group for which incidence data exists.
    Returned dataframe is 1 row per location.
    """
    cause_name='Neural tube defects'
    location_ids=list(location_ids)
    cause_id=list_ids('cause', cause_name)
    gbd_round_id=list_ids('gbd_round', '2019')
    sex_id=list_ids('sex', 'Male', 'Female', 'Both')
#     age_group_id=list_ids('age_group', 'Birth', 'Under 5')
#     birth_age_group_id=list_ids('age_group', 'Birth')
#     under_5_age_group_id=list_ids('age_group', 'Under 5')

    key_prefix='data_summary'
    cause_name_formatted = cause_name.lower().replace(' ', '_')
#     age_group_formatted = age_group.lower().replace(' ', '_')
    
    # Pulls birth prevalence for neural tube defects
    prevalence = get_draws(
        gbd_id_type='cause_id',
        gbd_id=cause_id,
        source='como',
        location_id=list(location_ids),
        year_id=2019,
#         age_group_id=list_ids('age_group', 'Birth'), # Passing Birth age group throws an error
        sex_id=sex_id,
        measure_id=list_ids('measure', 'Incidence'), # Birth is only age group for which there is incidence
        gbd_round_id=list_ids('gbd_round', '2019'),
        status='best',
        decomp_step='step5',
    )
    if hdf_filepath is not None:
        prevalence.to_hdf(hdf_filepath, f'{key_prefix}/{cause_name_formatted}/prevalence')

    # Pulls population for Birth and Under 5 age groups - 2 row per location per sex id
    population = get_population(
        age_group_id=list_ids('age_group', 'Birth', 'Under 5'),
        sex_id=sex_id,
        location_id=location_ids,
        year_id=2019,
        gbd_round_id=gbd_round_id,
        decomp_step='step4',
        with_ui=True,
    )
    if hdf_filepath is not None:
        population.to_hdf(hdf_filepath, f'{key_prefix}/{cause_name_formatted}/population')

    # Pulls DALY count and percent for 
    daly_burden = get_draws(
        gbd_id_type='cause_id',
        gbd_id=cause_id,
        source='dalynator',
        measure_id=find_ids('measure', 'DALYs'),
        metric_id=None, # Returns Number, Percent, and Rate
        location_id=list(location_ids),
        year_id=2019,
        sex_id=sex_id,
        age_group_id=list_ids('age_group', 'Under 5'),
        gbd_round_id=list_ids('gbd_round', '2019'),
        status='best',
        decomp_step='step5',
    )
    if hdf_filepath is not None:
        daly_burden.to_hdf(hdf_filepath, f'{key_prefix}/{cause_name_formatted}/daly_burden')

    return prevalence, population, daly_burden

def preprocess_neural_tube_defects_data(ntd_data):
    """Processes raw GBD data from `pull_neural_tube_defects_summary_input_data_for_locations`
    to pass into `get_binary_outcome_data_summary`.
    """
    prevalence, population, daly_burden = ntd_data
    
    birth_age_group_id=list_ids('age_group', 'Birth')
    under_5_age_group_id=list_ids('age_group', 'Under 5')
    
    sex_id=list_ids('sex', 'Both')
    number_id=list_ids('metric', 'Number')
    percent_id=list_ids('metric', 'Percent')
    rate_id=list_ids('metric', 'Rate')
    
    # Filter prevalence to Birth prevalence
    birth_prevalence = prevalence.query("sex_id == @sex_id and age_group_id == @birth_age_group_id")
    birth_population = population.query("sex_id == @sex_id and age_group_id == @birth_age_group_id")
    # Filter DALYs to Under-5 age group, and record the age group name
    dalys = (daly_burden
             .query("sex_id == @sex_id and age_group_id == @under_5_age_group_id")
             .assign(age_group='Under 5')
            )
    daly_count = dalys.query("metric_id == @number_id")
    daly_rate = dalys.query("metric_id == @rate_id")
    daly_percent = dalys.query("metric_id == @percent_id")
    BinaryCauseSummaryPreprocessedData = namedtuple(
        'BinaryCauseSummaryPreprocessedData','prevalence, population, daly_count, daly_rate, daly_percent')
    return BinaryCauseSummaryPreprocessedData(birth_prevalence, birth_population, daly_count, daly_rate, daly_percent)

# Main function to create data summaries

def get_binary_outcome_data_summary(preprocessed_data, outcome_name, location_ids=None, save_filepath=None):
    """Creates a dataframe summarizing GBD data for the specified binary outcome.
    Namely, the summary has columns for prevalence, number with outcome, DALYs attributable to outcome,
    and DALYs attributable to outcome per 100,000 person-years, in appropriate age groups.
    Currently works for Vitamin A deficiency, Zinc deficiency, and Neural tube defects.
    """
    prevalence, population, daly_count, daly_rate, daly_percent = preprocessed_data
    
    if location_ids is None:
        location_ids = list(population.location_id.unique())
    
    draw_cols = [f'draw_{i}' for i in range(1000)]
    index_cols = ['location_id']

    def make_column(df, colname):
        """Converts the draw data in a dataframe returned by get_draws into a pandas Series
        indexed by location_id and draw, with the specified name. The data is assumed to
        contain a unique gbd entity, age group, sex, measure, and metric.
        """
        column = (df
                  .query(f"location_id in {location_ids}")
                  .set_index(index_cols)[draw_cols]
                  .rename_axis(columns='draw')
                  .stack()
                  .rename(colname)
                 )
        return column

    # Create prevalence column
    prevalence_age_group = ids_to_names('age_group', *prevalence.age_group_id.unique())
    assert len(prevalence_age_group) == 1
    prevalence = make_column(prevalence, f'Prevalence of {outcome_name} in age group {prevalence_age_group.iloc[0]}')

    # Re-index the population to match other columns for calculations
    population = (population
                  .query("location_id in @location_ids")
                  .set_index(index_cols)['population']
                 )
    
    # Create column for number with the outcome
    number_with_outcome = (prevalence * population).rename(
        f'Number in age group {prevalence_age_group.iloc[0]} with {outcome_name}')

    # Create DALY count column
    daly_age_group = daly_count.age_group.unique()[0] # Age group column was added in preprocessing
    daly_count = make_column(daly_count, f'DALYs attributable to {outcome_name} in age group {daly_age_group}')
    
    # Create DALY rate column
    daly_multiplier = 100_000
    daly_rate_colname = (f'DALYs attributable to {outcome_name} '
                         f'per {daly_multiplier:,} person-years in age group {daly_age_group}'
                        )
    if daly_rate is None: # For risks, we can't pull DALY rate from burdenator, so we need to compute it
        daly_rate = (daly_count / population).rename(daly_rate_colname) * daly_multiplier
    else: # For causes, DALY rate was pulled from dalynator
        daly_rate = make_column(daly_rate, daly_rate_colname) * daly_multiplier
        
#     daly_percent = make_column(daly_percent, 'Percent of total DALYs in age group {daly_age_group} attributable {outcome_name}')

    # Concatenate columns, summarize draws, and add location names
    data_summary = pd.concat([prevalence, number_with_outcome, daly_count, daly_rate], axis=1)
    locations = ids_to_names('location', *location_ids)

    data_summary = (data_summary
                    .groupby(index_cols)
                    .pipe(aggregate_mean_lower_upper)
                    .pipe(flatten_column_levels) # Flatten the multi-index to avoid messing it up when joining
                    .join(locations)
                    .sort_values('location_name')
                    .set_index('location_name', append=True)
                    .pipe(move_global_to_end)
                    .pipe(expand_column_levels, names=['measure', 'estimate']) # Re-expand flattened multi-index
                   )
    if save_filepath is not None:
        data_summary.to_csv(save_filepath)
    return data_summary

# Helper functions for above data summary function

def aggregate_mean_lower_upper(df_or_groupby):
    """"""
    def lower(x): return x.quantile(0.025)
    def upper(x): return x.quantile(0.975)
    return df_or_groupby.agg(['mean', lower, upper])

def move_global_to_end(df):
    """"""
    return pd.concat(split_global_from_other_locations(df))

def flatten_column_levels(df):
    df.columns = df.columns.to_flat_index()
    return df

def expand_column_levels(df, sortorder=None, names=None):
    df.columns = pd.MultiIndex.from_tuples(df.columns, sortorder=sortorder, names=names)
    return df

# Function to format summarized data nicely for a table in Word or PowerPoint

def format_data_summary(data_summary,
                        prevalence_multiplier=100,
                        prevalence_decimals=1,
                        prevalent_cases_units=1000,
                        daly_count_units=1,
                        daly_count_decimals=0,
                        daly_rate_multiplier=1,
                        sep='\n',
                        save_filename=None):
    """Formats a data summary produced by `get_binary_outcome_data_summary` to combine
    each triple of mean, lower, upper columns into a single column of the form "mean (lower, upper)",
    with options to change the units and number of decimal places for each measure.
    """
    def print_mean_lower_upper(mean, lower, upper, number_format):
        return f"{mean:{number_format}}{sep}({lower:{number_format}}, {upper:{number_format}})"
    
    cols = []
    for measure in data_summary.columns.get_level_values('measure').unique():
        df = data_summary[measure]
        if measure.startswith('Prevalence'):
            df *= prevalence_multiplier
            number_format = f'.{prevalence_decimals}f'
            suffix = f" (per {prevalence_multiplier})"
        elif measure.startswith('Number'):
            df /= prevalent_cases_units
            number_format = ',.0f'
            suffix = f" ({prevalent_cases_units}s)" if prevalent_cases_units>1 else ''
#         elif measure.startswith('Mean'):
#             number_format = '.1f'
#             suffix = ''
        elif measure.startswith('DALYs'):
            if 'person-year' in measure:
                df *= daly_rate_multiplier
                number_format = f',.0f'
                suffix = f" (x {daly_rate_multiplier:,})" if daly_rate_multiplier !=1 else ''
            else:
                df /= daly_count_units
                number_format = f',.{daly_count_decimals}f'
                suffix = f" ({daly_count_units}s)" if daly_count_units>1 else ''
        else:
            number_format = ',.0f'
            suffix = ''
        col = (df
               .apply(lambda row: print_mean_lower_upper(row['mean'], row['lower'], row['upper'], number_format)
                      , axis=1)
               .rename(measure + suffix)
              )
        cols.append(col)
        
    formatted_summary = pd.concat(cols, axis=1)
    if save_filename is not None:
        formatted_summary.to_csv(save_filename)
    return formatted_summary

