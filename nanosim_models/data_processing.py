import pandas as pd, numpy as np
from collections import namedtuple

# Assumes the path to vivarium_research_lsff is in sys.path
from multiplication_models import mult_model_fns

def get_gbd_input_data(hdfstore=None, exposure_key=None, rr_key=None, yll_key=None):
    if hdfstore is None:
        hdfstore = '/share/scratch/users/ndbs/vivarium_lsff/gbd_data/lbwsg_data.hdf'
    if exposure_key is None:
        exposure_key = '/gbd_2019/exposure/bmgf_25_countries'
    if rr_key is None:
        rr_key = '/gbd_2019/relative_risk/diarrheal_diseases'
    if yll_key is None:
        yll_key = '/gbd_2019/burden/ylls/bmgf_25_countries_all_subcauses'

    exposure_data = pd.read_hdf(hdfstore, exposure_key)
    rr_data = pd.read_hdf(hdfstore, rr_key)
    yll_data = pd.read_hdf(hdfstore, yll_key)
    GBDInputData = namedtuple("GBDInputData", "exposure_data, rr_data, yll_data")
    return GBDInputData(exposure_data, rr_data, yll_data)

def get_fortification_input_data(vivarium_research_lsff_path='..', locations_path=None, coverage_data_path=None, consumption_data_path=None, concentration_data_path=None):
    """Reads input data from files and returns a tuple of input dataframes."""
    if locations_path is None:
        locations_path = f'{vivarium_research_lsff_path}/gbd_data_summary/input_data/bmgf_top_25_countries_20201203.csv'
    if coverage_data_path is None:
        coverage_data_path = f'{vivarium_research_lsff_path}/data_prep/outputs/lsff_input_coverage_data.csv'
    if consumption_data_path is None:
        consumption_data_path = f'{vivarium_research_lsff_path}/data_prep/outputs/lsff_input_coverage_data.csv'
    if concentration_data_path is None:
        concentration_data_path = '/share/scratch/users/ndbs/vivarium_lsff/gfdx_data/flour_fortification_standards_mg_per_kg.csv'

    locations = pd.read_csv(locations_path)
    coverage_df = pd.read_csv(coverage_data_path).pipe(mult_model_fns.create_marginal_uncertainty)
    consumption_df = pd.read_csv(consumption_data_path)
    concentration_df = (
        pd.read_csv(concentration_data_path)
        .pipe(process_concentration_data, locations)
    )
    FortificationInputData = namedtuple(
        "FortificationInputData",
        "locations, coverage_df, consumption_df, concentration_df"
    )
    return FortificationInputData(locations, coverage_df, consumption_df, concentration_df)

def process_concentration_data(concentration_df, locations):
    names_to_ids = locations.set_index('location_name').squeeze()
    names_to_ids["Cote d'Ivoire"]=205
    names_to_ids["Tanzania"]=189
    concentration_df = (
        concentration_df
        .merge(names_to_ids, left_on='Country', right_index=True)
        .merge(locations)
        .assign(vehicle=lambda df: df['Food Vehicle'].str.lower())
    )
    # Could the following be done using .groupby().transform() instead?
    vehicle_dfs = []
    for vehicle in concentration_df.vehicle.unique():
        df = concentration_df.query("vehicle==@vehicle")
#         if len(df.loc[df['NaFeEDTA']])==0 or len(df.loc[~df['NaFeEDTA']])==0:
#             # This may or may not make sense because it doesn't standardize if there are multiple compounds...
#             df['value'] = df['Indicator Value']
#             continue
        # Compute a multiplier to standardize all mg/kg values to NaFeEDTA.
        # Note: This assumes that for each vehicle, at least one location has NaFeEDTA and one has something else.
        absorption_multiplier = (
            df.loc[df['NaFeEDTA'], 'Indicator Value'].mean()
            / df.loc[~df['NaFeEDTA'], 'Indicator Value'].mean()
        )
        df['value'] = df['Indicator Value']
        df.loc[~df['NaFeEDTA'], 'value'] *= absorption_multiplier
        vehicle_dfs.append(df)
    concentration_df = pd.concat(vehicle_dfs)
    return concentration_df

def create_bw_dose_response_distribution():
    """Define normal distribution representing parameter uncertainty of dose-response on birthweight.
    mean = 16.7 g per 10 mg daily iron, 95% CI = (7.29,26.11).
    Effect size comes from Haider et al. (2013)
    """
    # mean and 0.975-quantile of normal distribution for mean difference (MD)
    mean = 16.7 # g per 10 mg daily iron
    q_025, q_975 = 7.29, 26.11 # 2.5th and 97.5th percentiles
    std = (q_975 - q_025) / (2*stats.norm.ppf(0.975))
    # Frozen normal distribution for MD, representing uncertainty in our effect size
    return stats.norm(mean, std)

def generate_normal_draws(mean, lower, upper, shape=1, quantile_ranks=(0.025,0.975), random_state=None):
    random_state = np.random.default_rng(random_state)
    stdev = (upper - lower) / (stats.norm.ppf(quantile_ranks[1]) - stats.norm.ppf(quantile_ranks[0]))
    return stats.norm.rvs(mean, stdev, size=shape, random_state=rng)

def generate_truncnorm_draws(mean, lower, upper, shape=1, interval=(0,1), quantile_ranks=(0.025,0.975), random_state=None):
    random_state = np.random.default_rng(random_state) # Create a generator object if random_state is a seed
    stdev = (upper-lower) / (stats.norm.ppf(quantile_ranks[1]) - stats.norm.ppf(quantile_ranks[0]))
    a = (interval[0] - mean) / stdev
    b = (interval[1] - mean) / stdev
    return stats.truncnorm.rvs(a, b, mean, stdev, size=shape, random_state=random_state)

def get_mean_consumption_draws(consumption_df, location_id, vehicle, draws, random_state):
    consumption_df = consumption_df.query("location_id==@location_id and vehicle==@vehicle")
    values = generate_normal_draws(
        consumption_df['value_mean'], consumption_df['lower'], consumption_df['upper'],
        shape=len(draws), random_state=random_state
    )
    assert (values >= 0).all(), f"Negative {vehicle} consumption values!"
    mean_consumption = pd.Series(
        values, index=draws, name=f"mean_{vehicle.lower().replace(' ', '_')}_consumption")
    return mean_consumption

def get_coverage_draws(coverage_df, location_id, vehicle, draws, random_state):
    # This line assumes Beatrix has made the appropriate update...
    coverage_df = coverage_df.query("location_id==@location_id and vehicle==@vehicle and wra_applicable==True")
    fortified = coverage_df.query("nutrient=='iron' and value_description == 'percent of population eating fortified vehicle'")
    fortifiable = coverage_df.query("value_description == 'percent of population eating industrially produced vehicle'")
    assert len(fortified)==1 and len(fortifiable)==1, f"Coverage dataframe has wrong number of rows for iron!"
    
#     fortified_draws = generate_truncnorm_draws(
#         fortified.value_mean, fortified.value_025_percentile, fortified.value_975_percentile,
#         interval=(0,100), random_state=global_data.random_generator)
    
    # Use rejection sampling to get valid draws with fortified <= fortifiable
    data = pd.concat(fortified, fortifiable)
    values = np.empty(shape=(0,len(data)))
    while(len(values) < len(draws)):
        values = np.append(values, generate_truncnorm_draws(
            data.value_mean, data.value_025_percentile, data.value_975_percentile,
            shape=(len(draws)**2,len(data)), interval=(0,100), random_state=random_state
        ), axis=0)
        values = values[values[:,0] <= values[:,1]] #1st column is fortified, 2nd column is fortifiable
    values = values[:len(draws)]
    eats_fortified = pd.Series(values[:,0], index=draws, name='eats_fortified')
    eats_fortifiable = pd.Series(values[:,1], index=draws, name='eats_fortifiable')
    return eats_fortified, eats_fortifiable

def get_global_data(effect_size_seed, random_generator, draws, take_mean=False):#mean_draws_name=None):
    """
    Information shared between locations and scenarios. May vary by draw.
    
    Returns
    -------
    draw_numbers
    draws - a pandas Index object
    dose-response of birthweight for iron (g per additional 10mg iron per day)
    """
    draw_numbers = tuple(draws) # Save original values in case we take the mean over draws
#     if mean_draws_name is None:
#         draws = pd.Index(draws, dtype='int64', name='draw')
#     else:
#         # Use the single mean value as the only "draw", labeled by `mean_draws_name`
#         draws = pd.CategoricalIndex([mean_draws_name], name='draw')
    if take_mean:
        # Use the single mean value as the only "draw", labeled by `mean_draws_name`
        mean_draws_name = f'mean_of_{len(draws)}_draws'
        draws = pd.CategoricalIndex([mean_draws_name], name='draw')
    else:
        mean_draws_name = None
        draws = pd.Index(draws, dtype='int64', name='draw')

    bw_dose_response_distribution = create_bw_dose_response_distribution()

    # Use our best guess if there's only one draw or we took the mean
    effect_size_rng = np.default_rng(effect_size_seed)
    birthweight_dose_response = pd.Series(
        bw_dose_response_distribution.rvs(size=len(draws), random_state=effect_size_rng) if len(draws)>1
        else bw_dose_response_distribution.mean(),
        index=draws,
        name='birthweight_dose_response'
    )
    random_generator = np.default_rng(random_generator)
    GlobalIronFortificationData = namedtuple(
        'GlobalIronFortificationData',
        "effect_size_seed, draws, draw_numbers, mean_draws_name, birthweight_dose_response, random_generator"
    )
    return GlobalIronFortificationData(
        effect_size_seed, draws, draw_numbers, mean_draws_name, birthweight_dose_response, random_generator
    )

def get_local_data(global_data, input_data, location, vehicle):
    """
    Information shared between scenarios for a specific location. May vary by draw and location.
    
    Returns
    -------
    location
    iron concentration in flour (mg iron as NaFeEDTA per kg flour)
    ?? mean flour consumption (g per day) -- unneeded if we have mean birthweight shift
    mean birthweight shift (g)
    baseline fortification coverage (proportion of population)
    target fortification coverage (proportion of population)
    """
    if isinstance(location, int):
        location_id = location
        location_name = input_data.locations.set_index('location_id').loc[location_id, 'location_name']
    else:
        location_name = location
        location_id = input_data.locations.set_index('location_name').loc[location_name, 'location_id']
    iron_concentration = get_iron_concentration(
        input_data.concentration_df, location_id, global_data.draws, global_data.random_generator)
    mean_daily_flour = get_mean_consumption_draws(
        input_data.consumption_df, location_id, vehicle, global_data.draws, global_data.random_generator)
    # Check data dimensions (scalar vs. Series) to make sure multiplication will work
    mean_birthweight_shift = calculate_birthweight_shift(
        global_data.birthweight_dose_response, # indexed by draw
        iron_concentration, # scalar or indexed by draw
        mean_daily_flour # indexed by draw
    ) # returns a Series since global_data.birthweight_dose_response is a Series
    mean_birthweight_shift.rename('mean_birthweight_shift', inplace=True)
    # Load coverage data
    eats_fortified, eats_fortifiable = get_coverage_draws(
        input_data.coverage_df, location_id, vehicle, global_data.draws, global_data.random_generator)

    LocalIronFortificationData = namedtuple(
        'LocalIronFortificationData',
        ['location_name',
         'location_id',
         'vehicle',
         'iron_concentration', # scalar or indexed by draw
         'mean_daily_flour', # indexed by draw
         'mean_birthweight_shift', # indexed by draw
         'eats_fortified', # scalar or indexed by draw
         'eats_fortifiable', # scalar or indexed by draw
        ]
    )
    return LocalIronFortificationData(
        location_name,
        location_id,
        vehicle,
        iron_concentration,
        mean_daily_flour,
        mean_birthweight_shift,
        eats_fortified,
        eats_fortifiable,
    )