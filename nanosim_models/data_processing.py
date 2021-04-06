import pandas as pd, numpy as np
from collections import namedtuple
from scipy import stats

# Assumes the path to vivarium_research_lsff is in sys.path
from multiplication_models import mult_model_fns

COMPLIANCE_MULTIPLIER = 0.5 # Value by which to scale iron concentration to account for non-compliance to standards

def get_gbd_input_data(hdfstore=None, exposure_key=None, rr_key=None, yll_key=None):
    if hdfstore is None:
        hdfstore = '/share/scratch/users/ndbs/vivarium_lsff/gbd_data/lbwsg_data.hdf'
    if exposure_key is None:
        exposure_key = '/gbd_2019/exposure/bmgf_25_countries'
    if rr_key is None:
        rr_key = '/gbd_2019/relative_risk/diarrheal_diseases'
    if yll_key is None:
        yll_key = '/gbd_2019/burden/ylls/bmgf_25_countries_all_subcauses'

    lbwsg_exposure = pd.read_hdf(hdfstore, exposure_key)
    lbwsg_rrs = pd.read_hdf(hdfstore, rr_key)
    lbwsg_ylls = pd.read_hdf(hdfstore, yll_key)
    GBDInputData = namedtuple("GBDInputData", "lbwsg_exposure, lbwsg_rrs, lbwsg_ylls")
    return GBDInputData(lbwsg_exposure, lbwsg_rrs, lbwsg_ylls)

def get_fortification_input_data(vivarium_research_lsff_path='..', locations_path=None, coverage_data_path=None, consumption_data_path=None, concentration_data_path=None):
    """Reads input data from files and returns a tuple of input dataframes."""
    if locations_path is None:
        locations_path = f'{vivarium_research_lsff_path}/gbd_data_summary/input_data/bmgf_top_25_countries_20201203.csv'
    if coverage_data_path is None:
        coverage_data_path = f'{vivarium_research_lsff_path}/data_prep/outputs/lsff_input_coverage_data.csv'
    if consumption_data_path is None:
        consumption_data_path = f'{vivarium_research_lsff_path}/data_prep/outputs/lsff_gday_data.csv'
    if concentration_data_path is None:
        concentration_data_path = '/share/scratch/users/ndbs/vivarium_lsff/gfdx_data/flour_fortification_standards_mg_per_kg.csv'

    locations = pd.read_csv(locations_path)
    coverage = pd.read_csv(coverage_data_path).pipe(mult_model_fns.create_marginal_uncertainty)
    consumption = pd.read_csv(consumption_data_path).pipe(process_consumption_data, coverage)
    concentration = pd.read_csv(concentration_data_path).pipe(process_concentration_data, locations)
    FortificationInputData = namedtuple(
        "FortificationInputData",
        "locations, coverage, consumption, concentration"
    )
    return FortificationInputData(locations, coverage, consumption, concentration)

def process_consumption_data(consumption_df, coverage_df):
    """Merges percent eating vehicle from coverage_df into consumption_df so that consumption among consumers
    can be computed from consumption per capita. The actual computation will happen in `get_mean_consumption_draws`
    in order to properly handle uncertainty.
    """
    percent_eating_vehicle = (
        coverage_df
        .query("value_description == 'percent of population eating vehicle' and wra_applicable==True")
        [['location_id', 'location_name', 'vehicle', 'value_mean', 'value_025_percentile', 'value_975_percentile']]
    )
    consumption_df = (
        consumption_df
        .drop(columns='location_name') # Use location name from coverage to replace 'Vietnam' with 'Viet Nam'
        .merge(percent_eating_vehicle, on=['location_id', 'vehicle'], suffixes=('_gday', '_coverage'))
    )
    assert consumption_df.pop_denom.isin(['capita', 'consumers']).all(), \
        f"Unexpected population denominator in g/day data! {consumption_df.pop_denom.unique()=}"
#     consumption_df['value'] = consumption_df['value_mean_gday']
#     consumption_df.loc[consumption_df.pop_denom=='capita', 'value'] = (
#         consumption_df.loc[consumption_df.pop_denom=='capita', 'value']
#         / (consumption_df.loc[consumption_df.pop_denom=='capita', 'value_mean_coverage'] / 100)
#     )
    return consumption_df

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
    concentration_df = pd.concat(vehicle_dfs, ignore_index=True)
    return concentration_df

def create_bw_dose_response_distribution():
    """Define normal distribution representing parameter uncertainty of dose-response on birthweight.
    mean = 16.7 g per 10 mg daily iron, 95% CI = (7.29,26.11).
    Effect size comes from Haider et al. (2013)
    """
    # mean and 0.975-quantile of normal distribution for mean difference (MD)
    mean = 16.7 # g per 10 mg daily iron
    q_025, q_975 = 7.29, 26.11 # 2.5th and 97.5th percentiles
    stdev = (q_975 - q_025) / (2*stats.norm.ppf(0.975))
    # Frozen normal distribution for MD, representing uncertainty in our effect size
    return stats.norm(mean, stdev)

def calculate_birthweight_shift(dose_response, iron_concentration, daily_consumption):
    """
    Computes the increase in birthweight (in grams) given the following:

    dose_response: g of birthweight increase per 10 mg daily iron
    iron_concentration: mg iron as NaFeEDTA per kg vehicle (wheat flour or maize flour)
    daily_consumption: g of vehicle eaten per day by pregnant mother
    """
    # TODO: Update this to handle different consumption units for salt and bouillon if necessary
    return (dose_response/10)*(iron_concentration)*(daily_consumption/1_000)

def generate_normal_draws(mean, lower, upper, shape=1, quantile_ranks=(0.025,0.975), random_state=None):
    random_state = np.random.default_rng(random_state)
    std_quantiles = stats.norm.ppf(quantile_ranks)
    stdev = (upper - lower) / (std_quantiles[1] - std_quantiles[0])
    return stats.norm.rvs(mean, stdev, size=shape, random_state=random_state)

def generate_truncnorm_draws(mean, lower, upper, shape=1, interval=(0,1), quantile_ranks=(0.025,0.975), random_state=None):
    random_state = np.random.default_rng(random_state) # Create a generator object if random_state is a seed
    std_quantiles = stats.norm.ppf(quantile_ranks)
    stdev = (upper - lower) / (std_quantiles[1] - std_quantiles[0])
    a = (interval[0] - mean) / stdev # a = left endpoint of standardized distribution
    b = (interval[1] - mean) / stdev # b = right endpoint of standardized distribution
    return stats.truncnorm.rvs(a, b, mean, stdev, size=shape, random_state=random_state)

def get_iron_concentration_draws(concentration_df, location_id, vehicle, compliance_multiplier, draws, random_state):
    """
    Get the iron concentration in specified vehicle for specified location (mg iron as NaFeEDTA per kg flour),
    with parameter uncertainty if there is any.

    Returns
    -------
    scalar if there is no uncertainty, or pandas Series indexed by draw if there is uncertainty.
    """
    concentration = concentration_df.query("location_id==@location_id and vehicle==@vehicle")
    assert len(concentration) <= 1, \
        f"Unexpected extra rows of iron concentration in {vehicle} for {location}! {concentration=}"
    if len(concentration) == 1:
        # Return the single value we have for iron concentration
        concentration_draws = concentration.squeeze()['value'] # scalar
    elif len(concentration) == 0:
        # Sample from emperical distribution for vehicle over locations
        rng = np.random.default_rng(random_state)
        possible_values = concentration_df.query("vehicle==@vehicle")['value']
        values = rng.choice(possible_values, size=len(draws), replace=True)
        concentration_draws = pd.Series(values, index=draws, name='iron_concentration')
    return compliance_multiplier * concentration_draws

def get_mean_consumption_draws(consumption_df, location_id, vehicle, draws, random_state):
    consumption = consumption_df.query("location_id==@location_id and vehicle==@vehicle")
    assert len(consumption)==1, \
        f"Consumption data has wrong number of rows for iron vehicle! {consumption=}"
    consumption = consumption.squeeze() # Convert single row to Series
    # Use rejection sampling to guarantee positive draws
    consumption_draws = generate_truncnorm_draws(
        consumption['value_mean_gday'], consumption['lower'], consumption['upper'],
        shape=len(draws), interval=(0,np.inf), random_state=random_state # Truncate at 0 to ensure positive consumption
    )
    # If consumption is per capita, convert to consumption among consumers
    if consumption['pop_denom'] == 'capita':
        coverage_draws = generate_truncnorm_draws(
            consumption['value_mean_coverage'],
            consumption['value_025_percentile'],
            consumption['value_975_percentile'],
            shape=len(draws), interval=(0,100), random_state=random_state # Truncate at 0% and 100%
        ) / 100 # convert percent to proportion
        consumption_draws /= coverage_draws
    # Note: This assert should now be unnecessary because I used truncnorm distributions
    assert (consumption_draws >= 0).all(), f"Negative {vehicle} consumption values for {location_id=}!"
    mean_consumption = pd.Series(
        consumption_draws, index=draws, name=f"mean_{vehicle.lower().replace(' ', '_')}_consumption")
    return mean_consumption

def get_coverage_draws(coverage_df, location_id, vehicle, draws, random_state):
    coverage_df = coverage_df.query("location_id==@location_id and vehicle==@vehicle and wra_applicable==True")
    fortified = coverage_df.query("nutrient=='iron' and value_description == 'percent of population eating fortified vehicle'")
    fortifiable = coverage_df.query("value_description == 'percent of population eating industrially produced vehicle'")
    assert len(fortified)==1 and len(fortifiable)==1, \
        f"Coverage data has wrong number of rows for iron vehicle! {fortified=}, {fortifiable=}"
    
#     fortified_draws = generate_truncnorm_draws(
#         fortified.value_mean, fortified.value_025_percentile, fortified.value_975_percentile,
#         interval=(0,100), random_state=global_data.random_generator)
    
    # Use rejection sampling to get valid draws with fortified <= fortifiable
    data = pd.concat([fortified, fortifiable])
    values = np.empty(shape=(0,len(data)))
    while(len(values) < len(draws)):
        values = np.append(values, generate_truncnorm_draws(
            data.value_mean, data.value_025_percentile, data.value_975_percentile,
            # The number of failures before reaching r successful draws is Negative-Binomial(r,p),
            # where p is the probability of success on one trial. Expected value is r*(1-p)/p,
            # i.e. proportional to r=len(draws), so start with a constant times len(draws) trials.
            shape=(10*len(draws),len(data)), interval=(0,100), random_state=random_state
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
    effect_size_rng = np.random.default_rng(effect_size_seed)
    birthweight_dose_response = pd.Series(
        bw_dose_response_distribution.rvs(size=len(draws), random_state=effect_size_rng) if len(draws)>1
        else bw_dose_response_distribution.mean(),
        index=draws,
        name='birthweight_dose_response'
    )
    random_generator = np.random.default_rng(random_generator)
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
    iron_concentration = get_iron_concentration_draws(
        input_data.concentration, location_id, vehicle, COMPLIANCE_MULTIPLIER,
        global_data.draws, global_data.random_generator
    )
    mean_daily_consumption = get_mean_consumption_draws(
        input_data.consumption, location_id, vehicle, global_data.draws, global_data.random_generator)
    # Check data dimensions (scalar vs. Series) to make sure multiplication will work
    mean_birthweight_shift = calculate_birthweight_shift(
        global_data.birthweight_dose_response, # indexed by draw
        iron_concentration, # scalar or indexed by draw
        mean_daily_consumption # indexed by draw
    ) # returns a Series since global_data.birthweight_dose_response is a Series
    mean_birthweight_shift.rename('mean_birthweight_shift', inplace=True)
    # Load coverage data
    eats_fortified, eats_fortifiable = get_coverage_draws(
        input_data.coverage, location_id, vehicle, global_data.draws, global_data.random_generator)

    LocalIronFortificationData = namedtuple(
        'LocalIronFortificationData',
        ['location_name',
         'location_id',
         'vehicle',
         'iron_concentration', # scalar or indexed by draw
         'mean_daily_consumption', # indexed by draw
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
        mean_daily_consumption,
        mean_birthweight_shift,
        eats_fortified,
        eats_fortifiable,
    )