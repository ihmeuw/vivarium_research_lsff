import numpy as np, pandas as pd
from db_queries import get_ids, get_population

# Values of age_start and age_end used in Artifacts:
# seven_days = 0.01917808 # 7/365, rounded to 8 decimal places
# twenty_eight_days = 0.07671233 # 28/365, rounded to 8 decimal places

def get_age_group_data(birth_age_end=None, birth_age_end_description=None):
    if birth_age_end is None:
        if birth_age_end_description is not None:
            raise ValueError("Value of 'birth_age_end_description' cannot be specified if 'birth_age_end' is None.")
        birth_age_end = np.round(5/(365*24*60), 8)
        birth_age_end_description = '5 minutes = 5/(365*24*60) years, rounded to 8 decimals'
    elif birth_age_end_description is None:
        raise ValueError("Value of 'birth_age_end_description' must be specified if 'birth_age_end' is not None.")

    # Define boundaries between age groups, with descriptions
    age_breaks_and_descriptions = [
        (0, "0 days = 0 years"),
        (birth_age_end, birth_age_end_description),
        *((np.round(d/365, 8), f"{d} days = {d}/365 years, rounded to 8 decimals") for d in (7,28)),
        (1, "1 year"),
        *((n, f"{n} years") for n in range(5,96,5)),
        (np.inf, "infinity!"),
    ]
    # Unzip the list of 2-tuples to get two lists
    age_breaks, age_descriptions = zip(*age_breaks_and_descriptions)
    
    # TODO: Maybe add a hardcoded version of this to avoid calling get_ids unless requested
    # Get age group names for the age group id's corresponding to the intervals between the age breaks
    # Birth, ENN, LNN, PNN, 1-4, 5-9,...,75-79, 80-85,...,90-94, 95+
    age_group_ids = [164, *range(2,21), *range(30,33), 235]
    age_group_df = (get_ids('age_group')
                    .set_index('age_group_id')
                    .loc[age_group_ids]
                    .reset_index()
                   )
    age_group_df.index = pd.IntervalIndex.from_breaks(age_breaks, closed='left', name='age_group_interval')
    
    # Record the age group start and end for each interval
    age_group_df['age_group_start'] =  age_breaks[:-1]
    age_group_df['age_group_end'] = age_breaks[1:]
    age_group_df['age_start_description'] = age_descriptions[:-1]
    age_group_df['age_end_description'] = age_descriptions[1:]
    return age_group_df

# def get_age_to_age_id_map(age_group_ids=None):
#     age_data = get_age_group_data()
#     if age_group_ids is not None:
#         age_data = age_data.query("age_group_id in @age_group_ids")
#     return age_data['age_group_id']

def get_age_to_age_id_map(birth_age_end=None):
    birth_age_end_description = None if birth_age_end is None else 'irrelevant'
    return get_age_group_data(birth_age_end, birth_age_end_description)['age_group_id']

def get_sex_id_to_sex_map(source=None):
    """Returns a pandas Series with index 'sex_id' (int) and name 'sex' (Categorical)."""
    if source is None:
        sex_id_to_sex = pd.Series({1: 'Male', 2: 'Female', 3: 'Both', 4: 'Unknown'}, name='sex', dtype='category')
        sex_id_to_sex.rename_axis('sex_id', inplace=True)
    elif source=='get_ids':
        sex_id_to_sex = get_ids('sex').set_index('sex_id')['sex'].astype('category')
    else:
        raise ValueError(f"Unknown source: {source}")
    return sex_id_to_sex


def initialize_population_table(draws, num_simulants, cohort_age=0.0):
    """Creates populations for baseline scenario and iron fortification intervention scenario,
    assigns birthweights and gestational ages to each simulant, shifts birthweights appropriately,
    and assigns relative risks for mortality based on resulting LBWSG categories.
    """
    # Create baseline population and assign demographic data
    pop = pd.DataFrame(index=pd.MultiIndex.from_product(
        [draws, range(num_simulants)], names=['draw', 'simulant_id']))
    assign_sex(pop)
    assign_age_to_cohort(pop, cohort_age)
    return pop

def assign_simulant_property(pop, property_name, choice_function=None):
    # Default is to assign uniform propensities
    if choice_function is None:
        choice_function = lambda size: np.random.uniform(size=size)
    simulant_index = pop.index.unique(level='simulant_id')
    simulant_values = pd.Series(choice_function(len(simulant_index)), index=simulant_index, name=property_name)
    # Join or reindex simulant values with pop.index to broadcast the same values over all draws
#     pop[property_name] = pop[[]].join(simulant_values)
    pop[property_name] = simulant_values.reindex(pop.index, level='simulant_id')

def assign_sex(pop):
#     pop['sex'] = np.random.choice(['Male', 'Female'], size=len(pop))
#     simulant_index = pop.index.unique(level='simulant_id')
#     sexes = pd.Series(np.random.choice(['Male', 'Female'], size=len(simulant_ids)), index=simulant_index, name='sex')
#     pop['sex'] = pop[[]].join(sexes)
#     sex_id_map = get_sex_id_to_sex_map()
#     male_female = sex_id_map.loc[sex_id_map.isin(['Male', 'Female'])].cat.remove_unused_categories()
    def choose_random_sex(size): return np.random.choice(['Male', 'Female'], size=size)
    assign_simulant_property(pop, 'sex', choose_random_sex)
    pop['sex'] = pop['sex'].astype('category', copy=False)

def assign_age_to_cohort(pop, cohort_age=0.0):
    pop['age'] = cohort_age
    pop['age_group_id'] = get_age_to_age_id_map().reindex(pop['age']).array

def increase_age(pop, age_increment):
    pop['age'] += age_increment
    pop['age_group_id'] = get_age_to_age_id_map().reindex(pop['age']).array

def assign_propensity(pop, propensity_name):
    """Assigns an independent uniform random number to each simulant.
    Enables sharing randomness across draws and scenarios.
    """
#     pop[propensity_name] = np.random.uniform(size=len(pop))
    assign_simulant_property(pop, propensity_name, choice_function=None)

def assign_propensities(pop, propensity_names):
    """Assigns propensities for each name in a list of propensity names.
    """
    for propensity_name in propensity_names:
        assign_simulant_property(pop, propensity_name)