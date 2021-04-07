import pandas as pd, numpy as np

# Is data_processing actually the right place for calculate_birthweight_shift, or should it be here?
# In addition to getting used in the intervention, it gets used to create local_data.
# Maybe creating global_data and local_data should be moved back to this module.
from data_processing import calculate_birthweight_shift

# Convenience function for testing. Not used below.
def sample_consumption(mean_consumption, sample_size, rng=321):
    """Sample from distribution of daily vehicle consumption (based on flour consumption in Ethiopia).
    The distribution is uniform between each quartile: min=0, q1=77.5, q2=100, q3=200, max=350.5
    This distribution represents individual heterogeneity and currently has no parameter uncertainty.

    Currently, the data is hardcoded, but eventually it should be location-dependent.
    The Ethiopia data comes from the Ethiopian National Food Consumption Survey (2013).
    """
    rng = np.random.default_rng(rng)
    return get_vehicle_consumption_from_propensity(mean_consumption, rng.uniform(0,1,size=sample_size))

def get_vehicle_consumption_from_propensity(mean_consumption, propensity):
    """Get distribution of daily consumption of vehicle based on the specified propensity array.
    This distribution represents individual heterogeneity with no parameter uncertainty,
    but mean_draws can incorporate parameter uncertainty.
    Currently, heterogeneity in consumption is modeled based on the consumption of wheat flour
    in Ethiopia (see `get_ethiopia_wheat_flour_consumption_from_propensity()` below), but
    the goal is to eventually incorporate additional data to account for different vehicles and
    locations.
    """
    # Sample the Ethiopia wheat flour consumption distribution, then rescale to have the correct mean
    ethiopia_wheat_consumption = get_ethiopia_wheat_flour_consumption_from_propensity(propensity)
    standardized_consumption = ethiopia_wheat_consumption / get_ethiopia_wheat_flour_consumption_mean()
    return mean_consumption * standardized_consumption

def get_ethiopia_wheat_flour_consumption_from_propensity(propensity):
    """Get distribution of daily wheat flour consumption in Ethiopia based on the specified propensity array.
    The distribution is uniform between each quartile: min=0, q1=77.5, q2=100, q3=200, max=350.5
    This distribution represents individual heterogeneity and currently has no parameter uncertainty.

    The quartile data come from the Ethiopian National Food Consumption Survey (2013).
    """
    # Define quartiles in g of flour per day
    q = (0, 77.5, 100, 200, 350.5) # q0=min=0, q1=77.5, q2=100, q3=200, q4=max=350.5
    u = propensity # use shorter name for readibility - u ~ uniform(0,1) random number
    # Scale the uniform random number to the correct interval based on its quartile
    consumption = np.select(
        [u<0.25, u<0.5, u<0.75, u<1],
        [q[1]*u/0.25, q[1]+(q[2]-q[1])*(u-0.25)/0.25, q[2]+(q[3]-q[2])*(u-0.5)/0.25, q[3]+(q[4]-q[3])*(u-0.75)/0.25]
    )
    # If propensities are in a Series, convert the result to a like-indexed Series so broadcasting will work later
    if isinstance(propensity, pd.Series):
        consumption = pd.Series(consumption, index=propensity.index)
    return consumption

def get_ethiopia_wheat_flour_consumption_mean():
    return 138.08844717769867 # Calculated by taking the mean of 1_000_000 samples

class IronFortificationIntervention:
    """
    Class for applying iron fortification intervention to simulants.
    """

    def __init__(self, global_data, local_data):
        """Initializes an IronFortificationIntervention with the specified global and local data."""
        self.global_data = global_data
        self.local_data = local_data

      # TODO: Perhaps create a Population class that can assign the propensities by getting called from here...
#     def assign_propensities(self, pop):
#         """
#         Assigns propensities to simualants for quantities relevant to this intervention.
#         """
#         propensities = np.random.uniform(size=(len(pop),2))
#         pop['iron_fortification_propensity'] = propensities[:,0]
#         pop['mother_flour_consumption_propensity'] = propensities[:,1]

    def get_propensity_names(self):
        """Get the names of the propensities used by this object."""
        return 'iron_fortification_propensity', 'mother_iron_vehicle_consumption_propensity'

    def assign_treatment_deleted_birthweight(self, pop, lbwsg_distribution, baseline_coverage):
        """
        Assigns "treatment-deleted" birthweights to each simulant in the population,
        i.e. birthweights assuming the absence of iron fortification.
        """
        # Shift everyone's birthweight down by the average shift
        lbwsg_distribution.apply_birthweight_shift(
            pop, -baseline_coverage * self.local_data.mean_birthweight_shift, shifted_col_prefix='treatment_deleted')

    def assign_treated_birthweight(self, pop, lbwsg_distribution, target_coverage):
        """
        Assigns birthweights resulting after iron fortification is implemented.
        Assumes `assign_propensities` and `assign_treatment_deleted_birthweight` have already been called on pop.
        `target_coverage` is assumed to be either a single number or a Series indexed by draw.
        """
        # We need to make sure the Series indices are lined up with pop by broadcasting draws over simulant ids
        if isinstance(target_coverage, pd.Series):
        # The level argument of .reindex is not implemented for CategoricalIndex, so we can't always do this:
        #    target_coverage = target_coverage.reindex(pop.index, level='draw')
        # Need to ensure the Series has a name so that .join will work
            target_coverage = pop[[]].join(target_coverage.rename('target_coverage')).squeeze()

        pop['mother_is_iron_fortified'] = pop['iron_fortification_propensity'] < target_coverage
        pop['mother_daily_consumption'] = pop['mother_is_iron_fortified'].astype(float)
        # This call to `get_vehicle_consumption_from_propensity` broadcasts the index draws of mean_daily_consumption
        # over the simulant_id's in pop.index
        pop.loc[pop.mother_is_iron_fortified, 'mother_daily_consumption'] = get_vehicle_consumption_from_propensity(
            self.local_data.mean_daily_consumption,
            pop.loc[pop.mother_is_iron_fortified, 'mother_iron_vehicle_consumption_propensity']
        )
        # This broadcasts draws of dose response and iron concentration over simulant ids indexing daily consumption
        pop['birthweight_shift'] = calculate_birthweight_shift(
            self.global_data.birthweight_dose_response, self.local_data.iron_concentration, pop['mother_daily_consumption'])
        # Note: The new columns will be called 'treated_treatment_deleted_birthweight' and 'treated_lbwsg_cat'...
        # TODO: Update to enable passing entire column names instead of just a prefix
        shifted_pop = lbwsg_distribution.apply_birthweight_shift(
            pop, pop['birthweight_shift'], bw_col='treatment_deleted_birthweight', shifted_col_prefix='treated')

