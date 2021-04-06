import pandas as pd, numpy as np

def sample_flour_consumption(mean_consumption, sample_size):
    """Sample from distribution of daily flour consumption (in Ethiopia).
    The distribution is uuniform between each quartile: min=0, q1=77.5, q2=100, q3=200, max=350.5
    This distribution represents individual heterogeneity and currently has no parameter uncertainty.

    Currently, the data is hardcoded, but eventually it should be location-dependent.
    The Ethiopia data comes from the Ethiopian National Food Consumption Survey (2013).
    """
    return get_flour_consumption_from_propensity(mean_consumption, np.random.uniform(0,1,size=sample_size))

def get_flour_consumption_from_propensity(mean_consumption, propensity):
    """Get distribution of daily flour consumption (in Ethiopia) based on the specified propensity array.
    The distribution is uuniform between each quartile: min=0, q1=77.5, q2=100, q3=200, max=350.5
    This distribution represents individual heterogeneity and currently has no parameter uncertainty.

    Currently, the data is hardcoded, but eventually it should be location-dependent.
    The Ethiopia data comes from the Ethiopian National Food Consumption Survey (2013).
    """
    # Define quartiles in g of flour per day
    q = (0, 77.5, 100, 200, 350.5) # q0=min=0, q1=77.5, q2=100, q3=200, q4=max=350.5
    u = propensity # use shorter name for readibility - u ~ uniform(0,1) random number
    # Scale the uniform random number to the correct interval based on its quartile
    standardized_consumption = np.select(
        [u<0.25, u<0.5, u<0.75, u<1],
        [q[1]*u/0.25, q[1]+(q[2]-q[1])*(u-0.25)/0.25, q[2]+(q[3]-q[2])*(u-0.5)/0.25, q[3]+(q[4]-q[3])*(u-0.75)/0.25]
    )
    if isinstance(propensity, pd.Series):
        standardized_consumption = pd.Series(standardized_consumption, index=propensity.index)
    return mean_consumption * standardized_consumption / get_standardized_consumption_mean()

def get_standardized_consumption_mean():
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
        return 'iron_fortification_propensity', 'mother_flour_consumption_propensity'

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
        `target_coverage` is assumed to be either a single number or a named Series indexed by draw.
        """
        # We need to make sure the Series indices are lined up with pop by broadcasting draws over simulant id's
        if isinstance(target_coverage, pd.Series):
        # The level argument of .reindex is not implemented for CategoricalIndex, so we can't always do this:
        #    target_coverage = target_coverage.reindex(pop.index, level='draw')
            target_coverage = pop[[]].join(target_coverage).squeeze()

        pop['mother_is_iron_fortified'] = pop['iron_fortification_propensity'] < target_coverage
        pop['mother_daily_flour'] = pop['mother_is_iron_fortified'].astype(float)
        # This call to `get_flour_consumption_from_propensity` broadcasts the index draws of mean_daily_flour
        # over the simulant_id's in pop.index
        pop.loc[pop.mother_is_iron_fortified, 'mother_daily_flour'] = get_flour_consumption_from_propensity(
            self.local_data.mean_daily_flour,
            pop.loc[pop.mother_is_iron_fortified, 'mother_flour_consumption_propensity']
        )
        # This broadcasts draws of dose response and iron concentration over simulant ids indexing daily flour
        pop['birthweight_shift'] = calculate_birthweight_shift(
            self.global_data.birthweight_dose_response, self.local_data.iron_concentration, pop['mother_daily_flour'])
        # Note: The new columns will be called 'treated_treatment_deleted_birthweight' and 'treated_lbwsg_cat'...
        # TODO: Update to enable passing entire column names instead of just a prefix
        shifted_pop = lbwsg_distribution.apply_birthweight_shift(
            pop, pop['birthweight_shift'], bw_col='treatment_deleted_birthweight', shifted_col_prefix='treated')

