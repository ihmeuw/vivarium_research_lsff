import pandas as pd, numpy as np
from scipy import stats
from collections import namedtuple

import lsff_plots
from vivarium_helpers import prob_utils
from vivarium_helpers import id_helper

import functions_for_all_nutrients

def sample_flour_consumption(mean_consumption, sample_size):
    """Sample from distribution of daily flour consumption (in Ethiopia).
    The distribution is uuniform between each quartile: min=0, q1=77.5, q2=100, q3=200, max=350.5
    This distribution represents individual heterogeneity and currently has no parameter uncertainty.

    Currently, the data is hardcoded, but eventually it should be location-dependent.
    The Ethiopia data comes from the Ethiopian National Food Consumption Survey (2013).
    """
#     # TODO: Edit this to simply call the propensity version below
#     # Define quartiles in g of flour per day
#     q = (0, 77.5, 100, 200, 350.5) # min=0, q1=77.5, q2=100, q3=200, max=350.5
#     u = np.random.uniform(0,1,size=sample_size)
#     # Scale the uniform random number to the correct interval based on its quartile
#     return np.select(
#         [u<0.25, u<0.5, u<0.75, u<1],
#         [q[1]*u/0.25, q[1]+(q[2]-q[1])*(u-0.25)/0.25, q[2]+(q[3]-q[2])*(u-0.5)/0.25, q[3]+(q[4]-q[3])*(u-0.75)/0.25]
#     )
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
#     propensity_name = 'iron_fortification_propensity'
    
    def __init__(self, global_data, local_data):
        """Initializes an IronFortificationIntervention with the specified global and local data."""
        self.global_data = global_data
        self.local_data = local_data
#         # OLD VERSION, pre-common random numbers:
#         # __init__(self, location, baseline_coverage, target_coverage)
#         # TODO: Eliminate the distributions in favor of storing a value for each draw (see below)
#         self.iron_conc_distribution = iron_conc_distributions[location]
#         self.bw_dose_response_distribution = create_bw_dose_response_distribution()
#         # TODO: Change constructor to accept the pre-retrieved data instead of looking it up here
#         # OR: Instead, pass baseline and target coverage into the functions below...
#         self.baseline_coverage = baseline_coverage
#         self.target_coverage = target_coverage
        
#         # Currently these distributions are sampling one value for all draws.
#         # TODO: Update to sample a different value for each draw (need to pass draws to constructor).
#         self.dose_response = self.bw_dose_response_distribution.rvs()
#         self.iron_concentration = self.iron_conc_distribution.rvs()

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
#         # NOTE: We don't necessarily need to sample flour consumption every time if we could
#         # compute the mean ahead of time... I need to think more about which data varies by draw vs. population...
#         flour_consumption = sample_flour_consumption(10_000)
#         mean_bw_shift = calculate_birthweight_shift(self.global_data.dose_response, self.iron_concentration, flour_consumption).mean()
        # Shift everyone's birthweight down by the average shift
#         # Old version that couldn't modify in place:
#         # TODO: actually, maybe we don't need to store the treatment-deleted category, only the treated categories
#         # 2021-03-07: Actually, I now lean towards just recording everything, which is the default in the new lbwsg funcions.
#         shifted_pop = lbwsg_distribution.apply_birthweight_shift(
#             pop, -baseline_coverage * self.local_data.mean_birthweight_shift)
#         pop['treatment_deleted_birthweight'] = shifted_pop['new_birthweight']
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
        # DONE: Can this line be rewritten to avoid sampling flour consumption for rows that will get set to 0?
        # Yes, initialize the column with pop['mother_is_iron_fortified'].astype(float),
        # then index to the relevant rows and reassign.
        # NOTE: If we want to compare intervention to baseline by simulant, we can't sample flour consumption
        # every time, because simulants who are fortified in baseline should have the same flour consumption
        # in the intrvention scenario. We'd have to use propensities instead... which is easy because my
        # current sampling function is based on quantiles - I just need to pass it the propensity instead.
        # However, if we don't care about comparability between scenarios, the current implementation should
        # be sufficient for large enough populations.
#         pop['mother_daily_flour'] = pop['mother_is_iron_fortified'] * sample_flour_consumption(len(pop))
        pop['mother_daily_flour'] = pop['mother_is_iron_fortified'].astype(float)
        # This call to `get_flour_consumption_from_propensity` should broadcast the index draws of mean_daily_flour
        # over the simulant_id's in pop.index
#         print(pop.loc[pop.mother_is_iron_fortified, 'mother_flour_consumption_propensity'])
        pop.loc[pop.mother_is_iron_fortified, 'mother_daily_flour'] = get_flour_consumption_from_propensity(
            self.local_data.mean_daily_flour,
            pop.loc[pop.mother_is_iron_fortified, 'mother_flour_consumption_propensity']
        )
        # This should broadcast draws of dose response and iron concentration over simulant id's indexing daily flour
        pop['birthweight_shift'] = calculate_birthweight_shift(
            self.global_data.birthweight_dose_response, self.local_data.iron_concentration, pop['mother_daily_flour'])
#         # Old version that couldn't modify in place:
#         shifted_pop = lbwsg_distribution.apply_birthweight_shift(
#             pop, pop['birthweight_shift'], bw_col='treatment_deleted_birthweight')
#         pop['treated_birthweight'] = shifted_pop['new_birthweight']
#         pop['treated_lbwsg_cat'] = shifted_pop['new_lbwsg_cat']
        # Note: The new columns will be called 'treated_treatment_deleted_birthweight' and 'treated_lbwsg_cat'...
        shifted_pop = lbwsg_distribution.apply_birthweight_shift(
            pop, pop['birthweight_shift'], bw_col='treatment_deleted_birthweight', shifted_col_prefix='treated')

