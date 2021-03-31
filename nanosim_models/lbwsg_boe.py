import pandas as pd, numpy as np
from collections import namedtuple

import demography, lbwsg, lsff_interventions
from lbwsg import LBWSGDistribution, LBWSGRiskEffect
from lsff_interventions import IronFortificationIntervention

# Assumes the path to vivarium_research_lsff is in sys.path
from multiplication_models import mult_model_fns

class IronBirthweightCalculator:
    """Class to run nanosimulations for the effect of iron on low birthweight."""
    
    def from_stuff(location, vehicle, draws, random_seed, take_mean):
        return

#     def __init__(self, location, artifact_path, year, draws, vehicle, covered_proportion_of_eats_fortifiable,
#                  take_mean=False, risk_effect_class=lbwsg.LBWSGRiskEffect, random_seed=None):
    def __init__(
        self,
        global_data,
        local_data,
        gbd_data,
        age, # pass an int, list of ints, or eventually enable passing an age distribution
        gap_coverage_proportions,
        risk_effect_class=lbwsg.LBWSGRiskEffectRBVSpline,
        random_seed=None,
        compute_maximum_effect=True,
        take_mean=False
    ):
        """
        """
        # Save input parameters so we can look them up later if necessary/desired
        self.location = location
        self.artifact_path = artifact_path
        self.year = year
#         self.draws = draws # These will also be stored in global_data, unless take_mean is True
        
        # TODO: Perhaps create and save a numpy random generator, and share it via global_data
        # random_generator = np.random.default_rng(random_seed) # Now do something with this...
        np.random.seed(random_seed)
        
#         if take_mean:
#             mean_draws_name = f'mean_of_{len(draws)}_draws'
#             draws = [mean_draws_name]
            
        mean_draws_name = f'mean_of_{len(draws)}_draws' if take_mean else None

        # Load iron intervention data
#         flour_coverage_df = lsff_interventions.get_flour_coverage_df()
#         baseline_coverage = flour_coverage_df.loc[location, ('eats_fortified', 'mean')] / 100
#         intervention_coverage = flour_coverage_df.loc[location, ('eats_fortifiable', 'mean')] / 100
        self.global_data = lsff_interventions.get_global_data(draws, mean_draws_name=mean_draws_name)
        self.local_data = lsff_interventions.get_local_data(
            self.global_data, location, vehicle, covered_proportion_of_eats_fortifiable)
        
        # Load LBWSG data
        if year==2017:
            exposure_data = lbwsg.read_lbwsg_data(
                artifact_path, 'exposure', "age_end < 1", f"year_start == {year}", draws=draws)
            rr_data = lbwsg.read_lbwsg_data(
                artifact_path, 'relative_risk', "age_end < 1", f"year_start == {year}", draws=draws)
        elif year==2019:
            exposure_data = pd.read_hdf(artifact_path, f"/gbd_2019/exposure/bmgf_25_countries")
#             location_id = id_helper.list_ids('location', location)
            exposure_data = lbwsg.preprocess_gbd_data(
                exposure_data, draws=draws,
                filter_terms=[f"location_id == {self.local_data.location_id}"],
                mean_draws_name=mean_draws_name
            )
            rr_data = pd.read_hdf(artifact_path, '/gbd_2019/relative_risk/diarrheal_diseases')
            # For now we only need early neonatal RR's
            rr_data = lbwsg.preprocess_gbd_data(
                rr_data, draws=draws, filter_terms=["age_group_id==2"], mean_draws_name=mean_draws_name)
        
#         if take_mean:
#             exposure_data = exposure_data.mean(axis=1).rename(mean_draws_name).to_frame()
#             rr_data = rr_data.mean(axis=1).rename(mean_draws_name).to_frame()

        # Create model components
        self.lbwsg_distribution = LBWSGDistribution(exposure_data)
        self.lbwsg_effect = risk_effect_class(rr_data, paf_data=None) # We don't need PAFs to initialize the pop tables with RR's

#         self.baseline_fortification = IronFortificationIntervention(location, baseline_coverage, baseline_coverage)
#         self.intervention_fortification = IronFortificationIntervention(location, baseline_coverage, intervention_coverage)
        self.iron_intervention = IronFortificationIntervention(self.global_data, self.local_data)
    
        # Declare variables for baseline and intervention populations,
        # which will be initialized in initialize_population_tables
        self.baseline_pop = None
        self.intervention_pop = None
        self.potential_impact_fraction = None

    def initialize_population_tables(self, num_simulants):
        """Creates populations for baseline scenario and iron fortification intervention scenario,
        assigns birthweights and gestational ages to each simulant, shifts birthweights appropriately,
        and assigns relative risks for mortality based on resulting LBWSG categories.
        """
        # Create baseline population and assign demographic data
        self.baseline_pop = demography.initialize_population_table(self.global_data.draws, num_simulants, 0)

        # Assign propensities to share between scenarios
#         assign_propensity(self.baseline_pop, IronFortificationIntervention.propensity_name)
#         self.lbwsg_distribution.assign_propensities(self.baseline_pop)
#         self.iron_intervention.assign_propensities(self.baseline_pop)
        propensity_names = [name for component in (self.lbwsg_distribution, self.iron_intervention)
                            for name in component.get_propensity_names()]
        demography.assign_propensities(self.baseline_pop, propensity_names)
        
        # Create intervention population - all the above data will be the same in intervention scenario
        self.intervention_pop = self.baseline_pop.copy()
        
        # Reset PIF to None until we're ready to recompute it
        self.potential_impact_fraction = None

    def assign_lbwsg_exposure(self):
        # Assign baseline exposure - ideally this would be done with a propensity to share between scenarios,
        # but that's more complicated to implement, so I'll just copy the table after assigning lbwsg exposure.
        self.lbwsg_distribution.assign_exposure(self.baseline_pop, category_col='lbwsg_category')
        self.intervention_pop = self.baseline_pop.copy() # Hack to deal with slow assign_exposure function
        
    def assign_iron_treatment_deleted_birthweights(self):  
        # Apply the birthweight shifts in baseline and intervention scenarios:
        # First compute treatment-deleted birthweight, then birthweight with iron fortification.
#         self.baseline_fortification.assign_treatment_deleted_birthweight(baseline_pop, self.lbwsg_distribution)
#         self.intervention_fortification.assign_treatment_deleted_birthweight(intervention_pop, self.lbwsg_distribution)
        self.iron_intervention.assign_treatment_deleted_birthweight(
            self.baseline_pop, self.lbwsg_distribution, self.local_data.eats_fortified)
        self.iron_intervention.assign_treatment_deleted_birthweight(
            self.intervention_pop, self.lbwsg_distribution, self.local_data.eats_fortified)

    def assign_iron_treated_birthweights(self):
#         self.baseline_fortification.assign_treated_birthweight(baseline_pop, self.lbwsg_distribution)
#         self.intervention_fortification.assign_treated_birthweight(intervention_pop, self.lbwsg_distribution)
        self.iron_intervention.assign_treated_birthweight(
            self.baseline_pop, self.lbwsg_distribution, self.local_data.eats_fortified)
        self.iron_intervention.assign_treated_birthweight(
            self.intervention_pop, self.lbwsg_distribution, self.local_data.eats_fortifiable)

    def age_populations(self, age_increment=1/365):
        demography.increase_age(self.baseline_pop, age_increment)
        demography.increase_age(self.intervention_pop, age_increment)

    def assign_lbwsg_relative_risks(self):
        # Compute the LBWSG relative risks in both scenarios - these will be used to compute the PIF
        # TODO: Maybe have lbwsg return the RR values instead, and assign them to the appropriate column here
        if type(self.lbwsg_effect) == lbwsg.LBWSGRiskEffect:
            self.lbwsg_effect.assign_relative_risk(self.baseline_pop, cat_colname='treated_lbwsg_category')
            self.lbwsg_effect.assign_relative_risk(self.intervention_pop, cat_colname='treated_lbwsg_category')
        else:
            self.lbwsg_effect.assign_relative_risk(
                self.baseline_pop, bw_colname='treated_treatment_deleted_birthweight')
            self.lbwsg_effect.assign_relative_risk(
                self.intervention_pop, bw_colname='treated_treatment_deleted_birthweight')
    
    def calculate_potential_impact_fraction(self):
        self.potential_impact_fraction = potential_impact_fraction(
            self.baseline_pop, self.intervention_pop, 'lbwsg_relative_risk')
        
    def do_back_of_envelope_calculation(self, num_simulants):
        """
        """
        self.initialize_population_tables(num_simulants)
        self.assign_lbwsg_exposure()
        self.assign_iron_treatment_deleted_birthweights()
        self.assign_iron_treated_birthweights()
        self.age_populations() # Necessary because there are no relative risks for birth age group
        self.assign_lbwsg_relative_risks()
        self.calculate_potential_impact_fraction()
    
def potential_impact_fraction(baseline_pop, counterfactual_pop, rr_colname):
    """Computes the population impact fraction for the specified baseline and counterfactual populations."""
    baseline_mean_rr = baseline_pop.groupby('draw')[rr_colname].mean()
    counterfactual_mean_rr = counterfactual_pop.groupby('draw')[rr_colname].mean()
    return (baseline_mean_rr - counterfactual_mean_rr) / baseline_mean_rr

def parse_args(args):
    """"""
    # Class to store and name the arguments passed to main()
    ParsedArgs = namedtuple('ParsedArgs', "location, artifact_path, year, draws, take_mean, random_seed, num_simulants")
    if len(args)>0:
        # Don't do any parsing for now, just make args into a named tuple
        args = ParsedArgs._make(args)
    else:
        # Hardcode some values for testing
        location = "Nigeria"
        artifact_path = f'/share/costeffectiveness/artifacts/vivarium_conic_lsff/{location.lower()}.hdf'
        year=2017
        draws = [0,50,100]
        take_mean = False
        random_seed = 191919
        num_simulants = 10
        args = ParsedArgs(location, artifact_path, year, draws, take_mean, random_seed, num_simulants)
    return args

# def main(args=None):
#     """
#     Does a back of the envelope calculation for the given arguments
#     """
#     if args is None:
#         args = sys.argv[1:]
        
#     args = parse_args(args)
#     # Old code:
#     sim = IronBirthweightCalculator(args.location, args.artifact_path, args.year, args.draws, args.take_mean)
#     baseline_pop, intervention_pop = sim.initialize_population_tables(args.num_simulants)
#     pif = population_impact_fraction(baseline_pop, intervention_pop, IronBirthweightNanoSim.treated_lbwsg_rr_colname)
    # do something with pif...
    
    # Iterate over... locations, vehicles, coverage levels
    # (Note: different age groups and sexes can be handled in one Calculator)
    # Parameters for calculator:
    # global_data (including draws), location (or local_data), vehicle, coverage levels
    # whether to take mean (or mean_draws_name)
    # "compliance" multiplier for iron concentration
    # ages or age groups, with ratios (can be passed to initialize_population_tables)
    # sexes or sex ratio (can be passed to initialize_population_tables)
    # 
    # Arguments for main():
    # location, draws, take_mean
    
def main(location, num_simulants, random_seed, draws, take_mean):
    """Computes the PIF for each vehicle for the specified location, for the gap_coverage levels [0.2, 0.5, 0.8]."""
    fortification_data = get_fortification_input_data()
    gbd_data = get_gbd_input_data()
    
    gap_coverage_levels = [0.2, 0.5, 0.8]
    absolute_coverage_levels = [1]
    
    coverage_df = fortification_data.coverage_df
    coverage_df = coverage_df.query("location_id == @location_id and nutrient=='iron' and wra==True")
    # Now loop through all iron vehicles for location:
    # Actually, might need to check that vehicle exists in all input dataframes, or exclude some vehicles...
    for vehicle in coverage_df.vehicles.unique():
        # do stuff...
        pass

if __name__ == "main":
    import sys
    args = sys.argv[1:]
    main(*args)
    
    

