import pandas as pd, numpy as np
# from collections import namedtuple

import demography, lbwsg, lsff_interventions, data_processing
from lbwsg import LBWSGDistribution, LBWSGRiskEffect
from lsff_interventions import IronFortificationIntervention

# Assumes the path to vivarium_research_lsff is in sys.path
# from multiplication_models import mult_model_fns

class IronBirthweightCalculator:
    """Class to run nanosimulations for the effect of iron on low birthweight."""
    
    def from_parameters(
        location, # location can be a location_name (str) or location_id (int)
        vehicle, # str
        draws, # iterable of ints
        risk_effect_class=lbwsg.LBWSGRiskEffectRBVSpline,
        effect_size_seed=5678, # This can be a seed or a numpy.random.Generator
        random_generator=23, # This can be a seed or a numpy.random.Generator
        take_mean=False,
        vivarium_research_lsff_path='..',
    ):
        global_data = data_processing.get_global_data(effect_size_seed, random_generator, draws, take_mean)
        fortification_input_data = data_processing.get_fortification_input_data(vivarium_research_lsff_path)
        local_data = data_processing.get_local_data(global_data, fortification_input_data, location, vehicle)
        gbd_data = data_processing.get_gbd_input_data()
        return IronBirthweightCalculator(global_data, local_data, gbd_data, risk_effect_class)

#     def __init__(self, location, artifact_path, year, draws, vehicle, covered_proportion_of_eats_fortifiable,
#                  take_mean=False, risk_effect_class=lbwsg.LBWSGRiskEffect, random_seed=None):
    def __init__(self, global_data, local_data, gbd_data, risk_effect_class=lbwsg.LBWSGRiskEffectRBVSpline):
        """
        """
        # Store needed data
        self.global_data = global_data
        self.local_data = local_data
#         self.lbwsg_dalys = gbd_data.lbwsg_dalys.query(
#             f"location_id=={self.local_data.location_id} and cause_id==294"
#         )
        self.lbwsg_dalys = lbwsg.preprocess_gbd_data(
            gbd_data.lbwsg_dalys,
            draws=global_data.draw_numbers,
            filter_terms=[
                f"location_id == {self.local_data.location_id}",
                "cause_id==294",
                "metric_id==1",
                f"age_group_id in [164,2,3]"
            ],
            mean_draws_name=global_data.mean_draws_name
        )
        # Preprocess GBD data for LBWSG classes
        exposure_data = lbwsg.preprocess_gbd_data(
            gbd_data.lbwsg_exposure,
            draws=global_data.draw_numbers,
            filter_terms=[f"location_id == {self.local_data.location_id}"],
            mean_draws_name=global_data.mean_draws_name
        )
        rr_data = lbwsg.preprocess_gbd_data(
            gbd_data.lbwsg_rrs,
            draws=global_data.draw_numbers,
            filter_terms=None,
            mean_draws_name=global_data.mean_draws_name
        )
        # Create model components
        self.lbwsg_distribution = LBWSGDistribution(exposure_data)
        self.lbwsg_effect = risk_effect_class(rr_data, paf_data=None) # We don't need PAFs to initialize the pop tables with RRs
        self.iron_intervention = IronFortificationIntervention(global_data, local_data)
    
        # Declare variables for baseline and intervention populations,
        # which will be initialized in initialize_population_tables
        self.baseline_pop = None
        self.intervention_pop = None
#         self.potential_impact_fraction = None

    def initialize_population_tables(self, num_simulants, ages):
        """Creates populations for baseline scenario and iron fortification intervention scenario,
        assigns birthweights and gestational ages to each simulant, shifts birthweights appropriately,
        and assigns relative risks for mortality based on resulting LBWSG categories.
        """
        # Create baseline population and assign demographic data
        self.baseline_pop = demography.initialize_population_table(self.global_data.draws, num_simulants, ages)

        # Assign propensities to share between scenarios
#         self.lbwsg_distribution.assign_propensities(self.baseline_pop)
#         self.iron_intervention.assign_propensities(self.baseline_pop)
        propensity_names = [name for component in (self.lbwsg_distribution, self.iron_intervention)
                            for name in component.get_propensity_names()]
        demography.assign_propensities(self.baseline_pop, propensity_names)
        
        # Create intervention population - all the above data will be the same in intervention scenario
        self.intervention_pop = self.baseline_pop.copy()
        
#         # Reset PIF to None until we're ready to recompute it
#         self.potential_impact_fraction = None

    def assign_lbwsg_exposure(self):
        # Assign baseline exposure
        self.lbwsg_distribution.assign_exposure(self.baseline_pop, category_col='lbwsg_category')
        self.intervention_pop = self.baseline_pop.copy() # Hack to deal with slow assign_exposure function
        
    def assign_iron_treatment_deleted_birthweights(self):  
        # Apply the birthweight shifts in baseline and intervention scenarios:
        # First compute treatment-deleted birthweight, then birthweight with iron fortification.
        self.iron_intervention.assign_treatment_deleted_birthweight(
            self.baseline_pop, self.lbwsg_distribution, self.local_data.eats_fortified)
        self.iron_intervention.assign_treatment_deleted_birthweight(
            self.intervention_pop, self.lbwsg_distribution, self.local_data.eats_fortified)

    def assign_iron_treated_birthweights(self, target_coverage=None):
        if target_coverage is None:
            target_coverage = self.local_data.eats_fortifiable
        self.iron_intervention.assign_treated_birthweight(
            self.baseline_pop, self.lbwsg_distribution, self.local_data.eats_fortified)
        #TODO: Implement passing in a column name
        self.iron_intervention.assign_treated_birthweight(
            self.intervention_pop, self.lbwsg_distribution, target_coverage)

    def age_populations(self, age_increment=1/365):
        demography.increase_age(self.baseline_pop, age_increment)
        demography.increase_age(self.intervention_pop, age_increment)

    def assign_lbwsg_relative_risks(self):
        # Compute the LBWSG relative risks in both scenarios - these will be used to compute the PIF
        # TODO: Maybe have lbwsg return the RR values instead, and assign them to the appropriate column here
#         if type(self.lbwsg_effect) == lbwsg.LBWSGRiskEffect:
#             self.lbwsg_effect.assign_relative_risk(self.baseline_pop, cat_colname='treated_lbwsg_category')
#             self.lbwsg_effect.assign_relative_risk(self.intervention_pop, cat_colname='treated_lbwsg_category')
#         else:
#             self.lbwsg_effect.assign_relative_risk(
#                 self.baseline_pop, bw_colname='treated_treatment_deleted_birthweight')
#             self.lbwsg_effect.assign_relative_risk(
#                 self.intervention_pop, bw_colname='treated_treatment_deleted_birthweight')
        if type(self.lbwsg_effect) == lbwsg.LBWSGRiskEffect:
            kwargs = dict(cat_colname='treated_lbwsg_category')
        else:
            kwargs = dict(bw_colname='treated_treatment_deleted_birthweight',
                          cat_colname='treated_lbwsg_category') # Really this shouldn't be necessary...
        self.lbwsg_effect.assign_relative_risk(self.baseline_pop, **kwargs)
        self.lbwsg_effect.assign_relative_risk(self.intervention_pop, **kwargs)

    def calculate_potential_impact_fraction(self, groupby='draw'):
        pif = potential_impact_fraction(
            self.baseline_pop, self.intervention_pop, 'lbwsg_relative_risk', groupby)
        return pif
        
    def calculate_averted_dalys(self, groupby='draw', drop_non_groupby_levels=False):
        pif = self.calculate_potential_impact_fraction(['draw', 'age_group_id', 'sex'])
        averted_dalys = (
            (self.lbwsg_dalys * pif)
            .groupby(groupby)
            .sum()
            .rename('averted_dalys')
        )
        if drop_non_groupby_levels:
            averted_dalys.index = averted_dalys.index.droplevel(averted_dalys.index.names.difference(groupby))
        return averted_dalys

    def do_back_of_envelope_calculation(self, num_simulants, ages, target_coverage=None):
        """
        """
        self.initialize_population_tables(num_simulants, ages)
        self.assign_lbwsg_exposure()
        self.assign_iron_treatment_deleted_birthweights()
        self.assign_iron_treated_birthweights(target_coverage)
#         self.age_populations() # Necessary because there are no relative risks for birth age group
        self.assign_lbwsg_relative_risks()
        # TODO: Maybe concatenate PIF and DALYs averted and return that...
        return self.calculate_potential_impact_fraction()
    
def potential_impact_fraction(baseline_pop, counterfactual_pop, rr_colname, groupby='draw'):
    """Computes the population impact fraction for the specified baseline and counterfactual populations."""
    baseline_mean_rr = baseline_pop.groupby(groupby)[rr_colname].mean()
    counterfactual_mean_rr = counterfactual_pop.groupby(groupby)[rr_colname].mean()
    return (baseline_mean_rr - counterfactual_mean_rr) / baseline_mean_rr

def main(vivarium_research_lsff_path, out_directory, location, num_simulants, num_draws=1000, draw_start_idx=0, random_seed=43, take_mean=False):
    """Computes the PIF for each vehicle for the specified location, for the gap_coverage levels [0.2, 0.5, 0.8]."""
#     fortification_data = get_fortification_input_data()
#     gbd_data = get_gbd_input_data()
    # Use the same random sequence of draws on all runs, starting at the specified location in the sequence
    # That way, we can add more draws to existing results if we want
    draws = np.random.default_rng(1234).permutation(1000)[draw_start_idx:num_draws]
    draws.sort()
    # Make sure the effect size is consistent across locations by setting the effect size seed
    # TODO: Hmm, will this work correctly if we try adding to existing draws as above...?
    effect_size_seed = draw_start_idx+5678
#     vehicles = ['wheat flour', 'maize flour'] # Restrict to these vehicles for now
    ages = [4/365, 14/365] # Choose one age in ENN (0-7 days) and one in LNN (7-28 days)
    age_group_ids = [2,3] # demography.get_age_to_age_id_map()[ages] # 2=ENN, 3=LNN
    gap_coverage_proportions = [0.2, 0.5, 0.8] # Low, medium, high scenarios
#     year = 2025 # Needed in index for Ali's code

    global_data = data_processing.get_global_data(effect_size_seed, random_seed, draws, take_mean)
    fortification_input_data = data_processing.get_fortification_input_data(vivarium_research_lsff_path)
    gbd_data = data_processing.get_gbd_input_data()

    # If take_mean==True, global_data.draws will consist of a single value
    output_index = pd.MultiIndex.from_product([global_data.draws, age_group_ids], names=['draw', 'age_group_id'])
    output = pd.DataFrame(index=output_index)

    # Now loop through all iron vehicles for location:
    # coverage_df = coverage_df.query("location_id == @location_id and nutrient=='iron' and wra==True")
    # for vehicle in coverage_df.vehicles.unique():
    # Actually, might need to check that vehicle exists in all input dataframes, or exclude some vehicles...

    # Use consumption df to determine which location-vehicle pairs we have data for,
    # since it has been inner joined with coverage df
    vehicles_for_location = fortification_input_data.consumption.query("location_name==@location")['vehicle'].unique()
    for vehicle in vehicles_for_location:
#         if vehicle not in vehicles:
#             continue
        # Get local data for vehicle
        local_data = data_processing.get_local_data(global_data, fortification_input_data, location, vehicle)
        # Create a calculator and set up its populations
        calc = IronBirthweightCalculator(global_data, local_data, gbd_data)
        calc.initialize_population_tables(num_simulants, ages)
        calc.assign_lbwsg_exposure()
        calc.assign_iron_treatment_deleted_birthweights()

        # Compute population coverage proportions from gap coverage proportions
        target_coverage_levels = [
            local_data.eats_fortifiable
                + proportion * (local_data.eats_fortifiable - local_data.eats_fortified)
            for proportion in gap_coverage_proportions
        ]

        # Also compute maximum impact, i.e. impact when coverage = 100% of the population
        target_coverage_levels.append(1)
        # Compute the corresponding proportion to use as an index key (we could potentially get NaN's...)
        proportions = gap_coverage_proportions + [
            ((1 - local_data.eats_fortified) / (local_data.eats_fortifiable - local_data.eats_fortified)).mean()
        ]

        # Compute the PIF for each coverage level
        for proportion, target_coverage in zip(proportions, target_coverage_levels):
            calc.assign_iron_treated_birthweights(target_coverage)
            calc.assign_lbwsg_relative_risks()
            pif = calc.calculate_potential_impact_fraction(['draw', 'age_group_id'])
            output[(local_data.location_id, vehicle, proportion, 'pif')] = pif
            
            bpop, ipop = calc.baseline_pop, calc.intervention_pop
            for pop in (bpop, ipop):
                calc.lbwsg_effect.assign_categorical_relative_risk(
                    pop, cat_colname='treated_lbwsg_category', rr_colname='lbwsg_relative_risk_for_category'
                )
            output[(local_data.location_id, vehicle, proportion, 'categorical_pif')] = (
                potential_impact_fraction(bpop, ipop, 'lbwsg_relative_risk_for_category', ['draw', 'age_group_id'])
            )
            # Now also do DALYs...
            averted_dalys = calc.calculate_averted_dalys(['draw', 'age_group_id'], drop_non_groupby_levels=True)
#             averted_dalys.index = averted_dalys.index.droplevel(
#                 averted_dalys.index.names.difference(['draw', 'age_group_id'])
#             )
            output[(local_data.location_id, vehicle, proportion, 'averted_dalys')] = averted_dalys

    # Make sure index level names and values match plotting function in Ali's code
    if not take_mean:
        output.index = output.index.set_levels([f"draw_{n}" for n in draws], level='draw') # age_group_id level stays the same
    output.columns = pd.MultiIndex.from_tuples(output.columns, names=['location_id', 'vehicle', 'coverage_level', 'measure'])

    # Save output with draws as columns and all other identifiers in index
    output.unstack('age_group_id').T.to_csv(
        f"{out_directory}/iron_bw_results_location_id_{local_data.location_id}.csv")

if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    main(*args)
    
    

