import sys

import pandas as pd, numpy as np

import data_processing, lbwsg_boe
# from lbwsg_boe import IronBirthweightCalculator
# from lbwsg import LBWSGDistribution, LBWSGRiskEffect
# from lsff_interventions import IronFortificationIntervention

def get_name_and_id_for_location(location_df, location):
    """Returns (location_name, location_id) if location is either a name or an id,
    using the one-to-one mapping specified by locaiton_df.
    """
    if isinstance(location, int) or (isinstance(location, str) and location.isdigit()):
        location_id = int(location)
        location_name = location_df.set_index('location_id').loc[location_id, 'location_name']
    elif isinstance(location, str):
        location_name = location
        location_id = location_df.set_index('location_name').loc[location_name, 'location_id']
    else:
        raise ValueError(f"What kind of input are you giving me?! {location=} {type(location)=}")
    return location_name, location_id

def main(vivarium_research_lsff_path, out_directory, location, num_simulants, num_draws=1000, random_seed=43, take_mean=False):
    """Computes the PIF for each vehicle for the specified location, for the gap_coverage levels [0.2, 0.5, 0.8]."""
#     fortification_data = get_fortification_input_data()
#     gbd_data = get_gbd_input_data()
    # Use the same random sequence of draws on all runs, starting at the specified location in the sequence
    # That way, we can add more draws to existing results if we want
#     print(num_draws, draw_start_idx)
#     print(type(num_draws), type(draw_start_idx), type(num_simulants))
    with open(f'{out_directory}/{location}_parallel.txt', 'w+') as outfile:
        outfile.write(f'{vivarium_research_lsff_path}, {out_directory}, {location}, {num_simulants}, {random_seed}, {num_draws}, {take_mean}')
    draw_start_idx = 0
    draws = np.random.default_rng(1234).permutation(1000)[draw_start_idx:draw_start_idx+num_draws]
    draws.sort()
    # Make sure the effect size is consistent across locations by setting the effect size seed
    # TODO: Hmm, will this work correctly if we try adding to existing draws as above...?
    effect_size_seed = draw_start_idx+5678
#     vehicles = ['wheat flour', 'maize flour'] # Restrict to these vehicles for now
    ages = [4/365, 14/365] # Choose one age in ENN (0-7 days) and one in LNN (7-28 days)
    age_group_ids = [2,3] # demography.get_age_to_age_id_map()[ages] # 2=ENN, 3=LNN
    gap_coverage_proportions = [] #[0.2, 0.5, 0.8] # Low, medium, high scenarios
#     year = 2025 # Needed in index for Ali's code
    
    # Get the name and id for the location
    locations = pd.read_csv(f'{vivarium_research_lsff_path}/gbd_data_summary/input_data/bmgf_top_25_countries_20201203.csv')
    location_name, location_id = get_name_and_id_for_location(locations, location)

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
#     vehicles_for_location = fortification_input_data.consumption.query("location_name==@location")['vehicle'].unique()
    vehicles_for_location = fortification_input_data.consumption.query("location_id==@location_id")['vehicle'].unique()
    for vehicle in vehicles_for_location:
#         if vehicle not in vehicles:
#             continue
        # Get local data for vehicle
        local_data = data_processing.get_local_data(global_data, fortification_input_data, location_id, vehicle)
        # Create a calculator and set up its populations
        calc = lbwsg_boe.IronBirthweightCalculator(global_data, local_data, gbd_data)
        calc.initialize_population_tables(num_simulants, ages)
        calc.assign_lbwsg_exposure()
        calc.assign_iron_treatment_deleted_birthweights()

        # Compute population coverage proportions from gap coverage proportions
        target_coverage_levels = [
            local_data.eats_fortifiable
                + proportion * (local_data.eats_fortifiable - local_data.eats_fortified)
            for proportion in gap_coverage_proportions
        ]

        # Also compute miminmum and maximum impact, i.e. impact when coverage = 0% and 100% of the population
        target_coverage_levels.append(0)
#         target_coverage_levels.append(1)
        # Compute the corresponding proportion to use as an index key (we could potentially get NaN's...)
        proportions = gap_coverage_proportions + [
            ((0 - local_data.eats_fortified) / (local_data.eats_fortifiable - local_data.eats_fortified)).mean(),
#             ((1 - local_data.eats_fortified) / (local_data.eats_fortifiable - local_data.eats_fortified)).mean()
        ]

        # Compute the PIF for each coverage level
        for proportion, target_coverage in zip(proportions, target_coverage_levels):
            calc.assign_iron_treated_birthweights(target_coverage)
            calc.assign_lbwsg_relative_risks()
            pif = calc.calculate_potential_impact_fraction(['draw', 'age_group_id'])
            output[(location_id, vehicle, proportion, 'pif')] = pif
            
#             bpop, ipop = calc.baseline_pop, calc.intervention_pop
#             for pop in (bpop, ipop):
#                 calc.lbwsg_effect.assign_categorical_relative_risk(
#                     pop, cat_colname='treated_lbwsg_category', rr_colname='lbwsg_relative_risk_for_category'
#                 )
#             output[(local_data.location_id, vehicle, proportion, 'categorical_pif')] = (
#                 lbwsg_boe.potential_impact_fraction(bpop, ipop, 'lbwsg_relative_risk_for_category', ['draw', 'age_group_id'])
#             )
            # Now also do DALYs...
            averted_dalys = calc.calculate_averted_dalys(['draw', 'age_group_id'], drop_non_groupby_levels=True)
#             averted_dalys.index = averted_dalys.index.droplevel(
#                 averted_dalys.index.names.difference(['draw', 'age_group_id'])
#             )
            output[(location_id, vehicle, proportion, 'averted_dalys')] = averted_dalys

    # Make sure index level names and values match plotting function in Ali's code
    if not take_mean:
        output.index = output.index.set_levels([f"draw_{n}" for n in draws], level='draw') # age_group_id level stays the same
    output.columns = pd.MultiIndex.from_tuples(output.columns, names=['location_id', 'vehicle', 'coverage_level', 'measure'])

    # Save output with draws as columns and all other identifiers in index
    output.unstack('age_group_id').T.to_csv(
        f"{out_directory}/iron_bw_results__{location_name}_{location_id}.csv")


if __name__ == "__main__":
#     main(*sys.argv[1:])
    args = sys.argv[1:]
    vivarium_research_lsff_path, out_directory, location, num_simulants, num_draws, random_seed, take_mean = args
    main(vivarium_research_lsff_path, out_directory, location, int(num_simulants), int(num_draws), int(random_seed), eval(take_mean))
