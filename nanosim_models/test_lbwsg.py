import pandas as pd, numpy as np
import demography
import lbwsg

def create_test_population(draws=None, sex='Female', age=1/365):
    """Create a test population to test LBWSG relative risk interpolation.
    
    The test population will have 1 simulant per LBWSG category, of the specified age and sex,
    repeated for each specified draw. Each simulant will be assigned a different LBWSG category,
    with gestational age and birthweight equal to the midpoint of the category.
    
    The test population can be used to check whether the interpolated RR values match
    the original RR values at each category midpoint like they're supposed to.
    """
    if draws is None:
        draws = [0]

    # Create a population with 1 simulant per category, of the specified age and sex,
    # repeated for each specified draw
    cat_df = lbwsg.get_category_data()
    num_simulants = len(cat_df)
    test_pop = demography.initialize_population_table(draws, num_simulants, cohort_age=age)
    test_pop['sex'] = sex
    test_pop['sex'] = test_pop['sex'].astype('category')

    # Assign a different LBWSG category to each simulant,
    # and give them the ga and bw equal to the midpoint of the category
    test_pop['lbwsg_category'] = cat_df['lbwsg_category'].array
    test_pop['gestational_age'] = cat_df['ga_midpoint'].array
    test_pop['birthweight'] = cat_df['bw_midpoint'].array
    return test_pop

def check_interpolated_rrs(rr_data, rr_interpolator_class=lbwsg.LBWSGRiskEffectInterp2d, test_pop=None, draws=None, tolerance=0.01):
    """rr_data is assumed to be relative risk data for LBWSG returned by get_draws"""
    if draws is None:
        draws = [0]
    rr_preprocessed = lbwsg.preprocess_gbd_data(rr_data, draws=draws)
    lbwsg_effect = lbwsg.LBWSGRiskEffect(rr_preprocessed)
    lbwsg_rr_interp = rr_interpolator_class(rr_preprocessed)
    
    if test_pop is None:
        test_pop = create_test_population(draws=draws)
    lbwsg_effect.assign_relative_risk(test_pop, rr_colname='lbwsg_relative_risk')
    lbwsg_rr_interp.assign_relative_risk(test_pop, rr_colname='interpolated_lbwsg_relative_risk')
    
    # Or use np.allclose with specified rtol and atol...
    test_pop[f"delta_rr_more_than_{tolerance:.2e}"] = (
        (np.abs(test_pop['interpolated_lbwsg_relative_risk'] - test_pop['lbwsg_relative_risk'])) > tolerance
    )
    return test_pop
    