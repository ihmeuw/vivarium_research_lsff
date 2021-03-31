import pandas as pd, numpy as np
from collections import namedtuple

import demography, lbwsg, lsff_interventions
from lbwsg import LBWSGDistribution, LBWSGRiskEffect
from lsff_interventions import IronFortificationIntervention

# Assumes the path to vivarium_research_lsff is in sys.path
from multiplication_models import mult_model_fns


def get_default_fortification_input_data(vivarium_research_lsff_path='..'):
    locations = pd.read_csv(f'{vivarium_research_lsff_path}/gbd_data_summary/input_data/bmgf_top_25_countries_20201203.csv')
    location_ids = locations.location_id.to_list()
    coverage_data_path = f'{vivarium_research_lsff_path}/data_prep/outputs/lsff_input_coverage_data.csv'
    consumption_data_path = f'{vivarium_research_lsff_path}/data_prep/outputs/lsff_input_coverage_data.csv'
    concentration_data_path = '/share/scratch/users/ndbs/vivarium_lsff/gfdx_data/gfdx_full_dataset.csv'
    return get_fortification_input_data(location_ids, vehicle, coverage_data_path, consumption_data_path, concentration_data_path)

def get_fortification_input_data(location_ids, vehicle, coverage_data_path, consumption_data_path, concentration_data_path):
    """Reads input data from files and returns a tuple of input dataframes."""
    location_ids = list(location_ids)
    coverage_df = (mult_model_fns.pull_coverage_data(coverage_data_path, 'iron', vehicle, location_ids, 'wra')
                   .pipe(mult_model_fns.create_marginal_uncertainty)
                  )
    consumption_df = pd.read_csv(consumption_data_path)
    consumption_df = consumption_df.query("location_id in @location_ids and vehicle==@vehicle")
    
    concentration_df = pd.read_csv(concentration_data_path)
    return coverag_df, consumption_df, concentration_df

def get_gbd_input_data(location_id, hdfstore=None, exposure_key=None, rr_key=None, yll_key=None):
    if hdfstore is None:
        hdfstore = '/share/scratch/users/ndbs/vivarium_lsff/gbd_data/lbwsg_data.hdf'
    if exposure_key is None:
        exposure_key = '/gbd_2019/exposure/bmgf_25_countries'
    if rr_key is None:
        rr_key = '/gbd_2019/relative_risk/diarrheal_diseases'
    if yll_key is None:
        yll_key = '/gbd_2019/burden/ylls/bmgf_25_countries_all_subcauses'
        
    exposure_data = pd.read_hdf(hdfstore, exposure_key)
    exposure_data = lbwsg.preprocess_gbd_data(
        exposure_data, draws=draws,
        filter_terms=[f"location_id == {location_id}"],
        mean_draws_name=mean_draws_name
    )
    rr_data = pd.read_hdf(hdfstore, rr_key)
    # For now we only need early neonatal RR's
    rr_data = lbwsg.preprocess_gbd_data(
        rr_data, draws=draws, filter_terms=["age_group_id==2"], mean_draws_name=mean_draws_name)
    yll_data = pd.read_hdf(hdfstore, yll_key)
    return exposure_data, rr_data, yll_data