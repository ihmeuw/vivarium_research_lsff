"""
Module for drawing plots related to the Vivarium CoNIC Large Scale Food Fortification project.

Fortification coverage data is from the following sources:

Grant J Aaron, Valerie M Friesen, Svenja Jungjohann, Greg S Garrett,
    Lynnette M Neufeld, Mark Myatt, Coverage of Large-Scale Food Fortification
    of Edible Oil, Wheat Flour, and Maize Flour Varies Greatly by Vehicle and
    Country but Is Consistently Lower among the Most Vulnerable: Results from
    Coverage Surveys in 8 Countries, The Journal of Nutrition, Volume 147, Issue
    5, May 2017, Pages 984S–994S, https://doi.org/10.3945/jn.116.245753
    
Ethiopian Federal Ministry of Health. Assessment of Feasibility and
    Potential Benefits of Food Fortification. 2011.
    http://www.ffinetwork.org/about/calendar/2011/documents%202011/Ethiopia.pdf
"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt

### Coverage function ###

def get_raw_flour_coverage_df():
    """
    Returns a dataframe with the flour fortification coverage data.
    """
    coverage_levels = ['eats_fortified', 'eats_fortifiable', 'eats_vehicle']
    coverage_stats = ['mean', 'lower', 'upper']

    flour_df = pd.DataFrame(
        {
            'Ethiopia': [1.0, 0.0, 10.0, 15.0, 10.0, 20.0, 28.0, 23.0, 33.0],
            'India (Rajasthan)': [6.3, 4.8, 7.9, 7.1, 5.6, 9.1, 83.2, 79.5, 86.5],
            'Nigeria (Kano)': [22.7, 20.0, 25.5, 83.8, 81.4, 86.2, 83.9, 81.5, 86.3],
            'Nigeria (Lagos)': [5.4, 3.8, 6.9, 13.8, 11.5, 16.1, 14.2, 11.8, 16.5],
        },
        index = pd.MultiIndex.from_product([coverage_levels, coverage_stats])
    )
    return flour_df

def get_raw_oil_coverage_df():
    """
    Returns a dataframe with the oil fortification coverage data.
    """
    coverage_levels = ['eats_fortified', 'eats_fortifiable', 'eats_vehicle']
    coverage_stats = ['mean', 'lower', 'upper']

    oil_df = pd.DataFrame(
        {
            'Ethiopia': [1, 0, 10, 44, 34, 54, 55, 45, 65],
            'India (Rajasthan)': [24.3, 21.1, 27.9, 89.4, 87.0, 91.8, 100, 100, 100],
            'Nigeria (Kano)': [7.6, 5.9, 9.4, 35.9, 32.7, 39.1, 98.4, 97.6, 99.3],
            'Nigeria (Lagos)': [7.2, 5.5, 8.9, 22.7, 19.9, 25.5, 98.6, 97.8, 99.3],
        },
        index = pd.MultiIndex.from_product([coverage_levels, coverage_stats])
    )
    return oil_df

def add_national_estimates(coverage_df):
    """
    Adds rows to coverage_df for national estimates of coverage in India and Nigeria,
    by taking a weighted avereage of the subnational estimates.
    """
    rajasthan_weight = 1 # Rajasthan is currently the only estimate we have for India
    kano_weight = 4/25 # population of Kano is ~4 million
    lagos_weight = 21/25 # population of Lagos is ~21 million

    coverage_df['India'] = rajasthan_weight*coverage_df['India (Rajasthan)']
    coverage_df['Nigeria'] = kano_weight*coverage_df['Nigeria (Kano)'] + lagos_weight*coverage_df['Nigeria (Lagos)']
    
def get_coverage_dfs():
    """Get a dictionary containing the flour and oil coverage dataframes with the national estimates added."""
    flour_df = get_raw_flour_coverage_df()
    oil_df = get_raw_oil_coverage_df()
    
    add_national_estimates(flour_df)
    add_national_estimates(oil_df)
    
    return {'flour': flour_df, 'oil': oil_df}

def coverage(t, a, b, c, t_start=1, r=0.1):
    """
    Returns fortification coverage as a function of time.
    
    a='eats_fortified', b='eats_fortifiable', c='eats_vehicle'
    t_start is the time at which the intervention starts.
    r is the proportion per year of unfortifiable vehicle that can be converted to a fortified version.
    """
    # return a if t < t_start else b + (c-b)*(1-(1-r)**(t-t_start)) # non-vectorized version (works for scalars)
    return np.where(t < t_start, a, b + (c-b)*(1-(1-r)**(t-t_start))) # vectorized version (t can be an array)

def plot_mean_lower_upper_ribbon(ax, x, mean, lower, upper, plt_kwargs={}, fill_kwargs={}):
    """Plot a line for the mean and a ribbon between lower and upper."""
    ax.plot(x, mean, **plt_kwargs)
    ax.fill_between(x, lower, upper, **fill_kwargs)
    return ax

def plot_coverage(ax, start_year, end_year, t_start, locations, coverage_df, vehicle):
    """Plot coverage as a function of time."""
    
    # Get a pandas IndexSlice for easier multi-indexing of coverage_df
    idx = pd.IndexSlice
    
    # Define the time range and coverage start time
    t = np.linspace(start_year, end_year, num=100)
    
    for location in locations:
        # Get the mean values of a,b,c to include in the plot legend
        a,b,c = coverage_df.loc[idx[:,'mean'], location]
    
        # Specify options for the plot
        plt_params = {'label':f'{location} (a={a:.1f}, b={b:.1f}, c={c:.1f})'}
        fill_params = {'alpha':0.1}
        
        # Compute 3 versions of the coverage function for the location,
        # based on mean, upper, and lower parameters
        mean, lower, upper = [coverage(t, *coverage_df.loc[idx[:,val], location], t_start)
                              for val in ['mean', 'lower', 'upper']]
        
        # Plot a line for the mean and a ribbon between lower and upper
        plot_mean_lower_upper_ribbon(ax, t, mean, lower, upper,
                                     plt_kwargs=plt_params, fill_kwargs=fill_params)
        
    # Add title, axis labels, and legend (should this go in the calling function insteead?)
    title_fontsize = 16
    axis_fontsize = 14
    legend_fontsize = 12

    ax.set_title(f'Population coverage of fortified {vehicle}', fontsize=title_fontsize)
    ax.set_xlabel('Year', fontsize=axis_fontsize)
    ax.set_ylabel('Percent of population', fontsize=axis_fontsize)
#     ax.set_ylim(0,50)
    ax.legend(fontsize=legend_fontsize)

    return ax

def make_coverage_plots():
    """Plot flour and oil coverage in the same figure."""
    fig, axs = plt.subplots(1,2, figsize=(14,5))
    
    locations = ['Ethiopia', 'India', 'Nigeria']
    
    for vehicle, df, ax in zip(['flour', 'oil'], get_coverage_dfs(), axs.flatten()):
        plot_coverage(ax, 2020, 2025, 2021, locations, df, vehicle)

    fig.tight_layout()

### Hemoglobin effect size ###
 
def hb_age_fraction(age):
    """
    Multiplier on Hb effect size due to children eating less food at younger
    ages. (`age` is current age in years).
    """
#     return 0 if age<0.5 else (age-0.5)/1.5 if age<2 else 1 # scalar version
    return np.select([age>2, age>0.5], [1, (age-0.5)/1.5]) # default=0 when both conditions are false

def hb_lag_fraction(time_since_fortified):
    """
    Multiplier on Hb effect size due to lag in response time to iron fortification.
    The argument `time_since_fortified` is the time (in years) since a simulant
    started eating fortified food (note that a negative value of `time_since_fortified`
    indicates the child has not yet started eating fortified food).
    """
# # scalar version:
#     return (0                         if time_since_fortified < 0   else
#           time_since_fortified/0.5  if time_since_fortified < 0.5 else
#           1)
    # vectorized version (default=0 when both conditions are false):
    return np.select([time_since_fortified > 0.5, time_since_fortified > 0], [1, time_since_fortified/0.5])

