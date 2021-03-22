import pandas as pd, numpy as np
from db_queries import get_ids, get_outputs, get_location_metadata, get_population, get_covariate_estimates
from get_draws.api import get_draws
import scipy.stats
import scipy.integrate as integrate
import matplotlib.pyplot as plt

def generate_coverage_parameter_draws(df):
    """This function is used to generate 1000 draws of nutrient/vehicle coverage parameters based on
    the mean value and confidence intervals. This function assumes a normal distribution of uncertainty
    within the confidence interval centered around the mean and is truncated at the bounds of 0 and 100%"""
    data_frame = df.copy()
    np.random.seed(11)
    for i in list(range(0,1000)):
        data_frame[f'draw_{i}'] = scipy.stats.truncnorm.rvs(data_frame.a,
                                                            data_frame.b,
                                                            data_frame.value_mean,
                                                            data_frame.value_std) / 100
    data_frame = (data_frame
                  .set_index(['location_id'])
                  .drop(columns=[c for c in data_frame.columns if 'draw' not in c
                                and c not in ['location_id','value_description']]))
    return data_frame


def generate_overall_coverage_rates(filepath,
                                    nutrient,
                                    vehicle,
                                    coverage_levels,
                                    years,
                                    location_ids,
                                    subpopulations):
    """This function generates baseline and counterfactual coverage rates of fortification for a specified
    nutrient and vehicle pair. The baseline coverage rates are assumed to remain constant from over all of
    the specified years. The alternative coverage rates are assumed to jump from the baseline rate in the
    first specified year to either 20/50/80 percent (or the defined coverage levels) of the proportion
    of the population consuming fortifiable/industrially produced vehicle in the second year of the specified
    years and then remains constant for the remaining years.

    The location_ids paramter can be set to 'all' or a list of specific location IDs.

    The subpopulation parameter should be carefully evaluated and set to a list of the desired subpopulations
    of interest that may be specific to women of reproductive age or children under 5, etc.
    """

    data = pd.read_csv(filepath).sort_values(by='location_id')
    if location_ids == 'all':
        data = (data.loc[data.sub_population.isin(subpopulations)].drop_duplicates())
    else:
        data = (data.loc[data.location_id.isin(location_ids)]
            .loc[data.sub_population.isin(subpopulations)].drop_duplicates())

    # the following is a transformation for a potential data issue and should be removed when resolved
    data['value_mean'] = data['value_mean'].replace(100, 100 - 0.00001 * 2).replace(0, 0 + 0.00001 * 2)
    data['value_025_percentile'] = data['value_025_percentile'].replace(100, 100 - 0.00001 * 3).replace(0, 0 + 0.00001)
    data['value_975_percentile'] = data['value_975_percentile'].replace(100, 100 - 0.00001).replace(0, 0 + 0.00001 * 3)

    data = data.loc[data.vehicle == vehicle].loc[data.nutrient.isin([nutrient, 'na'])]
    data['value_std'] = (data.value_975_percentile - data.value_025_percentile) / (2 * 1.96)
    data['a'] = (0 - data.value_mean) / data.value_std
    data['b'] = (100 - data.value_mean) / data.value_std

    cov_a = data.loc[data.value_description == 'percent of population eating fortified vehicle'].drop(
        columns='value_description')
    cov_b = data.loc[data.value_description == 'percent of population eating industrially produced vehicle'].drop(
        columns='value_description')
    cov_a = generate_coverage_parameter_draws(cov_a)
    cov_b = generate_coverage_parameter_draws(cov_b)

    assert np.all(cov_a <= cov_b), "Error: coverage parameters are not logically ordered"

    baseline_coverage = pd.DataFrame()
    for year in years:
        temp = cov_a.copy()
        temp['year'] = year
        baseline_coverage = pd.concat([baseline_coverage, temp])
    baseline_coverage = baseline_coverage.reset_index().set_index(['location_id', 'year']).sort_index()

    counterfactual_coverage = pd.DataFrame()
    for level in coverage_levels:
        cov = cov_a.copy()
        cov['year'] = years[0]
        for year in years[1:len(years)]:
            temp = cov_b * level
            temp['year'] = year
            cov = pd.concat([cov, temp])
        cov['coverage_level'] = level
        counterfactual_coverage = pd.concat([counterfactual_coverage, cov])
    counterfactual_coverage = (counterfactual_coverage.reset_index()
                               .set_index(['location_id', 'year', 'coverage_level']).sort_index())

    return baseline_coverage, counterfactual_coverage

def generate_rr_deficiency_nofort_draws(mean, std, location_ids):
    """This function takes a distribution for the relative risk
    for lack of fortification of a particular nutrient and generates
    1,000 draws based on a lognormal distribution of uncertainty. 
    The data is the duplicated so that it is the same for each location 
    so that it can be easily used later in the calculations."""
    data = pd.DataFrame()
    np.random.seed(7)
    data['rr'] = np.random.lognormal(mean, std, size=1000)
    draws = []
    for i in list(range(0, 1000)):
        draws.append(f'draw_{i}')
    data['draws'] = draws
    data = pd.DataFrame.pivot_table(data, values='rr', columns='draws').reset_index().drop(columns=['index'])
    df = pd.DataFrame(np.repeat(data.values, len(location_ids), axis=0))
    df.columns = data.columns
    df['location_id'] = location_ids
    df = df.set_index('location_id')
    return df


def make_dot_plots(data, nutrient, measure, coverage_levels, subtitle, output_filename, wra=False):
    """This function takes a dataframe,
    nutrient (as a string),
    and measure (as a string, either: 'rates', 'counts', or 'pifs').
    """

    f, ax = plt.subplots(figsize=(7, 4), dpi=120)
    colors = ['tab:red', 'tab:orange', 'tab:green']

    location_spacer = 0.15
    coverage_spacer = 0.025
    df = data.apply(pd.DataFrame.describe, percentiles=[0.025, 0.975], axis=1).reset_index()

    order = df.reset_index()
    order = list(
        order.loc[order.coverage_level == 0.8].loc[order.year == 2025].sort_values(by='mean').location_id.values)
    nums = list(range(0, len(order)))
    orders = pd.DataFrame()
    orders['location_id'] = order
    orders['order'] = nums
    df = df.merge(orders, on='location_id').sort_values(by='order', ascending=False)

    for n in list(range(0, len(coverage_levels))):
        rate = (df.loc[df.year == 2025]
            .loc[df.coverage_level == coverage_levels[n]])
        for i in list(range(0, len(rate))):
            plt.plot([location_spacer * i + coverage_spacer * n, location_spacer * i + coverage_spacer * n],
                     [rate['2.5%'].values[i], rate['97.5%'].values[i]], c='black')
            plt.scatter([location_spacer * i + coverage_spacer * n], rate['2.5%'].values[i], s=50, marker='_',
                        c='black')
            plt.scatter([location_spacer * i + coverage_spacer * n], rate['97.5%'].values[i], s=50, marker='_',
                        c='black')

        x_vals = []
        for x in list(range(0, len(rate))):
            x_vals.append(location_spacer * x + coverage_spacer * n)
        plt.scatter(x_vals, rate['mean'], s=50,
                    label=f'{int(coverage_levels[n] * 100)} percent coverage', color=colors[n])

    plt.hlines(0, 0 - coverage_spacer * 2,
               location_spacer * (len(rate)) - coverage_spacer * 2,
               linestyle='dashed', color='grey', alpha=0.5)

    plt.plot()

    if wra == True:
        subpop = 'Women of Reproductive Age'
    else:
        subpop = 'Children Under Five'

    if measure == 'rates':
        plt.title(f'DALYs Averted per 100,000 Person-Years due to\n{nutrient} Fortication Among {subpop}\n{subtitle}')
        plt.ylabel('DALYs Averted per 100,000')
    elif measure == 'counts':
        plt.title(f'DALYs Averted due to\n{nutrient} Fortication Among {subpop}\n{subtitle}')
        plt.ylabel('DALYs')
    elif measure == 'pifs':
        plt.title(f'Population Impact Fraction of {nutrient} Fortication\non DALYs Among {subpop}\n{subtitle}')
        plt.ylabel('Population Impact Fraction (Percent)')

    plt.legend(bbox_to_anchor=[1.5, 1])

    x_ticks = []
    for x in list(range(0, len(rate))):
        x_ticks.append(location_spacer * x + coverage_spacer)
    ax.set_xticks(x_ticks)
    plt.xticks(rotation=90)
    l = get_ids('location')
    l_names = df.loc[df.coverage_level == coverage_levels[0]].loc[df.year == 2025]
    l_names = list(l_names.reset_index().merge(l, on='location_id')['location_name'].values)
    ax.set_xticklabels(l_names)
    plt.savefig(f'results_plots/{output_filename}.png', bbox_inches='tight')