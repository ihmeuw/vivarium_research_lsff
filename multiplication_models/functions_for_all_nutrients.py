import pandas as pd, numpy as np
from db_queries import get_ids, get_outputs, get_location_metadata, get_population, get_covariate_estimates
from get_draws.api import get_draws
import scipy.stats
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from matplotlib.backends.backend_pdf import PdfPages

def generate_coverage_parameter_draws(df, random_seed, n_draws):
    data_frame = df.copy()
    np.random.seed(random_seed)
    for i in list(range(0, n_draws)):
        data_frame[f'draw_{i}'] = scipy.stats.truncnorm.rvs(data_frame.a,
                                                            data_frame.b,
                                                            data_frame.value_mean,
                                                            data_frame.value_std) / 100
    data_frame = (data_frame
                  .set_index(['location_id'])
                  .drop(columns=[c for c in data_frame.columns if 'draw' not in c
                                 and c not in ['location_id', 'value_description']]))
    return data_frame


def generate_logical_coverage_draws(coverage_data_dir, location_ids, nutrient, vehicle):
    data = pd.read_csv(coverage_data_dir).sort_values(by='location_id').drop_duplicates().drop(209)
    print('WARNING: dropped observation 209 of input data. Fix this once Vietnam data has been updated')
    data = data.loc[data.location_id.isin(location_ids)].loc[data.sub_population != 'women of reproductive age']
    print('WARNING: excluded all women of reproductive age observations. Fix this once WRA/U5 data update has been made')

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

    cov_a_draws = generate_coverage_parameter_draws(cov_a, 11, 1_000)
    cov_b_draws = generate_coverage_parameter_draws(cov_b, 11, 1_000)

    # check to see if any draws are illogically ordered
    test = (cov_a_draws.stack().reset_index().rename(columns={'level_1': 'draw', 0: 'cov_a'})
            .merge(cov_b_draws.stack().reset_index().rename(columns={'level_1': 'draw', 0: 'cov_b'}),
                   on=['location_id', 'draw']))
    test = test.loc[test.cov_a > test.cov_b]
    issue_locs = list(test.location_id.unique())
    # if no logically ordered draws are possible, exclude from analysis
    excepts = (cov_a.set_index('location_id')['value_025_percentile'] > cov_b.set_index('location_id')[
        'value_975_percentile'])
    excepts = list(excepts.loc[excepts == True].reset_index().location_id.unique())
    locs = [loc for loc in issue_locs if loc not in excepts]
    if len(excepts) > 0:
        print(f'Excluded {excepts}/{nutrient}/{vehicle} due to impossible logical values')

    # for locations/nutrients/vehicles with overlapping parameter distributions, run the following
    # to select logical draws only
    if len(locs) > 0:
        reruns = pd.DataFrame()
        for loc in locs:
            cov_a_sub = generate_coverage_parameter_draws(cov_a.loc[cov_a.location_id == loc], 234, 5_000)
            cov_b_sub = generate_coverage_parameter_draws(cov_b.loc[cov_b.location_id == loc], 341, 5_000)
            check = (cov_a_sub.stack().reset_index().rename(columns={'level_1': 'draw', 0: 'cov_a'})
                     .merge(cov_b_sub.stack().reset_index().rename(columns={'level_1': 'draw', 0: 'cov_b'}),
                            on=['location_id', 'draw']))
            check['logical'] = np.where(check.cov_a < check.cov_b, 1, 0)
            check = check.loc[check.logical == 1]
            while len(check) < 1000:
                cov_a_sub = generate_coverage_parameter_draws(cov_a.loc[cov_a.location_id == loc], 565, 1_000)
                cov_b_sub = generate_coverage_parameter_draws(cov_b.loc[cov_b.location_id == loc], 333, 1_000)
                check_sub = (cov_a_sub.stack().reset_index().rename(columns={'level_1': 'draw', 0: 'cov_a'})
                             .merge(cov_b_sub.stack().reset_index().rename(columns={'level_1': 'draw', 0: 'cov_b'}),
                                    on=['location_id', 'draw']))
                check_sub['logical'] = np.where(check_sub.cov_a < check_sub.cov_b, 1, 0)
                check_sub = check_sub.loc[check_sub.logical == 1]
                check = pd.concat([check, check_sub])

            out = check[0:1_000].drop(columns=['draw', 'logical']).set_index('location_id')
            draws = []
            for i in list(range(0, 1000)):
                draws.append(f'draw_{i}')
            out['draw'] = draws
            reruns = reruns.append(out)

        reruns_a = pd.pivot_table(reruns[['cov_a', 'draw']].reset_index(), index='location_id',
                                  columns='draw', values='cov_a').reset_index()
        reruns_b = pd.pivot_table(reruns[['cov_b', 'draw']].reset_index(), index='location_id',
                                  columns='draw', values='cov_b').reset_index()

        cov_a_draws_sub = (cov_a_draws.reset_index()
            .loc[cov_a_draws.reset_index()
            .location_id
            .isin([loc for loc in location_ids if loc not in issue_locs])])
        cov_b_draws_sub = (cov_b_draws.reset_index()
            .loc[cov_b_draws.reset_index()
            .location_id
            .isin([loc for loc in location_ids if loc not in issue_locs])])

        cov_a_final = pd.concat([cov_a_draws_sub, reruns_a], ignore_index=True, sort=True).set_index('location_id')
        cov_b_final = pd.concat([cov_b_draws_sub, reruns_b], ignore_index=True, sort=True).set_index('location_id')
    else:
        cov_a_final = (cov_a_draws.reset_index()
            .loc[cov_a_draws.reset_index()
            .location_id
            .isin([loc for loc in location_ids if loc not in excepts])]).set_index('location_id')
        cov_b_final = (cov_b_draws.reset_index()
            .loc[cov_b_draws.reset_index()
            .location_id
            .isin([loc for loc in location_ids if loc not in excepts])]).set_index('location_id')

    assert np.all(cov_a_final <= cov_b_final), "Illogically ordered"

    return cov_a_final, cov_b_final


def generate_coverage_dfs(cov_a, cov_b, years, coverage_levels):
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
            temp = cov_a + (cov_b - cov_a) * level
            temp['year'] = year
            cov = pd.concat([cov, temp])
        cov['coverage_level'] = level
        counterfactual_coverage = pd.concat([counterfactual_coverage, cov])
    counterfactual_coverage = (counterfactual_coverage.reset_index()
                               .set_index(['location_id', 'year', 'coverage_level']).sort_index())

    return baseline_coverage, counterfactual_coverage

def get_baseline_and_counterfactual_coverage(coverage_data_dir,
                                             location_ids,
                                             nutrient,
                                             vehicles,
                                             years,
                                             coverage_levels):
    baseline_coverage_final = pd.DataFrame()
    counterfactual_coverage_final = pd.DataFrame()
    for vehicle in vehicles:
        cov_a, cov_b = generate_logical_coverage_draws(coverage_data_dir, location_ids, nutrient, vehicle)
        baseline_coverage, counterfactual_coverage = generate_coverage_dfs(cov_a, cov_b, years, coverage_levels)
        baseline_coverage['vehicle'] = vehicle
        counterfactual_coverage['vehicle'] = vehicle
        baseline_coverage_final = pd.concat([baseline_coverage_final, baseline_coverage.reset_index()],
                                            ignore_index=True, sort=True)
        counterfactual_coverage_final = pd.concat([counterfactual_coverage_final, counterfactual_coverage.reset_index()],
                                            ignore_index=True, sort=True)

    return baseline_coverage_final, counterfactual_coverage_final

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
    df = data.drop(columns='measure', errors='ignore').apply(pd.DataFrame.describe, percentiles=[0.025, 0.975], axis=1).reset_index()

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
    #plt.savefig(f'results_plots/{output_filename}.png', bbox_inches='tight')
