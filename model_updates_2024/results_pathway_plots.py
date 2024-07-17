import numpy as np
import matplotlib.pyplot as plt

from vivarium_helpers.id_helper import ids_to_names

def lower(x):
    return x.quantile(.025)

def upper(x):
    return x.quantile(0.975)

def lsff_pathway_stacked_bar_plot(df, vehicle, age_group, legend=True, ax=None):
    if ax is None:
        ax = plt.gca()
    
    pathways = df.value_counts(['nutrient', 'cause']).sort_index().index.to_flat_index().unique()
    draw_cols = df.filter(like='draw').columns.to_list()
    year = 2025
#     coverage_levels = [0.2, 0.5, 0.8]
    x_variable = 'coverage_level' # x could be country instead, for a fixed coverage level
    measure = 'counts_averted' # y_variable -- needs to be additive for a stacked bar chart
    filters = [
        f"coverage_level != {0.2 if 'zero' in vehicle else 1.0}", # since gap coverage 1.0 is missing from iron->BW results
#         f"coverage_level != 1.0", # since gap coverage 1.0 is missing from iron->BW results
        "location_id != 11", # since location_id 11 (Indonesia) is missing from iron->anemia results
        "vehicle==@vehicle",
        "age_group==@age_group",
        "measure==@measure",
        "year==@year", # since iron only has results for 2025
    ]
#     if 'zero' in vehicle:
#         filters.append('coverage_level == -1.0')
    df = df.query(" and ".join(filters))
    # If there's no data to plot, just exit
    if df.empty:
        return ax
    location_ids = df['location_id'].unique()
    locations = "\n".join(ids_to_names('location', *location_ids))
    pathway_variables = ['nutrient', 'cause']
    grouped = df.groupby([x_variable, *pathway_variables])
    # For some reason, doing .agg(['mean', lower, upper], axis=1) dropped the
    # names of the levels in the index, so I use .T.describe().T instead
    data = grouped[draw_cols].sum().T.describe(percentiles=[.025,.975]).T#.agg(['mean', lower, upper], axis=1)
    assert len(location_id_lists := grouped['location_id'].unique().map(sorted).map(tuple).unique()) == 1, \
        f"Locations don't match between nutrient pathways: {location_id_lists}"

    x_values = data.index.unique(x_variable)
#     x = np.arange(len(x_values))
    # Make values evenly spaced, and put them in consistent locations for multiple axes plotted together
    x = x_values.map({-1: 0, 0.2: 1, 0.5: 2, 0.8: 3, 1.0: 4})
    bottom = np.zeros(len(x_values))
    # Each pathway is a (nutrient, cause) pair -- find unique pathways
    # by ignoring coverage level in index and finding all such pairs
    # pathways = data.index.droplevel('coverage_level').to_flat_index().unique()
    for pathway in pathways:
        try:
            mean = data.xs(pathway, level=pathway_variables)['mean']
        except KeyError as e:
            # If pathway does not exist for the given data, plot an empty bar
            # (of height 0) so that the color cycle counts all pathways,
            # thus keepint the same color for each pathway in all plots
            mean = 0
            # No need to filter x-values since scalar 0 will be broadcast across all
            x_mask = (x == x)
            # print(e)
        else:
            # Filter to coverage levels present for this pathway
            x_mask = np.isin(x_values, mean.index)

#         print(pathway, x, x_values, x[x_mask], mean)
        ax.bar(x[x_mask], mean, bottom=bottom[x_mask], label=fr'{pathway[0]} $\to$ {pathway[1]}')
        bottom[x_mask] += mean
        
    ax.set_xticks(x)
    ax.set_xticklabels(x_values.map(lambda x: 'remove baseline' if x==-1 else x))
    ax.set_xlabel("Fortification 'gap' coverage level")
    ax.set_ylabel('DALYs averted')

    if legend:
        # https://stackoverflow.com/questions/25068384/bbox-to-anchor-and-loc-in-matplotlib
#         ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title=r'nutrient $\to$ DALYs pathway')
        ax.legend(title=r'nutrient $\to$ DALYs pathway')
    location_string = (
        f" ({len(location_ids)} countries)" if len(location_ids) > 1 else
        f" ({ids_to_names('location', *location_ids).iloc[0]})"
    )
    ax.set_title(
        f"DALYs averted by '{vehicle.upper()}' fortification in age group {age_group.upper()}"
        + location_string
#         f"\nin {len(location_ids)} location{'s' if len(location_ids) > 1 else ' (' + ids_to_names('location', *location_ids).iloc[0] + ')'}"
    )
    return ax
    
def plot_all_vehicles_and_age_groups(df):
    vehicles = [ 'oil', 'wheat flour', 'maize flour']
#     grid = np.array((
#         (('oil', 'u5'), ('industry oil', 'u5')),
#         (('salt', 'u5'), ('industry salt', 'u5')), # Note: industry salt doesn't exist
#         (('wheat flour', 'u5'), ('industry wheat', 'u5')),
#         (('wheat flour', 'wra'), ('industry wheat', 'wra')),
#         (('maize flour', 'u5'), ('maize flour', 'wra')), # This row breaks pattern because there's no 'industry maize'
#     ))
    rows = (
        ('oil', 'u5'),
        ('salt', 'u5'), # Note: industry salt doesn't exist
        ('wheat flour', 'u5'),
        ('wheat flour', 'wra'),
        ('maize flour', 'u5'), # This row breaks pattern because there's no 'industry maize'
#         ('zero maize flour', 'u5')
    )
#     [
#         ['oil', 'industry oil'],
#         ['wheat flour', 'industry wheat'],
#         ['maize flour', '']
#     ]
    age_groups = ['u5', 'wra']
    
    n_rows = len(rows)
    n_cols = 2 # 1st column for regular, 2nd column for 'industry' version
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows), sharex=True, sharey=True)
    for row_num, (vehicle, age_group) in enumerate(rows):
        if vehicle in ['maize flour', 'zero maize flour']:
            vehicle2 = vehicle
            age_group2 = 'wra'
        else:
            vehicle2 = f"industry {vehicle[:-6] if vehicle.endswith(' flour') else vehicle}"
            age_group2 = age_group

        lsff_pathway_stacked_bar_plot(df, vehicle, age_group, legend=True, ax=axs[row_num, 0])
        lsff_pathway_stacked_bar_plot(df, vehicle2, age_group2, legend=True, ax=axs[row_num, 1])
#     fig.suptitle(r"DALYs averted by nutrient $\to$ cause pathway for each vehicle")
    fig.tight_layout()
    return fig
