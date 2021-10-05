import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# import lbwsg
# import test_lbwsg

# cat_df = lbwsg.get_category_data()

def draw_lbwsg_categories(cat_df):
    fig, ax = plt.subplots(figsize=(14,12))

    def draw_category_rectangle(row):
        rectangle = Rectangle(
            (row.bw_start, row.ga_start),
            row.bw_width, row.ga_width,
    #         label=row.lbwsg_category,
            color='tab:blue',
            fill=False
        )
        ax.add_patch(rectangle)

    def add_category_label(row, xy=None):
        x, y = (row.bw_midpoint, row.ga_midpoint) if xy is None else xy
        ax.text(
            x, y, row.lbwsg_category,
            horizontalalignment='center',
            verticalalignment='center'
        )

    cat_df.apply(draw_category_rectangle, axis=1)
    cat_df.apply(add_category_label, axis=1)

    for cat in ['cat2', 'cat8']:
        row = cat_df.loc[cat_df.lbwsg_category==cat,:].squeeze()
        add_category_label(row, xy=(row.bw_midpoint, row.ga_end-2))

    ax.set_xlabel('Birthweight')
    ax.set_xlim(0,4500)
    ax.set_xticks(range(0,5000,500))
    ax.set_ylabel('Gestational age')
    ax.set_ylim(20,42)
    ax.set_yticks(range(20,44,2))
    plt.show()

def draw_lbwsg_categories2(cat_df):
    fig, ax = plt.subplots(figsize=(18,8))

    def draw_rectangle(row):
        rectangle = Rectangle(
            (row.ga_start, row.bw_start),
            row.ga_width, row.bw_width,
    #         label=row.lbwsg_category,
            color='tab:blue',
            fill=False
        )
        ax.add_patch(rectangle)
        ax.text(
            row.ga_midpoint, row.bw_midpoint, row.lbwsg_category,
            horizontalalignment='center',
            verticalalignment='center'
        )

    cat_df.apply(draw_rectangle, axis=1)

    ax.set_xlabel('Gestational age')
    ax.set_xlim(0,42)
    ax.set_xticks(range(0,42,2))
    ax.set_ylabel('Birthweight')
    ax.set_ylim(0,4500)
    ax.set_yticks(range(0,4500,500))

    plt.show()

def plot_log_rrs(
    ax,
    gai,
    bwi,
    logrri,
    cat_df=None,
    title="",
    x_is_ga=True,
    logrri_xy_matches_axes=True,
    draw_category_midpoints=True,
    draw_grid_midpoints=False,
    draw_grid_boundary_points=False,
    draw_category_rectangles=False,
    grid_color='tab:blue',
    rectangle_boundary_color='tab:blue',
    contour_levels=15, # default if 'levels' not specified in contour_kwargs or contourf_kwargs
    contour_linewidths=0.5, # default if 'linewidths' not specified in contour_kwargs
    contour_colors='k', # default if 'colors' not specified in contour_kwargs
    contourf_cmap='RdBu_r', # default if neither 'colors' nor 'cmap' is specified in contour_kwargs
    contour_kwargs=None,
    contourf_kwargs=None,
):
    """Make a contour plot of interpolated log RR's for LBWSG."""
    
    def draw_category_rectangle(row, x_prefix, y_prefix, boundary_color):
        rectangle = Rectangle(
            (row[f"{x_prefix}_start"], row[f"{y_prefix}_start"]),
            row[f"{x_prefix}_width"], row[f"{y_prefix}_width"],
            color=boundary_color,
            fill=False
        )
        ax.add_patch(rectangle)

#     if contour_levels is None:
#         contour_levels = 15
    if contour_kwargs is None:
        contour_kwargs = {}
    if contourf_kwargs is None:
        contourf_kwargs = {}
    if 'levels' not in contour_kwargs:
        contour_kwargs['levels'] = contour_levels
    if 'linewidths' not in contour_kwargs:
        contour_kwargs['linewidths'] = contour_linewidths
    if 'colors' not in contour_kwargs:
        contour_kwargs['colors'] = contour_colors
    if 'levels' not in contourf_kwargs:
        contourf_kwargs['levels'] = contour_levels
    if 'colors' not in contourf_kwargs and 'cmap' not in contourf_kwargs:
        contourf_kwargs['cmap'] = contourf_cmap

#     fig, ax = plt.subplots(figsize=(10,8))
    
    ga_params = ['Gestational age (weeks)', (0,42), range(0,42,2), gai, 'ga']
    bw_params = ['Birthweight (g)', (0,4500), range(0,4500,500), bwi, 'bw']
    
    xy_params = zip(ga_params, bw_params) if x_is_ga else zip(bw_params, ga_params)
    
    (xlabel, ylabel), (xlim, ylim), (xticks, yticks), (xi, yi), (x_prefix, y_prefix) = xy_params
    
    if not logrri_xy_matches_axes: 
        logrri = logrri.T
    if cat_df is not None:
        x_mid, y_mid = cat_df[f"{x_prefix}_midpoint"], cat_df[f"{y_prefix}_midpoint"]
        x_min, y_min = cat_df[[f'{x_prefix}_start', f'{y_prefix}_start']].min()
        x_max, y_max = cat_df[[f'{x_prefix}_end', f'{y_prefix}_end']].max()

        x_unique = np.append(np.unique(x_mid), [x_min, x_max]); x_unique.sort()
        y_unique = np.append(np.unique(y_mid), [y_min, y_max]); y_unique.sort()

#         grid_color = 'tab:blue'
#         rectangle_boundary_color = 'tab:blue'

        if draw_grid_midpoints:
            x_grid, y_grid = np.meshgrid(sorted(x_mid.unique()), sorted(y_mid.unique()))
            ax.plot(x_grid.flatten(), y_grid.flatten(), 'o', color='none', markeredgecolor=grid_color, mew=1)
        if draw_grid_boundary_points:
            ax.plot(x_min, y_unique[None,:], 'o', color='none', mec=grid_color, mew=1)
            ax.plot(x_max, y_unique[None,:], 'o', color='none', mec=grid_color, mew=1)
            ax.plot(x_unique[None,:], y_min, 'o', color='none', mec=grid_color, mew=1)
            ax.plot(x_unique[None,:], y_max, 'o', color='none', mec=grid_color, mew=1)
        if draw_category_midpoints:
            ax.plot(x_mid, y_mid, 'o', color=grid_color)
        if draw_category_rectangles:
            cat_df.apply(draw_category_rectangle, args=(x_prefix, y_prefix, rectangle_boundary_color), axis=1)

    ax.contour(xi, yi, logrri, **contour_kwargs)
    cntr = ax.contourf(xi, yi, logrri, **contourf_kwargs)
#     fig.colorbar(cntr, ax=ax, label='log(RR)')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    
    ax.set_title(title)
    return cntr
#     return fig, ax

def single_log_rr_plot(
    gai,
    bwi,
    logrri,
    cat_df=None,
    title="",
    x_is_ga=True,
    logrri_xy_matches_axes=True,
    draw_category_midpoints=True,
    draw_grid_midpoints=False,
    draw_grid_boundary_points=False,
    draw_category_rectangles=False,
):
    fig, ax = plt.subplots(figsize=(10,8))

    cntr = plot_log_rrs(
    ax=ax,
    gai=gai,
    bwi=bwi,
    logrri=logrri,
    cat_df=cat_df,
    title=title,
    x_is_ga=x_is_ga,
    logrri_xy_matches_axes=logrri_xy_matches_axes,
    draw_category_midpoints=draw_category_midpoints,
    draw_grid_midpoints=draw_grid_midpoints,
    draw_grid_boundary_points=draw_grid_boundary_points,
    draw_category_rectangles=draw_category_rectangles,
    )
    fig.colorbar(cntr, ax=ax, label='log(RR)')
    return fig, ax

def plot_log_rrs_by_age_sex(
    gai,
    bwi,
    logrri_by_age_sex,
    cat_df=None,
    suptitle="",
    x_is_ga=True,
    logrri_xy_matches_axes=True,
    draw_category_midpoints=True,
    draw_grid_midpoints=False,
    draw_grid_boundary_points=False,
    draw_category_rectangles=False,
):
    # Using constrained_layout=True instead of fig.tight_layout() because of colorbar
    fig, axs = plt.subplots(2,2, figsize=(16,14), constrained_layout=True)

    # Set same vmax for contourf function in each axes to use same scale on all plots
    # vmax = max(logrri.max() for logrri in logrri_by_age_sex.values()) # if logrri_by_age_sex is a dict
    vmax = logrri_by_age_sex.map(np.max).max() # if logrri_by_age_sex is a Series

    age_ids_to_names = {2: 'Early Neonatal', 3: 'Late Neonatal'}
    sex_ids_to_names = {1: 'Male', 2: 'Female'}

    cntrs = []
    for age in 2,3:
        for sex in 1,2:
            ax = axs[age-2,sex-1] # Top row is ENN, bottom row is LNN; 1st col is Male, 2nd col is Female
            cntr = plot_log_rrs(
                ax=ax,
                gai=gai,
                bwi=bwi,
                logrri=logrri_by_age_sex[(age,sex)],
                cat_df=cat_df,
                title=f"{age_ids_to_names[age]}, {sex_ids_to_names[sex]}",
                x_is_ga=x_is_ga,
                logrri_xy_matches_axes=logrri_xy_matches_axes,
                draw_category_midpoints=draw_category_midpoints,
                draw_grid_midpoints=draw_grid_midpoints,
                draw_grid_boundary_points=draw_grid_boundary_points,
                draw_category_rectangles=draw_category_rectangles,
                contourf_kwargs = dict(vmin=0, vmax=vmax),
            )
            cntrs += [cntr]
            ax.title.set_fontsize(16)
    # Find the ContourSet object with the maximum level, and use it to draw the colorbar.
    # It seems like this shouldn't be necessary since I passed the same vmin and vmax
    # to all the contourf calls, but the colorbar limits didn't go up to the maximum
    # when I just passed the last used cntr.
    max_cntr = max(cntrs, key=lambda cntr: cntr.levels.max())
    fig.colorbar(max_cntr, ax=axs, label='log(RR)')
    fig.suptitle(suptitle, fontsize=20)
    return fig, axs, cntrs
