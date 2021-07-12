import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import lbwsg
import test_lbwsg

cat_df = lbwsg.get_category_data()

def draw_lbwsg_categories():
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

def draw_lbwsg_categories2():
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
    interpolation_type="some type of",
    subtitle=None,
    x_is_ga=True,
    logrri_xy_matches_axes=True,
    draw_category_midpoints=True,
    draw_grid_midpoints=False,
    draw_grid_boundary_points=False,
    draw_category_rectangles=False,
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

#     fig, ax = plt.subplots(figsize=(10,8))
    
    ga_params = ['Gestational age', (0,42), range(0,42,2), gai, 'ga']
    bw_params = ['Birthweight', (0,4500), range(0,4500,500), bwi, 'bw']
    
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

        grid_color = 'tab:blue'
        rectangle_boundary_color = 'tab:blue'

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

    ax.contour(xi, yi, logrri, levels=15, linewidths=0.5, colors='k')
    cntr = ax.contourf(xi, yi, logrri, levels=15, cmap="RdBu_r")
#     fig.colorbar(cntr, ax=ax, label='log(RR)')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    
    subtitle = f"\n{subtitle}" if subtitle is not None else ""
    ax.set_title(f"Contour plot of {interpolation_type} interpolation of log(RR)" f"{subtitle}")
    return cntr
#     return fig, ax

def single_log_rr_plot(
    gai,
    bwi,
    logrri,
    cat_df=None,
    interpolation_type="some type of",
    subtitle=None,
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
    interpolation_type=interpolation_type,
    subtitle=subtitle,
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
    interpolation_type="some type of",
    subtitle=None,
    x_is_ga=True,
    logrri_xy_matches_axes=True,
    draw_category_midpoints=True,
    draw_grid_midpoints=False,
    draw_grid_boundary_points=False,
    draw_category_rectangles=False,
):
    fig, axs = plt.subplots(2,2, figsize=(16,14), constrained_layout=True)

    for age in (2,3):
        for sex in (1,2):
            ax = axs[age-2,sex-1]
            cntr = plot_log_rrs(
            ax=ax,
            gai=gai,
            bwi=bwi,
            logrri=logrri_by_age_sex[(age,sex)],
            cat_df=cat_df,
            interpolation_type=interpolation_type,
            subtitle=subtitle,
            x_is_ga=x_is_ga,
            logrri_xy_matches_axes=logrri_xy_matches_axes,
            draw_category_midpoints=draw_category_midpoints,
            draw_grid_midpoints=draw_grid_midpoints,
            draw_grid_boundary_points=draw_grid_boundary_points,
            draw_category_rectangles=draw_category_rectangles,
            )
    fig.colorbar(cntr, ax=axs, label='log(RR)')
#     fig.tight_layout()
    return fig, ax
