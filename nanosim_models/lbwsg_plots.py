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

def plot_log_rrs(gai, bwi, logrri,
                 cat_df=None,
                 interpolation_type="some type of",
                 subtitle=None,
                 x_is_ga=True,
                 logrri_xy_matches_axes=True,
                 draw_category_midpoints=True,
                 draw_all_gridpoints=False,
                 draw_category_rectangles=False,
                ):
    """Make a contour plot of interpolated log RR's for LBWSG."""
    
    def draw_category_rectangle(row, x_prefix, y_prefix):
        rectangle = Rectangle(
            (row[f"{x_prefix}_start"], row[f"{y_prefix}_start"]),
            row[f"{x_prefix}_width"], row[f"{y_prefix}_width"],
            color='tab:blue',
            fill=False
        )
        ax.add_patch(rectangle)

    fig, ax = plt.subplots(figsize=(10,8))
    
    ga_params = ['Gestational age', (0,42), range(0,42,2), gai, 'ga']
    bw_params = ['Birthweight', (0,4500), range(0,4500,500), bwi, 'bw']
    
    xy_params = zip(ga_params, bw_params) if x_is_ga else zip(bw_params, ga_params)
    
    (xlabel, ylabel), (xlim, ylim), (xticks, yticks), (xi, yi), (x_prefix, y_prefix) = xy_params
    
    if not logrri_xy_matches_axes: 
        logrri = logrri.T
    if cat_df is not None:
        x_mid, y_mid = cat_df[f"{x_prefix}_midpoint"], cat_df[f"{y_prefix}_midpoint"]
        if draw_all_gridpoints:
            x_grid, y_grid = np.meshgrid(sorted(x_mid.unique()), sorted(y_mid.unique()))
            ax.plot(x_grid.flatten(), y_grid.flatten(), 'o', color='tab:blue')
        if draw_category_midpoints:
            ax.plot(x_mid, y_mid, 'o', color='tab:green')
        if draw_category_rectangles:
            cat_df.apply(draw_category_rectangle, args=(x_prefix, y_prefix), axis=1)

    ax.contour(xi, yi, logrri, levels=15, linewidths=0.5, colors='k')
    cntr = ax.contourf(xi, yi, logrri, levels=15, cmap="RdBu_r")
    fig.colorbar(cntr, ax=ax, label='log(RR)')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    
    subtitle = f"\n{subtitle}" if subtitle is not None else ""
    ax.set_title(f"Contour plot of {interpolation_type} interpolation of log(RR)" f"{subtitle}")
    return fig, ax