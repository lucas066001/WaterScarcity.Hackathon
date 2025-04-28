"""
Water Management Analysis Module

This module provides visualization and analysis functions for water allocation
simulation results. It includes tools for analyzing the relationships between
ecological impacts, economic impacts, and various simulation parameters.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
from typing import Dict, List, Optional, Tuple


# Set consistent styling for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")


# =====================================================================
# Common Utilities and Color Schemes
# =====================================================================

# Color schemes for different plot elements
COLOR_SCHEMES = {
    'scarcity': {
        "low": "#2E8B57",     # Sea Green - Calm, good conditions
        "medium": "#FFD700",  # Gold - Caution
        "high": "#B22222"     # Firebrick - Danger, severe conditions
    },
    'station': {
        1: "#4682B4",  # Steel Blue - Small river basin
        2: "#8A2BE2"   # Blue Violet - Large river basin
    },
    'scenario': {
        "0.yml": {"color": "#1E90FF", "marker": "o", "name": "Base 0"},           # Dodger Blue
        "1.yml": {"color": "#FF6347", "marker": "o", "name": "Base 1"},           # Tomato
        "0-v.yml": {"color": "#1E90FF", "marker": "s", "name": "Variant 0-v"},    # Dodger Blue, square
        "1-v.yml": {"color": "#FF6347", "marker": "s", "name": "Variant 1-v"},    # Tomato, square
        "0-b.yml": {"color": "#1E90FF", "marker": "^", "name": "Variant 0-b"},    # Dodger Blue, triangle
        "1-b.yml": {"color": "#FF6347", "marker": "^", "name": "Variant 1-b"},    # Tomato, triangle
        "0-c.yml": {"color": "#1E90FF", "marker": "D", "name": "Variant 0-c"},    # Dodger Blue, diamond
        "1-c.yml": {"color": "#FF6347", "marker": "D", "name": "Variant 1-c"}     # Tomato, diamond
    },
    'correlation': 'coolwarm',
    'boxplot': 'GnBu',
    'violin': {
        "low": "#2E8B57",     # Sea Green
        "medium": "#FFD700",  # Gold
        "high": "#B22222"     # Firebrick
    }
}


def create_styled_legend(ax, handles, labels, title, position=None, ncol=1):
    """
    Create a consistently styled legend.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to add the legend to
    handles : list
        Legend handles
    labels : list
        Legend labels
    title : str
        Legend title
    position : tuple, optional
        Position to place legend (default: top-right outside plot)
    ncol : int
        Number of columns in the legend
        
    Returns:
    --------
    legend : matplotlib.legend.Legend
        The created legend object
    """
    if position is None:
        legend = ax.legend(
            handles=handles,
            labels=labels,
            title=title,
            fontsize=10,
            title_fontsize=11,
            frameon=True,
            facecolor='white',
            edgecolor='gray',
            framealpha=0.95,
            loc='best',
            ncol=ncol
        )
    else:
        legend = ax.legend(
            handles=handles,
            labels=labels,
            title=title,
            fontsize=10,
            title_fontsize=11,
            frameon=True,
            facecolor='white',
            edgecolor='gray',
            framealpha=0.95,
            bbox_to_anchor=position,
            ncol=ncol
        )
    legend.get_frame().set_linewidth(0.5)
    return legend


def add_impact_plot_elements(ax, results_df=None):
    """
    Add common elements to impact plots like diagonal line and best/worst regions.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to annotate
    results_df : pandas.DataFrame, optional
        Results data for calculating min/max values
        
    Returns:
    --------
    None
    """
    # Get current axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Draw diagonal reference line using current limits
    # Inverse relationship: lower ecological impact, higher economic impact
    ax.plot([xlim[0], xlim[1]], [ylim[1], ylim[0]], 'k--', alpha=0.3, linewidth=1)
    
    # Calculate ranges
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    
    # Add shaded rectangles for best/worst regions
    if results_df is not None:
        # Best region: low ecological impact, high economic impact
        x_best = np.linspace(xlim[0], xlim[0] + 0.33 * x_range, 50)
        y_best_bottom = np.ones_like(x_best) * (ylim[1] - 0.33 * y_range)
        y_best_top = np.ones_like(x_best) * ylim[1]
        ax.fill_between(x_best, y_best_top, y_best_bottom, color='#E5F5E0', alpha=0.2)
        
        # Worst region: high ecological impact, low economic impact
        x_worst = np.linspace(xlim[0] + 0.67 * x_range, xlim[1], 50)
        y_worst_bottom = np.ones_like(x_worst) * ylim[0]
        y_worst_top = np.ones_like(x_worst) * (ylim[0] + 0.33 * y_range)
        ax.fill_between(x_worst, y_worst_top, y_worst_bottom, color='#FEE5D9', alpha=0.2)
        
        # Add labels for best/worst regions
        ax.text(
            xlim[0] + 0.10 * x_range, 
            ylim[1] - 0.10 * y_range, 
            "Best", 
            fontsize=10, 
            ha='center', 
            va='center',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='green', boxstyle='round,pad=0.3')
        )
        ax.text(
            xlim[1] - 0.10 * x_range, 
            ylim[0] + 0.10 * y_range, 
            "Worst", 
            fontsize=10, 
            ha='center', 
            va='center',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='red', boxstyle='round,pad=0.3')
        )
    
    # Improve grid appearance
    ax.grid(True, alpha=0.2, linestyle='-')
    
    # Enhance boundary lines
    for spine in ax.spines.values():
        spine.set_edgecolor('gray')
        spine.set_linewidth(0.5)
    
    # Common axis labels for impact plots
    ax.set_xlabel('Ecological Impact (lower is better)', fontsize=12)
    ax.set_ylabel('Economic Impact (higher is better)', fontsize=12)


# =====================================================================
# Primary Analysis Functions
# =====================================================================

def analyze_scenario_impacts(results_df):
    """
    Analyze impacts across different scenarios, stations, and scarcity levels
    with improved visualization aesthetics.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame containing simulation results with ecological_impact, economic_impact,
        and scenario metadata
        
    Returns:
    --------
    None
        Displays visualizations showing impact relationships
    """
    # 1. Analysis by scarcity level
    create_impact_by_category_plot(
        results_df, 
        category='scarcity',
        color_map=COLOR_SCHEMES['scarcity'],
        title='Trade-off Between Ecological and Economic Impacts by Scarcity Level',
        point_size=60
    )
    
    # 2. Analysis by station (river basin)
    create_impact_by_category_plot(
        results_df, 
        category='station',
        color_map=COLOR_SCHEMES['station'],
        label_format=lambda s: f'Station {s}: {"Small" if s==1 else "Large"} Basin',
        title='Impact by River Basin Size',
        point_size=60
    )
    
    # 3. Analysis by scenario
    create_impact_by_scenario_plot(results_df)
    
    # 4. Combined multi-dimensional analysis
    create_multidimensional_impact_plot(results_df)


def analyze_forecast_effects(results_df):
    """
    Analyze the effects of forecast bias and uncertainty on simulation outcomes.
    Creates matrix of plots showing relationships between bias, uncertainty, 
    and impact measures.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame containing simulation results
        
    Returns:
    --------
    None
        Displays visualizations showing forecast effects
    """
    # Extract base scenarios (without variants) for clearer analysis
    base_df = results_df[results_df['scenario'].isin(['0.yml', '1.yml'])]
    
    # Create forecast analysis plots (2x2 grid)
    fig, axs = plt.subplots(2, 2, figsize=(18, 14))
    
    # Define variables and labels for each subplot
    plot_configs = [
        {'x': 'bias', 'y': 'economic_impact', 'size_var': 'uncertainty', 'abs_size': False, 'ax': axs[0, 0]},
        {'x': 'bias', 'y': 'ecological_impact', 'size_var': 'uncertainty', 'abs_size': False, 'ax': axs[0, 1]},
        {'x': 'uncertainty', 'y': 'economic_impact', 'size_var': 'bias', 'abs_size': True, 'ax': axs[1, 0]},
        {'x': 'uncertainty', 'y': 'ecological_impact', 'size_var': 'bias', 'abs_size': True, 'ax': axs[1, 1]}
    ]
    
    # Create each subplot
    for config in plot_configs:
        create_forecast_effect_plot(base_df, **config)
    
    # Create a common legend for all subplots instead of individual legends
    scarcity_handles = []
    for scarcity, color in COLOR_SCHEMES['scarcity'].items():
        handle = Line2D([0], [0], marker='o', color=color, markersize=10, 
                        linestyle='None', label=f'Scarcity: {scarcity}')
        scarcity_handles.append(handle)
    
    # Create size legend for uncertainty
    size_handles = []
    sizes = [0.1, 0.3, 0.5]
    size_scale = (50, 200)
    scale_range = size_scale[1] - size_scale[0]
    for size in sizes:
        marker_size = size_scale[0] + scale_range * (size / 0.5)
        handle = Line2D([0], [0], marker='o', color='gray', markersize=np.sqrt(marker_size/3), 
                        linestyle='None', label=f'{size:.1f}')
        size_handles.append(handle)
    
    # Add the legends to the figure, not to any specific axes
    fig.legend(
        handles=scarcity_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        fontsize=12,
        frameon=True,
        facecolor='white',
        edgecolor='gray',
        title='Scarcity Level'
    )
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Make room for the legend at the bottom
    plt.show()
    
    # Create visualization of impact distributions using violin plots
    fig, axs = plt.subplots(1, 2, figsize=(18, 8))
    
    # Create enhanced violin plots for better distribution visualization
    for i, impact in enumerate(['ecological_impact', 'economic_impact']):
        # Create violin plot
        sns.violinplot(
            data=results_df,
            x='scarcity',
            y=impact,
            palette=COLOR_SCHEMES['violin'],
            ax=axs[i],
            inner='quartile',  # Show quartiles inside violin
            density_norm='width',     # Scale violins to have same width
        )
        
        # Add individual data points using stripplot
        sns.stripplot(
            data=results_df,
            x='scarcity',
            y=impact,
            ax=axs[i],
            size=4,
            alpha=0.6,
            jitter=True,
            color='black',
            zorder=1  # Ensure points are above violin plot
        )
        
        # Improve appearance
        axs[i].set_title(f'{impact.replace("_", " ").title()} Distribution by Scarcity Level', fontsize=15)
        axs[i].set_xlabel('Scarcity Level', fontsize=12)
        axs[i].set_ylabel(impact.replace("_", " ").title(), fontsize=12)
        
        # Add mean markers
        mean_values = results_df.groupby('scarcity')[impact].mean()
        for j, scarcity in enumerate(sorted(results_df['scarcity'].unique())):
            axs[i].plot(j, mean_values[scarcity], 'o', color='white', markersize=8, zorder=3)
            axs[i].plot(j, mean_values[scarcity], 'x', color='black', markersize=6, zorder=4)
    
    plt.tight_layout()
    plt.show()


def correlation_analysis(results_df):
    """
    Perform correlation analysis on simulation results and visualize
    relationships between factors and outcomes.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame containing simulation results
        
    Returns:
    --------
    None
        Displays correlation heatmaps
    """
    # Prepare data for correlation analysis
    # Select numeric columns for correlation
    numeric_df = results_df[['ecological_impact', 'economic_impact', 'bias', 'uncertainty']]
    
    # Add dummy variables for categorical features
    cat_df = pd.get_dummies(results_df[['scarcity', 'scenario', 'station']])
    
    # Combine for correlation analysis
    analysis_df = pd.concat([numeric_df, cat_df], axis=1)
    
    # Compute correlation matrix
    corr_matrix = analysis_df.corr()
    
    # Plot full correlation heatmap
    plt.figure(figsize=(14, 12))
    
    # Create mask for upper triangle to declutter the plot
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Create improved heatmap
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap=COLOR_SCHEMES['correlation'], 
        center=0,
        fmt='.2f',
        linewidths=0.5,
        mask=mask,  # Only show lower triangle
        annot_kws={"size": 8}
    )
    plt.title('Correlation Matrix of Simulation Factors and Impacts', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # Focus on key impact correlations - more targeted analysis
    # Extract only columns that have significant correlation with impacts
    impact_columns = ['ecological_impact', 'economic_impact']
    
    # Filter for columns with meaningful correlations
    threshold = 0.1
    corr_with_impacts = corr_matrix[impact_columns].abs().max(axis=1) >= threshold
    filtered_corr = corr_matrix.loc[corr_with_impacts, impact_columns]
    
    # Sort by magnitude of ecological impact correlation
    filtered_corr = filtered_corr.sort_values(by='ecological_impact', key=abs, ascending=False)
    
    # Create focused heatmap
    plt.figure(figsize=(10, 12))
    sns.heatmap(
        filtered_corr,
        annot=True,
        cmap=COLOR_SCHEMES['correlation'],
        center=0,
        fmt='.2f',
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    plt.title('Key Factors Correlated with Impact Metrics', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Additional scatter plot matrix for key numeric variables
    plt.figure(figsize=(12, 10))
    key_vars = ['ecological_impact', 'economic_impact', 'bias', 'uncertainty']
    g = sns.pairplot(
        results_df,
        vars=key_vars,
        hue='scarcity',
        palette=COLOR_SCHEMES['scarcity'],
        plot_kws={'alpha': 0.6, 's': 50, 'edgecolor': 'w', 'linewidth': 0.5},
        diag_kind='kde',
        corner=True
    )
    g.fig.suptitle('Relationships Between Key Variables', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()


# =====================================================================
# Helper Functions for Plot Creation
# =====================================================================

def create_impact_by_category_plot(results_df, category, color_map, 
                                 title, point_size=60,
                                 label_format=None):
    """
    Create a plot showing ecological vs economic impact by a categorical variable.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame containing simulation results
    category : str
        Category column to group by (e.g., 'scarcity', 'station')
    color_map : dict
        Mapping of category values to colors
    title : str
        Plot title
    point_size : int
        Size of scatter points
    label_format : callable, optional
        Function to format category value for label
        
    Returns:
    --------
    None
        Displays the created plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for value, group in results_df.groupby(category):
        # Format label if function provided
        if label_format:
            label = label_format(value)
        else:
            label = f'{category.title()}: {value}'
            
        # Plot group
        ax.scatter(
            group['ecological_impact'], 
            group['economic_impact'], 
            c=color_map[value],
            label=label,
            alpha=0.75,
            s=point_size,
            edgecolor='white',
            linewidth=0.5
        )
    
    # Add common elements
    add_impact_plot_elements(ax, results_df)
    ax.set_title(title, fontsize=14)
    
    # Add legend
    create_styled_legend(
        ax, 
        handles=ax.get_legend_handles_labels()[0],
        labels=ax.get_legend_handles_labels()[1],
        title=category.title()
    )
    
    plt.tight_layout()
    plt.show()


def create_impact_by_scenario_plot(results_df):
    """
    Create a plot showing ecological vs economic impact by scenario.
    Uses different markers and colors for better differentiation.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame containing simulation results
        
    Returns:
    --------
    None
        Displays the created plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get scenario settings
    scenario_settings = COLOR_SCHEMES['scenario']
    
    # Plot each scenario
    for scenario in results_df['scenario'].unique():
        group = results_df[results_df['scenario'] == scenario]
        settings = scenario_settings[scenario]
        
        ax.scatter(
            group['ecological_impact'], 
            group['economic_impact'], 
            c=settings["color"],
            marker=settings["marker"],
            label=settings["name"],
            alpha=0.75,
            s=60,
            edgecolor='white',
            linewidth=0.5
        )
    
    # Add common elements
    add_impact_plot_elements(ax, results_df)
    ax.set_title('Impact by Actor Scenario Configuration', fontsize=14)
    
    # Get handles and labels for sorted legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    sorted_labels = sorted(by_label.keys())
    sorted_handles = [by_label[label] for label in sorted_labels]
    
    # Create legend
    create_styled_legend(
        ax,
        handles=sorted_handles,
        labels=sorted_labels,
        title="Scenario",

    )
    
    plt.tight_layout()
    plt.show()


def create_multidimensional_impact_plot(results_df):
    """
    Create a plot showing ecological vs economic impact with
    multiple dimensions encoded (scenario, station, scarcity).
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame containing simulation results
        
    Returns:
    --------
    None
        Displays the created plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get color schemes
    scenario_settings = COLOR_SCHEMES['scenario']
    scarcity_colors = COLOR_SCHEMES['scarcity']
    
    # Plot data points encoding multiple dimensions
    for scenario in results_df['scenario'].unique():
        for scarcity in results_df['scarcity'].unique():
            subset = results_df[(results_df['scenario'] == scenario) & 
                               (results_df['scarcity'] == scarcity)]
            
            if not subset.empty:
                settings = scenario_settings[scenario]
                
                # Create a label that only shows on the first occurrence
                label = f"{settings['name']} ({scarcity})" if scarcity == "medium" else None
                
                ax.scatter(
                    subset['ecological_impact'], 
                    subset['economic_impact'], 
                    c=scarcity_colors[scarcity],
                    marker=settings["marker"],
                    label=label,
                    alpha=0.8,
                    s=70,
                    edgecolor=settings["color"],
                    linewidth=1.5
                )
    
    # Add common elements
    add_impact_plot_elements(ax, results_df)
    ax.set_title('Combined Analysis: Scenario, Basin Size and Scarcity Level', fontsize=14)
    
    # Create custom legend elements
    
    # Marker legends (scenario variants)
    scenario_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Base Scenarios'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=10, label='V-Variant'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=10, label='B-Variant'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='gray', markersize=10, label='C-Variant')
    ]
    
    # Color legends (scarcity)
    scarcity_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=scarcity_colors["low"], markersize=10, label='Low Scarcity'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=scarcity_colors["medium"], markersize=10, label='Medium Scarcity'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=scarcity_colors["high"], markersize=10, label='High Scarcity')
    ]
    
    # Edge color legends (scenario type)
    type_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='w', markersize=10, 
               markeredgecolor=scenario_settings["0.yml"]["color"], markeredgewidth=2, label='Type 0'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='w', markersize=10, 
               markeredgecolor=scenario_settings["1.yml"]["color"], markeredgewidth=2, label='Type 1')
    ]
    
    # Add legends at different positions
    first_legend = create_styled_legend(
        ax, 
        handles=scenario_elements,
        labels=[e.get_label() for e in scenario_elements],
        title="Scenario Variant",
        position=(.85, 0.30)
    )
    ax.add_artist(first_legend)
    
    second_legend = create_styled_legend(
        ax, 
        handles=scarcity_elements,
        labels=[e.get_label() for e in scarcity_elements],
        title="Scarcity Level",
        position=(.85, 0.45)
    )
    ax.add_artist(second_legend)
    
    third_legend = create_styled_legend(
        ax, 
        handles=type_elements,
        labels=[e.get_label() for e in type_elements],
        title="Scenario Type",
        position=(.85, 0.8)
    )
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust right margin for legends
    plt.show()


def create_forecast_effect_plot(df, x, y, ax, size_var=None, abs_size=False):
    """
    Create a scatter plot showing the effect of forecast parameters
    on impact measures.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing simulation results
    x : str
        Column name for x-axis
    y : str
        Column name for y-axis
    ax : matplotlib.axes.Axes
        Axes to plot on
    size_var : str, optional
        Column name to use for point sizes
    abs_size : bool
        Whether to use absolute values for sizing
        
    Returns:
    --------
    None
        Modifies the provided axes
    """
    # Set up size variable
    if size_var:
        sizes = (50, 200)
        if abs_size:
            size_values = df[size_var].abs()
        else:
            size_values = df[size_var]
    else:
        sizes = 80
        size_values = None
    
    # Create the scatter plot
    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        hue='scarcity',
        size=size_var,
        sizes=sizes,
        palette=COLOR_SCHEMES['scarcity'],
        ax=ax,
        legend=False  # Don't show legend for individual plots
    )
    
    # Add reference line through origin
    if min(ax.get_xlim()) <= 0 <= max(ax.get_xlim()):
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # Improve aesthetics
    ax.set_title(f'{x.title()} vs {y.replace("_", " ").title()}', fontsize=15)
    ax.set_xlabel(x.title(), fontsize=12)
    ax.set_ylabel(y.replace("_", " ").title(), fontsize=12)


# =====================================================================
# Example Usage
# =====================================================================

if __name__ == "__main__":
    # This is a demonstration of how to use these functions
    # Create some synthetic data to show the plots
    
    np.random.seed(42)
    
    # Generate synthetic data
    n = 100
    scenarios = ['0.yml', '1.yml', '0-v.yml', '1-v.yml', '0-b.yml', '1-b.yml', '0-c.yml', '1-c.yml']
    scarcities = ['low', 'medium', 'high']
    stations = [1, 2]
    
    # Create data frame
    data = {
        'ecological_impact': np.random.uniform(0, 1, n),
        'economic_impact': np.random.uniform(0, 1, n),
        'bias': np.random.uniform(-0.5, 0.5, n),
        'uncertainty': np.random.uniform(0, 0.5, n),
        'scenario': np.random.choice(scenarios, n),
        'scarcity': np.random.choice(scarcities, n),
        'station': np.random.choice(stations, n)
    }
    
    # Introduce some correlations
    for i in range(n):
        if data['scarcity'][i] == 'high':
            data['ecological_impact'][i] += 0.2
            data['economic_impact'][i] -= 0.1
        elif data['scarcity'][i] == 'low':
            data['ecological_impact'][i] -= 0.1
            data['economic_impact'][i] += 0.1
            
        # Bias affects economic impact
        data['economic_impact'][i] += data['bias'][i] * 0.2
        
        # Station affects ecological impact
        if data['station'][i] == 1:
            data['ecological_impact'][i] -= 0.05
    
    # Ensure values are in valid range
    data['ecological_impact'] = np.clip(data['ecological_impact'], 0, 1)
    data['economic_impact'] = np.clip(data['economic_impact'], 0, 1)
    
    df = pd.DataFrame(data)
    
    # Run the analysis functions
    print("Analyzing scenario impacts...")
    analyze_scenario_impacts(df)
    
    print("Analyzing forecast effects...")
    analyze_forecast_effects(df)
    
    print("Performing correlation analysis...")
    correlation_analysis(df)