from typing import List, Literal
from matplotlib import pyplot as plt
import pyarrow as pa
import pyarrow.parquet as pq

from visualization.eeg_attribution import plot_attribution_summary, plot_attribution_over_time, plot_attribution_with_additional_curve, attribution_topomap
import pandas as pd
import numpy as np

def plot_summary_grid(attributions, 
                      titles, 
                      channel_names,
                      suptitle=None,
                      supxlabel="Runs",
                      supylabel="Configurations",
                      figsize=(8.27,11.69),
                      show_channel_names=False,
                      first_channel_name_size=None,
                      spatial_to_time_ratio=9
                      ):
    """
    attr_means: List of mean attribution values for each experiment. I assume we have k runs of n configurations. Shape of attr_means is (n, k, (shape of attribution))
    """
    fig, ax = plt.subplots(
        len(attributions)*2, len(attributions[0]), 
        gridspec_kw={"height_ratios": [1, spatial_to_time_ratio]*3},
        figsize=figsize,
        constrained_layout=True,
        )
    for run in range(0, len(attributions[0])):
        for conf in range(0, len(attributions)):
            if conf == len(attributions) -1 and run == len(attributions[0]) -1 and first_channel_name_size is not None:
                show_channel_names_ = True
            else:
                show_channel_names_ = show_channel_names
            plot_attribution_summary(
                attributions[conf][run], 
                channel_names, 
                time_series_title=titles[conf][run],
                axes=(ax[conf*2,run], ax[conf*2+1, run]),
                show_channel_names=show_channel_names_,
            )
            ax[conf*2, run].tick_params("x", labelsize="small", direction="in")
            for text in ax[conf*2, run].texts:
                text.set_fontsize(first_channel_name_size)
    fig.suptitle(suptitle, y=1.02)
    fig.supxlabel(supxlabel, y=-0.02)
    fig.supylabel(supylabel, x=-0.02)
    #fig.execute_constrained_layout()
    return fig

def calculate_attribution_statistics(df: pd.DataFrame):
    """
    Calculate statistics for the attribution evaluation the DataFrame (which can also be a view).
    Args:
        df (pd.DataFrame): DataFrame containing the attribution results with a column named 'attribution'.
    Returns:
        dict: A dictionary containing the mean and standard deviation of the attribution values.
    """
    mean_attribution = df.attribution.mean()
    var_attribution = df.attribution.apply(
        lambda x: (x - mean_attribution) ** 2
    ).sum() / len(df)
    std_attribution = np.sqrt(var_attribution)
    return {
        "mean": mean_attribution,
        "std": std_attribution
    }

def calculate_classwise_statistics(
        df: pd.DataFrame, 
        prediction_col: str, 
        label_col: str, 
        class_names: dict):
    """
    Calculate statistics for attribution results grouped by class and correctness.
    Args:
        df (pd.DataFrame): DataFrame containing the attribution results with columns for predictions and labels.
        prediction_col (str): Name of the column containing model predictions.
        label_col (str): Name of the column containing true labels.
        class_names (dict): Mapping from class indices to human-readable names.
    Returns:
        dict: A dictionary containing statistics for each class, including mean and std of attribution values for
                correct and incorrect predictions. 
    """
    
    correct_mask = df[prediction_col] == df[label_col]
    classes = df[label_col].unique()
    class_stats = {}
    for class_ in classes:
        class_mask = df[label_col] == class_
        class_df = df[class_mask]
        
        correct_class_df = class_df[correct_mask[class_mask]]
        incorrect_class_df = class_df[~correct_mask[class_mask]]
        
        correct_stats = calculate_attribution_statistics(correct_class_df)
        all_stats = calculate_attribution_statistics(class_df)
        incorrect_stats = calculate_attribution_statistics(incorrect_class_df)

        class_stats[class_] = {
            "class_name": class_names.get(class_, str(class_)),
            "correct_attribution_mean": correct_stats["mean"],
            "correct_attribution_std": correct_stats["std"],
            "all_attribution_mean": all_stats["mean"],
            "all_attribution_std": all_stats["std"],
            "incorrect_attribution_mean": incorrect_stats["mean"],
            "incorrect_attribution_std": incorrect_stats["std"]
        }
    return class_stats

def plot_summary_correctness(
        stats: dict,
        channel_names: List[str],
        stat_to_plot: Literal["mean", "std"] = "mean",
        subsecs_to_plot: List[Literal["correct", "incorrect", "all"]] = ["all", "correct"],
        views_to_plot: List[Literal["spatial", "temporal"]] = ["spatial"]
):
    n_cols = len(stats) * len (views_to_plot)
    n_rows = len(subsecs_to_plot)
    fig, axes = plt.subplots(
        n_rows, n_cols, 
        figsize=(n_cols * 3, n_rows * 3),
        gridspec_kw={'height_ratios': [.2, .8] * len(stats) * 2} if len(views_to_plot) > 1 else None,
    )

    for i, (class_, class_stats) in enumerate(stats.items()):
        for j, subseq in enumerate(subsecs_to_plot):
            if subseq == "correct":
                value = class_stats["correct_attribution_" + stat_to_plot]
            elif subseq == "incorrect":
                value = class_stats["incorrect_attribution_" + stat_to_plot]
            else: # "all"
                value = class_stats["all_attribution_" + stat_to_plot]
            if "spatial" in views_to_plot:
                spatial_col = j*2+1 if len(views_to_plot) > 1 else j
                attribution_topomap(
                    value,
                    channel_names=channel_names,
                    title=f"{class_stats['class_name']} - {subseq.capitalize()} Attribution {stat_to_plot.capitalize()}",
                    ax=axes[spatial_col, i]
                )
            if "temporal" in views_to_plot:
                temporal_col = j*2 if len(views_to_plot) > 1 else j
                plot_attribution_over_time(
                    value,
                    title=f"{class_stats['class_name']} - {subseq.capitalize()} Attribution {stat_to_plot.capitalize()}",
                    ax=axes[temporal_col, i]
                )
    
