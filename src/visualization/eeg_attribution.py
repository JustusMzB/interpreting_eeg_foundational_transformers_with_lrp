"""
Visualization suite for EEG feature attributions.
This module provides utilities for mapping Layer-wise Relevance Propagation (LRP) 
scores back to neuroscientifically grounded visualizations, including topographic 
maps (Voronoi-based) and temporal relevance profiles.
"""
from plotly import graph_objects as go
import plotly
from plotly.subplots import make_subplots
import numpy as np
import torch
from typing import Literal, Union, List, Tuple, Optional
import matplotlib.pyplot as plt
import mne
import logging

logger = logging.getLogger(__name__)
default_channels = mne.channels.make_standard_montage("standard_1020").ch_names

def plot_channel_plotly(
    eeg_data: Union[np.ndarray, torch.Tensor], 
    relevance_data: Optional[Union[np.ndarray, torch.Tensor]], 
    channel_name: str, 
    min_relevance: Optional[float] = None, 
    max_relevance: Optional[float] = None, 
    center_colorbar: bool = True,
    fig: Optional[go.Figure] = None, 
    row: Optional[int] = None, 
    col: Optional[int] = None,
    y_min: Optional[float] = None, 
    y_max: Optional[float] = None,
    additional_traces: Optional[List[go.Trace]] = None,
    display_name: bool = True,
    heatmap_ticks: Literal["numerical", "+-"] = "+-",
    show_cbar: bool = True
) -> go.Figure:
    """
    Plots a single EEG channel's time-series overlaid with its temporal 
    relevance (attribution) heatmap using Plotly.

    This visualization allows for the direct identification of signal components 
    (e.g., specific ERP/ERD features or artifacts) that drive model decisions.

    Args:
        eeg_data: EEG signal magnitude, shape (num_timepoints,).
        relevance_data: LRP relevance scores, shape (num_timepoints,).
        channel_name: Name of the EEG electrode.
        min_relevance: Min value for relevance normalization.
        max_relevance: Max value for relevance normalization.
        center_colorbar: Whether to center the relevance color scale on zero.
        fig: Optional existing Plotly figure for subplots.
        row: Subplot row index.
        col: Subplot column index.
        y_min: Min y-axis limit for the signal plot.
        y_max: Max y-axis limit for the signal plot.
        additional_traces: Extra Plotly traces to overlay.
        display_name: Whether to show the channel name as a title.
        heatmap_ticks: Format for heatmap colorbar ticks.
        show_cbar: Whether to display the colorbar.

    Returns:
        go.Figure: The Plotly figure with the updated channel plot.
    """
    # Validate and cast inputs
    assert isinstance(channel_name, str), "channel_name must be a string"
    assert isinstance(center_colorbar, bool), "center_colorbar must be a boolean"
    assert fig is None or isinstance(fig, go.Figure), "fig must be a Plotly Figure or None"
    assert row is None or isinstance(row, int), "row must be an integer or None"
    assert col is None or isinstance(col, int), "col must be an integer or None"
    assert y_min is None or isinstance(y_min, (int, float)), "y_min must be a number or None"
    assert y_max is None or isinstance(y_max, (int, float)), "y_max must be a number or None"

    if eeg_data is not None:
        assert isinstance(eeg_data, np.ndarray), "eeg_data must be a numpy array or torch tensor"
        assert eeg_data.ndim == 1, "eeg_data must be a 1D array"
    elif relevance_data is not None:
        assert isinstance(relevance_data, np.ndarray), "relevance_data must be a numpy array or torch tensor"
        assert relevance_data.ndim == 1, "relevance_data must be a 1D array"
    else:
        raise ValueError("either eeg or relevance data must be provided")
    if eeg_data is not None and relevance_data is not None:
        assert len(eeg_data) == len(relevance_data), "eeg_data and relevance_data must have the same length"
    
    if isinstance(eeg_data, torch.Tensor):
        eeg_data = eeg_data.numpy()
    if fig is None:
        fig = go.Figure()
    # Add EEG data line
    # Determine y-axis limits for the heatmap
    if eeg_data is not None:
        if y_min is None:
            y_min = 1.2 * min(eeg_data)
        if y_max is None:
            y_max = 1.2 * max(eeg_data)
    else:
        y_min = np.nan
        y_max = np.nan

    if np.isnan(y_min):
        y_min = 0
    if np.isnan(y_max):
        y_max = 0
        
    if y_min == y_max:
        if y_min == 0:
            y_min =-0.2
            y_max =+0.2
        else:
            y_max = y_max + 0.2*abs(y_max)
            y_min = y_min - 0.2*abs(y_min)



    # Handle Relevance optionally
    if relevance_data is not None:
        if isinstance(relevance_data, torch.Tensor):
            relevance_data = relevance_data.numpy()
        assert isinstance(relevance_data, np.ndarray), "relevance_data must be a numpy array or torch tensor"
        assert relevance_data.ndim == 1, "relevance_data must be a 1D array"
        
        if min_relevance is None:
            min_relevance = np.nanmin(relevance_data)
        if max_relevance is None:
            max_relevance = np.nanmax(relevance_data)
        lim_relevance = max(abs(min_relevance), abs(max_relevance))
        # Normalize relevance data for coloring
        norm_relevance = (relevance_data - min_relevance) / (max_relevance - min_relevance)
        # Center on 0
        if center_colorbar:
            median = np.median(norm_relevance)
            norm_relevance = norm_relevance - median + 0.5
            norm_relevance = np.clip(norm_relevance, 0, 1)  # Ensure values are between 0 and 1
        
        #y_lim = max(abs(y_min), abs(y_max))

        # Add background as a heatmap
        if heatmap_ticks == "numerical":
            color_ticks = [min_relevance, 0, max_relevance]
            ticktexts = [f"{min_relevance:.2f}", "0", f"{max_relevance:.2f}"]
        elif heatmap_ticks == "+-":
            color_ticks = [min_relevance, 0, max_relevance]
            ticktexts = ["-", "0", "+"]
        else:
            raise ValueError(f"Invalid heatmap_ticks value {heatmap_ticks}")
        
        fig.add_trace(go.Heatmap(
            z=[relevance_data, relevance_data], 
            x=np.arange(0, len(relevance_data)), 
            #y0=0,
            y=[y_min, y_max],  # Use provided y_min and y_max
            colorscale="rdbu_r",  # Change color gradient here
            showscale=show_cbar,  # Hide color legend
            opacity=0.8,
            name=f'Relevance of Channel {channel_name}',
            zmin=-lim_relevance,  # Enforce same heatmap scale
            zmax=lim_relevance,   # Enforce same heatmap scale
            colorbar=dict(
                title=dict(text="Attribution",side="right"),
                tickvals=color_ticks,
                ticktext=ticktexts,
                len=0.8,  # Length of the colorbar
                thickness=10,  # Thickness of the colorbar
            )
        ), row=row, col=col)
    if additional_traces is not None:
       for trace in additional_traces:
           fig.add_trace(trace, row=row, col=col)

    if eeg_data is not None:
        # Print EEG last for visibility
        fig.add_trace(go.Scatter(
            x=np.arange(len(eeg_data)),
            y=eeg_data,
            mode='lines',
            name=f'Channel {channel_name}',
            line=dict(color='black'),
        ), row=row, col=col)
    if display_name:
        fig.update_yaxes(title_text=channel_name, row=row, col=col)
    
    return fig

def plot_eeg_channels_with_relevance(
    eeg_data: Union[np.ndarray, torch.Tensor], 
    relevance_data: Optional[Union[np.ndarray, torch.Tensor]] = None, 
    channel_names: List[str] = default_channels, 
    selected_channels: Optional[List[str]] = None, 
    title: str = "EEG Plot with Relevance", 
    figsize: Tuple[int, Optional[int]] = (10, None), 
    shared_yaxis: bool = False,
    additional_traces: Optional[List[go.Trace]] = None,
    top_k_relevant: Optional[int] = None,
    display_names: Optional[List[str]] = True,
    show_cbar=True,
    show_ax_labels=True,
) -> go.Figure:
    """
    Plots multiple EEG channels with relevance data using Plotly subplots.

    Parameters:
    - eeg_data: Union[np.ndarray, torch.Tensor], shape (num_channels, num_timepoints)
    - relevance_data: Union[np.ndarray, torch.Tensor], shape (num_channels, num_timepoints)
    - channel_names: List[str], names of the channels
    - selected_channels: Optional[List[str]], names of the channels to plot (default: None, plots all channels)
    - title: str, title of the plot (default: "EEG Plot with Relevance")
    - figsize: Tuple[int, Optional[int]], size of the figure (default: (10, None))
    - shared_yaxis: bool, whether to share the y-axis among subplots (default: False)

    Returns:
    - fig: go.Figure, the figure with the plotted channels
    """
    # Validate and cast inputs
    assert isinstance(channel_names, list) and all(isinstance(ch, str) for ch in channel_names), "channel_names must be a list of strings"
    assert isinstance(shared_yaxis, bool), "shared_yaxis must be a boolean"
    # Prepare plotting EEG
    if selected_channels is not None:
        assert isinstance(selected_channels, list) and all(isinstance(ch, str) for ch in selected_channels), "selected_channels must be a list of strings or None"
    if eeg_data is not None:
        assert isinstance(eeg_data, np.ndarray) or isinstance(eeg_data, torch.Tensor), "eeg_data must be a numpy array or torch tensor"
        assert eeg_data.ndim == 2, "eeg_data must be a 2D array"
        assert len(channel_names) == eeg_data.shape[0], "eeg_data must have same dimension as channel names. Consider setting channel names explicitly"
    elif relevance_data is not None:
        assert len(channel_names) == relevance_data.shape[0], "relevance_data must have same dimension as channel names. Consider setting channel names explicitly"
    else:
        raise ValueError("either eeg or relevance data must be provided")
    if (eeg_data is not None) and (relevance_data is not None):
        assert eeg_data.shape == relevance_data.shape, "eeg_data and relevance_data must have the same shape"

    
    
    

    # Prepare plotting Relevance
    if relevance_data is not None:
        if isinstance(eeg_data, torch.Tensor):
            eeg_data = eeg_data.detach().numpy()
        if isinstance(relevance_data, torch.Tensor):
            relevance_data = relevance_data.detach().numpy()
        assert isinstance(relevance_data, np.ndarray), "relevance_data must be a numpy array or torch tensor"
        assert relevance_data.ndim == 2, "relevance_data must be a 2D array"
        
        if top_k_relevant is not None:
            assert isinstance(top_k_relevant, int) and top_k_relevant > 0, "top_k_relevant must be a positive integer"
            selected_indices = np.argsort(np.mean(np.abs(relevance_data), axis=1))[-top_k_relevant:]
            selected_channels = [channel_names[i] for i in reversed(selected_indices)]

        min_relevance = np.nanmin(relevance_data)
        max_relevance = np.nanmax(relevance_data)
    elif selected_channels is None:
        selected_channels = channel_names
    else:
        selected_indices = [channel_names.index(ch) for ch in selected_channels]

    if eeg_data is not None:
        if not (eeg_data == np.nan).all(): # Necessary to avoid type error
            max_eeg = np.nanmax(eeg_data)
            min_eeg = np.nanmin(eeg_data)
        else:
            min_eeg = -1.0
            max_eeg = 1.0
    else:
        logging.warning("EEG data contains only NaN values. Using default y-axis limits. Are you sure you wanted this?")
        max_eeg = 1.0
        min_eeg = -1.0
    
    if figsize[1] is None:
        figsize = (figsize[0], len(selected_channels))
    num_channels = len(selected_channels)
    fig = make_subplots(rows=num_channels, cols=1, shared_xaxes=True)
    fig.update_layout(
        height=figsize[1] * 100,
        width=figsize[0] * 100,
        title_text=title,
        showlegend=False
    )
    for i, ch_idx in enumerate(selected_indices):
        if isinstance(display_names, list):
            display_name = selected_channels[i] in display_names
        else:
            display_name = bool(display_names)
    
        fig = plot_channel_plotly(
            eeg_data[ch_idx] if eeg_data is not None else None,
            relevance_data[ch_idx] if relevance_data is not None else None,
            selected_channels[i],
            min_relevance=min_relevance if relevance_data is not None else None,
            max_relevance=max_relevance if relevance_data is not None else None,
            fig=fig,
            row=i+1,
            col=1,
            y_min=min_eeg if shared_yaxis else None,
            y_max=max_eeg if shared_yaxis else None,
            additional_traces=additional_traces,
            display_name=display_name,
            show_cbar=show_cbar,

        )

    if shared_yaxis:
        fig.update_yaxes(matches='y')



    if not show_cbar:
        fig.update_coloraxes(showscale=False)
    
    if not show_ax_labels:
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)

    return fig

def attribution_topomap(
    attributions, 
    channel_names, 
    title="Attribution Topomap", 
    montage="standard_1020",
    ax=None, 
    sfreq=200,
    show_channel_names=True,
    sensors=None,
):
    """
    Create a topomap plot for EEG channel attributions.

    Parameters:
        attributions (numpy.ndarray): The attribution values for each channel.
        channel_names (list): List of channel names.
        title (str): Title of the plot.
        montage (str): Montage type for the topomap (see MNE documentation for options).
        ax (matplotlib.axes.Axes): Axes to plot on. If None, a new figure is created.

    Returns:
        matplotlib.figure.Figure: The figure object containing the topomap plot.
    """
    # Create MNE info object
    info = mne.create_info(ch_names=channel_names, sfreq=200, ch_types="eeg")
    info.set_montage(montage)
    if sensors is None:
        sensors = not show_channel_names
    # Create topomap plot
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = ax.figure
    
    if len(attributions.shape) == 2:
        attributions = attributions.sum(axis=1)
    vlim = np.abs(attributions).max()
    im, _ = mne.viz.plot_topomap(
        attributions, 
        info, 
        axes=ax, 
        names=channel_names if show_channel_names else None,
        show=False,
        contours=0, 
        vlim=(-vlim, vlim), 
        cmap="RdBu_r", 
        outlines="head", 
        sphere="auto",
        image_interp="nearest",
        sensors=sensors,
    )
    ax.set_title(title)
    return fig

def plot_attribution_over_time(attributions, 
                               title, 
                               ax=None):
    """
    Plot the attribution values over time.

    Parameters:
        attributions (numpy.ndarray): The attribution values for each time point.
        title (str): Title of the plot.
        ax (matplotlib.axes.Axes): Axes to plot on. If None, a new figure is created.

    Returns:
        matplotlib.figure.Figure: The figure object containing the time series plot.
    """
    if isinstance(attributions, torch.Tensor):
        attributions = attributions.detach().cpu().numpy()

    attributions = attributions.sum(axis=0)
    attributions = attributions.reshape((1, attributions.shape[0]))  # Sum over channels
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    else:
        fig = ax.figure
    im = ax.imshow(
        attributions,
        cmap="RdBu_r",
        aspect="auto",
        vmin=-np.max(np.abs(attributions)),
        vmax=np.max(np.abs(attributions)),
    )
    ax.set_yticks([])
    ax.set_title(title)
    return fig

def plot_attribution_with_additional_curve(
    attributions,
    additional_curves,
    title="Attribution with Additional Curve",
    ax=None,
    additional_curve_labels=None,
    alphas=None,
    linetypes=None,
    colors=None,
    show_yticks:Optional[bool] = False
):
    """
    Plot attribution over time with additional curves (e.g., predictions, labels).

    Parameters:
    attributions (np.ndarray or torch.Tensor): Attribution values (channels x time).
    additional_curves (list or np.ndarray): List of additional 1D arrays to plot. Each must be same length as attribution (time axis) and in similar value ranges for meaningful overlay.
    title (str): Title of the plot.
    ax (matplotlib.axes.Axes): Axes to plot on. If None, a new figure is created.
    additional_curve_labels (list): List of labels for additional curves.
    alphas (list or float): Alpha (transparency) values for additional curves.
    linetypes (list or str): Line styles for additional curves.
    colors (list or str): Colors for additional curves.

    Returns:
    matplotlib.figure.Figure: The figure object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    else:
        fig = ax.figure

    fig = plot_attribution_over_time(
        attributions,
        title=title,
        ax=ax
    )

    ax2 = ax.twinx()
    if not show_yticks:
        ax2.set_yticks([])
    if additional_curves is not None:
        if not isinstance(additional_curves, list):
            additional_curves = [additional_curves]
        if additional_curve_labels is None:
            additional_curve_labels = [None] * len(additional_curves)
        if not isinstance(additional_curve_labels, list):
            additional_curve_labels = [additional_curve_labels]
        if alphas is None:
            alphas = [1.0] * len(additional_curves)
        if not isinstance(alphas, list):
            alphas = [alphas] * len(additional_curves)
        if linetypes is None:
            linetypes = ['-'] * len(additional_curves)
        if not isinstance(linetypes, list):
            linetypes = [linetypes] * len(additional_curves)
        if colors is None:
            colors = [None] * len(additional_curves)
        if not isinstance(colors, list):
            colors = [colors] * len(additional_curves)

        lower_bound = np.min(np.array(additional_curves)) - 0.1 * np.abs(np.max(np.array(additional_curves)))
        upper_bound = np.max(np.array(additional_curves)) + 0.1 * np.abs(np.max(np.array(additional_curves)))
        for additional_curve, label, alpha, linetype, color in zip(additional_curves, additional_curve_labels, alphas, linetypes, colors):
            assert attributions.shape[-1] == additional_curve.shape[0], "Attributions and additional curve must have the same length"
            if isinstance(additional_curve, torch.Tensor):
                additional_curve = additional_curve.detach().cpu().numpy()
            ax2.plot(additional_curve, label=label, alpha=alpha, linestyle=linetype, color=color)
            ax2.set_ylim(lower_bound, upper_bound)
        if any(label is not None for label in additional_curve_labels):
            ax2.legend(loc="upper right")
    return fig

def plot_attribution_summary(
    attributions,
    channel_names,
    title=None,
    topomap_title=None,
    time_series_title=None,
    additional_curves=None,
    additional_curve_labels=None,
    alphas=None,
    linetypes=None,
    colors=None,
    montage="standard_1020",
    sfreq=200,
    axes = None,
    show_channel_names=True,

):
    if axes is None:
        fig, (ax_time, ax_topo) = plt.subplots(2, 1, figsize=(8, 9), height_ratios=[2, 10])
        fig.suptitle(title)
    else:
        assert len(axes) == 2, "If axes are provided, they must be a tuple of (ax_time, ax_topo)"
        ax_time, ax_topo = axes
        fig = ax_time.figure
        if title is not None:
            logger.warning("Title is not None, but axes were provided. Title will not be set.")
    
    attribution_topomap(
        attributions,
        channel_names,
        title=topomap_title,
        montage=montage,
        ax=ax_topo,
        sfreq=sfreq,
        show_channel_names=show_channel_names,
    )
    plot_attribution_with_additional_curve(
        attributions,
        additional_curves=additional_curves,
        title=time_series_title,
        ax=ax_time,
        additional_curve_labels=additional_curve_labels,
        alphas=alphas,
        linetypes=linetypes,
        colors=colors
    )
    
    fig.tight_layout()
    return fig