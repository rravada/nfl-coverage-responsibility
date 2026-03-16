"""
Visualization utilities for NFL tracking data analysis.

This module provides functions for creating static plots, animations,
and visual validation of player tracking data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


def plot_frame_static(tracking_df, frame_id, offense_team, defense_team, 
                      game_id=None, play_id=None, figsize=(15, 8), 
                      show_jersey_numbers=True, ax=None):
    """
    Create a static plot of player positions at a specific frame.
    
    Parameters
    ----------
    tracking_df : pd.DataFrame
        Tracking data (can be full play or already filtered to one frame)
    frame_id : int
        The frame ID to visualize
    offense_team : str
        Team abbreviation for offense (e.g., 'TB')
    defense_team : str
        Team abbreviation for defense (e.g., 'DAL')
    game_id : int, optional
        Game ID for title display
    play_id : int, optional
        Play ID for title display
    figsize : tuple, optional
        Figure size (width, height) in inches
    show_jersey_numbers : bool, optional
        Whether to display jersey numbers on players
    ax : matplotlib.axes.Axes, optional
        Axes to plot on (creates new figure if None)
    
    Returns
    -------
    tuple of (fig, ax)
        Matplotlib figure and axes objects
        
    Examples
    --------
    >>> fig, ax = plot_frame_static(play_tracking, frame_id=6, 
    ...                             offense_team='TB', defense_team='DAL',
    ...                             game_id=2021090900, play_id=1267)
    >>> plt.show()
    """
    # Filter to specific frame if not already filtered
    frame_data = tracking_df[tracking_df['frameId'] == frame_id].copy()
    
    if len(frame_data) == 0:
        raise ValueError(f"No data found for frame_id={frame_id}")
    
    # Separate by team
    offense = frame_data[frame_data['team'] == offense_team]
    defense = frame_data[frame_data['team'] == defense_team]
    ball = frame_data[frame_data['team'] == 'football']
    
    # Create figure if axes not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Plot offense (red circles)
    ax.scatter(offense['x'], offense['y'], 
               c='red', s=200, marker='o', 
               edgecolors='darkred', linewidth=2,
               label=f'Offense ({offense_team})', zorder=3)
    
    # Plot defense (blue squares)
    ax.scatter(defense['x'], defense['y'], 
               c='blue', s=200, marker='s', 
               edgecolors='darkblue', linewidth=2,
               label=f'Defense ({defense_team})', zorder=3)
    
    # Plot ball (brown diamond)
    ax.scatter(ball['x'], ball['y'], 
               c='brown', s=150, marker='D', 
               edgecolors='black', linewidth=2,
               label='Football', zorder=4)
    
    # Add jersey numbers if requested
    if show_jersey_numbers:
        for _, player in offense.iterrows():
            if not np.isnan(player['jerseyNumber']):
                ax.text(player['x'], player['y'], str(int(player['jerseyNumber'])), 
                        fontsize=8, ha='center', va='center', 
                        color='white', weight='bold')
        
        for _, player in defense.iterrows():
            if not np.isnan(player['jerseyNumber']):
                ax.text(player['x'], player['y'], str(int(player['jerseyNumber'])), 
                        fontsize=8, ha='center', va='center', 
                        color='white', weight='bold')
    
    # Configure axes
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)
    ax.set_xlabel('X Coordinate (yards) — Field Length', fontsize=12, weight='bold')
    ax.set_ylabel('Y Coordinate (yards) — Field Width', fontsize=12, weight='bold')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add title
    title_parts = ['Player Positions']
    if game_id is not None:
        title_parts.append(f'GameId: {game_id}')
    if play_id is not None:
        title_parts.append(f'PlayId: {play_id}')
    title_parts.append(f'Frame: {frame_id}')
    ax.set_title(' | '.join(title_parts), fontsize=14, weight='bold', pad=20)
    
    # Add field markings
    ax.axhline(y=0, color='green', linewidth=3, alpha=0.3)
    ax.axhline(y=53.3, color='green', linewidth=3, alpha=0.3)
    ax.axvline(x=10, color='orange', linewidth=2, alpha=0.3, linestyle='--')
    ax.axvline(x=110, color='orange', linewidth=2, alpha=0.3, linestyle='--')
    
    # Add legend
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    
    return fig, ax


def plot_side_by_side_comparison(tracking_before, tracking_after, frame_id,
                                  offense_team, defense_team,
                                  title_before='BEFORE', title_after='AFTER',
                                  figsize=(20, 8)):
    """
    Create side-by-side comparison plots (e.g., before/after normalization).
    
    Parameters
    ----------
    tracking_before : pd.DataFrame
        Tracking data for left plot
    tracking_after : pd.DataFrame
        Tracking data for right plot
    frame_id : int
        Frame ID to visualize
    offense_team : str
        Offense team abbreviation
    defense_team : str
        Defense team abbreviation
    title_before : str, optional
        Title for left plot
    title_after : str, optional
        Title for right plot
    figsize : tuple, optional
        Figure size (width, height)
    
    Returns
    -------
    tuple of (fig, (ax1, ax2))
        Figure and tuple of both axes
        
    Examples
    --------
    >>> fig, (ax1, ax2) = plot_side_by_side_comparison(
    ...     play_tracking_original, play_tracking_normalized,
    ...     frame_id=6, offense_team='TB', defense_team='DAL',
    ...     title_before='BEFORE Normalization', 
    ...     title_after='AFTER Normalization')
    >>> plt.show()
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot BEFORE
    frame_before = tracking_before[tracking_before['frameId'] == frame_id]
    offense_before = frame_before[frame_before['team'] == offense_team]
    defense_before = frame_before[frame_before['team'] == defense_team]
    ball_before = frame_before[frame_before['team'] == 'football']
    
    ax1.scatter(offense_before['x'], offense_before['y'], 
                c='red', s=200, marker='o', edgecolors='darkred', linewidth=2,
                label=f'Offense ({offense_team})', zorder=3)
    ax1.scatter(defense_before['x'], defense_before['y'], 
                c='blue', s=200, marker='s', edgecolors='darkblue', linewidth=2,
                label=f'Defense ({defense_team})', zorder=3)
    ax1.scatter(ball_before['x'], ball_before['y'], 
                c='brown', s=150, marker='D', edgecolors='black', linewidth=2,
                label='Football', zorder=4)
    
    # Show play direction for BEFORE
    play_dir_before = frame_before['playDirection'].iloc[0]
    if play_dir_before == 'left':
        ax1.arrow(35, 10, -10, 0, head_width=3, head_length=2, 
                  fc='orange', ec='orange', linewidth=3, alpha=0.7)
    else:
        ax1.arrow(75, 10, 10, 0, head_width=3, head_length=2,
                  fc='orange', ec='orange', linewidth=3, alpha=0.7)
    
    ax1.set_xlim(0, 120)
    ax1.set_ylim(0, 53.3)
    ax1.set_xlabel('X Coordinate (yards)', fontsize=12, weight='bold')
    ax1.set_ylabel('Y Coordinate (yards)', fontsize=12, weight='bold')
    ax1.set_title(f'{title_before}\nplayDirection = "{play_dir_before}"', 
                  fontsize=13, weight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_aspect('equal', adjustable='box')
    ax1.legend(loc='upper right', fontsize=9)
    
    # Plot AFTER
    frame_after = tracking_after[tracking_after['frameId'] == frame_id]
    offense_after = frame_after[frame_after['team'] == offense_team]
    defense_after = frame_after[frame_after['team'] == defense_team]
    ball_after = frame_after[frame_after['team'] == 'football']
    
    ax2.scatter(offense_after['x'], offense_after['y'], 
                c='red', s=200, marker='o', edgecolors='darkred', linewidth=2,
                label=f'Offense ({offense_team})', zorder=3)
    ax2.scatter(defense_after['x'], defense_after['y'], 
                c='blue', s=200, marker='s', edgecolors='darkblue', linewidth=2,
                label=f'Defense ({defense_team})', zorder=3)
    ax2.scatter(ball_after['x'], ball_after['y'], 
                c='brown', s=150, marker='D', edgecolors='black', linewidth=2,
                label='Football', zorder=4)
    
    # Show play direction for AFTER
    play_dir_after = frame_after['playDirection'].iloc[0]
    if play_dir_after == 'left':
        ax2.arrow(35, 10, -10, 0, head_width=3, head_length=2,
                  fc='green', ec='green', linewidth=3, alpha=0.7)
    else:
        ax2.arrow(75, 10, 10, 0, head_width=3, head_length=2,
                  fc='green', ec='green', linewidth=3, alpha=0.7)
    
    ax2.set_xlim(0, 120)
    ax2.set_ylim(0, 53.3)
    ax2.set_xlabel('X Coordinate (yards)', fontsize=12, weight='bold')
    ax2.set_ylabel('Y Coordinate (yards)', fontsize=12, weight='bold')
    ax2.set_title(f'{title_after}\nplayDirection = "{play_dir_after}"',
                  fontsize=13, weight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_aspect('equal', adjustable='box')
    ax2.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    return fig, (ax1, ax2)


def animate_play(tracking_df, offense_team, defense_team, 
                 game_id=None, play_id=None, interval=100, 
                 figsize=(15, 8), repeat=True):
    """
    Create an animated visualization of a play frame-by-frame.
    
    Parameters
    ----------
    tracking_df : pd.DataFrame
        Tracking data for the play to animate (all frames)
    offense_team : str
        Team abbreviation for offense
    defense_team : str
        Team abbreviation for defense
    game_id : int, optional
        Game ID for title
    play_id : int, optional
        Play ID for title
    interval : int, optional
        Milliseconds between frames (100ms = real-time at 10fps)
    figsize : tuple, optional
        Figure size (width, height)
    repeat : bool, optional
        Whether to loop animation continuously
    
    Returns
    -------
    matplotlib.animation.FuncAnimation
        Animation object (can be displayed with IPython.display.HTML(anim.to_jshtml()))
        
    Examples
    --------
    >>> from IPython.display import HTML
    >>> anim = animate_play(play_tracking, offense_team='TB', defense_team='DAL',
    ...                     game_id=2021090900, play_id=1267)
    >>> HTML(anim.to_jshtml())
    """
    # Get all unique frames in order
    frames = sorted(tracking_df['frameId'].unique())
    
    # Initialize figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create initial empty scatter plots
    offense_scatter = ax.scatter([], [], c='red', s=200, marker='o', 
                                edgecolors='darkred', linewidth=2,
                                label=f'Offense ({offense_team})', zorder=3)
    defense_scatter = ax.scatter([], [], c='blue', s=200, marker='s',
                                edgecolors='darkblue', linewidth=2,
                                label=f'Defense ({defense_team})', zorder=3)
    ball_scatter = ax.scatter([], [], c='brown', s=150, marker='D',
                             edgecolors='black', linewidth=2,
                             label='Football', zorder=4)
    
    # Set fixed field boundaries
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)
    ax.set_xlabel('X Coordinate (yards) — Field Length', fontsize=12, weight='bold')
    ax.set_ylabel('Y Coordinate (yards) — Field Width', fontsize=12, weight='bold')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add field markings
    ax.axhline(y=0, color='green', linewidth=3, alpha=0.3)
    ax.axhline(y=53.3, color='green', linewidth=3, alpha=0.3)
    ax.axvline(x=10, color='orange', linewidth=2, alpha=0.3, linestyle='--')
    ax.axvline(x=110, color='orange', linewidth=2, alpha=0.3, linestyle='--')
    
    # Add legend
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Create title text object
    title = ax.text(0.5, 1.05, '', transform=ax.transAxes,
                   ha='center', fontsize=14, weight='bold')
    
    def init():
        """Initialize animation"""
        offense_scatter.set_offsets(np.empty((0, 2)))
        defense_scatter.set_offsets(np.empty((0, 2)))
        ball_scatter.set_offsets(np.empty((0, 2)))
        title.set_text('')
        return offense_scatter, defense_scatter, ball_scatter, title
    
    def update(frame_idx):
        """Update function for each frame"""
        current_frame = frames[frame_idx]
        
        # Filter to current frame
        frame_data = tracking_df[tracking_df['frameId'] == current_frame]
        
        # Separate by team
        offense = frame_data[frame_data['team'] == offense_team]
        defense = frame_data[frame_data['team'] == defense_team]
        ball = frame_data[frame_data['team'] == 'football']
        
        # Extract positions
        offense_positions = offense[['x', 'y']].values
        defense_positions = defense[['x', 'y']].values
        ball_position = ball[['x', 'y']].values
        
        # Update scatter plots
        offense_scatter.set_offsets(offense_positions)
        defense_scatter.set_offsets(defense_positions)
        ball_scatter.set_offsets(ball_position)
        
        # Update title
        title_parts = ['Play Animation']
        if game_id is not None:
            title_parts.append(f'GameId: {game_id}')
        if play_id is not None:
            title_parts.append(f'PlayId: {play_id}')
        title_parts.append(f'Frame {current_frame} / {frames[-1]}')
        title.set_text(' | '.join(title_parts))
        
        return offense_scatter, defense_scatter, ball_scatter, title
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(frames),
                        init_func=init, interval=interval, 
                        blit=True, repeat=repeat)
    
    return anim


def plot_random_frame_analysis(tracking_df, offense_team, defense_team, 
                                game_id=None, play_id=None, seed=None, 
                                figsize=(16, 9)):
    """
    Select and plot a random frame with detailed annotations for data quality validation.
    
    Parameters
    ----------
    tracking_df : pd.DataFrame
        Tracking data for a play (all frames)
    offense_team : str
        Offense team abbreviation
    defense_team : str
        Defense team abbreviation
    game_id : int, optional
        Game ID for title
    play_id : int, optional
        Play ID for title
    seed : int, optional
        Random seed for reproducibility
    figsize : tuple, optional
        Figure size
    
    Returns
    -------
    tuple of (fig, ax, frame_id, frame_data)
        Figure, axes, selected frame ID, and frame data
        
    Examples
    --------
    >>> fig, ax, frame_id, frame_data = plot_random_frame_analysis(
    ...     play_tracking, 'TB', 'DAL', seed=42)
    >>> print(f"Analyzed frame {frame_id}")
    >>> plt.show()
    """
    # Select random frame
    if seed is not None:
        np.random.seed(seed)
    
    available_frames = tracking_df['frameId'].unique()
    random_frame = np.random.choice(available_frames)
    
    # Filter to selected frame
    frame_data = tracking_df[tracking_df['frameId'] == random_frame].copy()
    
    # Separate teams
    offense = frame_data[frame_data['team'] == offense_team]
    defense = frame_data[frame_data['team'] == defense_team]
    ball = frame_data[frame_data['team'] == 'football']
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot entities
    ax.scatter(offense['x'], offense['y'], 
               c='red', s=250, marker='o', 
               edgecolors='darkred', linewidth=2.5,
               label=f'Offense ({offense_team})', zorder=3, alpha=0.8)
    
    ax.scatter(defense['x'], defense['y'], 
               c='blue', s=250, marker='s', 
               edgecolors='darkblue', linewidth=2.5,
               label=f'Defense ({defense_team})', zorder=3, alpha=0.8)
    
    ax.scatter(ball['x'], ball['y'], 
               c='brown', s=200, marker='D', 
               edgecolors='black', linewidth=3,
               label='Football', zorder=5)
    
    # Add jersey numbers
    for _, player in offense.iterrows():
        if not np.isnan(player['jerseyNumber']):
            ax.text(player['x'], player['y'], str(int(player['jerseyNumber'])), 
                    fontsize=9, ha='center', va='center', 
                    color='white', weight='bold', zorder=4)
    
    for _, player in defense.iterrows():
        if not np.isnan(player['jerseyNumber']):
            ax.text(player['x'], player['y'], str(int(player['jerseyNumber'])), 
                    fontsize=9, ha='center', va='center', 
                    color='white', weight='bold', zorder=4)
    
    # Configure axes
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)
    ax.set_xlabel('X Coordinate (yards) — Field Length [0=Endzone, 120=Endzone]', 
                  fontsize=13, weight='bold')
    ax.set_ylabel('Y Coordinate (yards) — Field Width [0=Sideline, 53.3=Sideline]', 
                  fontsize=13, weight='bold')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
    
    # Title
    title_parts = ['Random Frame Snapshot — Data Quality Validation']
    info_parts = []
    if game_id is not None:
        info_parts.append(f'GameId: {game_id}')
    if play_id is not None:
        info_parts.append(f'PlayId: {play_id}')
    info_parts.append(f'FrameId: {random_frame}')
    
    # Check for events
    events = frame_data['event'].dropna().unique()
    if len(events) > 0:
        info_parts.append(f'Event: {events[0]}')
    
    if info_parts:
        title_parts.append(' | '.join(info_parts))
    
    ax.set_title('\n'.join(title_parts), fontsize=14, weight='bold', pad=20)
    
    # Field markings
    ax.axhline(y=0, color='green', linewidth=4, alpha=0.5)
    ax.axhline(y=53.3, color='green', linewidth=4, alpha=0.5)
    ax.axvline(x=10, color='orange', linewidth=3, alpha=0.4, linestyle='--')
    ax.axvline(x=110, color='orange', linewidth=3, alpha=0.4, linestyle='--')
    
    # Legend
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95, 
              edgecolor='black', fancybox=True)
    
    plt.tight_layout()
    
    return fig, ax, random_frame, frame_data


def print_spatial_validation_summary(frame_data, offense_team, defense_team):
    """
    Print a detailed spatial validation analysis for a single frame.
    
    Analyzes:
    - Field boundary violations
    - Team formation statistics
    - Player clustering
    - Team separation
    - Outlier detection
    
    Parameters
    ----------
    frame_data : pd.DataFrame
        Tracking data for a single frame
    offense_team : str
        Offense team abbreviation
    defense_team : str
        Defense team abbreviation
    
    Returns
    -------
    dict
        Dictionary containing all validation metrics
        
    Examples
    --------
    >>> validation = print_spatial_validation_summary(frame_data, 'TB', 'DAL')
    >>> if not validation['field_valid']:
    ...     print("Warning: Boundary violations detected!")
    """
    print("=" * 80)
    print("SPATIAL VALIDATION ANALYSIS")
    print("=" * 80)
    
    # Separate teams
    offense = frame_data[frame_data['team'] == offense_team]
    defense = frame_data[frame_data['team'] == defense_team]
    ball = frame_data[frame_data['team'] == 'football']
    all_entities = pd.concat([offense, defense, ball])
    
    # Boundary checks
    x_violations = all_entities[(all_entities['x'] < 0) | (all_entities['x'] > 120)]
    y_violations = all_entities[(all_entities['y'] < 0) | (all_entities['y'] > 53.3)]
    
    print("\n1. FIELD BOUNDARY CHECKS:")
    print(f"   X-coordinate violations (outside 0-120): {len(x_violations)}")
    print(f"   Y-coordinate violations (outside 0-53.3): {len(y_violations)}")
    
    field_valid = len(x_violations) == 0 and len(y_violations) == 0
    if field_valid:
        print("   ✓ All entities within valid field boundaries")
    else:
        print("   ⚠️  WARNING: Entities detected outside field boundaries!")
    
    # Ball position
    ball_x = ball['x'].iloc[0] if len(ball) > 0 else None
    ball_y = ball['y'].iloc[0] if len(ball) > 0 else None
    
    print(f"\n2. BALL POSITION:")
    if ball_x is not None:
        print(f"   X = {ball_x:.2f} yards, Y = {ball_y:.2f} yards")
    else:
        print("   Ball not found in frame")
    
    # Offensive formation
    off_x_range = offense['x'].max() - offense['x'].min() if len(offense) > 0 else 0
    off_y_range = offense['y'].max() - offense['y'].min() if len(offense) > 0 else 0
    off_x_mean = offense['x'].mean() if len(offense) > 0 else 0
    off_x_std = offense['x'].std() if len(offense) > 0 else 0
    
    print(f"\n3. OFFENSIVE FORMATION ANALYSIS:")
    print(f"   X-axis spread: {off_x_range:.2f} yards (min: {offense['x'].min():.2f}, max: {offense['x'].max():.2f})")
    print(f"   Y-axis spread: {off_y_range:.2f} yards (formation width)")
    print(f"   Mean X position: {off_x_mean:.2f} ± {off_x_std:.2f} yards")
    
    # Defensive formation
    def_x_range = defense['x'].max() - defense['x'].min() if len(defense) > 0 else 0
    def_y_range = defense['y'].max() - defense['y'].min() if len(defense) > 0 else 0
    def_x_mean = defense['x'].mean() if len(defense) > 0 else 0
    def_x_std = defense['x'].std() if len(defense) > 0 else 0
    
    print(f"\n4. DEFENSIVE FORMATION ANALYSIS:")
    print(f"   X-axis spread: {def_x_range:.2f} yards (min: {defense['x'].min():.2f}, max: {defense['x'].max():.2f})")
    print(f"   Y-axis spread: {def_y_range:.2f} yards (coverage width)")
    print(f"   Mean X position: {def_x_mean:.2f} ± {def_x_std:.2f} yards")
    
    # Team separation
    separation = def_x_mean - off_x_mean
    print(f"\n5. TEAM SEPARATION:")
    print(f"   Average defensive depth behind offense: {separation:.2f} yards")
    
    if separation < -5:
        print("   ⚠️  SUSPICIOUS: Defense is ahead of offense (may indicate pass in flight)")
    elif separation > 30:
        print("   ⚠️  UNUSUAL: Very deep defensive alignment")
    else:
        print("   ✓ Reasonable defensive positioning")
    
    # Outlier detection
    print(f"\n6. OUTLIER DETECTION:")
    
    off_outliers = offense[np.abs(offense['x'] - off_x_mean) > 2.5 * off_x_std] if off_x_std > 0 else pd.DataFrame()
    if len(off_outliers) > 0:
        print(f"   Offensive outliers (X-axis): {len(off_outliers)} player(s)")
        for _, player in off_outliers.iterrows():
            dist = player['x'] - off_x_mean
            print(f"     - #{int(player['jerseyNumber'])} ({player['position']}): X={player['x']:.2f} ({dist:+.2f} from mean)")
    else:
        print(f"   Offensive outliers: None detected")
    
    def_outliers = defense[np.abs(defense['x'] - def_x_mean) > 2.5 * def_x_std] if def_x_std > 0 else pd.DataFrame()
    if len(def_outliers) > 0:
        print(f"   Defensive outliers (X-axis): {len(def_outliers)} player(s)")
        for _, player in def_outliers.iterrows():
            dist = player['x'] - def_x_mean
            print(f"     - #{int(player['jerseyNumber'])} ({player['position']}): X={player['x']:.2f} ({dist:+.2f} from mean)")
    else:
        print(f"   Defensive outliers: None detected")
    
    # Player clustering
    print(f"\n7. PLAYER CLUSTERING:")
    
    min_dist_off = float('inf')
    if len(offense) > 1:
        for i, p1 in offense.iterrows():
            for j, p2 in offense.iterrows():
                if i < j:
                    dist = np.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)
                    min_dist_off = min(min_dist_off, dist)
    
    min_dist_def = float('inf')
    if len(defense) > 1:
        for i, p1 in defense.iterrows():
            for j, p2 in defense.iterrows():
                if i < j:
                    dist = np.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)
                    min_dist_def = min(min_dist_def, dist)
    
    if min_dist_off < float('inf'):
        print(f"   Minimum offensive player separation: {min_dist_off:.2f} yards")
        if min_dist_off < 1.0:
            print(f"   ⚠️  WARNING: Offensive players extremely close (< 1 yard)")
        elif min_dist_off < 2.0:
            print(f"   ⚠️  NOTE: Tight offensive clustering (linemen likely)")
        else:
            print(f"   ✓ Reasonable offensive spacing")
    
    if min_dist_def < float('inf'):
        print(f"   Minimum defensive player separation: {min_dist_def:.2f} yards")
        if min_dist_def < 1.0:
            print(f"   ⚠️  WARNING: Defensive players extremely close (< 1 yard)")
        else:
            print(f"   ✓ Reasonable defensive spacing")
    
    print("\n" + "=" * 80)
    
    # Return summary dict
    return {
        'field_valid': field_valid,
        'x_violations': len(x_violations),
        'y_violations': len(y_violations),
        'ball_position': (ball_x, ball_y) if ball_x is not None else None,
        'offense_x_range': off_x_range,
        'offense_y_range': off_y_range,
        'offense_x_mean': off_x_mean,
        'defense_x_range': def_x_range,
        'defense_y_range': def_y_range,
        'defense_x_mean': def_x_mean,
        'team_separation': separation,
        'min_offense_separation': min_dist_off if min_dist_off < float('inf') else None,
        'min_defense_separation': min_dist_def if min_dist_def < float('inf') else None
    }


