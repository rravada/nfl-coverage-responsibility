"""
Data processing utilities for NFL tracking data analysis.

This module provides functions for filtering, normalizing, and preparing
NFL player tracking data for spatial and temporal analysis.
"""

import pandas as pd
import numpy as np


def filter_standard_passes(plays_df):
    """
    Filter plays dataframe to standard passing plays only.
    
    Excludes:
    - Non-pass plays (sacks, scrambles, runs)
    - Spikes (intentional clock management)
    - Trick plays (laterals, fumbles)
    - Prevent defense (desperation situations)
    
    Parameters
    ----------
    plays_df : pd.DataFrame
        The plays dataframe with columns: passResult, playDescription, pff_passCoverage
    
    Returns
    -------
    pd.DataFrame
        Filtered plays dataframe containing only standard passing plays
        
    Examples
    --------
    >>> standard_passes = filter_standard_passes(plays)
    >>> print(f"Found {len(standard_passes)} standard passing plays")
    """
    filtered = plays_df[
        # Must be a pass attempt (complete, incomplete, or intercepted)
        (plays_df['passResult'].isin(['C', 'I', 'IN'])) &
        # Exclude spikes
        (~plays_df['playDescription'].str.contains('spiked', case=False, na=False)) &
        # Exclude trick plays with laterals or fumbles
        (~plays_df['playDescription'].str.contains('lateral|FUMBLES', case=False, na=False, regex=True)) &
        # Exclude prevent defense (desperation situations)
        (plays_df['pff_passCoverage'] != 'Prevent')
    ].copy()
    
    return filtered


def filter_tracking_to_play(tracking_df, game_id, play_id):
    """
    Filter tracking data to a specific play.
    
    Parameters
    ----------
    tracking_df : pd.DataFrame
        The tracking dataframe with columns: gameId, playId, and all tracking fields
    game_id : int
        The game identifier
    play_id : int
        The play identifier within the game
    
    Returns
    -------
    pd.DataFrame
        Tracking data filtered to the specified play
        
    Examples
    --------
    >>> play_tracking = filter_tracking_to_play(tracking, 2021090900, 1267)
    >>> print(f"Play has {len(play_tracking)} rows across {play_tracking['frameId'].nunique()} frames")
    """
    play_data = tracking_df[
        (tracking_df['gameId'] == game_id) & 
        (tracking_df['playId'] == play_id)
    ].copy()
    
    return play_data


def normalize_play_direction(tracking_df):
    """
    Normalize tracking data so offense always moves left-to-right (increasing X).
    
    In the raw data, plays can go either direction depending on which endzone
    the offense is attacking. This function transforms all plays to move toward
    the right endzone (X increasing toward 120).
    
    For plays going left (playDirection='left'):
    - Flip X: x_new = 120 - x_old
    - Flip Y: y_new = 53.3 - y_old  
    - Flip angles: angle_new = (angle_old + 180) % 360
    
    For plays going right: no change needed (already left-to-right)
    
    Parameters
    ----------
    tracking_df : pd.DataFrame
        Tracking dataframe with columns: playDirection, x, y, o, dir
    
    Returns
    -------
    pd.DataFrame
        Normalized tracking dataframe with consistent left-to-right orientation
        
    Notes
    -----
    - Modifies x, y, o (orientation), and dir (direction) columns
    - Updates playDirection to 'right' for all rows
    - Preserves all other columns unchanged
    
    Examples
    --------
    >>> play_normalized = normalize_play_direction(play_tracking)
    >>> print(f"Direction: {play_normalized['playDirection'].iloc[0]}")
    >>> # Output: Direction: right
    """
    df_normalized = tracking_df.copy()
    
    # Identify plays going left (need flipping)
    going_left = df_normalized['playDirection'] == 'left'
    
    # Flip spatial coordinates
    df_normalized.loc[going_left, 'x'] = 120 - df_normalized.loc[going_left, 'x']
    df_normalized.loc[going_left, 'y'] = 53.3 - df_normalized.loc[going_left, 'y']
    
    # Flip orientation and direction angles
    df_normalized.loc[going_left, 'o'] = (df_normalized.loc[going_left, 'o'] + 180) % 360
    df_normalized.loc[going_left, 'dir'] = (df_normalized.loc[going_left, 'dir'] + 180) % 360
    
    # Update playDirection to 'right' for consistency
    df_normalized.loc[going_left, 'playDirection'] = 'right'
    
    return df_normalized


def get_play_details(plays_df, game_id, play_id):
    """
    Retrieve metadata for a specific play.
    
    Parameters
    ----------
    plays_df : pd.DataFrame
        The plays dataframe
    game_id : int
        The game identifier
    play_id : int
        The play identifier
    
    Returns
    -------
    pd.Series
        Series containing all play metadata fields for the specified play
        
    Examples
    --------
    >>> play_info = get_play_details(plays, 2021090900, 1267)
    >>> print(f"Pass result: {play_info['passResult']}")
    >>> print(f"Description: {play_info['playDescription']}")
    """
    play_details = plays_df[
        (plays_df['gameId'] == game_id) & 
        (plays_df['playId'] == play_id)
    ]
    
    if len(play_details) == 0:
        raise ValueError(f"No play found for gameId={game_id}, playId={play_id}")
    
    return play_details.iloc[0]


def get_frame_structure_summary(tracking_df):
    """
    Analyze the frame structure of tracking data for a play.
    
    Provides statistics on number of frames, entities per frame, and
    distribution of players/ball across frames.
    
    Parameters
    ----------
    tracking_df : pd.DataFrame
        Tracking data for a single play (already filtered to gameId/playId)
    
    Returns
    -------
    dict
        Dictionary with keys:
        - num_frames: Total unique frames
        - mean_entities: Average entities per frame
        - min_entities: Minimum entities per frame
        - max_entities: Maximum entities per frame
        - mode_entities: Most common number of entities per frame
        - frame_distribution: pd.Series with entity counts per frame
        
    Examples
    --------
    >>> summary = get_frame_structure_summary(play_tracking)
    >>> print(f"Play has {summary['num_frames']} frames")
    >>> print(f"Typically {summary['mode_entities']} entities per frame")
    """
    entities_per_frame = tracking_df.groupby('frameId').size()
    
    summary = {
        'num_frames': tracking_df['frameId'].nunique(),
        'mean_entities': entities_per_frame.mean(),
        'min_entities': entities_per_frame.min(),
        'max_entities': entities_per_frame.max(),
        'mode_entities': entities_per_frame.mode()[0] if len(entities_per_frame.mode()) > 0 else None,
        'frame_distribution': entities_per_frame.value_counts().sort_index()
    }
    
    return summary


def separate_teams(tracking_df, offense_team, defense_team):
    """
    Separate tracking data by team and ball.
    
    Parameters
    ----------
    tracking_df : pd.DataFrame
        Tracking data with 'team' column
    offense_team : str
        Team abbreviation for offense (e.g., 'TB', 'DAL')
    defense_team : str
        Team abbreviation for defense
    
    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame, pd.DataFrame)
        (offense_df, defense_df, ball_df)
        
    Examples
    --------
    >>> offense, defense, ball = separate_teams(frame_data, 'TB', 'DAL')
    >>> print(f"Offense: {len(offense)}, Defense: {len(defense)}, Ball: {len(ball)}")
    """
    offense = tracking_df[tracking_df['team'] == offense_team].copy()
    defense = tracking_df[tracking_df['team'] == defense_team].copy()
    ball = tracking_df[tracking_df['team'] == 'football'].copy()
    
    return offense, defense, ball


def validate_field_boundaries(tracking_df):
    """
    Check if all entities are within valid field boundaries.
    
    Valid boundaries:
    - X: 0 to 120 yards (endzone to endzone)
    - Y: 0 to 53.3 yards (sideline to sideline)
    
    Parameters
    ----------
    tracking_df : pd.DataFrame
        Tracking data with x and y columns
    
    Returns
    -------
    dict
        Dictionary with keys:
        - x_violations: DataFrame of rows with invalid X coordinates
        - y_violations: DataFrame of rows with invalid Y coordinates
        - is_valid: Boolean indicating if all coordinates are valid
        
    Examples
    --------
    >>> validation = validate_field_boundaries(play_tracking)
    >>> if validation['is_valid']:
    ...     print("All entities within field boundaries")
    ... else:
    ...     print(f"Found {len(validation['x_violations'])} X violations")
    """
    x_violations = tracking_df[(tracking_df['x'] < 0) | (tracking_df['x'] > 120)].copy()
    y_violations = tracking_df[(tracking_df['y'] < 0) | (tracking_df['y'] > 53.3)].copy()
    
    result = {
        'x_violations': x_violations,
        'y_violations': y_violations,
        'is_valid': len(x_violations) == 0 and len(y_violations) == 0
    }
    
    return result


def calculate_team_statistics(team_df):
    """
    Calculate spatial statistics for a team's positions at a given moment.
    
    Parameters
    ----------
    team_df : pd.DataFrame
        Tracking data for one team at one or more frames (must have x and y columns)
    
    Returns
    -------
    dict
        Dictionary with keys:
        - x_min, x_max, x_mean, x_std: X-axis statistics
        - y_min, y_max, y_mean, y_std: Y-axis statistics
        - x_range: Horizontal spread (max - min)
        - y_range: Vertical spread (max - min)
        
    Examples
    --------
    >>> offense_stats = calculate_team_statistics(offense)
    >>> print(f"Offensive spread: {offense_stats['x_range']:.1f} yards")
    >>> print(f"Mean position: X={offense_stats['x_mean']:.1f}, Y={offense_stats['y_mean']:.1f}")
    """
    if len(team_df) == 0:
        return None
    
    stats = {
        'x_min': team_df['x'].min(),
        'x_max': team_df['x'].max(),
        'x_mean': team_df['x'].mean(),
        'x_std': team_df['x'].std(),
        'y_min': team_df['y'].min(),
        'y_max': team_df['y'].max(),
        'y_mean': team_df['y'].mean(),
        'y_std': team_df['y'].std(),
        'x_range': team_df['x'].max() - team_df['x'].min(),
        'y_range': team_df['y'].max() - team_df['y'].min()
    }
    
    return stats


def find_minimum_player_separation(team_df):
    """
    Find the minimum distance between any two players on a team.
    
    Useful for detecting unrealistic clustering or tracking errors.
    
    Parameters
    ----------
    team_df : pd.DataFrame
        Tracking data for one team (must have x and y columns)
    
    Returns
    -------
    float
        Minimum Euclidean distance between any two players (in yards)
        Returns inf if fewer than 2 players
        
    Examples
    --------
    >>> min_sep = find_minimum_player_separation(offense)
    >>> if min_sep < 1.0:
    ...     print("Warning: Players extremely close (possible tracking error)")
    """
    if len(team_df) < 2:
        return float('inf')
    
    positions = team_df[['x', 'y']].values
    min_dist = float('inf')
    
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dist = np.sqrt(
                (positions[i][0] - positions[j][0])**2 + 
                (positions[i][1] - positions[j][1])**2
            )
            min_dist = min(min_dist, dist)
    
    return min_dist


