"""
Deterministic, explainable coverage assignment engine for NFL tracking data.

Uses Voronoi diagrams for zone coverage and motion correlation for man coverage
to assign each receiver to a responsible defender at snap (zone) or over route (man).
"""

import pandas as pd
import numpy as np
from scipy.spatial import Voronoi

from src.data_processing import (
    filter_tracking_to_play,
    normalize_play_direction,
    separate_teams,
)


# Offensive skill positions (exclude QB, OL)
SKILL_POSITIONS = {'WR', 'TE', 'RB'}

# Coverage defenders (DBs and LBs; exclude pure pass rushers unless in coverage)
COVERAGE_POSITIONS = {'CB', 'SS', 'FS', 'LB', 'MLB', 'OLB'}

# Pass rush positions (excluded unless depth > threshold)
PASS_RUSH_POSITIONS = {'DE', 'DT'}

# Depth (yards off line) above which DE/DT are considered in coverage
COVERAGE_DEPTH_YARDS = 3.0


def _get_xy_series(nflid, tracking_df, frames_in_range):
    """Build a per-frame (x, y) series for one player, reindexed to frames_in_range.

    Missing frames are forward/back-filled, and any remaining gaps are imputed
    with the column mean so downstream velocity/correlation math always sees
    a fully populated series.
    """
    sub = (
        tracking_df[tracking_df['nflId'] == nflid][['frameId', 'x', 'y']]
        .drop_duplicates('frameId')
        .set_index('frameId')
    )
    sub = sub.reindex(frames_in_range).sort_index().ffill().bfill()
    if sub.isna().any().any():
        sub = sub.fillna(sub.mean())
    return sub


def get_snap_frame(tracking_df):
    """
    Find the frameId where event == "ball_snap".

    Parameters
    ----------
    tracking_df : pd.DataFrame
        Tracking data for a single play with columns: frameId, event

    Returns
    -------
    int or None
        The frameId of the ball snap, or None if not found.

    Examples
    --------
    >>> snap = get_snap_frame(play_tracking)
    >>> if snap is not None:
    ...     frame_data = play_tracking[play_tracking['frameId'] == snap]
    """
    try:
        snap_rows = tracking_df[tracking_df['event'] == 'ball_snap']
        if len(snap_rows) == 0:
            return None
        return int(snap_rows['frameId'].iloc[0])
    except (KeyError, IndexError):
        return None


def get_pass_frame(tracking_df):
    """
    Find the frameId where event == "pass_forward".

    Parameters
    ----------
    tracking_df : pd.DataFrame
        Tracking data for a single play with columns: frameId, event

    Returns
    -------
    int or None
        The frameId of pass forward, or None if not found (e.g. scrambles).

    Examples
    --------
    >>> pass_f = get_pass_frame(play_tracking)
    >>> if pass_f is None:
    ...     print("No pass event (scramble or sack)")
    """
    try:
        pass_rows = tracking_df[tracking_df['event'] == 'pass_forward']
        if len(pass_rows) == 0:
            return None
        return int(pass_rows['frameId'].iloc[0])
    except (KeyError, IndexError):
        return None


def get_skill_players(frame_df, team):
    """
    From a single frame, extract offensive skill positions (WR, TE, RB) for the given team.

    Excludes QB and OL (C, G, T). If 'position' column is missing, returns all
    offensive players for that team except the ball (caller should merge position
    from roster for accurate filtering).

    Parameters
    ----------
    frame_df : pd.DataFrame
        Single-frame tracking data with columns: team, and optionally position
    team : str
        Team abbreviation (e.g. 'TB', 'DAL')

    Returns
    -------
    pd.DataFrame
        Filtered rows for skill players only.

    Examples
    --------
    >>> receivers = get_skill_players(snap_frame_data, 'TB')
    >>> print(receivers[['jerseyNumber', 'position', 'x', 'y']])
    """
    offense = frame_df[frame_df['team'] == team].copy()
    offense = offense[offense['team'] != 'football']
    if len(offense) == 0:
        return offense
    if 'position' not in offense.columns:
        return offense
    return offense[offense['position'].str.upper().str.strip().isin(SKILL_POSITIONS)].copy()


def get_coverage_defenders(frame_df, team, ball_x=None):
    """
    Extract defensive backs and linebackers (CB, SS, FS, LB, MLB, OLB).

    Excludes pure pass rushers (DE, DT) unless they are clearly in coverage
    (depth > COVERAGE_DEPTH_YARDS off the line). Line of scrimmage is
    approximated by the minimum defender x if ball_x is not provided.

    Parameters
    ----------
    frame_df : pd.DataFrame
        Single-frame tracking data with columns: team, x, and optionally position
    team : str
        Defensive team abbreviation
    ball_x : float, optional
        Ball x at snap (line of scrimmage). If None, use min defender x.

    Returns
    -------
    pd.DataFrame
        Filtered rows for coverage defenders only.

    Examples
    --------
    >>> defenders = get_coverage_defenders(snap_frame_data, 'DAL', ball_x=37.0)
    """
    defense = frame_df[frame_df['team'] == team].copy()
    if len(defense) == 0:
        return defense
    if 'position' not in defense.columns:
        return defense

    los = ball_x if ball_x is not None else defense['x'].min()
    pos_upper = defense['position'].str.upper().str.strip()

    # Include all DB/LB
    in_coverage_pos = pos_upper.isin(COVERAGE_POSITIONS)

    # DE/DT only if depth > threshold (yards behind LOS)
    depth = defense['x'] - los
    rush_pos = pos_upper.isin(PASS_RUSH_POSITIONS)
    in_coverage_depth = rush_pos & (depth > COVERAGE_DEPTH_YARDS)

    return defense[in_coverage_pos | in_coverage_depth].copy()


def compute_voronoi_assignment(receivers_df, defenders_df):
    """
    At snap, compute which defender is geometrically responsible for each receiver
    using Voronoi diagram of defender positions. Receivers outside the convex
    hull of defenders are assigned to nearest defender by Euclidean distance.

    Parameters
    ----------
    receivers_df : pd.DataFrame
        Skill players at snap with columns: nflId, jerseyNumber, position, x, y
    defenders_df : pd.DataFrame
        Coverage defenders at snap with same columns

    Returns
    -------
    pd.DataFrame
        One row per receiver with columns:
        receiver_nflId, receiver_jersey, receiver_position, receiver_x, receiver_y,
        assigned_defender_nflId, assigned_defender_jersey, assigned_defender_position,
        assigned_defender_x, assigned_defender_y, assignment_method, cushion_distance

    Examples
    --------
    >>> assignments = compute_voronoi_assignment(receivers, defenders)
    >>> print(assignments[['receiver_jersey', 'assigned_defender_jersey', 'cushion_distance']])
    """
    if len(receivers_df) == 0 or len(defenders_df) == 0:
        return pd.DataFrame()

    def_xy = defenders_df[['x', 'y']].values.astype(np.float64)
    rec_xy = receivers_df[['x', 'y']].values.astype(np.float64)

    # Build Voronoi on defenders
    vor = Voronoi(def_xy)

    from matplotlib.path import Path

    def point_in_region(pt, region_vertices, vor):
        if -1 in region_vertices or len(region_vertices) < 3:
            return False
        verts = vor.vertices[region_vertices]
        path = Path(verts)
        return path.contains_point(pt)

    receiver_rows = []
    for r in range(len(rec_xy)):
        pt = rec_xy[r]
        rec_row = receivers_df.iloc[r]
        assigned_idx = None
        method = 'nearest_neighbor'

        for d in range(len(def_xy)):
            reg_idx = vor.point_region[d]
            region = vor.regions[reg_idx]
            if -1 in region or len(region) < 3:
                continue
            if point_in_region(pt, region, vor):
                assigned_idx = d
                method = 'voronoi'
                break

        if assigned_idx is None:
            # Outside convex hull: assign nearest defender by Euclidean distance
            dists = np.sqrt(np.sum((def_xy - pt) ** 2, axis=1))
            assigned_idx = int(np.argmin(dists))
            method = 'nearest_neighbor'

        def_row = defenders_df.iloc[assigned_idx]
        cushion = np.sqrt((rec_row['x'] - def_row['x'])**2 + (rec_row['y'] - def_row['y'])**2)

        receiver_rows.append({
            'receiver_nflId': rec_row.get('nflId', np.nan),
            'receiver_jersey': rec_row.get('jerseyNumber', np.nan),
            'receiver_position': rec_row.get('position', 'Unknown'),
            'receiver_x': rec_row['x'],
            'receiver_y': rec_row['y'],
            'assigned_defender_nflId': def_row.get('nflId', np.nan),
            'assigned_defender_jersey': def_row.get('jerseyNumber', np.nan),
            'assigned_defender_position': def_row.get('position', 'Unknown'),
            'assigned_defender_x': def_row['x'],
            'assigned_defender_y': def_row['y'],
            'assignment_method': method,
            'cushion_distance': float(cushion),
        })

    return pd.DataFrame(receiver_rows)


def compute_man_coverage_assignment(receivers_df, defenders_df, tracking_df, snap_frame, pass_frame):
    """
    For man coverage, assign each receiver to the defender who most closely mirrors
    their route between snap and pass. Uses motion correlation (60%) and average
    distance (40%).

    Parameters
    ----------
    receivers_df : pd.DataFrame
        Skill players at snap (columns: nflId, jerseyNumber, position, x, y)
    defenders_df : pd.DataFrame
        Coverage defenders at snap (same columns)
    tracking_df : pd.DataFrame
        Full play tracking (all frames) with frameId, nflId, x, y
    snap_frame : int
        Frame ID of ball snap
    pass_frame : int or None
        Frame ID of pass_forward; if None, use last frame in tracking_df

    Returns
    -------
    pd.DataFrame
        Same schema as compute_voronoi_assignment plus: avg_distance,
        motion_correlation, composite_score
    """
    if len(receivers_df) == 0 or len(defenders_df) == 0:
        return pd.DataFrame()

    if pass_frame is None:
        pass_frame = int(tracking_df['frameId'].max())
    frames = sorted(tracking_df['frameId'].unique())
    frames_in_range = [f for f in frames if snap_frame <= f <= pass_frame]
    if len(frames_in_range) < 2:
        # Not enough frames for correlation; fall back to Voronoi
        out = compute_voronoi_assignment(receivers_df, defenders_df)
        out['avg_distance'] = np.nan
        out['motion_correlation'] = np.nan
        out['composite_score'] = np.nan
        return out

    rec_nflids = receivers_df['nflId'].dropna().astype(int).tolist()
    def_nflids = defenders_df['nflId'].dropna().astype(int).tolist()

    if not rec_nflids or not def_nflids:
        out = compute_voronoi_assignment(receivers_df, defenders_df)
        out['avg_distance'] = np.nan
        out['motion_correlation'] = np.nan
        out['composite_score'] = np.nan
        return out

    rec_series = {}
    for n in rec_nflids:
        try:
            rec_series[n] = _get_xy_series(n, tracking_df, frames_in_range)
        except Exception:
            continue
    def_series = {}
    for n in def_nflids:
        try:
            def_series[n] = _get_xy_series(n, tracking_df, frames_in_range)
        except Exception:
            continue

    # Velocity: diff in x and y between consecutive frames
    def velocities(series_dict, _fallback=None):
        out = {}
        for n, df in series_dict.items():
            if df is None or len(df) < 2:
                out[n] = (np.zeros(len(frames_in_range)), np.zeros(len(frames_in_range)))
                continue
            dx = df['x'].diff().values
            dy = df['y'].diff().values
            dx = np.nan_to_num(dx, nan=0.0, posinf=0.0, neginf=0.0)
            dy = np.nan_to_num(dy, nan=0.0, posinf=0.0, neginf=0.0)
            if len(dx) > 1 and (dx[0] == 0 and dy[0] == 0):
                dx[0] = dx[1]
                dy[0] = dy[1]
            out[n] = (dx, dy)
        return out

    rec_vel = velocities(rec_series, None)
    def_vel = velocities(def_series, None)

    def pearson_corr(a, b):
        if np.std(a) < 1e-10 or np.std(b) < 1e-10:
            return 0.0
        return np.corrcoef(a, b)[0, 1]

    def avg_dist(rec_nfl, def_nfl):
        r = rec_series.get(rec_nfl)
        d = def_series.get(def_nfl)
        if r is None or d is None or len(r) == 0 or len(d) == 0:
            return np.nan
        return np.sqrt((r['x'].values - d['x'].values)**2 + (r['y'].values - d['y'].values)**2).mean()

    def motion_corr(rec_nfl, def_nfl):
        rv = rec_vel.get(rec_nfl, (np.zeros(len(frames_in_range)), np.zeros(len(frames_in_range))))
        dv = def_vel.get(def_nfl, (np.zeros(len(frames_in_range)), np.zeros(len(frames_in_range))))
        cx = pearson_corr(rv[0], dv[0])
        cy = pearson_corr(rv[1], dv[1])
        return (cx + cy) / 2.0

    receiver_rows = []
    for r_idx, rec_row in receivers_df.iterrows():
        rec_nfl = rec_row.get('nflId')
        if pd.isna(rec_nfl):
            continue
        rec_nfl = int(rec_nfl)
        best_def_idx = None
        best_score = -np.inf
        best_avg_dist = np.nan
        best_corr = np.nan

        for d_idx, def_row in defenders_df.iterrows():
            def_nfl = def_row.get('nflId')
            if pd.isna(def_nfl):
                continue
            def_nfl = int(def_nfl)
            avg_d = avg_dist(rec_nfl, def_nfl)
            corr = motion_corr(rec_nfl, def_nfl)
            if np.isnan(avg_d):
                continue
            # Lower distance is better; higher correlation is better. Normalize and combine.
            # Composite: 0.6 * correlation (in [-1,1]) + 0.4 * (1 - normalized_dist)
            # For distance use inverse: e.g. 1 / (1 + avg_d) so smaller dist = higher score
            dist_score = 1.0 / (1.0 + avg_d)  # 0 to 1
            corr_norm = (corr + 1) / 2  # 0 to 1
            composite = 0.6 * corr_norm + 0.4 * dist_score
            if composite > best_score:
                best_score = composite
                best_def_idx = d_idx
                best_avg_dist = avg_d
                best_corr = corr

        if best_def_idx is None:
            continue
        def_row = defenders_df.loc[best_def_idx]
        cushion = np.sqrt((rec_row['x'] - def_row['x'])**2 + (rec_row['y'] - def_row['y'])**2)

        receiver_rows.append({
            'receiver_nflId': rec_row.get('nflId', np.nan),
            'receiver_jersey': rec_row.get('jerseyNumber', np.nan),
            'receiver_position': rec_row.get('position', 'Unknown'),
            'receiver_x': rec_row['x'],
            'receiver_y': rec_row['y'],
            'assigned_defender_nflId': def_row.get('nflId', np.nan),
            'assigned_defender_jersey': def_row.get('jerseyNumber', np.nan),
            'assigned_defender_position': def_row.get('position', 'Unknown'),
            'assigned_defender_x': def_row['x'],
            'assigned_defender_y': def_row['y'],
            'assignment_method': 'man_motion',
            'cushion_distance': float(cushion),
            'avg_distance': best_avg_dist,
            'motion_correlation': best_corr,
            'composite_score': best_score,
        })

    return pd.DataFrame(receiver_rows)


def compute_man_correlation_matrix(receivers_df, defenders_df, tracking_df, snap_frame, pass_frame):
    """
    Compute the motion correlation matrix (receivers x defenders) between
    snap and pass for use in heatmaps. Same logic as in compute_man_coverage_assignment.

    Parameters
    ----------
    receivers_df : pd.DataFrame
        Skill players at snap (must have nflId, x, y)
    defenders_df : pd.DataFrame
        Coverage defenders at snap
    tracking_df : pd.DataFrame
        Full play tracking
    snap_frame : int
    pass_frame : int or None

    Returns
    -------
    tuple of (pd.DataFrame, np.ndarray)
        (assignments_df from compute_man_coverage_assignment, 2D array shape (n_receivers, n_defenders))
    """
    assignments = compute_man_coverage_assignment(
        receivers_df, defenders_df, tracking_df, snap_frame, pass_frame
    )
    if len(receivers_df) == 0 or len(defenders_df) == 0:
        return assignments, np.array([[]])

    pass_frame = pass_frame or int(tracking_df['frameId'].max())
    frames = sorted(tracking_df['frameId'].unique())
    frames_in_range = [f for f in frames if snap_frame <= f <= pass_frame]
    if len(frames_in_range) < 2:
        return assignments, np.zeros((len(receivers_df), len(defenders_df)))

    rec_nflids = receivers_df['nflId'].dropna().astype(int).tolist()
    def_nflids = defenders_df['nflId'].dropna().astype(int).tolist()

    rec_series = {n: _get_xy_series(n, tracking_df, frames_in_range) for n in rec_nflids}
    def_series = {n: _get_xy_series(n, tracking_df, frames_in_range) for n in def_nflids}

    def velocities(series_dict):
        out = {}
        for n, df in series_dict.items():
            if df is None or len(df) < 2:
                out[n] = (np.zeros(len(frames_in_range)), np.zeros(len(frames_in_range)))
                continue
            dx = np.nan_to_num(df['x'].diff().values, nan=0.0)
            dy = np.nan_to_num(df['y'].diff().values, nan=0.0)
            if len(dx) > 1 and dx[0] == 0 and dy[0] == 0:
                dx[0], dy[0] = dx[1], dy[1]
            out[n] = (dx, dy)
        return out

    rec_vel = velocities(rec_series)
    def_vel = velocities(def_series)

    def pearson_corr(a, b):
        if np.std(a) < 1e-10 or np.std(b) < 1e-10:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    def motion_corr(rec_nfl, def_nfl):
        rv = rec_vel.get(rec_nfl, (np.zeros(len(frames_in_range)), np.zeros(len(frames_in_range))))
        dv = def_vel.get(def_nfl, (np.zeros(len(frames_in_range)), np.zeros(len(frames_in_range))))
        return (pearson_corr(rv[0], dv[0]) + pearson_corr(rv[1], dv[1])) / 2.0

    n_r, n_d = len(receivers_df), len(defenders_df)
    mat = np.zeros((n_r, n_d))
    for i, rec_nfl in enumerate(rec_nflids):
        for j, def_nfl in enumerate(def_nflids):
            mat[i, j] = motion_corr(rec_nfl, def_nfl)
    return assignments, mat


def assign_coverage_responsibility(tracking_df, plays_df, game_id, play_id):
    """
    Master function: filter play, normalize, get snap/pass frame, then assign
    coverage using Voronoi (zone) or motion correlation (man) based on
    pff_passCoverageType from plays_df.

    Parameters
    ----------
    tracking_df : pd.DataFrame
        Full tracking data (e.g. week1)
    plays_df : pd.DataFrame
        Plays metadata with possessionTeam, defensiveTeam, pff_passCoverage,
        pff_passCoverageType
    game_id : int
        Game identifier
    play_id : int
        Play identifier

    Returns
    -------
    dict or None
        Keys: game_id, play_id, coverage_type, coverage_scheme, assignments_df,
        snap_frame, pass_frame, offense_team, defense_team. Returns None on failure.
    """
    try:
        play_tracking = filter_tracking_to_play(tracking_df, game_id, play_id)
        if len(play_tracking) == 0:
            return None
        play_tracking = normalize_play_direction(play_tracking)

        snap_frame = get_snap_frame(play_tracking)
        pass_frame = get_pass_frame(play_tracking)
        if snap_frame is None:
            return None

        play_meta = plays_df[(plays_df['gameId'] == game_id) & (plays_df['playId'] == play_id)]
        if len(play_meta) == 0:
            return None
        play_meta = play_meta.iloc[0]
        offense_team = play_meta['possessionTeam']
        defense_team = play_meta['defensiveTeam']
        coverage_scheme = play_meta.get('pff_passCoverage', None)
        coverage_type = play_meta.get('pff_passCoverageType', None)

        snap_df = play_tracking[play_tracking['frameId'] == snap_frame]
        ball = snap_df[snap_df['team'] == 'football']
        ball_x = float(ball['x'].iloc[0]) if len(ball) > 0 else None

        receivers = get_skill_players(snap_df, offense_team)
        defenders = get_coverage_defenders(snap_df, defense_team, ball_x=ball_x)

        if len(receivers) == 0 or len(defenders) == 0:
            return {
                'game_id': game_id,
                'play_id': play_id,
                'coverage_type': coverage_type,
                'coverage_scheme': coverage_scheme,
                'assignments_df': pd.DataFrame(),
                'snap_frame': snap_frame,
                'pass_frame': pass_frame,
                'offense_team': offense_team,
                'defense_team': defense_team,
            }

        if coverage_type == 'Zone':
            assignments_df = compute_voronoi_assignment(receivers, defenders)
        elif coverage_type == 'Man':
            assignments_df = compute_man_coverage_assignment(
                receivers, defenders, play_tracking, snap_frame, pass_frame
            )
        else:
            # NaN or unknown: run both, return Voronoi as default
            assignments_df = compute_voronoi_assignment(receivers, defenders)
            if assignments_df is not None and len(assignments_df) > 0:
                pass  # keep voronoi
            else:
                assignments_df = compute_man_coverage_assignment(
                    receivers, defenders, play_tracking, snap_frame, pass_frame
                )

        return {
            'game_id': game_id,
            'play_id': play_id,
            'coverage_type': coverage_type,
            'coverage_scheme': coverage_scheme,
            'assignments_df': assignments_df,
            'snap_frame': snap_frame,
            'pass_frame': pass_frame,
            'offense_team': offense_team,
            'defense_team': defense_team,
        }
    except Exception:
        return None
