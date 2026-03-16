"""
Feature engineering for coverage responsibility ML phase.

Extracts pre-snap alignment and coverage structure features from tracking data
for use in Man/Zone and coverage scheme classification.
"""

import pandas as pd
import numpy as np

from src.data_processing import filter_tracking_to_play, normalize_play_direction
from src.coverage_assignment import (
    get_snap_frame,
    get_skill_players,
    get_coverage_defenders,
)


# Box width (y) for "in box" defenders: between the tackles ~ 53.3/2 ± 5 yards
TACKLES_Y_HALF = 5.0  # yards from center (26.65) each way
FIELD_WIDTH = 53.3
CENTER_Y = FIELD_WIDTH / 2
BOX_Y_LO = CENTER_Y - TACKLES_Y_HALF
BOX_Y_HI = CENTER_Y + TACKLES_Y_HALF
IN_BOX_X_YARDS = 5.0  # within 5 yards of LOS
HIGH_SAFETY_X_YARDS = 8.0  # safeties with x > ball_x + 8


def extract_presnap_features(tracking_df, plays_df, game_id, play_id):
    """
    At snap frame, compute features describing the defensive alignment.

    Parameters
    ----------
    tracking_df : pd.DataFrame
        Full tracking data (e.g. week1) with frameId, team, x, y, event
    plays_df : pd.DataFrame
        Plays metadata with gameId, playId, possessionTeam, defensiveTeam
    game_id : int
        Game identifier
    play_id : int
        Play identifier

    Returns
    -------
    dict or None
        Flat dict of features (one row in feature matrix). Keys: avg_cushion,
        defender_depth, defenders_in_box, coverage_width, num_high_safeties,
        cb_alignment, safety_depth, plus game_id, play_id. Returns None on failure.

    Examples
    --------
    >>> feats = extract_presnap_features(tracking, plays, 2021090900, 97)
    >>> if feats:
    ...     print(f"avg_cushion={feats['avg_cushion']:.2f}")
    """
    try:
        play_tracking = filter_tracking_to_play(tracking_df, game_id, play_id)
        if len(play_tracking) == 0:
            return None
        play_tracking = normalize_play_direction(play_tracking)

        snap_frame = get_snap_frame(play_tracking)
        if snap_frame is None:
            return None

        play_meta = plays_df[(plays_df['gameId'] == game_id) & (plays_df['playId'] == play_id)]
        if len(play_meta) == 0:
            return None
        play_meta = play_meta.iloc[0]
        offense_team = play_meta['possessionTeam']
        defense_team = play_meta['defensiveTeam']

        snap_df = play_tracking[play_tracking['frameId'] == snap_frame]
        ball = snap_df[snap_df['team'] == 'football']
        if len(ball) == 0:
            return None
        ball_x = float(ball['x'].iloc[0])
        ball_y = float(ball['y'].iloc[0])

        receivers = get_skill_players(snap_df, offense_team)
        defenders = get_coverage_defenders(snap_df, defense_team, ball_x=ball_x)

        if len(receivers) == 0 or len(defenders) == 0:
            return None

        rec_xy = receivers[['x', 'y']].values
        def_xy = defenders[['x', 'y']].values

        # avg_cushion: mean distance from each CB/safety to nearest WR (or all coverage defenders to nearest WR)
        from scipy.spatial import cKDTree
        wr_tree = cKDTree(rec_xy)
        dists, _ = wr_tree.query(def_xy, k=1)
        avg_cushion = float(np.mean(dists))

        # defender_depth: mean yards defenders are behind line of scrimmage (ball_x)
        defender_depth = float(np.mean(defenders['x'] - ball_x))

        # defenders_in_box: count defenders within 5 yards of LOS and between the tackles (y in [BOX_Y_LO, BOX_Y_HI])
        in_box = (
            (np.abs(defenders['x'].values - ball_x) <= IN_BOX_X_YARDS) &
            (defenders['y'].values >= BOX_Y_LO) &
            (defenders['y'].values <= BOX_Y_HI)
        )
        defenders_in_box = int(np.sum(in_box))

        # coverage_width: y-axis spread of coverage defenders
        coverage_width = float(defenders['y'].max() - defenders['y'].min()) if len(defenders) > 0 else 0.0

        # num_high_safeties: safeties with x > ball_x + 8
        if 'position' in defenders.columns:
            pos = defenders['position'].str.upper().str.strip()
            safeties = defenders[pos.isin({'SS', 'FS'})]
            num_high_safeties = int(np.sum(safeties['x'].values > ball_x + HIGH_SAFETY_X_YARDS))
            safety_depth = float(safeties['x'].mean()) if len(safeties) > 0 else np.nan
        else:
            num_high_safeties = 0
            safety_depth = np.nan

        # cb_alignment: mean lateral (y) distance between each CB and their nearest WR
        if 'position' in defenders.columns:
            pos = defenders['position'].str.upper().str.strip()
            cbs = defenders[pos.isin({'CB'})]
            if len(cbs) > 0 and len(receivers) > 0:
                cb_xy = cbs[['x', 'y']].values
                _, idx = wr_tree.query(cb_xy, k=1)
                nearest_wr_y = rec_xy[idx, 1]
                cb_alignment = float(np.mean(np.abs(cb_xy[:, 1] - nearest_wr_y)))
            else:
                cb_alignment = np.nan
        else:
            # no position: use all defenders' lateral distance to nearest WR
            _, idx = wr_tree.query(def_xy, k=1)
            nearest_wr_y = rec_xy[idx, 1]
            cb_alignment = float(np.mean(np.abs(def_xy[:, 1] - nearest_wr_y)))

        out = {
            'game_id': game_id,
            'play_id': play_id,
            'avg_cushion': avg_cushion,
            'defender_depth': defender_depth,
            'defenders_in_box': defenders_in_box,
            'coverage_width': coverage_width,
            'num_high_safeties': num_high_safeties,
            'cb_alignment': cb_alignment,
            'safety_depth': safety_depth,
        }
        return out
    except Exception:
        return None


def build_feature_matrix(tracking_df, plays_df, game_ids=None):
    """
    Build a feature matrix for all plays (or filtered by game_ids).

    Parameters
    ----------
    tracking_df : pd.DataFrame
        Full tracking data
    plays_df : pd.DataFrame
        Plays metadata with pff_passCoverageType, pff_passCoverage
    game_ids : array-like, optional
        If provided, only include these game IDs

    Returns
    -------
    pd.DataFrame
        Shape (n_plays, n_features + 2). Columns include all keys from
        extract_presnap_features plus pff_passCoverageType and pff_passCoverage.
        Plays with missing tracking or no snap frame are skipped.

    Examples
    --------
    >>> X = build_feature_matrix(tracking, plays)
    >>> print(X.shape)
    >>> X_week1 = build_feature_matrix(tracking, plays, game_ids=[2021090900])
    """
    if game_ids is not None:
        plays_sub = plays_df[plays_df['gameId'].isin(game_ids)].copy()
    else:
        plays_sub = plays_df.copy()

    rows = []
    for _, row in plays_sub.iterrows():
        gid = row['gameId']
        pid = row['playId']
        feats = extract_presnap_features(tracking_df, plays_df, gid, pid)
        if feats is None:
            continue
        feats['pff_passCoverageType'] = row.get('pff_passCoverageType', np.nan)
        feats['pff_passCoverage'] = row.get('pff_passCoverage', np.nan)
        rows.append(feats)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df
