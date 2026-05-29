"""
NFL Coverage Feature Extraction Engine
=======================================

Processes a normalized tracking DataFrame (the output of
``src.data.normalizer.build_normalized_tracking`` / ``load_and_build``) to
extract frame-by-frame spatial geometry metrics that serve as the foundation
for coverage-responsibility modelling.

Feature families
----------------
Receiver Cushion
    For each secondary defender at every tracking frame, the Euclidean
    distance (yards) to their assigned offensive receiver.

Spatial Leverage
    Signed coordinate offsets between a coverage defender and their assigned
    receiver at every tracking frame::

        leverage_x = x_defender − x_receiver
        leverage_y = y_defender − y_receiver

    A positive ``leverage_x`` means the defender is deeper than the receiver
    in the normalized left→right orientation (over-the-top).  A positive
    ``leverage_y`` means the defender is toward the far sideline (outside
    leverage).

Pre-Snap Safety Depth
    Per-play mean X distance between safeties (FS, SS) and the line of
    scrimmage, measured across all tracking frames at or before the
    ``ball_snap`` event.

Spatial Indexing
----------------
Per-frame ``scipy.spatial.cKDTree`` instances are constructed for all
nearest-receiver lookups.  Each tree is built once per
``(gameId, playId, frameId)`` group, keyed against the pre-grouped tracking
dictionary, and reused for both cushion and leverage calculations when PFF
assignment data is absent.

Stateless contract
------------------
Every public function:

* accepts an explicit DataFrame argument and returns a **new** DataFrame;
* never mutates any input argument;
* never uses ``inplace=True``;
* reads no module-level mutable state.

Expected input schema
---------------------
``tracking_df`` must be the output of
``src.data.normalizer.build_normalized_tracking`` or ``load_and_build``,
guaranteeing the following columns:

    gameId, playId, nflId, frameId, event, x, y, team,
    officialPosition, pff_role, o_rad, dir_rad

Optional columns that enhance accuracy when present (joined via
``build_normalized_tracking``'s optional ``plays_df`` / ``games_df``
arguments):

    pff_primaryDefensiveCoveredReceiverId
        PFF-recorded coverage target; used for direct defender→receiver
        pairing before falling back to nearest-receiver cKDTree resolution.

    absoluteYardlineNumber
        Field-position of the snap (from ``plays.csv``); secondary
        line-of-scrimmage reference when the ball entity is not tracked.

Usage
-----
::

    from src.features.coverage_features import (
        filter_secondary,
        build_frame_spatial_index,
        compute_receiver_cushion,
        compute_spatial_leverage,
        compute_presnap_safety_depth,
        compute_velocity_angular_divergence,
    )

    cushion  = compute_receiver_cushion(tracking_df)
    leverage = compute_spatial_leverage(tracking_df)
    depth    = compute_presnap_safety_depth(tracking_df)
    vad      = compute_velocity_angular_divergence(tracking_df)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Defensive positions that constitute the pass-coverage shell.
SECONDARY_POSITIONS: frozenset[str] = frozenset({"CB", "FS", "SS", "ILB", "OLB"})

#: Positions used to identify safeties for the pre-snap depth metric.
SAFETY_POSITIONS: frozenset[str] = frozenset({"FS", "SS"})

#: Offensive positions eligible as receivers in nearest-receiver fallback.
OFFENSIVE_SKILL_POSITIONS: frozenset[str] = frozenset({"WR", "TE", "RB", "FB", "HB"})

#: Event label marking the ball snap in the NGS tracking stream.
SNAP_EVENT: str = "ball_snap"

#: PFF scouting column linking a coverage defender to their coverage target.
PFF_COVERAGE_TARGET_COL: str = "pff_primaryDefensiveCoveredReceiverId"

#: Plays-schema column used as a fallback line-of-scrimmage reference.
SCRIMMAGE_X_COL: str = "absoluteYardlineNumber"

# ---------------------------------------------------------------------------
# Private module-level constants (not part of the public API)
# ---------------------------------------------------------------------------

#: Composite columns that uniquely identify a single tracking frame.
_FRAME_KEY: list[str] = ["gameId", "playId", "frameId"]

#: Length of each NFL end zone in yards; added to absoluteYardlineNumber to
#: convert a yard-line value (0–100) into the normalized x coordinate (0–120).
_ENDZONE_DEPTH: float = 10.0

#: officialPosition sentinel assigned to the football entity by the normalizer.
_BALL_SENTINEL: str = "BALL"


# ---------------------------------------------------------------------------
# Public utility: defensive secondary filter
# ---------------------------------------------------------------------------


def filter_secondary(tracking_df: pd.DataFrame) -> pd.DataFrame:
    """
    Isolate tracking rows belonging to the defensive passing shell.

    Filters to CB, FS, SS, ILB, and OLB positions, preventing interior
    defensive-linemen coordinates from polluting pass-shell spatial
    calculations.

    Parameters
    ----------
    tracking_df : pd.DataFrame
        Normalized tracking DataFrame with an ``officialPosition`` column.
        **Not mutated.**

    Returns
    -------
    pd.DataFrame
        Subset of *tracking_df* where ``officialPosition`` is in
        ``SECONDARY_POSITIONS``.  Returns an empty DataFrame (preserving all
        original columns) when no matching rows exist.

    Examples
    --------
    >>> secondary = filter_secondary(tracking_df)
    >>> set(secondary["officialPosition"].unique()).issubset(SECONDARY_POSITIONS)
    True
    """
    mask: pd.Series = tracking_df["officialPosition"].isin(SECONDARY_POSITIONS)
    return tracking_df.loc[mask].copy()


# ---------------------------------------------------------------------------
# Public primitive: per-frame spatial index
# ---------------------------------------------------------------------------


def build_frame_spatial_index(
    frame_df: pd.DataFrame,
    position_filter: Optional[frozenset[str]] = None,
) -> tuple[cKDTree, pd.DataFrame]:
    """
    Construct a ``cKDTree`` spatial index over the (x, y) positions in a
    single tracking frame.

    This is the core spatial-lookup primitive that backs all nearest-neighbor
    feature queries.  Callers may also invoke it directly for custom spatial
    queries beyond the built-in metrics.

    Parameters
    ----------
    frame_df : pd.DataFrame
        Tracking rows for a **single** ``(gameId, playId, frameId)`` slice.
        Must contain numeric ``x`` and ``y`` columns and, when
        *position_filter* is supplied, an ``officialPosition`` column.
        **Not mutated.**
    position_filter : frozenset[str], optional
        When provided, only rows whose ``officialPosition`` is in this set
        are indexed.  Pass ``None`` to index every row in *frame_df*.

    Returns
    -------
    tree : cKDTree
        Spatial index built on the filtered (x, y) coordinates.  Point *i*
        in the tree corresponds to row *i* in *indexed_df*.
    indexed_df : pd.DataFrame
        The (possibly filtered) subset of *frame_df* backing the tree, with
        its index reset to zero-based integers for safe positional access
        after tree queries.

    Raises
    ------
    ValueError
        If no rows survive the position filter.  An empty cKDTree would
        silently corrupt downstream queries; this guard surfaces the issue
        explicitly.

    Examples
    --------
    Build a receiver tree and query the nearest WR/TE for a defender:

    >>> frame = tracking_df[
    ...     (tracking_df["gameId"] == gid) &
    ...     (tracking_df["playId"] == pid) &
    ...     (tracking_df["frameId"] == fid)
    ... ]
    >>> tree, indexed = build_frame_spatial_index(
    ...     frame, position_filter=frozenset({"WR", "TE"})
    ... )
    >>> dists, idx = tree.query([[x_def, y_def]], k=1)
    >>> nearest_id = indexed.iloc[idx[0]]["nflId"]
    """
    df: pd.DataFrame = frame_df.copy()

    if position_filter is not None:
        df = df.loc[df["officialPosition"].isin(position_filter)].copy()

    if df.empty:
        raise ValueError(
            "build_frame_spatial_index: no rows remain after applying "
            f"position_filter={position_filter!r}.  "
            "Cannot construct an empty cKDTree."
        )

    coords: np.ndarray = df[["x", "y"]].to_numpy(dtype=np.float64)
    tree: cKDTree = cKDTree(coords)
    return tree, df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Private helpers – receiver-assignment resolution
# ---------------------------------------------------------------------------


def _receiver_coords_from_assignment(
    defenders_df: pd.DataFrame,
    tracking_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Resolve assigned receiver coordinates via the PFF coverage-target column.

    For each defender row, matches the ``PFF_COVERAGE_TARGET_COL`` value to
    the receiver's tracked position in the **same** ``(gameId, playId,
    frameId)`` triple via a vectorized merge.

    Parameters
    ----------
    defenders_df : pd.DataFrame
        Secondary defender rows that carry a non-NaN value in
        ``PFF_COVERAGE_TARGET_COL``.  **Not mutated.**
    tracking_df : pd.DataFrame
        Full normalized tracking DataFrame used as the coordinate lookup
        source.  **Not mutated.**

    Returns
    -------
    pd.DataFrame
        Copy of *defenders_df* — with ``PFF_COVERAGE_TARGET_COL`` renamed to
        ``assigned_receiver_id`` — augmented with two additional columns:

        ``receiver_x`` : float64
        ``receiver_y`` : float64

        Rows where the assigned receiver has no tracking entry in that frame
        receive ``np.nan`` in both coordinate columns.
    """
    # Build a slim receiver-coordinate lookup keyed by frame + nflId.
    all_coords: pd.DataFrame = (
        tracking_df[_FRAME_KEY + ["nflId", "x", "y"]]
        .copy()
        .rename(
            columns={
                "nflId": "assigned_receiver_id",
                "x": "receiver_x",
                "y": "receiver_y",
            }
        )
    )

    result: pd.DataFrame = defenders_df.copy()
    result = result.rename(columns={PFF_COVERAGE_TARGET_COL: "assigned_receiver_id"})

    result = result.merge(
        all_coords,
        on=_FRAME_KEY + ["assigned_receiver_id"],
        how="left",
        suffixes=("", "_recv_lookup"),
    )

    # Drop any duplicate coordinate columns introduced by the merge suffix.
    duplicate_cols = [c for c in result.columns if c.endswith("_recv_lookup")]
    if duplicate_cols:
        result = result.drop(columns=duplicate_cols)

    return result


def _receiver_coords_from_nearest(
    tracking_df: pd.DataFrame,
    defenders_df: pd.DataFrame,
    receiver_positions: frozenset[str],
) -> pd.DataFrame:
    """
    Resolve receiver coordinates for unassigned defenders via per-frame
    ``cKDTree`` nearest-neighbor queries.

    Pre-groups *tracking_df* by ``(gameId, playId, frameId)`` into a dict
    for O(1) per-frame lookups, then iterates over unique defender frames,
    batching all defenders in each frame into a single ``tree.query`` call.

    Parameters
    ----------
    tracking_df : pd.DataFrame
        Full normalized tracking DataFrame.  **Not mutated.**
    defenders_df : pd.DataFrame
        Secondary defender rows that require nearest-receiver resolution
        (i.e. those lacking a PFF assignment).  **Not mutated.**
    receiver_positions : frozenset[str]
        ``officialPosition`` labels that qualify as eligible receivers for
        nearest-neighbor lookup.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the same rows as *defenders_df* plus three new
        columns::

            assigned_receiver_id : float64   (nflId of nearest receiver)
            receiver_x           : float64
            receiver_y           : float64

        When no eligible receiver exists in a frame, all three columns
        contain ``np.nan`` for that defender.
    """
    # Pre-group the full tracking data for O(1) per-frame access.
    tracking_by_frame: dict[tuple, pd.DataFrame] = {
        key: group for key, group in tracking_df.groupby(_FRAME_KEY)
    }

    result_parts: list[pd.DataFrame] = []

    for frame_key, def_group in defenders_df.groupby(_FRAME_KEY):
        frame_group: Optional[pd.DataFrame] = tracking_by_frame.get(frame_key)

        if frame_group is None:
            # Frame exists in defenders but not in full tracking — emit NaN.
            nan_chunk = def_group.copy()
            nan_chunk["assigned_receiver_id"] = np.nan
            nan_chunk["receiver_x"] = np.nan
            nan_chunk["receiver_y"] = np.nan
            result_parts.append(nan_chunk)
            continue

        receivers: pd.DataFrame = frame_group[
            frame_group["officialPosition"].isin(receiver_positions)
        ]

        if receivers.empty:
            nan_chunk = def_group.copy()
            nan_chunk["assigned_receiver_id"] = np.nan
            nan_chunk["receiver_x"] = np.nan
            nan_chunk["receiver_y"] = np.nan
            result_parts.append(nan_chunk)
            continue

        try:
            tree, indexed_receivers = build_frame_spatial_index(
                receivers, position_filter=None
            )
        except ValueError:
            nan_chunk = def_group.copy()
            nan_chunk["assigned_receiver_id"] = np.nan
            nan_chunk["receiver_x"] = np.nan
            nan_chunk["receiver_y"] = np.nan
            result_parts.append(nan_chunk)
            continue

        def_coords: np.ndarray = def_group[["x", "y"]].to_numpy(dtype=np.float64)
        _, nn_indices = tree.query(def_coords, k=1)

        nearest_rows: pd.DataFrame = indexed_receivers.iloc[nn_indices]

        frame_result: pd.DataFrame = def_group.copy()
        frame_result["assigned_receiver_id"] = nearest_rows["nflId"].to_numpy()
        frame_result["receiver_x"] = nearest_rows["x"].to_numpy()
        frame_result["receiver_y"] = nearest_rows["y"].to_numpy()
        result_parts.append(frame_result)

    if not result_parts:
        empty_result: pd.DataFrame = defenders_df.copy()
        empty_result["assigned_receiver_id"] = np.nan
        empty_result["receiver_x"] = np.nan
        empty_result["receiver_y"] = np.nan
        return empty_result

    return pd.concat(result_parts, ignore_index=True)


def _resolve_receiver_assignments(
    tracking_df: pd.DataFrame,
    fallback_to_nearest: bool,
    receiver_positions: frozenset[str],
) -> pd.DataFrame:
    """
    Return all secondary defender rows enriched with their matched receiver's
    coordinates at each tracking frame.

    Resolution priority
    -------------------
    1. ``pff_primaryDefensiveCoveredReceiverId`` — used when the column
       exists **and** contains a non-NaN value.
    2. Nearest eligible receiver via per-frame ``cKDTree`` query, when
       *fallback_to_nearest* is ``True`` and either the PFF column is
       absent or the value is NaN.
    3. ``np.nan`` in ``receiver_x`` / ``receiver_y`` for defenders whose
       assignment cannot be resolved by either method.

    Parameters
    ----------
    tracking_df : pd.DataFrame
        Full normalized tracking DataFrame.  **Not mutated.**
    fallback_to_nearest : bool
        Whether to invoke cKDTree nearest-receiver resolution for defenders
        that lack a valid PFF assignment.
    receiver_positions : frozenset[str]
        Eligible receiver positions for the nearest-receiver fallback.

    Returns
    -------
    pd.DataFrame
        One row per secondary defender per tracking frame with the original
        tracking columns plus::

            assigned_receiver_id : float64
            receiver_x           : float64
            receiver_y           : float64
    """
    secondary_df: pd.DataFrame = filter_secondary(tracking_df)

    has_pff_col: bool = PFF_COVERAGE_TARGET_COL in secondary_df.columns

    if has_pff_col:
        has_assignment: pd.Series = secondary_df[PFF_COVERAGE_TARGET_COL].notna()
        with_assignment: pd.DataFrame = secondary_df.loc[has_assignment].copy()
        without_assignment: pd.DataFrame = secondary_df.loc[~has_assignment].copy()
    else:
        with_assignment = pd.DataFrame(columns=secondary_df.columns)
        without_assignment = secondary_df.copy()

    result_parts: list[pd.DataFrame] = []

    # --- Segment 1: PFF-assigned defenders ---
    if not with_assignment.empty:
        pff_resolved: pd.DataFrame = _receiver_coords_from_assignment(
            with_assignment, tracking_df
        )
        result_parts.append(pff_resolved)

    # --- Segment 2: unassigned defenders ---
    if not without_assignment.empty:
        if fallback_to_nearest:
            nearest_resolved: pd.DataFrame = _receiver_coords_from_nearest(
                tracking_df, without_assignment, receiver_positions
            )
            result_parts.append(nearest_resolved)
        else:
            # Caller opted out of nearest-receiver fallback.
            nan_result: pd.DataFrame = without_assignment.copy()
            nan_result["assigned_receiver_id"] = np.nan
            nan_result["receiver_x"] = np.nan
            nan_result["receiver_y"] = np.nan
            result_parts.append(nan_result)

    if not result_parts:
        empty: pd.DataFrame = secondary_df.copy()
        empty["assigned_receiver_id"] = np.nan
        empty["receiver_x"] = np.nan
        empty["receiver_y"] = np.nan
        return empty

    return pd.concat(result_parts, ignore_index=True)


def _resolve_los_x(play_group: pd.DataFrame, snap_frame_id: int) -> float:
    """
    Determine the line-of-scrimmage X coordinate for a single play.

    Resolution order:

    1. X position of the ball entity (``officialPosition == "BALL"``) at the
       snap frame — spatially authoritative and already in normalized
       coordinates.
    2. ``absoluteYardlineNumber + _ENDZONE_DEPTH`` when ball tracking is
       absent.  The ``+10`` offset converts the yard-line value (0–100 range
       in plays.csv) into the normalized tracking coordinate space (0–120),
       where x = 10 is the offensive goal line.

    Parameters
    ----------
    play_group : pd.DataFrame
        All tracking rows for a single ``(gameId, playId)``.
    snap_frame_id : int
        The ``frameId`` at which ``event == SNAP_EVENT`` was first recorded.

    Returns
    -------
    float
        LOS X coordinate in normalized tracking units (0–120 yards), or
        ``np.nan`` when neither source is available.
    """
    snap_frame: pd.DataFrame = play_group[play_group["frameId"] == snap_frame_id]
    ball_rows: pd.DataFrame = snap_frame[snap_frame["officialPosition"] == _BALL_SENTINEL]

    if not ball_rows.empty:
        return float(ball_rows["x"].iloc[0])

    if SCRIMMAGE_X_COL in play_group.columns:
        yard_series: pd.Series = play_group[SCRIMMAGE_X_COL].dropna()
        if not yard_series.empty:
            return float(yard_series.iloc[0]) + _ENDZONE_DEPTH

    return np.nan


# ---------------------------------------------------------------------------
# Public feature: Receiver Cushion
# ---------------------------------------------------------------------------


def compute_receiver_cushion(
    tracking_df: pd.DataFrame,
    fallback_to_nearest: bool = True,
    receiver_positions: frozenset[str] = OFFENSIVE_SKILL_POSITIONS,
) -> pd.DataFrame:
    """
    Compute the Euclidean distance from each secondary defender to their
    assigned (or nearest) offensive receiver at every tracking frame.

    Assignment resolution order
    ---------------------------
    1. ``pff_primaryDefensiveCoveredReceiverId`` — when the column exists and
       contains a valid player ID.
    2. Nearest player whose ``officialPosition`` is in *receiver_positions*,
       resolved via a per-frame ``cKDTree`` nearest-neighbor query, when
       *fallback_to_nearest* is ``True``.
    3. ``np.nan`` when neither source yields a match (e.g. no eligible
       receivers tracked in the frame, or fallback is disabled).

    Parameters
    ----------
    tracking_df : pd.DataFrame
        Fully normalized tracking DataFrame (output of
        ``src.data.normalizer.build_normalized_tracking``).  Must contain
        ``officialPosition``, ``x``, ``y``, ``nflId``, ``gameId``,
        ``playId``, ``frameId``.  **Not mutated.**
    fallback_to_nearest : bool, default True
        When ``True``, defenders without a PFF assignment are matched to the
        spatially nearest eligible receiver via cKDTree.  When ``False``,
        those defenders receive ``np.nan`` cushion values.
    receiver_positions : frozenset[str], default OFFENSIVE_SKILL_POSITIONS
        The ``officialPosition`` labels treated as valid receivers in the
        nearest-receiver fallback.

    Returns
    -------
    pd.DataFrame
        One row per secondary defender per tracking frame.  Columns:

        ``gameId``, ``playId``, ``frameId``
            Frame identifier triple.
        ``nflId`` : float64
            Defender player ID.
        ``officialPosition`` : str
            Defender position label (one of ``SECONDARY_POSITIONS``).
        ``team`` : str
            Defender team abbreviation.
        ``assigned_receiver_id`` : float64
            ``nflId`` of the matched receiver; ``np.nan`` if unresolved.
        ``receiver_cushion`` : float64
            Euclidean distance in yards to the matched receiver; ``np.nan``
            if unresolved.

    Examples
    --------
    >>> cushion_df = compute_receiver_cushion(tracking_df)
    >>> # Identify press-coverage frames (cushion under 1.5 yards)
    >>> press_frames = cushion_df[cushion_df["receiver_cushion"] < 1.5]
    >>> print(press_frames[["nflId", "officialPosition", "receiver_cushion"]])
    """
    resolved: pd.DataFrame = _resolve_receiver_assignments(
        tracking_df, fallback_to_nearest, receiver_positions
    )

    result: pd.DataFrame = resolved.copy()
    result["receiver_cushion"] = np.sqrt(
        (result["x"] - result["receiver_x"]) ** 2
        + (result["y"] - result["receiver_y"]) ** 2
    )

    output_cols: list[str] = _FRAME_KEY + [
        "nflId",
        "officialPosition",
        "team",
        "assigned_receiver_id",
        "receiver_cushion",
    ]
    available: list[str] = [c for c in output_cols if c in result.columns]
    return result[available].copy()


# ---------------------------------------------------------------------------
# Public feature: Spatial Leverage
# ---------------------------------------------------------------------------


def compute_spatial_leverage(
    tracking_df: pd.DataFrame,
    fallback_to_nearest: bool = True,
    receiver_positions: frozenset[str] = OFFENSIVE_SKILL_POSITIONS,
) -> pd.DataFrame:
    """
    Compute signed coordinate offsets (leverage) between each secondary
    defender and their assigned receiver at every tracking frame.

    Leverage components
    -------------------
    ``leverage_x``
        ``x_defender − x_receiver``.
        Positive → defender is playing over-the-top / deeper than the
        receiver in the normalized left→right orientation.
        Negative → defender is playing underneath.

    ``leverage_y``
        ``y_defender − y_receiver``.
        Positive → defender is toward the far sideline (outside leverage).
        Negative → defender is to the inside.

    Assignment resolution order is identical to ``compute_receiver_cushion``:
    PFF target ID first, then cKDTree nearest-receiver fallback.

    Parameters
    ----------
    tracking_df : pd.DataFrame
        Fully normalized tracking DataFrame.  **Not mutated.**
    fallback_to_nearest : bool, default True
        When ``True``, defenders without a PFF assignment are matched to the
        spatially nearest eligible receiver via cKDTree nearest-neighbor
        query.
    receiver_positions : frozenset[str], default OFFENSIVE_SKILL_POSITIONS
        Eligible receiver positions for the nearest-receiver fallback.

    Returns
    -------
    pd.DataFrame
        One row per secondary defender per tracking frame.  Columns:

        ``gameId``, ``playId``, ``frameId``
            Frame identifier triple.
        ``nflId`` : float64
            Defender player ID.
        ``officialPosition`` : str
            Defender position label.
        ``team`` : str
            Defender team abbreviation.
        ``assigned_receiver_id`` : float64
            ``nflId`` of the matched receiver; ``np.nan`` if unresolved.
        ``leverage_x`` : float64
            X-axis signed offset (defender minus receiver); ``np.nan`` if
            unresolved.
        ``leverage_y`` : float64
            Y-axis signed offset (defender minus receiver); ``np.nan`` if
            unresolved.

    Examples
    --------
    >>> leverage_df = compute_spatial_leverage(tracking_df)
    >>> # Deep safeties playing over-the-top (positive leverage_x > 5 yards)
    >>> over_top = leverage_df[
    ...     leverage_df["officialPosition"].isin(["FS", "SS"]) &
    ...     (leverage_df["leverage_x"] > 5.0)
    ... ]
    """
    resolved: pd.DataFrame = _resolve_receiver_assignments(
        tracking_df, fallback_to_nearest, receiver_positions
    )

    result: pd.DataFrame = resolved.copy()
    result["leverage_x"] = result["x"] - result["receiver_x"]
    result["leverage_y"] = result["y"] - result["receiver_y"]

    output_cols: list[str] = _FRAME_KEY + [
        "nflId",
        "officialPosition",
        "team",
        "assigned_receiver_id",
        "leverage_x",
        "leverage_y",
    ]
    available: list[str] = [c for c in output_cols if c in result.columns]
    return result[available].copy()


# ---------------------------------------------------------------------------
# Public feature: Pre-Snap Safety Depth
# ---------------------------------------------------------------------------


def compute_presnap_safety_depth(
    tracking_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute the average pre-snap depth of safeties relative to the line of
    scrimmage for every play in the tracking DataFrame.

    "Pre-snap" encompasses all tracking frames from the first frame of the
    play up to and including the frame whose ``event`` equals ``ball_snap``.
    Depth is defined as ``x_safety − los_x`` in the normalized coordinate
    system (positive values indicate the safety is behind the line,
    i.e. further downfield in a defensive sense).

    Line-of-scrimmage reference
    ---------------------------
    1. X position of the ball entity (``officialPosition == "BALL"``) at the
       snap frame — preferred, since it is already in normalized coordinates.
    2. ``absoluteYardlineNumber + 10.0`` when ball-entity rows are absent.
       The ``+10`` offset converts the absolute yard-line (0–100 in
       ``plays.csv``) into the normalized tracking X space (0–120, where
       x = 10 is the offensive goal line).

    Parameters
    ----------
    tracking_df : pd.DataFrame
        Fully normalized tracking DataFrame.  Must contain
        ``officialPosition``, ``x``, ``gameId``, ``playId``, ``frameId``,
        ``event``.  The ``absoluteYardlineNumber`` column is used as a
        fallback LOS reference when present.  **Not mutated.**

    Returns
    -------
    pd.DataFrame
        One row per ``(gameId, playId)`` with columns:

        ``gameId``, ``playId``
            Play identifiers.
        ``los_x`` : float64
            Line-of-scrimmage X coordinate used for depth calculations;
            ``np.nan`` if neither resolution source was available.
        ``safety_depth_mean`` : float64
            Mean of ``(x_safety − los_x)`` across all safeties and all
            pre-snap frames.  ``np.nan`` if no safeties are present or no
            snap event is found.
        ``safety_depth_std`` : float64
            Standard deviation of the per-(safety, frame) depth values.
            ``np.nan`` when fewer than two observations are available.
        ``n_safety_frames`` : int
            Total number of (safety × frame) observations contributing to
            the statistics.

    Examples
    --------
    >>> depth_df = compute_presnap_safety_depth(tracking_df)
    >>> # Plays with a deep two-high shell (safeties > 12 yards from LOS)
    >>> two_high_candidates = depth_df[depth_df["safety_depth_mean"] > 12.0]
    >>> print(two_high_candidates[["gameId", "playId", "safety_depth_mean"]])
    """
    df: pd.DataFrame = tracking_df.copy()

    # ── Step 1: Identify the snap frameId per (gameId, playId) ───────────────
    # Minimum frameId where event == SNAP_EVENT matches the original
    # `iloc[0]` first-occurrence semantics (plays are time-ordered).
    snap_frame_map: pd.DataFrame = (
        df[df["event"] == SNAP_EVENT]
        .groupby(["gameId", "playId"])["frameId"]
        .min()
        .rename("snap_frameId")
        .reset_index()
    )

    if snap_frame_map.empty:
        return pd.DataFrame(
            columns=[
                "gameId", "playId", "los_x",
                "safety_depth_mean", "safety_depth_std", "n_safety_frames",
            ]
        )

    # ── Step 2: Resolve LOS X from the ball entity at the snap frame ─────────
    # The ball entity (officialPosition == "BALL") is always at the snap frame;
    # `.first()` safely handles the rare edge case of multiple ball rows.
    ball_los: pd.DataFrame = (
        df[
            (df["officialPosition"] == _BALL_SENTINEL)
            & (df["event"] == SNAP_EVENT)
        ]
        .groupby(["gameId", "playId"])["x"]
        .first()
        .rename("los_x_ball")
        .reset_index()
    )

    # ── Step 3: Assemble play-level metadata (snap frameId + LOS X) ──────────
    play_info: pd.DataFrame = snap_frame_map.merge(
        ball_los, on=["gameId", "playId"], how="left"
    )

    # Fallback LOS from absoluteYardlineNumber when ball-entity tracking is absent.
    if SCRIMMAGE_X_COL in df.columns:
        yard_src = (
            df[["gameId", "playId", SCRIMMAGE_X_COL]]
            .dropna(subset=[SCRIMMAGE_X_COL])
            .drop_duplicates(subset=["gameId", "playId"])
            .copy()
        )
        yard_src["los_x_yard"] = yard_src[SCRIMMAGE_X_COL] + _ENDZONE_DEPTH
        yard_los: pd.DataFrame = yard_src[["gameId", "playId", "los_x_yard"]]
        play_info = play_info.merge(yard_los, on=["gameId", "playId"], how="left")
        play_info["los_x"] = play_info["los_x_ball"].combine_first(
            play_info["los_x_yard"]
        )
        play_info = play_info.drop(columns=["los_x_ball", "los_x_yard"])
    else:
        play_info = play_info.rename(columns={"los_x_ball": "los_x"})

    # ── Step 4: Broadcast snap_frameId and los_x to all tracking rows ─────────
    # Left join preserves every tracking row; plays without snap events receive
    # NaN snap_frameId, which the boolean mask below excludes cleanly.
    merged: pd.DataFrame = df.merge(
        play_info[["gameId", "playId", "snap_frameId", "los_x"]],
        on=["gameId", "playId"],
        how="left",
    )

    # ── Step 5: Vector-filter to pre-snap safety rows ─────────────────────────
    presnap_safety: pd.DataFrame = merged[
        merged["officialPosition"].isin(SAFETY_POSITIONS)
        & merged["snap_frameId"].notna()
        & (merged["frameId"] <= merged["snap_frameId"])
        & merged["los_x"].notna()
    ].copy()

    # Per-row depth from the line of scrimmage.
    presnap_safety["depth"] = presnap_safety["x"] - presnap_safety["los_x"]

    # ── Step 6: Single vectorized groupby aggregation ─────────────────────────
    # pandas .std() defaults to ddof=1; returns NaN when the group has one row.
    if presnap_safety.empty:
        safety_agg: pd.DataFrame = pd.DataFrame(
            columns=[
                "gameId", "playId",
                "safety_depth_mean", "safety_depth_std", "n_safety_frames",
            ]
        )
    else:
        safety_agg = (
            presnap_safety
            .groupby(["gameId", "playId"])["depth"]
            .agg(
                safety_depth_mean="mean",
                safety_depth_std="std",
                n_safety_frames="count",
            )
            .reset_index()
        )

    # ── Step 7: Left-join back to all plays that had a snap event ─────────────
    # Plays with snap events but no pre-snap safeties receive NaN depth stats
    # and n_safety_frames = 0 — same semantics as the original loop.
    result: pd.DataFrame = (
        play_info[["gameId", "playId", "los_x"]]
        .merge(safety_agg, on=["gameId", "playId"], how="left")
    )
    result["n_safety_frames"] = result["n_safety_frames"].fillna(0).astype(int)

    return result


# ---------------------------------------------------------------------------
# Public feature: Velocity Angular Divergence
# ---------------------------------------------------------------------------


def compute_velocity_angular_divergence(
    tracking_df: pd.DataFrame,
    fallback_to_nearest: bool = True,
    receiver_positions: frozenset[str] = OFFENSIVE_SKILL_POSITIONS,
) -> pd.DataFrame:
    """
    Compute the cosine similarity between each secondary defender's velocity
    vector and the velocity vector of their assigned (or nearest) receiver.

    Interpretation
    --------------
    +1.0  — defender and receiver are moving in exactly the same direction
             (mirror coverage, trailing a route).
     0.0  — perpendicular motion (cross-field cut vs. lateral defender drift).
    -1.0  — directly opposing trajectories (press bail vs. receiver release).

    Man defenders typically exhibit high positive cosine similarity because
    they mirror route stems; zone defenders maintain spatial territory
    regardless of receiver direction, producing values near zero or negative.

    Velocity vector construction
    ----------------------------
    For any tracked player::

        v = (s · cos(dir_rad),  s · sin(dir_rad))

    where ``s`` is speed in yards/second and ``dir_rad`` is the movement
    direction in standard Cartesian radians (output of
    ``convert_angles_to_radians``).

    Assignment resolution order
    ---------------------------
    Identical to :func:`compute_receiver_cushion`:
    1. ``pff_primaryDefensiveCoveredReceiverId`` (when available).
    2. Nearest player in *receiver_positions* via per-frame cKDTree (when
       *fallback_to_nearest* is ``True``).
    3. ``np.nan`` when neither source yields a valid match.

    Parameters
    ----------
    tracking_df : pd.DataFrame
        Fully normalized tracking DataFrame.  Must contain ``s``,
        ``dir_rad``, ``officialPosition``, ``nflId``, ``gameId``,
        ``playId``, ``frameId``.  **Not mutated.**
    fallback_to_nearest : bool, default True
        When ``True``, defenders without a PFF assignment are matched to
        the spatially nearest eligible receiver.
    receiver_positions : frozenset[str], default OFFENSIVE_SKILL_POSITIONS
        Eligible receiver positions for the nearest-receiver fallback.

    Returns
    -------
    pd.DataFrame
        One row per secondary defender per tracking frame.  Columns:

        ``gameId``, ``playId``, ``frameId``
            Frame identifier triple.
        ``nflId`` : float64
            Defender player ID.
        ``velocity_angular_divergence`` : float64
            Cosine similarity in [−1, +1]; ``np.nan`` when the receiver
            assignment is unresolved or either player has zero speed
            (making direction undefined).

    Examples
    --------
    >>> vad_df = compute_velocity_angular_divergence(tracking_df)
    >>> # Frames where defender mirrors receiver motion (likely man coverage)
    >>> mirroring = vad_df[vad_df["velocity_angular_divergence"] > 0.85]
    """
    # Resolve receiver position assignments (x, y, assigned_receiver_id).
    resolved: pd.DataFrame = _resolve_receiver_assignments(
        tracking_df, fallback_to_nearest, receiver_positions
    )

    # Build a slim receiver-velocity lookup keyed by frame + receiver nflId.
    # The rename maps tracking nflId → assigned_receiver_id so the merge key
    # aligns with the resolved DataFrame's column name.
    recv_vel: pd.DataFrame = (
        tracking_df[_FRAME_KEY + ["nflId", "s", "dir_rad"]]
        .rename(
            columns={
                "nflId": "assigned_receiver_id",
                "s": "receiver_s",
                "dir_rad": "receiver_dir_rad",
            }
        )
    )

    result: pd.DataFrame = resolved.merge(
        recv_vel,
        on=_FRAME_KEY + ["assigned_receiver_id"],
        how="left",
    )

    # Defender velocity components (Cartesian).
    def_vx: pd.Series = result["s"] * np.cos(result["dir_rad"])
    def_vy: pd.Series = result["s"] * np.sin(result["dir_rad"])

    # Receiver velocity components (Cartesian).
    rec_vx: pd.Series = result["receiver_s"] * np.cos(result["receiver_dir_rad"])
    rec_vy: pd.Series = result["receiver_s"] * np.sin(result["receiver_dir_rad"])

    # Drop the receiver velocity lookup columns immediately — all information
    # has been extracted into the local vector blocks above.
    result = result.drop(columns=["receiver_s", "receiver_dir_rad"], errors="ignore")

    dot: pd.Series = def_vx * rec_vx + def_vy * rec_vy
    def_mag: pd.Series = np.sqrt(def_vx ** 2 + def_vy ** 2)
    rec_mag: pd.Series = np.sqrt(rec_vx ** 2 + rec_vy ** 2)
    denom: pd.Series = def_mag * rec_mag

    # Cosine similarity is undefined when either speed is zero; emit NaN.
    result["velocity_angular_divergence"] = np.where(
        denom > 0,
        dot / denom,
        np.nan,
    )

    output_cols: list[str] = _FRAME_KEY + ["nflId", "velocity_angular_divergence"]
    available: list[str] = [c for c in output_cols if c in result.columns]
    return result[available].copy()
