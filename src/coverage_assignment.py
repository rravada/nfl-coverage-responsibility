"""
NFL Kinematic Voronoi Coverage Assignment Engine
=================================================

Ingests the normalized tracking DataFrame produced by
``src.data.normalizer.build_normalized_tracking`` (or ``load_and_build``) and
computes frame-by-frame **Kinematic Voronoi** field partitions to determine
which defensive player "owns" each region of the field.  Ownership is then
intersected with offensive skill-player locations to emit an explicit
coverage-responsibility mapping table.

Algorithm overview
------------------
1. **Field grid** – the 120 × 53.3-yard playing surface is discretised into a
   uniform rectangular mesh at a configurable resolution (default 1.0 yd).
   All grid construction happens once and the resulting array is reused across
   every play and frame.

2. **Time-to-Intercept (TTI)** – for every *(defender, grid-point)* pair the
   minimum time for the defender to physically reach that point is estimated
   using a 1-D kinematic model along the defender-to-point direction::

       d = v_proj · t + ½ · a_eff · t²

   where *d* is the Euclidean distance to the grid point, *v_proj* is the
   component of the defender's current velocity vector projected onto the
   defender-to-point unit vector, and *a_eff = max(a, 0)* is the defender's
   instantaneous acceleration (floored at zero so we always model a player
   trying to reach the target).  The positive root of the quadratic gives the
   TTI.  When acceleration is negligible the equation degrades gracefully to
   ``t = d / max(v_proj, _MIN_SPEED)``.

   The **entire** (D × G) TTI matrix for one frame is computed in a single
   NumPy expression using axis-level broadcasting — no Python loop iterates
   over individual defenders or grid points.

3. **Voronoi partition** – ``np.argmin`` over the D axis assigns each grid
   point to the defender with the minimum TTI, partitioning the field into
   kinematic Voronoi territories.

4. **Coverage assignment** – each offensive skill player's location is mapped
   to its nearest grid point (vectorised over all receivers simultaneously);
   that point's owning defender becomes the coverage-responsibility holder for
   that receiver in that frame.

5. **Direct-TTI assignment** – an alternative ``assign_coverage_direct_tti``
   function computes TTI from every defender directly to every receiver
   position (bypassing the grid entirely) for faster single-play diagnostics.

Stateless contract
------------------
Every public function:

* accepts explicit DataFrame arguments and returns a **fresh** DataFrame;
* never mutates any input argument;
* never uses ``inplace=True``;
* reads no module-level mutable state after import time.

Expected input schema
---------------------
``tracking_df`` must be the output of
``src.data.normalizer.build_normalized_tracking`` or ``load_and_build``,
guaranteeing the following columns:

    ``gameId``, ``playId``, ``nflId``, ``frameId``,
    ``x``, ``y``, ``s``, ``a``, ``dir_rad``, ``officialPosition``, ``team``

The ``pff_role`` column is used as a supplementary defensive-player filter
when present.

Usage
-----
Minimal end-to-end example::

    from src.data.normalizer import load_and_build
    from src.coverage_assignment import build_coverage_assignments

    tracking = load_and_build(
        tracking_path="data/raw/week1.csv",
        players_path="data/raw/players.csv",
        pff_path="data/raw/pffScoutingData.csv",
        plays_path="data/raw/plays.csv",
    )

    assignments = build_coverage_assignments(tracking)
    # → DataFrame[gameId, playId, frameId, defender_nflId, assigned_receiver_nflId]

Fine-grained control::

    from src.coverage_assignment import (
        build_field_grid,
        compute_tti_matrix,
        partition_voronoi_frame,
        compute_defender_territories,
        assign_coverage_direct_tti,
        build_coverage_assignments,
    )

    # Pre-build the grid once and reuse across functions.
    grid = build_field_grid(resolution=0.5)

    # Inspect kinematic Voronoi territories for one frame.
    territories = compute_defender_territories(tracking, grid_xy=grid)

    # Full assignment table using the grid-based approach.
    assignments = build_coverage_assignments(tracking, grid_xy=grid)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Field-geometry and physics constants (immutable after module load)
# ---------------------------------------------------------------------------

#: End-zone-to-end-zone field length in yards (X axis in normalised coords).
FIELD_LENGTH: float = 120.0

#: Sideline-to-sideline field width in yards (Y axis in normalised coords).
FIELD_WIDTH: float = 53.3

#: Default mesh-grid resolution in yards.  Smaller values give finer spatial
#: resolution at the cost of higher memory and CPU usage per frame.
DEFAULT_GRID_RESOLUTION: float = 1.0

#: Defensive positions eligible as coverage defenders in the kinematic model.
#: Includes all DB/LB designations used across different data-provider schemas.
COVERAGE_POSITIONS: frozenset[str] = frozenset(
    {"CB", "FS", "SS", "ILB", "OLB", "LB", "MLB", "DB"}
)

#: Offensive positions that represent valid coverage-responsibility targets.
SKILL_POSITIONS: frozenset[str] = frozenset({"WR", "TE", "RB"})

# ---------------------------------------------------------------------------
# Private constants
# ---------------------------------------------------------------------------

#: Sentinel label assigned to the football entity by the normaliser.
_BALL_SENTINEL: str = "BALL"

#: Composite key columns that uniquely identify one tracking frame.
_FRAME_KEY: list[str] = ["gameId", "playId", "frameId"]

#: Minimum effective speed (yd/s) used in the linear TTI branch to prevent
#: divide-by-zero when a defender is moving orthogonally to or away from the
#: target point.  Approximates a slow-but-reachable baseline.
_MIN_SPEED: float = 0.5

#: Acceleration magnitude threshold (yd/s²) below which the quadratic TTI
#: branch degrades to the linear (constant-speed) formula.
_ACC_EPSILON: float = 1e-6

#: Large sentinel TTI value (seconds) assigned when a grid point is
#: kinematically unreachable under the current model state (e.g. negative
#: discriminant when decelerating hard away from the target).
_UNREACHABLE_TTI: float = 999.0

#: Euclidean distance threshold (yards) below which a defender is considered
#: already at the grid point and receives TTI = 0.
_AT_POINT_THRESHOLD: float = 1e-9

#: Required columns for the core assignment engine.  Used for early validation.
_REQUIRED_COLS: frozenset[str] = frozenset(
    {"gameId", "playId", "frameId", "nflId", "x", "y", "s", "a", "dir_rad",
     "officialPosition"}
)


# ---------------------------------------------------------------------------
# Public: Field-grid construction
# ---------------------------------------------------------------------------


def build_field_grid(resolution: float = DEFAULT_GRID_RESOLUTION) -> np.ndarray:
    """
    Build a flat (G, 2) array of (x, y) grid-point coordinates covering the
    entire 120 × 53.3-yard normalised field.

    The grid spans ``x ∈ [0, FIELD_LENGTH]`` and ``y ∈ [0, FIELD_WIDTH]``
    inclusive, sampled at *resolution*-yard intervals.  The resulting array
    is the input consumed by :func:`compute_tti_matrix` and
    :func:`partition_voronoi_frame`.

    Parameters
    ----------
    resolution : float, default 1.0
        Grid-cell spacing in yards.  Common choices:

        * ``1.0`` yd – fast; adequate for play-level analysis (~6 360 points).
        * ``0.5`` yd – finer; better spatial precision (~25 440 points).
        * ``0.25`` yd – high-resolution; use only for single-play deep-dives.

    Returns
    -------
    np.ndarray of shape (G, 2), dtype float64
        Row *i* is the ``(x, y)`` coordinate of grid point *i*.  The total
        number of points G ≈ ``(FIELD_LENGTH / resolution + 1) ×
        (FIELD_WIDTH / resolution + 1)``.

    Examples
    --------
    >>> grid = build_field_grid(resolution=1.0)
    >>> grid.shape
    (7442, 2)
    >>> grid[0]
    array([ 0.,  0.])
    """
    if resolution <= 0.0:
        raise ValueError(
            f"build_field_grid: resolution must be positive; got {resolution!r}."
        )
    xs: np.ndarray = np.arange(0.0, FIELD_LENGTH + resolution, resolution)
    ys: np.ndarray = np.arange(0.0, FIELD_WIDTH + resolution, resolution)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel()])


# ---------------------------------------------------------------------------
# Private: kinematic array extraction
# ---------------------------------------------------------------------------


def _extract_kinematics(
    df: pd.DataFrame,
    fallback_speed: float = 0.0,
    fallback_acc: float = 0.0,
    fallback_dir_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract (xy, velocity, acceleration) NumPy arrays from a player DataFrame.

    NaN values in the kinematic columns are filled with the supplied fallback
    scalars before conversion, preventing silent NaN propagation through the
    TTI algebra.

    Parameters
    ----------
    df : pd.DataFrame
        Player tracking rows.  Must contain ``x``, ``y``, ``s``, ``a``,
        ``dir_rad``.  **Not mutated.**
    fallback_speed : float, default 0.0
        Replacement for NaN ``s`` (speed) values.
    fallback_acc : float, default 0.0
        Replacement for NaN ``a`` (acceleration) values.
    fallback_dir_rad : float, default 0.0
        Replacement for NaN ``dir_rad`` values (0.0 = pointing along +X).

    Returns
    -------
    xy : np.ndarray of shape (N, 2)
        Player positions.
    vel : np.ndarray of shape (N, 2)
        Velocity vectors ``(s·cos(dir_rad), s·sin(dir_rad))``.
    acc : np.ndarray of shape (N,)
        Scalar acceleration magnitudes.
    """
    x: np.ndarray = df["x"].to_numpy(dtype=np.float64)
    y: np.ndarray = df["y"].to_numpy(dtype=np.float64)

    s: np.ndarray = (
        pd.to_numeric(df["s"], errors="coerce")
        .fillna(fallback_speed)
        .to_numpy(dtype=np.float64)
    )
    a: np.ndarray = (
        pd.to_numeric(df["a"], errors="coerce")
        .fillna(fallback_acc)
        .to_numpy(dtype=np.float64)
    )
    dir_rad: np.ndarray = (
        pd.to_numeric(df["dir_rad"], errors="coerce")
        .fillna(fallback_dir_rad)
        .to_numpy(dtype=np.float64)
    )

    xy: np.ndarray = np.column_stack([x, y])                                 # (N, 2)
    vel: np.ndarray = np.column_stack(
        [s * np.cos(dir_rad), s * np.sin(dir_rad)]
    )                                                                          # (N, 2)
    return xy, vel, a


# ---------------------------------------------------------------------------
# Public: Core TTI computation (fully vectorised)
# ---------------------------------------------------------------------------


def compute_tti_matrix(
    def_xy: np.ndarray,
    def_vel: np.ndarray,
    def_acc: np.ndarray,
    grid_xy: np.ndarray,
) -> np.ndarray:
    """
    Compute the (D × G) Time-to-Intercept matrix using NumPy broadcasting.

    For every defender *d* and grid point *g*, the TTI is the minimum time
    (in seconds) for defender *d* to physically reach grid point *g* given
    their current position, velocity direction, and acceleration, under a
    1-D kinematic model along the defender-to-point axis.

    Physics model
    ~~~~~~~~~~~~~
    Let:

    * ``dist`` – Euclidean distance from defender *d* to grid point *g*.
    * ``v_proj`` – speed component of defender *d* projected onto the unit
      vector pointing from *d* toward *g* (positive = already moving
      toward target).
    * ``a_eff = max(a_d, 0)`` – effective acceleration (non-negative; we
      model the defender always trying to reach the target).

    Kinematic equation::

        dist = v_proj · t + ½ · a_eff · t²

    **Quadratic branch** (``a_eff > _ACC_EPSILON``)::

        discriminant = v_proj² + 2 · a_eff · dist   (≥ 0 when a_eff ≥ 0, dist ≥ 0)
        TTI = (−v_proj + √discriminant) / a_eff

    **Linear branch** (``a_eff ≈ 0``, constant speed)::

        TTI = dist / max(v_proj, _MIN_SPEED)

    A defender already at the grid point (``dist < _AT_POINT_THRESHOLD``)
    receives ``TTI = 0``.  Any result that is negative, infinite, or NaN is
    clamped to ``_UNREACHABLE_TTI = 999.0`` seconds.

    Parameters
    ----------
    def_xy : np.ndarray of shape (D, 2)
        Defender (x, y) positions.
    def_vel : np.ndarray of shape (D, 2)
        Defender velocity vectors ``(vx, vy)`` in yd/s.
    def_acc : np.ndarray of shape (D,)
        Defender acceleration magnitudes in yd/s².
    grid_xy : np.ndarray of shape (G, 2)
        Grid-point (x, y) coordinates (from :func:`build_field_grid`).

    Returns
    -------
    np.ndarray of shape (D, G), dtype float64
        ``tti[d, g]`` is the estimated seconds for defender *d* to reach
        grid point *g*.

    Notes
    -----
    * No Python loop iterates over individual players or grid points; all
      arithmetic is performed as a single sequence of NumPy array operations.
    * Memory footprint per frame: ``D × G × 8`` bytes  (e.g. 11 defenders ×
      6 360 grid points × 8 B ≈ 560 KB at 1-yd resolution).

    Examples
    --------
    >>> grid = build_field_grid(1.0)
    >>> tti = compute_tti_matrix(def_xy, def_vel, def_acc, grid)
    >>> assigned = np.argmin(tti, axis=0)   # Voronoi partition: shape (G,)
    """
    # ── (D, G, 2): vector from each defender to each grid point ────────────
    diff: np.ndarray = (
        grid_xy[np.newaxis, :, :]          # (1, G, 2)
        - def_xy[:, np.newaxis, :]         # (D, 1, 2)
    )                                      # → (D, G, 2)

    # ── (D, G): Euclidean distance ──────────────────────────────────────────
    dist: np.ndarray = np.sqrt(np.sum(diff ** 2, axis=2))

    # ── (D, G, 2): unit direction from defender to grid point ──────────────
    # Avoid divide-by-zero at coincident locations (dist == 0).
    dist_safe: np.ndarray = np.where(
        dist > _AT_POINT_THRESHOLD,
        dist,
        1.0,
    )[:, :, np.newaxis]                   # (D, G, 1) for broadcasting
    unit: np.ndarray = diff / dist_safe   # (D, G, 2)

    # ── (D, G): speed projected onto direction to grid point ───────────────
    # def_vel[:, np.newaxis, :] broadcasts to (D, G, 2) × (D, G, 2) → sum → (D, G)
    v_proj: np.ndarray = np.sum(
        def_vel[:, np.newaxis, :] * unit, axis=2
    )                                      # (D, G)

    # ── (D, G): effective non-negative acceleration ─────────────────────────
    a_eff: np.ndarray = np.maximum(
        def_acc[:, np.newaxis], 0.0        # (D, 1) broadcasts to (D, G)
    )

    # ── Quadratic branch: a_eff > ε ────────────────────────────────────────
    discriminant: np.ndarray = v_proj ** 2 + 2.0 * a_eff * dist
    # Clamp discriminant to 0 to prevent sqrt of tiny negatives from rounding.
    disc_safe: np.ndarray = np.maximum(discriminant, 0.0)
    # Avoid divide-by-zero in branch denominator (overwritten by np.where).
    a_for_div: np.ndarray = np.where(a_eff > _ACC_EPSILON, a_eff, 1.0)
    tti_quad: np.ndarray = (-v_proj + np.sqrt(disc_safe)) / a_for_div

    # ── Linear branch: a_eff ≈ 0, constant-speed model ─────────────────────
    # Floor v_proj at _MIN_SPEED so a defender moving away still "eventually"
    # reaches the point (at a penalised pace), keeping TTI finite.
    v_eff: np.ndarray = np.maximum(v_proj, _MIN_SPEED)
    tti_lin: np.ndarray = dist / v_eff

    # ── Select branch via mask ──────────────────────────────────────────────
    tti: np.ndarray = np.where(a_eff > _ACC_EPSILON, tti_quad, tti_lin)

    # ── Zero distance: already at grid point ───────────────────────────────
    tti = np.where(dist < _AT_POINT_THRESHOLD, 0.0, tti)

    # ── Sanitise: clamp non-finite / negative values to sentinel ───────────
    tti = np.where(
        np.isfinite(tti) & (tti >= 0.0),
        tti,
        _UNREACHABLE_TTI,
    )

    return tti


# ---------------------------------------------------------------------------
# Public: Frame-level Voronoi partition
# ---------------------------------------------------------------------------


def partition_voronoi_frame(
    defenders_df: pd.DataFrame,
    grid_xy: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Compute the kinematic Voronoi grid assignment for a single tracking frame.

    Builds the (D × G) TTI matrix via :func:`compute_tti_matrix` and assigns
    each grid point to the defender with the minimum TTI (``np.argmin`` along
    the defender axis).

    Parameters
    ----------
    defenders_df : pd.DataFrame
        Tracking rows for **defensive players only** in one
        ``(gameId, playId, frameId)`` slice.  Must contain ``x``, ``y``,
        ``s``, ``a``, ``dir_rad``, ``nflId``.  **Not mutated.**
    grid_xy : np.ndarray of shape (G, 2)
        Field mesh produced by :func:`build_field_grid`.

    Returns
    -------
    assignment : np.ndarray of shape (G,), dtype int
        For each grid point *g*, the **zero-based row index** into
        *indexed_defenders* of the defender with the minimum TTI.
    indexed_defenders : pd.DataFrame
        Copy of *defenders_df* with a fresh zero-based integer index aligned
        to the values in *assignment*.

    Raises
    ------
    ValueError
        If *defenders_df* is empty (cannot build a Voronoi partition with no
        defenders) or if *grid_xy* has incorrect shape.

    Examples
    --------
    >>> frame = tracking[(tracking.gameId==gid) & (tracking.playId==pid)
    ...                  & (tracking.frameId==fid)]
    >>> defenders = frame[frame.officialPosition.isin(COVERAGE_POSITIONS)]
    >>> grid = build_field_grid(1.0)
    >>> assignment, indexed_def = partition_voronoi_frame(defenders, grid)
    >>> # Count grid points owned by the first defender
    >>> int(np.sum(assignment == 0))
    """
    if defenders_df.empty:
        raise ValueError(
            "partition_voronoi_frame: defenders_df must not be empty — "
            "cannot construct a kinematic Voronoi partition without defenders."
        )
    if grid_xy.ndim != 2 or grid_xy.shape[1] != 2:
        raise ValueError(
            f"partition_voronoi_frame: grid_xy must have shape (G, 2); "
            f"got {grid_xy.shape!r}."
        )

    indexed: pd.DataFrame = defenders_df.reset_index(drop=True).copy()
    def_xy, def_vel, def_acc = _extract_kinematics(indexed)

    tti: np.ndarray = compute_tti_matrix(def_xy, def_vel, def_acc, grid_xy)  # (D, G)
    assignment: np.ndarray = np.argmin(tti, axis=0)                           # (G,)

    return assignment, indexed


# ---------------------------------------------------------------------------
# Public: Defender territory areas across all frames
# ---------------------------------------------------------------------------


def compute_defender_territories(
    tracking_df: pd.DataFrame,
    grid_xy: Optional[np.ndarray] = None,
    grid_resolution: float = DEFAULT_GRID_RESOLUTION,
    coverage_positions: frozenset[str] = COVERAGE_POSITIONS,
) -> pd.DataFrame:
    """
    Compute each defender's kinematic Voronoi territory area for every
    ``(gameId, playId, frameId)`` in the tracking DataFrame.

    The territory is measured as both a raw count of owned grid points and an
    approximate area in square yards (``count × resolution²``).

    Parameters
    ----------
    tracking_df : pd.DataFrame
        Fully normalised tracking DataFrame.  **Not mutated.**
    grid_xy : np.ndarray of shape (G, 2), optional
        Pre-built field grid.  When ``None`` a fresh grid is constructed from
        *grid_resolution*.  Passing a pre-built grid avoids repeated
        allocation across calls.
    grid_resolution : float, default 1.0
        Grid resolution in yards — used only when *grid_xy* is ``None``.
    coverage_positions : frozenset[str], default COVERAGE_POSITIONS
        Positions to treat as coverage defenders.

    Returns
    -------
    pd.DataFrame
        One row per ``(gameId, playId, frameId, defender_nflId)`` with
        columns:

        ``gameId``, ``playId``, ``frameId``
            Frame identifier triple.
        ``defender_nflId`` : float64
            Defender player ID.
        ``territory_grid_points`` : int
            Number of grid points in the defender's Voronoi territory.
        ``territory_area_sq_yd`` : float64
            Approximate territory area in square yards
            (``territory_grid_points × resolution²``).

        Frames with no eligible defenders are silently skipped.

    Examples
    --------
    >>> grid = build_field_grid(1.0)
    >>> territories = compute_defender_territories(tracking, grid_xy=grid)
    >>> # Largest single-frame territory
    >>> territories.nlargest(5, "territory_area_sq_yd")
    """
    if grid_xy is None:
        grid_xy = build_field_grid(grid_resolution)

    grid_cell_area: float = grid_resolution ** 2
    records: list[dict] = []

    for frame_key, frame_group in tracking_df.groupby(_FRAME_KEY):
        game_id, play_id, frame_id = frame_key

        defenders: pd.DataFrame = frame_group[
            frame_group["officialPosition"].isin(coverage_positions)
        ]
        if defenders.empty:
            continue

        try:
            assignment, indexed_def = partition_voronoi_frame(defenders, grid_xy)
        except (ValueError, Exception):
            continue

        n_defenders: int = len(indexed_def)
        for d_idx in range(n_defenders):
            n_points: int = int(np.sum(assignment == d_idx))
            records.append(
                {
                    "gameId": game_id,
                    "playId": play_id,
                    "frameId": frame_id,
                    "defender_nflId": indexed_def.iloc[d_idx]["nflId"],
                    "territory_grid_points": n_points,
                    "territory_area_sq_yd": float(n_points * grid_cell_area),
                }
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "gameId", "playId", "frameId", "defender_nflId",
                "territory_grid_points", "territory_area_sq_yd",
            ]
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Public: Direct-TTI coverage assignment (no grid intermediary)
# ---------------------------------------------------------------------------


def assign_coverage_direct_tti(
    tracking_df: pd.DataFrame,
    coverage_positions: frozenset[str] = COVERAGE_POSITIONS,
    skill_positions: frozenset[str] = SKILL_POSITIONS,
) -> pd.DataFrame:
    """
    Assign coverage responsibility by computing TTI **directly** from every
    defender to every receiver's current position (no grid intermediary).

    This is a faster, grid-free alternative to :func:`build_coverage_assignments`
    that trades spatial territory visualisation for lower compute cost — useful
    for single-play diagnostics or as a cross-validation signal.

    Parameters
    ----------
    tracking_df : pd.DataFrame
        Fully normalised tracking DataFrame.  **Not mutated.**
    coverage_positions : frozenset[str], default COVERAGE_POSITIONS
        Positions treated as coverage defenders.
    skill_positions : frozenset[str], default SKILL_POSITIONS
        Positions treated as coverage targets.

    Returns
    -------
    pd.DataFrame
        Columns: ``gameId``, ``playId``, ``frameId``,
        ``defender_nflId``, ``assigned_receiver_nflId``.

        Each row means: in that frame, *defender_nflId* has the minimum TTI
        to *assigned_receiver_nflId*'s current location (and therefore owns
        primary coverage responsibility for that receiver).  Frames with no
        defenders or no receivers are silently skipped.

    Notes
    -----
    Unlike the grid-based approach, one receiver can be assigned to multiple
    defenders only when their TTI values are exactly tied (rare in practice).
    The ``np.argmin`` selection ensures each receiver receives exactly one
    defender assignment.

    Examples
    --------
    >>> direct = assign_coverage_direct_tti(tracking)
    >>> direct.head()
    """
    records: list[dict] = []

    for frame_key, frame_group in tracking_df.groupby(_FRAME_KEY):
        game_id, play_id, frame_id = frame_key

        defenders: pd.DataFrame = frame_group[
            frame_group["officialPosition"].isin(coverage_positions)
        ].reset_index(drop=True)

        receivers: pd.DataFrame = frame_group[
            frame_group["officialPosition"].isin(skill_positions)
        ].reset_index(drop=True)

        if defenders.empty or receivers.empty:
            continue

        def_xy, def_vel, def_acc = _extract_kinematics(defenders)
        rec_xy: np.ndarray = receivers[["x", "y"]].to_numpy(dtype=np.float64)

        # Compute TTI from each defender (D) to each receiver point (R).
        tti: np.ndarray = compute_tti_matrix(
            def_xy, def_vel, def_acc, rec_xy
        )                                      # (D, R)

        # Each receiver is assigned to the defender with minimum TTI.
        assigned_def_idx: np.ndarray = np.argmin(tti, axis=0)  # (R,)

        for r_idx in range(len(receivers)):
            d_idx: int = int(assigned_def_idx[r_idx])
            records.append(
                {
                    "gameId": game_id,
                    "playId": play_id,
                    "frameId": frame_id,
                    "defender_nflId": defenders.iloc[d_idx]["nflId"],
                    "assigned_receiver_nflId": receivers.iloc[r_idx]["nflId"],
                }
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "gameId", "playId", "frameId",
                "defender_nflId", "assigned_receiver_nflId",
            ]
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Private: single-frame grid-based assignment helper
# ---------------------------------------------------------------------------


def _assign_coverage_for_frame(
    frame_group: pd.DataFrame,
    grid_xy: np.ndarray,
    coverage_positions: frozenset[str],
    skill_positions: frozenset[str],
) -> list[dict]:
    """
    Compute grid-based kinematic Voronoi coverage assignments for one frame.

    Steps
    -----
    1. Partition the field grid among defenders via kinematic TTI Voronoi.
    2. Map each receiver's position to its nearest grid point.
    3. Look up which defender owns that grid point → that defender is
       responsible for that receiver.

    Parameters
    ----------
    frame_group : pd.DataFrame
        All tracking rows for a single ``(gameId, playId, frameId)`` slice.
        **Not mutated.**
    grid_xy : np.ndarray of shape (G, 2)
        Pre-built field mesh.
    coverage_positions : frozenset[str]
        Defensive position filter.
    skill_positions : frozenset[str]
        Offensive receiver position filter.

    Returns
    -------
    list[dict]
        Each dict has keys:
        ``gameId``, ``playId``, ``frameId``,
        ``defender_nflId``, ``assigned_receiver_nflId``.
        Returns an empty list when there are no defenders or no receivers in
        the frame, or when the Voronoi partition raises an exception.
    """
    game_id = frame_group["gameId"].iloc[0]
    play_id = frame_group["playId"].iloc[0]
    frame_id = frame_group["frameId"].iloc[0]

    defenders: pd.DataFrame = frame_group[
        frame_group["officialPosition"].isin(coverage_positions)
    ]
    receivers: pd.DataFrame = frame_group[
        frame_group["officialPosition"].isin(skill_positions)
    ].reset_index(drop=True)

    if defenders.empty or receivers.empty:
        return []

    try:
        assignment, indexed_def = partition_voronoi_frame(defenders, grid_xy)
    except (ValueError, Exception):
        return []

    # Receiver (x, y) array: (R, 2).
    rec_xy: np.ndarray = receivers[["x", "y"]].to_numpy(dtype=np.float64)

    # Squared distances from each receiver to every grid point: (R, G).
    diff_rg: np.ndarray = (
        grid_xy[np.newaxis, :, :]          # (1, G, 2)
        - rec_xy[:, np.newaxis, :]         # (R, 1, 2)
    )                                      # → (R, G, 2)
    sq_dist_rg: np.ndarray = np.sum(diff_rg ** 2, axis=2)  # (R, G)

    # For each receiver, index of its nearest grid point.
    nearest_grid_idx: np.ndarray = np.argmin(sq_dist_rg, axis=1)  # (R,)

    # Defender index that owns each receiver's nearest grid point.
    assigned_def_idx: np.ndarray = assignment[nearest_grid_idx]  # (R,)

    records: list[dict] = []
    for r_idx in range(len(receivers)):
        d_idx: int = int(assigned_def_idx[r_idx])
        records.append(
            {
                "gameId": game_id,
                "playId": play_id,
                "frameId": frame_id,
                "defender_nflId": indexed_def.iloc[d_idx]["nflId"],
                "assigned_receiver_nflId": receivers.iloc[r_idx]["nflId"],
            }
        )

    return records


# ---------------------------------------------------------------------------
# Public: Main entry point — full frame-by-frame assignment table
# ---------------------------------------------------------------------------


def build_coverage_assignments(
    tracking_df: pd.DataFrame,
    grid_resolution: float = DEFAULT_GRID_RESOLUTION,
    coverage_positions: frozenset[str] = COVERAGE_POSITIONS,
    skill_positions: frozenset[str] = SKILL_POSITIONS,
    grid_xy: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Frame-by-frame kinematic Voronoi coverage assignment over the full
    normalised tracking DataFrame.

    This is the primary public entry point.  For each
    ``(gameId, playId, frameId)`` it:

    1. Partitions the field grid among defensive players using kinematic TTI
       Voronoi (fully vectorised with NumPy broadcasting).
    2. Maps each offensive skill player's location to its nearest grid point.
    3. Emits one row per receiver per frame recording which defender has
       primary coverage responsibility.

    Parameters
    ----------
    tracking_df : pd.DataFrame
        Fully normalised tracking DataFrame (output of
        ``src.data.normalizer.build_normalized_tracking``).  Must contain
        ``gameId``, ``playId``, ``frameId``, ``nflId``, ``x``, ``y``,
        ``s``, ``a``, ``dir_rad``, ``officialPosition``.  **Not mutated.**
    grid_resolution : float, default 1.0
        Mesh-grid spacing in yards.  Ignored when *grid_xy* is supplied.
    coverage_positions : frozenset[str], default COVERAGE_POSITIONS
        ``officialPosition`` labels treated as coverage defenders.
    skill_positions : frozenset[str], default SKILL_POSITIONS
        ``officialPosition`` labels treated as coverage targets.
    grid_xy : np.ndarray of shape (G, 2), optional
        Pre-built field grid.  Pass a grid built with :func:`build_field_grid`
        to avoid re-allocating it on every call (recommended for batch
        processing across multiple weeks or calls in a notebook).

    Returns
    -------
    pd.DataFrame
        One row per ``(receiver, frame)`` with columns:

        ``gameId`` : int / object
            Game identifier (matches source dtype).
        ``playId`` : int / object
            Play identifier.
        ``frameId`` : int
            Tracking frame index.
        ``defender_nflId`` : float64
            NFL player ID of the assigned coverage defender.
        ``assigned_receiver_nflId`` : float64
            NFL player ID of the offensive receiver being covered.

        Frames in which no eligible defenders or no eligible receivers are
        present are silently omitted from the output.

    Raises
    ------
    ValueError
        If *tracking_df* is missing any of the required kinematic columns
        listed in ``_REQUIRED_COLS``.

    Notes
    -----
    * The field grid is built once and reused for every frame, amortising
      the allocation cost.
    * The inner TTI computation for each frame uses no Python loops over
      individual players or grid points — all arithmetic is a single
      sequence of NumPy array operations.
    * For very large datasets (e.g. all 8 weeks), consider processing one
      week at a time and concatenating outputs to control peak memory usage.

    Examples
    --------
    One-call usage with default 1-yard grid::

        from src.data.normalizer import load_and_build
        from src.coverage_assignment import build_coverage_assignments

        tracking = load_and_build()
        assignments = build_coverage_assignments(tracking)
        print(assignments.head(10))

    Reuse a pre-built grid across multiple calls::

        from src.coverage_assignment import build_field_grid, build_coverage_assignments

        grid = build_field_grid(resolution=0.5)

        week1 = load_and_build(tracking_path="data/raw/week1.csv", ...)
        week2 = load_and_build(tracking_path="data/raw/week2.csv", ...)

        a1 = build_coverage_assignments(week1, grid_xy=grid)
        a2 = build_coverage_assignments(week2, grid_xy=grid)
    """
    missing_cols: frozenset[str] = _REQUIRED_COLS - frozenset(tracking_df.columns)
    if missing_cols:
        raise ValueError(
            f"build_coverage_assignments: tracking_df is missing required "
            f"columns: {sorted(missing_cols)!r}.  "
            "Ensure the DataFrame was produced by "
            "src.data.normalizer.build_normalized_tracking."
        )

    if grid_xy is None:
        grid_xy = build_field_grid(grid_resolution)

    all_records: list[dict] = []

    for _frame_key, frame_group in tracking_df.groupby(_FRAME_KEY):
        frame_records: list[dict] = _assign_coverage_for_frame(
            frame_group,
            grid_xy,
            coverage_positions,
            skill_positions,
        )
        all_records.extend(frame_records)

    if not all_records:
        return pd.DataFrame(
            columns=[
                "gameId", "playId", "frameId",
                "defender_nflId", "assigned_receiver_nflId",
            ]
        )

    result: pd.DataFrame = pd.DataFrame(all_records)
    return result
