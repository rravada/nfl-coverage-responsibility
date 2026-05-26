"""
NFL Next Gen Stats Tracking Normalization Module
================================================

Ingests raw NFL tracking frames (week*.csv) and enriches them with player
and PFF scouting metadata, then applies two layers of spatial normalization:

  1. **Play-direction standardization** – all plays are transformed so that
     the offense always attacks toward increasing X (left → right). Plays
     already going right are returned unchanged.

  2. **Kinematic angle alignment** – raw NFL tracking angles for orientation
     (``o``) and movement direction (``dir``) are expressed in a coordinate
     system where **0° points straight up the field along the positive Y-axis
     and angles increase clockwise**. These are re-expressed as standard
     mathematical Cartesian radians where **0 rad points along the positive
     X-axis and angles increase counter-clockwise**, enabling direct use with
     NumPy/SciPy trigonometric functions without caller-side conversion.

Stateless contract
------------------
Every public function operates on an **explicit copy** of its input – no
argument is ever mutated, no ``inplace=True`` is used, and no module-level
variable is re-bound after initialisation.

Output schema additions
-----------------------
After a full pipeline pass (``build_normalized_tracking``) the returned
DataFrame contains every original tracking column plus:

  ``officialPosition`` (str)
      Position label from ``players.csv``; ``"BALL"`` for the football entity.

  ``pff_role`` (str | NaN)
      PFF assignment role (e.g. ``"Coverage"``, ``"Pass Route"``) from
      ``pffScoutingData.csv``; NaN for the football or absent rows.

  ``pff_positionLinedUp`` (str | NaN)
      PFF pre-snap alignment slot (e.g. ``"CB-L"``, ``"TE-R"``).

  *(all other pff_ columns from pffScoutingData.csv are also retained)*

  ``o_rad`` (float64)
      Body orientation in standard Cartesian radians.

  ``dir_rad`` (float64)
      Movement direction in standard Cartesian radians.

  ``playDirection``
      Normalised to ``"right"`` for every row once direction standardisation
      has been applied.

Usage
-----
Minimal one-call entry point::

    from src.data.normalizer import load_and_build

    tracking = load_and_build(
        tracking_path="data/raw/week1.csv",
        players_path="data/raw/players.csv",
        pff_path="data/raw/pffScoutingData.csv",
        plays_path="data/raw/plays.csv",
        games_path="data/raw/games.csv",
    )

Or build the pipeline from pre-loaded DataFrames::

    from src.data.normalizer import build_normalized_tracking

    clean = build_normalized_tracking(tracking_df, players_df, pff_df)
"""

from __future__ import annotations

import pathlib
from typing import Optional, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Immutable field-geometry and angle-transform constants
# ---------------------------------------------------------------------------

#: Total field length in yards, endzone to endzone (X axis).
FIELD_LENGTH: float = 120.0

#: Total field width in yards, sideline to sideline (Y axis).
FIELD_WIDTH: float = 53.3

#: Degrees added to an angle when mirroring a left-travelling play.
DIRECTION_OFFSET: float = 180.0

#: Modulo base for wrapping angles back into [0, 360).
FULL_ROTATION: float = 360.0

#: Degrees offset used to convert NFL tracking angles (0° = +Y, clockwise)
#: to standard mathematical angles (0° = +X, counter-clockwise).
#: Derivation: math_deg = 90 − nfl_deg  →  math_rad = π/2 − nfl_rad
NFL_ANGLE_NORTH_OFFSET: float = 90.0

#: Sentinel position label assigned to the football entity (nflId is NaN in
#: the raw tracking data for the ball row).
BALL_POSITION_GROUP: str = "BALL"

# ---------------------------------------------------------------------------
# Internal path helpers
# ---------------------------------------------------------------------------

_PathLike = Union[str, pathlib.Path]
_DATA_RAW: pathlib.Path = pathlib.Path("data") / "raw"


def _resolve(path: Optional[_PathLike], default_filename: str) -> pathlib.Path:
    """Return *path* as a ``Path``, falling back to the default raw-data location."""
    if path is not None:
        return pathlib.Path(path)
    return _DATA_RAW / default_filename


# ---------------------------------------------------------------------------
# I/O – raw file loaders
# ---------------------------------------------------------------------------


def load_tracking(
    path: Optional[_PathLike] = None,
) -> pd.DataFrame:
    """
    Load a raw NFL Next Gen Stats tracking CSV (e.g. ``week1.csv``).

    Parameters
    ----------
    path:
        Explicit path to the tracking CSV.  Defaults to
        ``data/raw/week1.csv`` relative to the working directory.

    Returns
    -------
    pd.DataFrame
        Raw tracking frame with columns:
        ``gameId``, ``playId``, ``nflId``, ``frameId``, ``time``,
        ``jerseyNumber``, ``team``, ``playDirection``,
        ``x``, ``y``, ``s``, ``a``, ``dis``, ``o``, ``dir``, ``event``.

    Notes
    -----
    ``nflId`` is kept as a nullable float because the football row has no
    player ID (represented as NaN in the source file).
    """
    resolved = _resolve(path, "week1.csv")
    df = pd.read_csv(resolved, low_memory=False)
    return df.copy()


def load_players(
    path: Optional[_PathLike] = None,
) -> pd.DataFrame:
    """
    Load the ``players.csv`` metadata file.

    Parameters
    ----------
    path:
        Explicit path.  Defaults to ``data/raw/players.csv``.

    Returns
    -------
    pd.DataFrame
        Player metadata with at minimum:
        ``nflId``, ``officialPosition``, ``displayName``,
        ``height``, ``weight``, ``birthDate``, ``collegeName``.
    """
    resolved = _resolve(path, "players.csv")
    df = pd.read_csv(resolved, low_memory=False)
    return df.copy()


def load_pff_scouting(
    path: Optional[_PathLike] = None,
) -> pd.DataFrame:
    """
    Load the ``pffScoutingData.csv`` file containing per-play player roles.

    Parameters
    ----------
    path:
        Explicit path.  Defaults to ``data/raw/pffScoutingData.csv``.

    Returns
    -------
    pd.DataFrame
        PFF scouting data with at minimum:
        ``gameId``, ``playId``, ``nflId``, ``pff_role``,
        ``pff_positionLinedUp`` and associated blocking / pass-rush flags.
    """
    resolved = _resolve(path, "pffScoutingData.csv")
    df = pd.read_csv(resolved, low_memory=False)
    return df.copy()


def load_plays(
    path: Optional[_PathLike] = None,
) -> pd.DataFrame:
    """
    Load the ``plays.csv`` play-context file.

    Parameters
    ----------
    path:
        Explicit path.  Defaults to ``data/raw/plays.csv``.

    Returns
    -------
    pd.DataFrame
        Play context with at minimum:
        ``gameId``, ``playId``, ``possessionTeam``, ``defensiveTeam``,
        ``passResult``, ``pff_passCoverage``, ``pff_passCoverageType``.
    """
    resolved = _resolve(path, "plays.csv")
    df = pd.read_csv(resolved, low_memory=False)
    return df.copy()


def load_games(
    path: Optional[_PathLike] = None,
) -> pd.DataFrame:
    """
    Load the ``games.csv`` scheduling file.

    Parameters
    ----------
    path:
        Explicit path.  Defaults to ``data/raw/games.csv``.

    Returns
    -------
    pd.DataFrame
        Game scheduling data with at minimum:
        ``gameId``, ``season``, ``week``, ``gameDate``,
        ``homeTeamAbbr``, ``visitorTeamAbbr``.
    """
    resolved = _resolve(path, "games.csv")
    df = pd.read_csv(resolved, low_memory=False)
    return df.copy()


# ---------------------------------------------------------------------------
# Schema-merging pipeline
# ---------------------------------------------------------------------------


def merge_player_metadata(
    tracking_df: pd.DataFrame,
    players_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join tracking data with player metadata on ``nflId``.

    The football entity (whose ``nflId`` is NaN in raw tracking data) does
    not match any player row.  After the join its ``officialPosition`` cell
    is therefore NaN; this function fills it with ``BALL_POSITION_GROUP``
    (``"BALL"``) so every downstream filter can treat the ball as a named
    entity rather than a missing value.

    Parameters
    ----------
    tracking_df:
        Raw or partially-processed tracking DataFrame.  Must contain
        ``nflId`` (nullable float).  **Not mutated.**
    players_df:
        Player metadata DataFrame.  Must contain ``nflId`` and
        ``officialPosition``.  **Not mutated.**

    Returns
    -------
    pd.DataFrame
        Copy of *tracking_df* with ``officialPosition`` (and all other
        ``players_df`` columns) appended.  The ``displayName``,
        ``height``, ``weight``, ``birthDate``, and ``collegeName``
        columns are included but callers may drop them if not required.

    Notes
    -----
    * A left join is used so that every tracking row is preserved even if
      the player is absent from the roster file (edge case: practice squad
      elevation, tracking system ghost rows, etc.).
    * ``inplace`` operations are never used; the input DataFrames are
      not modified.
    """
    # Work on independent copies to guarantee caller frames are unchanged.
    tracking_copy = tracking_df.copy()
    players_copy = players_df.copy()

    # Avoid column collision if tracking already has a position column from
    # a previous pipeline pass.
    if "officialPosition" in tracking_copy.columns:
        tracking_copy = tracking_copy.drop(columns=["officialPosition"])

    merged = tracking_copy.merge(
        players_copy[
            [
                "nflId",
                "officialPosition",
                "displayName",
                "height",
                "weight",
                "birthDate",
                "collegeName",
            ]
        ],
        on="nflId",
        how="left",
    )

    # Football rows have no nflId and therefore no officialPosition after the
    # join.  Fill with the sentinel so filters never see NaN here.
    merged["officialPosition"] = merged["officialPosition"].fillna(
        BALL_POSITION_GROUP
    )

    return merged


def merge_pff_scouting(
    tracking_df: pd.DataFrame,
    pff_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join tracking data with PFF scouting assignments.

    Joins on the composite key ``['gameId', 'playId', 'nflId']`` so that
    each tracking row is annotated with the PFF role, pre-snap alignment,
    and pass-blocking metrics recorded for that player on that specific play.

    The football entity and any player absent from ``pffScoutingData.csv``
    will receive NaN values in all PFF columns; callers should treat these
    as "not applicable" rather than missing data errors.

    Parameters
    ----------
    tracking_df:
        Tracking DataFrame (may already have ``officialPosition`` appended
        by ``merge_player_metadata``).  Must contain ``gameId``, ``playId``,
        ``nflId``.  **Not mutated.**
    pff_df:
        PFF scouting DataFrame.  Must contain ``gameId``, ``playId``,
        ``nflId``.  **Not mutated.**

    Returns
    -------
    pd.DataFrame
        Copy of *tracking_df* with all PFF columns appended.  PFF columns
        include at minimum ``pff_role`` and ``pff_positionLinedUp``; all
        other columns present in *pff_df* are also attached.
    """
    tracking_copy = tracking_df.copy()
    pff_copy = pff_df.copy()

    # Identify which PFF columns to bring across (exclude the join keys so
    # they are not duplicated in the output).
    join_keys = ["gameId", "playId", "nflId"]
    pff_payload_cols = [c for c in pff_copy.columns if c not in join_keys]

    # Drop any PFF columns that are already present to allow idempotent calls.
    cols_to_drop = [c for c in pff_payload_cols if c in tracking_copy.columns]
    if cols_to_drop:
        tracking_copy = tracking_copy.drop(columns=cols_to_drop)

    merged = tracking_copy.merge(
        pff_copy[join_keys + pff_payload_cols],
        on=join_keys,
        how="left",
    )

    return merged


# ---------------------------------------------------------------------------
# Spatial normalization
# ---------------------------------------------------------------------------


def normalize_play_direction(tracking_df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize all tracking frames so that the offense always advances
    toward increasing X (left → right across the field).

    In the raw Next Gen Stats data, roughly half of all plays are recorded
    with the offense moving from right to left (``playDirection == 'left'``).
    Mixing both orientations makes spatial models position-dependent in a
    spurious way.  This function mirrors those plays so every frame shares a
    single canonical orientation.

    Transformation applied **only** to rows where ``playDirection == 'left'``:

    .. code-block:: text

        x_new   = FIELD_LENGTH  − x_old          (120.0 − x)
        y_new   = FIELD_WIDTH   − y_old           (53.3  − y)
        o_new   = (o_old   + DIRECTION_OFFSET) % FULL_ROTATION
        dir_new = (dir_old + DIRECTION_OFFSET) % FULL_ROTATION

    Rows where ``playDirection == 'right'`` pass through unchanged.

    Parameters
    ----------
    tracking_df:
        Tracking DataFrame with columns ``playDirection``, ``x``, ``y``,
        ``o``, ``dir``.  **Not mutated.**

    Returns
    -------
    pd.DataFrame
        Copy of *tracking_df* with spatial columns transformed for
        left-travelling plays and ``playDirection`` set to ``"right"``
        uniformly across all rows.

    Notes
    -----
    * All four numeric columns (``x``, ``y``, ``o``, ``dir``) are mutated
      exclusively via boolean-indexed ``.loc`` assignment on the returned
      copy; the original DataFrame is never touched.
    * ``inplace=True`` is not used anywhere in this function.
    """
    df = tracking_df.copy()

    going_left: pd.Series = df["playDirection"] == "left"

    # Spatial mirror across the field centre-point.
    df.loc[going_left, "x"] = FIELD_LENGTH - df.loc[going_left, "x"]
    df.loc[going_left, "y"] = FIELD_WIDTH - df.loc[going_left, "y"]

    # Angle mirror: add 180° and wrap into [0, 360).
    df.loc[going_left, "o"] = (
        df.loc[going_left, "o"] + DIRECTION_OFFSET
    ) % FULL_ROTATION
    df.loc[going_left, "dir"] = (
        df.loc[going_left, "dir"] + DIRECTION_OFFSET
    ) % FULL_ROTATION

    # Stamp every row as right-travelling for downstream consistency.
    df.loc[going_left, "playDirection"] = "right"

    return df


def convert_angles_to_radians(tracking_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert NFL tracking angles to standard Cartesian radians and append
    ``o_rad`` and ``dir_rad`` columns.

    Coordinate system mismatch
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    NFL Next Gen Stats records ``o`` (body orientation) and ``dir`` (movement
    direction) using a **field-centric clockwise** convention:

    * **0°** = pointing straight up the field (toward the positive Y-axis /
      far sideline when attacking rightward).
    * Angles **increase clockwise**: 90° = pointing toward the right sideline
      (positive X direction), 180° = pointing back toward own endzone, 270°
      = pointing toward the left sideline (negative X direction).

    Standard mathematical (Cartesian) radians use the opposite handedness:

    * **0 rad** = pointing along the positive X-axis (rightward).
    * Angles **increase counter-clockwise**: π/2 = pointing up (+Y), π =
      pointing left (−X), 3π/2 or −π/2 = pointing down (−Y).

    Conversion derivation
    ~~~~~~~~~~~~~~~~~~~~~
    Let ``θ_nfl`` be an angle in NFL degrees.  The equivalent Cartesian angle
    in degrees is ``90 − θ_nfl`` (shift the reference axis from +Y to +X and
    reverse the rotation sense).  Converting to radians::

        θ_cart_rad = radians(NFL_ANGLE_NORTH_OFFSET − θ_nfl)
                   = π/2 − radians(θ_nfl)

    Verification:

    +-----------+---------------------------+--------------------------------+
    | NFL (deg) | Expected Cartesian        | Formula result                 |
    +===========+===========================+================================+
    | 0         | π/2 (pointing up, +Y)     | radians(90 − 0) = π/2  ✓       |
    | 90        | 0 (pointing right, +X)    | radians(90 − 90) = 0   ✓       |
    | 180       | −π/2 (pointing down, −Y)  | radians(90 − 180) = −π/2 ✓     |
    | 270       | −π / π (pointing left, −X)| radians(90 − 270) = −π ✓       |
    +-----------+---------------------------+--------------------------------+

    Parameters
    ----------
    tracking_df:
        Tracking DataFrame with numeric ``o`` and ``dir`` columns expressing
        angles in NFL-convention degrees.  **Not mutated.**

    Returns
    -------
    pd.DataFrame
        Copy of *tracking_df* with two new float64 columns appended:

        ``o_rad``
            Body orientation in standard Cartesian radians.
        ``dir_rad``
            Movement direction in standard Cartesian radians.

    Notes
    -----
    * NaN values in ``o`` or ``dir`` (e.g. stationary football rows) are
      propagated transparently; ``np.radians(NaN)`` returns NaN.
    * The ``o`` and ``dir`` source columns are preserved unchanged so that
      callers requiring the original degree values are not broken.
    * ``inplace=True`` is not used.
    """
    df = tracking_df.copy()

    df["o_rad"] = np.radians(NFL_ANGLE_NORTH_OFFSET - df["o"])
    df["dir_rad"] = np.radians(NFL_ANGLE_NORTH_OFFSET - df["dir"])

    return df


# ---------------------------------------------------------------------------
# Full normalization pipeline
# ---------------------------------------------------------------------------


def build_normalized_tracking(
    tracking_df: pd.DataFrame,
    players_df: pd.DataFrame,
    pff_df: pd.DataFrame,
    plays_df: Optional[pd.DataFrame] = None,
    games_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Execute the full normalization pipeline on pre-loaded DataFrames.

    Pipeline stages
    ---------------
    1. Merge player position metadata (``officialPosition``) from
       *players_df* on ``nflId``.
    2. Merge PFF scouting assignments (``pff_role``, ``pff_positionLinedUp``,
       etc.) from *pff_df* on ``['gameId', 'playId', 'nflId']``.
    3. *(Optional)* Left-join play context from *plays_df* on
       ``['gameId', 'playId']`` if provided.
    4. *(Optional)* Left-join game scheduling data from *games_df* on
       ``'gameId'`` if provided.
    5. Mirror all left-travelling plays to canonical left-to-right
       orientation via ``normalize_play_direction``.
    6. Append ``o_rad`` and ``dir_rad`` Cartesian radian columns via
       ``convert_angles_to_radians``.

    Parameters
    ----------
    tracking_df:
        Raw tracking DataFrame (e.g. loaded from ``week1.csv``).
        **Not mutated.**
    players_df:
        Player metadata DataFrame (from ``players.csv``).  **Not mutated.**
    pff_df:
        PFF scouting DataFrame (from ``pffScoutingData.csv``).
        **Not mutated.**
    plays_df:
        Optional play-context DataFrame (from ``plays.csv``).  When
        provided it is left-joined on ``['gameId', 'playId']``.
        **Not mutated.**
    games_df:
        Optional game-scheduling DataFrame (from ``games.csv``).  When
        provided it is left-joined on ``'gameId'``.  **Not mutated.**

    Returns
    -------
    pd.DataFrame
        Fully normalised tracking DataFrame containing all original columns
        plus ``officialPosition``, all PFF columns, ``o_rad``, ``dir_rad``,
        and optionally play/game context columns.  Every row represents one
        entity (player or football) at one tracking frame.

    Notes
    -----
    * No global state is read from or written to.
    * No argument is mutated; every intermediate result is a fresh copy.
    * ``inplace=True`` is not used at any stage.
    """
    result: pd.DataFrame = tracking_df.copy()

    # --- Stage 1: player position metadata ---
    result = merge_player_metadata(result, players_df)

    # --- Stage 2: PFF scouting assignments ---
    result = merge_pff_scouting(result, pff_df)

    # --- Stage 3 (optional): play context ---
    if plays_df is not None:
        plays_copy = plays_df.copy()
        play_join_keys = ["gameId", "playId"]
        # Avoid duplicating columns already present in tracking.
        play_cols_to_add = [
            c
            for c in plays_copy.columns
            if c not in play_join_keys and c not in result.columns
        ]
        result = result.merge(
            plays_copy[play_join_keys + play_cols_to_add],
            on=play_join_keys,
            how="left",
        )

    # --- Stage 4 (optional): game scheduling context ---
    if games_df is not None:
        games_copy = games_df.copy()
        game_join_key = "gameId"
        game_cols_to_add = [
            c
            for c in games_copy.columns
            if c != game_join_key and c not in result.columns
        ]
        result = result.merge(
            games_copy[[game_join_key] + game_cols_to_add],
            on=game_join_key,
            how="left",
        )

    # --- Stage 5: play-direction normalisation ---
    result = normalize_play_direction(result)

    # --- Stage 6: kinematic angle alignment ---
    result = convert_angles_to_radians(result)

    return result


def load_and_build(
    tracking_path: Optional[_PathLike] = None,
    players_path: Optional[_PathLike] = None,
    pff_path: Optional[_PathLike] = None,
    plays_path: Optional[_PathLike] = None,
    games_path: Optional[_PathLike] = None,
) -> pd.DataFrame:
    """
    Convenience entry-point: load all source files and run the full pipeline.

    This function is the recommended single-call interface for notebooks and
    scripts that simply want a clean, normalised tracking matrix without
    managing individual DataFrames.

    Parameters
    ----------
    tracking_path:
        Path to the tracking CSV.  Defaults to ``data/raw/week1.csv``.
    players_path:
        Path to ``players.csv``.  Defaults to ``data/raw/players.csv``.
    pff_path:
        Path to ``pffScoutingData.csv``.  Defaults to
        ``data/raw/pffScoutingData.csv``.
    plays_path:
        Path to ``plays.csv``.  Defaults to ``data/raw/plays.csv``.
        Pass ``None`` explicitly to skip play-context merging.
    games_path:
        Path to ``games.csv``.  Defaults to ``data/raw/games.csv``.
        Pass ``None`` explicitly to skip game-scheduling merging.

    Returns
    -------
    pd.DataFrame
        Fully normalised tracking matrix (see ``build_normalized_tracking``
        for the complete output schema).

    Examples
    --------
    Default paths (project root as working directory)::

        from src.data.normalizer import load_and_build

        tracking = load_and_build()
        print(tracking.columns.tolist())
        print(tracking[["x", "y", "o_rad", "dir_rad", "officialPosition"]].head())

    Explicit paths::

        tracking = load_and_build(
            tracking_path="data/raw/week1.csv",
            players_path="data/raw/players.csv",
            pff_path="data/raw/pffScoutingData.csv",
            plays_path="data/raw/plays.csv",
            games_path="data/raw/games.csv",
        )
    """
    tracking_df = load_tracking(tracking_path)
    players_df = load_players(players_path)
    pff_df = load_pff_scouting(pff_path)
    plays_df = load_plays(plays_path)
    games_df = load_games(games_path)

    return build_normalized_tracking(
        tracking_df=tracking_df,
        players_df=players_df,
        pff_df=pff_df,
        plays_df=plays_df,
        games_df=games_df,
    )
