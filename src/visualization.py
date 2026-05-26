"""
NFL Kinematic Voronoi Visualization Engine
==========================================

Stateless visualization engine for NFL coverage-responsibility analysis.
Ingests normalized tracking data (``src.data.normalizer``) and computed
Kinematic Voronoi coverage assignments (``src.coverage_assignment``) to
generate static and animated field visualizations of defender territory maps.

Stateless contract
------------------
Every public plotting function either:

* Returns a ``matplotlib.figure.Figure`` object (optionally saved to disk), or
* Returns a ``matplotlib.animation.FuncAnimation``.

No input DataFrame is ever mutated, no ``inplace=True`` is used, and no
module-level mutable state is updated after import time.

Input schemas
-------------
``tracking_df``
    Output of ``src.data.normalizer.build_normalized_tracking``.
    Required columns: ``gameId``, ``playId``, ``frameId``, ``nflId``,
    ``x``, ``y``, ``s``, ``a``, ``dir_rad``, ``officialPosition``, ``team``.

``assignment_df``
    Output of ``src.coverage_assignment.build_coverage_assignments``.
    Required columns: ``gameId``, ``playId``, ``frameId``,
    ``defender_nflId``, ``assigned_receiver_nflId``.

Public API
----------
``draw_field(ax, dark_mode=True)``
    Draw a canonical 120 × 53.3-yard NFL field with yard lines,
    hashmarks, and endzone boundaries on *ax*.

``plot_voronoi_frame(tracking_df, assignment_df, game_id, play_id, frame_id, ...)``
    Render a single-frame Kinematic Voronoi territory map with player
    positions, nflId labels, and defender→receiver coverage-link overlays.

``animate_voronoi_play(tracking_df, assignment_df, game_id, play_id, ...)``
    Animate consecutive tracking frames of a play as Voronoi territory
    evolution; saves to mp4 or gif when *output_path* is provided.

``plot_frame_static(...)``
    Basic player-position snapshot (no Voronoi) — retained for backward
    compatibility with notebooks using the earlier API.

``plot_side_by_side_comparison(...)``
    Side-by-side before/after normalization comparison.

``animate_play(...)``
    Simple player-position animation (no Voronoi) — retained for
    backward compatibility.

``plot_random_frame_analysis(...)``
    Random-frame data-quality snapshot with detailed annotations.

``print_spatial_validation_summary(frame_data, offense_team, defense_team)``
    Print spatial validation metrics for a single tracking frame.
"""

from __future__ import annotations

import pathlib
from typing import Optional, Union

import matplotlib.animation as manimation
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from src.coverage_assignment import (
    COVERAGE_POSITIONS,
    DEFAULT_GRID_RESOLUTION,
    FIELD_LENGTH,
    FIELD_WIDTH,
    SKILL_POSITIONS,
    build_field_grid,
    partition_voronoi_frame,
)

# ---------------------------------------------------------------------------
# Module-level immutable constants
# ---------------------------------------------------------------------------

#: 12 perceptually distinct defender territory colours (cycled for > 12 defenders).
_DEFENDER_PALETTE: tuple[str, ...] = (
    "#4e79a7",  # steel blue
    "#f28e2b",  # tangerine
    "#e15759",  # coral red
    "#76b7b2",  # teal
    "#59a14f",  # leaf green
    "#edc948",  # golden yellow
    "#b07aa1",  # mauve
    "#ff9da7",  # salmon pink
    "#9c755f",  # warm brown
    "#bab0ac",  # cool gray
    "#d37295",  # magenta rose
    "#a0cbe8",  # sky blue
)

# Dark-mode colour tokens.
_DARK_FIELD_BG: str = "#1a4a1a"
_DARK_ENDZONE_BG: str = "#0d3311"
_DARK_LINE_COLOR: str = "#ffffff"
_DARK_NUMBER_ALPHA: float = 0.55

# Light-mode colour tokens.
_LIGHT_FIELD_BG: str = "#4caf50"
_LIGHT_ENDZONE_BG: str = "#2e7d32"
_LIGHT_LINE_COLOR: str = "#111111"
_LIGHT_NUMBER_ALPHA: float = 0.80

# NFL inbound-line (hashmark) Y-positions in yards from the left sideline.
# Regulation: 70 ft 9 in = 70.75 ft ÷ 3 ft/yd ≈ 23.583 yd from each sideline.
_LEFT_HASH_Y: float = 70.75 / 3.0
_RIGHT_HASH_Y: float = FIELD_WIDTH - _LEFT_HASH_Y

# Playing-field X boundaries (excluding endzones).
_GOAL_LINE_LEFT: float = 10.0
_GOAL_LINE_RIGHT: float = 110.0

# Hashmark half-tick length in yards.
_HASH_HALF_LEN: float = 0.5

# Player scatter sizes (points²).
_PLAYER_MARKER_SIZE: int = 160
_BALL_MARKER_SIZE: int = 100

# Coverage-link line style.
_COVERAGE_LINK_LW: float = 1.4
_COVERAGE_LINK_ALPHA: float = 0.75
_COVERAGE_LINK_LS: str = "--"

# Maximum coverage-link and label artists pre-allocated for animation.
_MAX_LINKS: int = 8
_MAX_LABELS: int = 28


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _grid_dims(grid_resolution: float) -> tuple[int, int]:
    """Return ``(n_xs, n_ys)`` for a grid built at *grid_resolution* yards."""
    xs = np.arange(0.0, FIELD_LENGTH + grid_resolution, grid_resolution)
    ys = np.arange(0.0, FIELD_WIDTH + grid_resolution, grid_resolution)
    return int(len(xs)), int(len(ys))


def _defender_rgb_map(
    all_nfl_ids: "Union[np.ndarray, pd.Series, list]",
) -> dict[float, tuple[float, float, float]]:
    """
    Assign a stable RGB colour to each unique defender nflId.

    Sorting ensures the colour assignment is deterministic regardless of
    DataFrame row order.

    Parameters
    ----------
    all_nfl_ids :
        All unique defender IDs across the scope (one play or one frame).

    Returns
    -------
    dict[float, (r, g, b)]
        Mapping from nflId → normalised RGB triple ``(r, g, b) ∈ [0, 1]``.
    """
    sorted_ids = sorted({float(n) for n in all_nfl_ids if not (isinstance(n, float) and np.isnan(n))})
    return {
        nfl_id: mcolors.to_rgb(_DEFENDER_PALETTE[i % len(_DEFENDER_PALETTE)])
        for i, nfl_id in enumerate(sorted_ids)
    }


def _build_rgba_territory(
    assignment: np.ndarray,
    indexed_defenders: pd.DataFrame,
    color_map: dict[float, tuple[float, float, float]],
    n_xs: int,
    n_ys: int,
    alpha: float,
) -> np.ndarray:
    """
    Build an RGBA image array encoding the Voronoi territory map.

    The field grid is stored in row-major ``(N_x, N_y)`` order (indexing
    ``"ij"``).  This function reshapes the flat assignment vector to
    ``(N_x, N_y)`` and transposes to ``(N_y, N_x)`` so the returned array
    is ready for ``ax.imshow(..., origin='lower')``.

    Parameters
    ----------
    assignment : np.ndarray of shape (G,)
        Per-grid-point defender row index from :func:`partition_voronoi_frame`.
    indexed_defenders : pd.DataFrame
        Reset-indexed defenders whose integer row index aligns with *assignment*.
    color_map : dict
        Mapping from ``nflId`` (float) → ``(r, g, b)`` triple.
    n_xs, n_ys : int
        Grid dimension counts along the X and Y axes.
    alpha : float
        Territory fill opacity ``∈ (0, 1]``.

    Returns
    -------
    np.ndarray of shape (n_ys, n_xs, 4), dtype float32
    """
    rgba = np.zeros((n_ys, n_xs, 4), dtype=np.float32)
    for d_idx in range(len(indexed_defenders)):
        nfl_id = float(indexed_defenders.iloc[d_idx]["nflId"])
        r, g, b = color_map.get(nfl_id, (0.5, 0.5, 0.5))
        # Reshape (G,) → (n_xs, n_ys) then transpose to (n_ys, n_xs).
        mask: np.ndarray = (assignment == d_idx).reshape(n_xs, n_ys).T
        rgba[mask, 0] = r
        rgba[mask, 1] = g
        rgba[mask, 2] = b
        rgba[mask, 3] = alpha
    return rgba


def _configure_axes(ax: Axes, *, dark_mode: bool) -> None:
    """Set limits, aspect, background, and suppress tick decorations."""
    ax.set_xlim(0.0, FIELD_LENGTH)
    ax.set_ylim(0.0, FIELD_WIDTH)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor(_DARK_FIELD_BG if dark_mode else _LIGHT_FIELD_BG)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_coverage_links(
    ax: Axes,
    frame_assignment: pd.DataFrame,
    frame_tracking: pd.DataFrame,
    color_map: dict[float, tuple[float, float, float]],
) -> list[Line2D]:
    """
    Draw dashed coverage-link lines from each defender to their assigned receiver.

    Parameters
    ----------
    ax : Axes
        Target axes.
    frame_assignment : pd.DataFrame
        Assignment rows filtered to this frame
        (columns: ``defender_nflId``, ``assigned_receiver_nflId``).
    frame_tracking : pd.DataFrame
        Tracking rows filtered to this frame (columns: ``nflId``, ``x``, ``y``).
    color_map : dict
        Defender nflId → RGB colour.

    Returns
    -------
    list[Line2D]
        All drawn Line2D artists (for blitting support if needed).
    """
    pos: dict[float, tuple[float, float]] = {
        float(row["nflId"]): (float(row["x"]), float(row["y"]))
        for _, row in frame_tracking.iterrows()
    }
    lines: list[Line2D] = []
    for _, row in frame_assignment.iterrows():
        def_id = float(row["defender_nflId"])
        rec_id = float(row["assigned_receiver_nflId"])
        if def_id not in pos or rec_id not in pos:
            continue
        dx, dy = pos[def_id]
        rx, ry = pos[rec_id]
        rgb = color_map.get(def_id, (0.9, 0.9, 0.9))
        (line,) = ax.plot(
            [dx, rx],
            [dy, ry],
            color=rgb,
            linewidth=_COVERAGE_LINK_LW,
            linestyle=_COVERAGE_LINK_LS,
            alpha=_COVERAGE_LINK_ALPHA,
            zorder=7,
        )
        lines.append(line)
    return lines


def _draw_player_labels(
    ax: Axes,
    frame_tracking: pd.DataFrame,
    dark_mode: bool,
) -> None:
    """
    Annotate each player with their nflId (last 4 digits) and position.

    Labels are placed 1.5 yards above each player marker to avoid overlap
    with the scatter point.

    Parameters
    ----------
    ax : Axes
        Target axes.
    frame_tracking : pd.DataFrame
        Tracking rows for this frame; must contain ``nflId``,
        ``x``, ``y``, ``officialPosition``.
    dark_mode : bool
        Selects text colour contrast.
    """
    text_color = "#ffffffcc" if dark_mode else "#111111cc"
    for _, row in frame_tracking.iterrows():
        pos_label = str(row.get("officialPosition", ""))
        if pos_label in ("BALL", "football"):
            continue
        nfl_id_str = str(int(row["nflId"]))[-4:]
        label = f"{nfl_id_str}\n{pos_label}"
        ax.text(
            float(row["x"]),
            float(row["y"]) + 1.6,
            label,
            fontsize=5.0,
            ha="center",
            va="bottom",
            color=text_color,
            fontfamily="monospace",
            zorder=9,
        )


# ---------------------------------------------------------------------------
# Public: Canonical field renderer
# ---------------------------------------------------------------------------


def draw_field(ax: Axes, *, dark_mode: bool = True) -> None:
    """
    Draw a canonical 120 × 53.3-yard NFL field on *ax*.

    Rendered elements
    -----------------
    * Playing-field background fill and endzone fills.
    * Sidelines and end-zone back boundaries.
    * Goal lines at ``x = 10`` and ``x = 110``.
    * Yard lines every 5 yards across the playing field
      (major lines every 10 yards, minor lines every 5 yards).
    * Yard-number annotations (10 … 50 … 10) above both hash rows.
    * NFL-regulation hashmarks (70 ft 9 in from each sideline)
      at every integer yard from ``x = 11`` to ``x = 109``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.  Mutated in place; limits are left unchanged.
    dark_mode : bool, default True
        Use dark-mode palette when ``True``, light-mode otherwise.

    Notes
    -----
    Call :func:`_configure_axes` before this function to set the axes limits
    and background colour.  The field background rectangle and all field-line
    artists are drawn at ``zorder`` levels 0–6, leaving ``zorder ≥ 7``
    available for data overlays.
    """
    lc = _DARK_LINE_COLOR if dark_mode else _LIGHT_LINE_COLOR
    n_alpha = _DARK_NUMBER_ALPHA if dark_mode else _LIGHT_NUMBER_ALPHA
    field_bg = _DARK_FIELD_BG if dark_mode else _LIGHT_FIELD_BG
    ez_bg = _DARK_ENDZONE_BG if dark_mode else _LIGHT_ENDZONE_BG

    # ── Background fill ────────────────────────────────────────────────────
    ax.add_patch(
        mpatches.Rectangle(
            (0.0, 0.0), FIELD_LENGTH, FIELD_WIDTH,
            color=field_bg, zorder=0,
        )
    )

    # ── Endzone fills ──────────────────────────────────────────────────────
    for x0 in (0.0, _GOAL_LINE_RIGHT):
        ax.add_patch(
            mpatches.Rectangle(
                (x0, 0.0), _GOAL_LINE_LEFT, FIELD_WIDTH,
                color=ez_bg, zorder=1,
            )
        )

    # ── Sidelines ──────────────────────────────────────────────────────────
    for y in (0.0, FIELD_WIDTH):
        ax.axhline(y, color=lc, linewidth=1.6, zorder=5)

    # ── End-zone back boundaries ───────────────────────────────────────────
    for x in (0.0, FIELD_LENGTH):
        ax.axvline(x, color=lc, linewidth=1.6, zorder=5)

    # ── Goal lines ─────────────────────────────────────────────────────────
    for x in (_GOAL_LINE_LEFT, _GOAL_LINE_RIGHT):
        ax.axvline(x, color=lc, linewidth=2.0, zorder=5)

    # ── Interior yard lines ────────────────────────────────────────────────
    for x in range(int(_GOAL_LINE_LEFT) + 5, int(_GOAL_LINE_RIGHT), 5):
        is_ten = (x % 10 == 0)
        ax.axvline(
            x,
            color=lc,
            linewidth=1.5 if is_ten else 1.0,
            alpha=0.5 if is_ten else 0.28,
            zorder=4,
        )

    # ── Yard-number annotations ────────────────────────────────────────────
    # Coordinate → display yard number.
    yard_labels: dict[int, str] = {
        20: "10", 30: "20", 40: "30", 50: "40", 60: "50",
        70: "40", 80: "30", 90: "20", 100: "10",
    }
    for x_coord, label in yard_labels.items():
        # Numbers below the left hash (rotated 0°) and above right hash (180°).
        for y_coord, rotation in (
            (_LEFT_HASH_Y - 2.5, 0),
            (_RIGHT_HASH_Y + 1.5, 180),
        ):
            ax.text(
                x_coord, y_coord, label,
                color=lc, alpha=n_alpha,
                fontsize=7, ha="center", va="center",
                rotation=rotation,
                fontfamily="monospace",
                zorder=6,
            )

    # ── NFL hashmarks at every integer yard ───────────────────────────────
    tick_xs = np.arange(_GOAL_LINE_LEFT + 1.0, _GOAL_LINE_RIGHT, 1.0)
    for y_center in (_LEFT_HASH_Y, _RIGHT_HASH_Y):
        for x in tick_xs:
            ax.plot(
                [x, x],
                [y_center - _HASH_HALF_LEN, y_center + _HASH_HALF_LEN],
                color=lc, linewidth=0.55, alpha=0.45, zorder=5,
            )


# ---------------------------------------------------------------------------
# Public: Static single-frame Voronoi map
# ---------------------------------------------------------------------------


def plot_voronoi_frame(
    tracking_df: pd.DataFrame,
    assignment_df: Optional[pd.DataFrame],
    game_id: Union[int, str],
    play_id: Union[int, str],
    frame_id: int,
    *,
    grid_xy: Optional[np.ndarray] = None,
    grid_resolution: float = DEFAULT_GRID_RESOLUTION,
    figsize: tuple[float, float] = (18, 9),
    dark_mode: bool = True,
    dpi: int = 150,
    territory_alpha: float = 0.45,
    coverage_positions: frozenset[str] = COVERAGE_POSITIONS,
    skill_positions: frozenset[str] = SKILL_POSITIONS,
    output_path: Optional[Union[str, pathlib.Path]] = None,
) -> Figure:
    """
    Render a single-frame Kinematic Voronoi territory map.

    The function partitions the field grid among coverage defenders using
    kinematic TTI Voronoi, fills each defender's territory with a distinct
    semi-transparent colour, and overlays player positions, nflId labels,
    and dashed coverage-assignment links.

    Parameters
    ----------
    tracking_df : pd.DataFrame
        Normalised tracking DataFrame (all frames for the target play are
        accepted; the function filters to *frame_id* internally).
    assignment_df : pd.DataFrame or None
        Coverage assignment table from
        :func:`src.coverage_assignment.build_coverage_assignments`.  When
        ``None`` the Voronoi territory fill is still drawn but no
        coverage-link arrows are rendered.
    game_id : int or str
        Game identifier used for title and output-file naming.
    play_id : int or str
        Play identifier used for title and output-file naming.
    frame_id : int
        Tracking frame to render.
    grid_xy : np.ndarray of shape (G, 2), optional
        Pre-built field grid.  Pass a grid from :func:`build_field_grid`
        to avoid re-allocation; otherwise a fresh grid is constructed from
        *grid_resolution*.
    grid_resolution : float, default 1.0
        Mesh spacing in yards — used only when *grid_xy* is ``None``.
    figsize : (float, float), default (18, 9)
        Figure width × height in inches.
    dark_mode : bool, default True
        Use dark colour palette.
    dpi : int, default 150
        Output resolution in dots per inch.
    territory_alpha : float, default 0.45
        Opacity of the Voronoi territory fills ``∈ (0, 1]``.
    coverage_positions : frozenset[str], default COVERAGE_POSITIONS
        Positions treated as coverage defenders.
    skill_positions : frozenset[str], default SKILL_POSITIONS
        Positions treated as coverage targets.
    output_path : str or Path, optional
        Destination file path for the saved figure.  The format is inferred
        from the file extension (png, pdf, svg, …).  When ``None`` the
        figure is returned without saving.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure.

    Raises
    ------
    ValueError
        If *frame_id* is not present in *tracking_df* for the given
        *game_id* / *play_id*, or if no coverage defenders are found.
    """
    # ── Filter tracking data ───────────────────────────────────────────────
    play_mask = (
        (tracking_df["gameId"] == game_id)
        & (tracking_df["playId"] == play_id)
        & (tracking_df["frameId"] == frame_id)
    )
    frame_df: pd.DataFrame = tracking_df.loc[play_mask]

    if frame_df.empty:
        raise ValueError(
            f"plot_voronoi_frame: no tracking rows found for "
            f"gameId={game_id!r}, playId={play_id!r}, frameId={frame_id!r}."
        )

    defenders: pd.DataFrame = frame_df[
        frame_df["officialPosition"].isin(coverage_positions)
    ]
    receivers: pd.DataFrame = frame_df[
        frame_df["officialPosition"].isin(skill_positions)
    ]
    offense: pd.DataFrame = frame_df[
        ~frame_df["officialPosition"].isin(coverage_positions)
        & (frame_df["officialPosition"] != "BALL")
        & (frame_df["officialPosition"] != "football")
    ]
    ball: pd.DataFrame = frame_df[
        frame_df["officialPosition"].isin({"BALL"})
        | (frame_df["team"] == "football")
    ]

    # ── Build or reuse field grid ─────────────────────────────────────────
    _grid_xy: np.ndarray = (
        grid_xy if grid_xy is not None else build_field_grid(grid_resolution)
    )
    _res = grid_resolution if grid_xy is None else (
        round(float(_grid_xy[1, 0] - _grid_xy[0, 0]) or grid_resolution, 6)
    )
    n_xs, n_ys = _grid_dims(_res)

    # ── Defender colour assignment (stable sort by nflId) ─────────────────
    all_def_ids: list[float] = defenders["nflId"].tolist() if not defenders.empty else []
    color_map = _defender_rgb_map(all_def_ids)

    # ── Figure setup ──────────────────────────────────────────────────────
    bg_color = "#0e1117" if dark_mode else "#f5f5f5"
    fig, ax = plt.subplots(figsize=figsize, facecolor=bg_color)
    _configure_axes(ax, dark_mode=dark_mode)
    draw_field(ax, dark_mode=dark_mode)

    # ── Voronoi territory fill ─────────────────────────────────────────────
    if not defenders.empty:
        try:
            assignment, indexed_def = partition_voronoi_frame(defenders, _grid_xy)
            rgba = _build_rgba_territory(
                assignment, indexed_def, color_map, n_xs, n_ys, territory_alpha
            )
            ax.imshow(
                rgba,
                origin="lower",
                extent=[0.0, FIELD_LENGTH, 0.0, FIELD_WIDTH],
                aspect="auto",
                interpolation="nearest",
                zorder=2,
            )
        except (ValueError, Exception):
            pass  # Guard: render without territory if partition fails.

    # ── Coverage-assignment links ──────────────────────────────────────────
    if assignment_df is not None and not assignment_df.empty:
        frame_asgn_mask = (
            (assignment_df["gameId"] == game_id)
            & (assignment_df["playId"] == play_id)
            & (assignment_df["frameId"] == frame_id)
        )
        frame_asgn: pd.DataFrame = assignment_df.loc[frame_asgn_mask]
        _draw_coverage_links(ax, frame_asgn, frame_df, color_map)

    # ── Player scatter plots ───────────────────────────────────────────────
    _scatter_players(ax, offense, defenders, receivers, ball, color_map, dark_mode)

    # ── Player labels ──────────────────────────────────────────────────────
    _draw_player_labels(ax, frame_df, dark_mode)

    # ── Legend ────────────────────────────────────────────────────────────
    legend_elements = _build_legend_handles(color_map, defenders, dark_mode)
    if legend_elements:
        legend = ax.legend(
            handles=legend_elements,
            loc="upper left",
            fontsize=7.5,
            framealpha=0.85,
            facecolor="#1a1a2e" if dark_mode else "#ffffff",
            edgecolor="#444" if dark_mode else "#ccc",
            labelcolor="#eee" if dark_mode else "#111",
        )
        legend.set_zorder(12)

    # ── Title ─────────────────────────────────────────────────────────────
    title_color = "#eeeeee" if dark_mode else "#111111"
    ax.set_title(
        f"Kinematic Voronoi Coverage  ·  Game {game_id}  |  Play {play_id}  |  Frame {frame_id}",
        fontsize=12,
        color=title_color,
        pad=10,
        fontfamily="monospace",
    )

    fig.tight_layout(pad=1.2)

    # ── Save if requested ─────────────────────────────────────────────────
    if output_path is not None:
        out = pathlib.Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=bg_color)

    return fig


# ---------------------------------------------------------------------------
# Public: Animated Voronoi play renderer
# ---------------------------------------------------------------------------


def animate_voronoi_play(
    tracking_df: pd.DataFrame,
    assignment_df: Optional[pd.DataFrame],
    game_id: Union[int, str],
    play_id: Union[int, str],
    *,
    grid_xy: Optional[np.ndarray] = None,
    grid_resolution: float = DEFAULT_GRID_RESOLUTION,
    figsize: tuple[float, float] = (18, 9),
    dark_mode: bool = True,
    dpi: int = 120,
    fps: int = 10,
    territory_alpha: float = 0.42,
    coverage_positions: frozenset[str] = COVERAGE_POSITIONS,
    skill_positions: frozenset[str] = SKILL_POSITIONS,
    output_path: Optional[Union[str, pathlib.Path]] = None,
) -> manimation.FuncAnimation:
    """
    Animate the Kinematic Voronoi territory map across all frames of a play.

    Each frame is rendered by re-partitioning the field grid and updating
    the territory imshow, player scatter plots, coverage-link lines, and
    player-ID text annotations.  The animation is stitched into an mp4 or
    gif when *output_path* is provided.

    Parameters
    ----------
    tracking_df : pd.DataFrame
        Normalised tracking DataFrame for the target play.
    assignment_df : pd.DataFrame or None
        Coverage assignment table.  When ``None`` territory fills are
        rendered but no defender→receiver links are drawn.
    game_id : int or str
        Game identifier.
    play_id : int or str
        Play identifier.
    grid_xy : np.ndarray of shape (G, 2), optional
        Pre-built field grid reused across all animation frames.
    grid_resolution : float, default 1.0
        Grid mesh spacing — used only when *grid_xy* is ``None``.
    figsize : (float, float), default (18, 9)
        Figure dimensions in inches.
    dark_mode : bool, default True
        Dark-mode colour palette.
    dpi : int, default 120
        Output resolution for saved video/gif.
    fps : int, default 10
        Frames per second for the saved output.
    territory_alpha : float, default 0.42
        Territory fill opacity.
    coverage_positions : frozenset[str], default COVERAGE_POSITIONS
        Defensive position labels for the Voronoi partition.
    skill_positions : frozenset[str], default SKILL_POSITIONS
        Offensive receiver position labels.
    output_path : str or Path, optional
        Destination file path.  The extension determines the format:
        ``.mp4`` uses FFMpegWriter; ``.gif`` uses PillowWriter.
        When ``None`` the animation object is returned without saving.

    Returns
    -------
    matplotlib.animation.FuncAnimation

    Raises
    ------
    ValueError
        If no tracking rows are found for the given *game_id* / *play_id*.
    RuntimeError
        If saving mp4 fails because FFmpeg is not installed on the system.
    """
    # ── Filter play ────────────────────────────────────────────────────────
    play_mask = (
        (tracking_df["gameId"] == game_id)
        & (tracking_df["playId"] == play_id)
    )
    play_df: pd.DataFrame = tracking_df.loc[play_mask]

    if play_df.empty:
        raise ValueError(
            f"animate_voronoi_play: no tracking rows found for "
            f"gameId={game_id!r}, playId={play_id!r}."
        )

    frame_ids: list[int] = sorted(play_df["frameId"].unique().tolist())

    # ── Build or reuse grid ────────────────────────────────────────────────
    _grid_xy: np.ndarray = (
        grid_xy if grid_xy is not None else build_field_grid(grid_resolution)
    )
    _res = grid_resolution if grid_xy is None else (
        round(float(np.diff(_grid_xy[:, 0][_grid_xy[:, 0] > 0]).min()) or grid_resolution, 6)
    )
    n_xs, n_ys = _grid_dims(_res)

    # ── Stable defender colour map across all play frames ─────────────────
    all_def_mask = play_df["officialPosition"].isin(coverage_positions)
    all_def_ids = play_df.loc[all_def_mask, "nflId"].tolist()
    color_map = _defender_rgb_map(all_def_ids)

    # ── Figure and static field setup ─────────────────────────────────────
    bg_color = "#0e1117" if dark_mode else "#f5f5f5"
    fig, ax = plt.subplots(figsize=figsize, facecolor=bg_color)
    _configure_axes(ax, dark_mode=dark_mode)
    draw_field(ax, dark_mode=dark_mode)

    # ── Pre-initialise mutable artists ────────────────────────────────────
    blank_rgba = np.zeros((n_ys, n_xs, 4), dtype=np.float32)
    territory_im = ax.imshow(
        blank_rgba,
        origin="lower",
        extent=[0.0, FIELD_LENGTH, 0.0, FIELD_WIDTH],
        aspect="auto",
        interpolation="nearest",
        zorder=2,
    )

    off_scatter = ax.scatter(
        [], [], c="#ff6b6b", s=_PLAYER_MARKER_SIZE,
        marker="o", edgecolors="#cc0000", linewidths=1.5,
        label="Offense", zorder=8,
    )
    def_scatter = ax.scatter(
        [], [], c="#6bc5ff", s=_PLAYER_MARKER_SIZE,
        marker="s", edgecolors="#005fa3", linewidths=1.5,
        label="Defense", zorder=8,
    )
    rec_scatter = ax.scatter(
        [], [], c="#ffd966", s=_PLAYER_MARKER_SIZE + 30,
        marker="^", edgecolors="#b8860b", linewidths=1.8,
        label="Receiver", zorder=9,
    )
    ball_scatter = ax.scatter(
        [], [], c="#c8a96e", s=_BALL_MARKER_SIZE,
        marker="D", edgecolors="#5a3e1b", linewidths=1.5,
        label="Football", zorder=10,
    )

    # Pre-allocated coverage-link Line2D objects (hidden initially).
    link_lines: list[Line2D] = []
    for _ in range(_MAX_LINKS):
        (ln,) = ax.plot(
            [], [],
            color="white", linewidth=_COVERAGE_LINK_LW,
            linestyle=_COVERAGE_LINK_LS, alpha=0.0, zorder=7,
        )
        link_lines.append(ln)

    # Pre-allocated player label text objects.
    label_texts = [
        ax.text(
            0.0, 0.0, "",
            fontsize=5.0, ha="center", va="bottom",
            color="#ffffffcc" if dark_mode else "#111111cc",
            fontfamily="monospace",
            alpha=0.0, zorder=9,
        )
        for _ in range(_MAX_LABELS)
    ]

    # Frame counter and title.
    title_color = "#eeeeee" if dark_mode else "#111111"
    frame_title = ax.text(
        0.5, 1.012, "",
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=11, color=title_color,
        fontfamily="monospace",
    )

    ax.legend(
        loc="upper right",
        fontsize=7.5,
        framealpha=0.85,
        facecolor="#1a1a2e" if dark_mode else "#ffffff",
        edgecolor="#444" if dark_mode else "#ccc",
        labelcolor="#eee" if dark_mode else "#111",
    )

    def _update(frame_idx: int) -> None:
        """Mutate pre-allocated artists for frame *frame_idx*."""
        fid = frame_ids[frame_idx]
        frame_df = play_df[play_df["frameId"] == fid]

        # ── Territory imshow ───────────────────────────────────────────────
        frame_defenders = frame_df[
            frame_df["officialPosition"].isin(coverage_positions)
        ]
        if not frame_defenders.empty:
            try:
                asgn, idx_def = partition_voronoi_frame(frame_defenders, _grid_xy)
                rgba = _build_rgba_territory(
                    asgn, idx_def, color_map, n_xs, n_ys, territory_alpha
                )
                territory_im.set_data(rgba)
            except (ValueError, Exception):
                territory_im.set_data(blank_rgba)
        else:
            territory_im.set_data(blank_rgba)

        # ── Player scatter positions ───────────────────────────────────────
        frame_rec = frame_df[frame_df["officialPosition"].isin(skill_positions)]
        frame_def = frame_df[frame_df["officialPosition"].isin(coverage_positions)]
        frame_off = frame_df[
            ~frame_df["officialPosition"].isin(coverage_positions)
            & ~frame_df["officialPosition"].isin(skill_positions)
            & ~frame_df["officialPosition"].isin({"BALL"})
            & (frame_df["team"] != "football")
        ]
        frame_ball = frame_df[
            frame_df["officialPosition"].isin({"BALL"})
            | (frame_df["team"] == "football")
        ]

        def _xy(sub: pd.DataFrame) -> np.ndarray:
            if sub.empty:
                return np.empty((0, 2))
            return sub[["x", "y"]].to_numpy(dtype=np.float64)

        off_scatter.set_offsets(_xy(frame_off))
        def_scatter.set_offsets(_xy(frame_def))
        rec_scatter.set_offsets(_xy(frame_rec))
        ball_scatter.set_offsets(_xy(frame_ball))

        # ── Coverage links ─────────────────────────────────────────────────
        # Reset all pre-allocated lines.
        for ln in link_lines:
            ln.set_data([], [])
            ln.set_alpha(0.0)

        if assignment_df is not None and not assignment_df.empty:
            fmask = (
                (assignment_df["gameId"] == game_id)
                & (assignment_df["playId"] == play_id)
                & (assignment_df["frameId"] == fid)
            )
            frame_asgn = assignment_df.loc[fmask]
            pos: dict[float, tuple[float, float]] = {
                float(r["nflId"]): (float(r["x"]), float(r["y"]))
                for _, r in frame_df.iterrows()
            }
            for link_idx, (_, row) in enumerate(frame_asgn.iterrows()):
                if link_idx >= _MAX_LINKS:
                    break
                def_id = float(row["defender_nflId"])
                rec_id = float(row["assigned_receiver_nflId"])
                if def_id not in pos or rec_id not in pos:
                    continue
                rgb = color_map.get(def_id, (0.9, 0.9, 0.9))
                ln = link_lines[link_idx]
                dx, dy = pos[def_id]
                rx, ry = pos[rec_id]
                ln.set_data([dx, rx], [dy, ry])
                ln.set_color(rgb)
                ln.set_alpha(_COVERAGE_LINK_ALPHA)

        # ── Player labels ──────────────────────────────────────────────────
        for txt in label_texts:
            txt.set_alpha(0.0)
            txt.set_text("")

        label_rows = frame_df[
            ~frame_df["officialPosition"].isin({"BALL"})
            & (frame_df["team"] != "football")
        ]
        for lbl_idx, (_, row) in enumerate(label_rows.iterrows()):
            if lbl_idx >= _MAX_LABELS:
                break
            nfl_id_str = str(int(row["nflId"]))[-4:]
            pos_label = str(row.get("officialPosition", ""))
            txt = label_texts[lbl_idx]
            txt.set_position((float(row["x"]), float(row["y"]) + 1.6))
            txt.set_text(f"{nfl_id_str}\n{pos_label}")
            txt.set_alpha(1.0)

        # ── Title ──────────────────────────────────────────────────────────
        frame_title.set_text(
            f"Kinematic Voronoi  ·  Game {game_id}  |  Play {play_id}"
            f"  |  Frame {fid} / {frame_ids[-1]}"
        )

    anim = manimation.FuncAnimation(
        fig,
        _update,
        frames=len(frame_ids),
        interval=int(1000 / fps),
        repeat=True,
    )

    # ── Save if requested ─────────────────────────────────────────────────
    if output_path is not None:
        out = pathlib.Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        ext = out.suffix.lower()
        if ext == ".gif":
            writer = manimation.PillowWriter(fps=fps)
        else:
            # Default to mp4 via FFmpeg.
            try:
                writer = manimation.FFMpegWriter(
                    fps=fps, bitrate=2000,
                    extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"],
                )
            except Exception as exc:
                raise RuntimeError(
                    "animate_voronoi_play: FFmpeg is required to save mp4 files. "
                    "Install FFmpeg or use output_path with a .gif extension."
                ) from exc
        anim.save(str(out), writer=writer, dpi=dpi, savefig_kwargs={"facecolor": bg_color})

    return anim


# ---------------------------------------------------------------------------
# Private scatter helper (used by static frame renderer)
# ---------------------------------------------------------------------------


def _scatter_players(
    ax: Axes,
    offense: pd.DataFrame,
    defenders: pd.DataFrame,
    receivers: pd.DataFrame,
    ball: pd.DataFrame,
    color_map: dict[float, tuple[float, float, float]],
    dark_mode: bool,
) -> None:
    """Scatter-plot all player groups and the football on *ax*."""
    def _xy(sub: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if sub.empty:
            return np.array([]), np.array([])
        return sub["x"].to_numpy(dtype=float), sub["y"].to_numpy(dtype=float)

    # Offense (non-receiver) — semi-transparent red circles.
    pure_off = offense[~offense["officialPosition"].isin(receivers["officialPosition"].unique())]
    ox, oy = _xy(pure_off)
    if len(ox):
        ax.scatter(
            ox, oy, c="#ff6b6b", s=_PLAYER_MARKER_SIZE,
            marker="o", edgecolors="#cc0000", linewidths=1.5,
            label="Offense", zorder=8, alpha=0.9,
        )

    # Receivers — golden triangles.
    rx, ry = _xy(receivers)
    if len(rx):
        ax.scatter(
            rx, ry, c="#ffd966", s=_PLAYER_MARKER_SIZE + 30,
            marker="^", edgecolors="#b8860b", linewidths=1.8,
            label="Receiver (WR/TE/RB)", zorder=9, alpha=0.95,
        )

    # Defenders — per-player colour from color_map (blue fallback).
    for _, row in defenders.iterrows():
        nfl_id = float(row["nflId"])
        rgb = color_map.get(nfl_id, (0.42, 0.77, 1.0))
        ax.scatter(
            float(row["x"]), float(row["y"]),
            c=[rgb], s=_PLAYER_MARKER_SIZE,
            marker="s", edgecolors="#003d66", linewidths=1.5,
            zorder=8, alpha=0.95,
        )

    # Football — brown diamond.
    bx, by = _xy(ball)
    if len(bx):
        ax.scatter(
            bx, by, c="#c8a96e", s=_BALL_MARKER_SIZE,
            marker="D", edgecolors="#5a3e1b", linewidths=1.5,
            label="Football", zorder=10,
        )


def _build_legend_handles(
    color_map: dict[float, tuple[float, float, float]],
    defenders: pd.DataFrame,
    dark_mode: bool,
) -> list:
    """Build matplotlib legend handle objects for the static frame."""
    handles = [
        mpatches.Patch(facecolor="#ff6b6b", edgecolor="#cc0000", label="Offense"),
        mpatches.Patch(facecolor="#ffd966", edgecolor="#b8860b", label="Receiver (WR/TE/RB)"),
        mpatches.Patch(facecolor="#c8a96e", edgecolor="#5a3e1b", label="Football"),
    ]
    for _, row in defenders.iterrows():
        nfl_id = float(row["nflId"])
        rgb = color_map.get(nfl_id, (0.5, 0.5, 0.5))
        pos_label = str(row.get("officialPosition", "DEF"))
        id_short = str(int(nfl_id))[-4:]
        handles.append(
            mpatches.Patch(
                facecolor=rgb, alpha=0.75,
                label=f"#{id_short} {pos_label}",
            )
        )
    return handles


# ---------------------------------------------------------------------------
# Legacy public API — retained for backward compatibility
# ---------------------------------------------------------------------------


def plot_frame_static(
    tracking_df: pd.DataFrame,
    frame_id: int,
    offense_team: str,
    defense_team: str,
    game_id: Optional[Union[int, str]] = None,
    play_id: Optional[Union[int, str]] = None,
    figsize: tuple[float, float] = (15, 8),
    show_jersey_numbers: bool = True,
    dark_mode: bool = True,
    ax: Optional[Axes] = None,
) -> tuple[Figure, Axes]:
    """
    Basic player-position snapshot (no Voronoi).

    Retained for backward compatibility with notebooks using the earlier
    ``visualization`` API.  For Voronoi territory maps use
    :func:`plot_voronoi_frame` instead.

    Parameters
    ----------
    tracking_df : pd.DataFrame
        Tracking data (full play or pre-filtered to one frame).
    frame_id : int
        Frame to render.
    offense_team : str
        Offense team abbreviation (e.g. ``'TB'``).
    defense_team : str
        Defense team abbreviation (e.g. ``'DAL'``).
    game_id : int or str, optional
        Game identifier for the plot title.
    play_id : int or str, optional
        Play identifier for the plot title.
    figsize : (float, float), default (15, 8)
        Figure dimensions in inches.
    show_jersey_numbers : bool, default True
        Annotate scatter points with jersey numbers when ``True``.
    dark_mode : bool, default True
        Dark-mode colour palette.
    ax : Axes, optional
        Existing axes to draw on; creates a new figure when ``None``.

    Returns
    -------
    (Figure, Axes)
    """
    frame_data = tracking_df[tracking_df["frameId"] == frame_id].copy()
    if frame_data.empty:
        raise ValueError(f"plot_frame_static: no data for frame_id={frame_id!r}.")

    offense = frame_data[frame_data["team"] == offense_team]
    defense = frame_data[frame_data["team"] == defense_team]
    ball = frame_data[frame_data["team"] == "football"]

    bg_color = "#0e1117" if dark_mode else "#f5f5f5"
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor=bg_color)
    else:
        fig = ax.get_figure()

    _configure_axes(ax, dark_mode=dark_mode)
    draw_field(ax, dark_mode=dark_mode)

    ax.scatter(
        offense["x"], offense["y"],
        c="#ff6b6b", s=200, marker="o",
        edgecolors="#cc0000", linewidth=2,
        label=f"Offense ({offense_team})", zorder=8,
    )
    ax.scatter(
        defense["x"], defense["y"],
        c="#6bc5ff", s=200, marker="s",
        edgecolors="#005fa3", linewidth=2,
        label=f"Defense ({defense_team})", zorder=8,
    )
    ax.scatter(
        ball["x"], ball["y"],
        c="#c8a96e", s=150, marker="D",
        edgecolors="#5a3e1b", linewidth=2,
        label="Football", zorder=10,
    )

    if show_jersey_numbers:
        tc = "#ffffffcc" if dark_mode else "#111111cc"
        for _, player in pd.concat([offense, defense]).iterrows():
            jn = player.get("jerseyNumber")
            if jn is not None and not (isinstance(jn, float) and np.isnan(jn)):
                ax.text(
                    float(player["x"]), float(player["y"]),
                    str(int(jn)),
                    fontsize=7, ha="center", va="center",
                    color=tc, weight="bold", zorder=9,
                )

    title_color = "#eeeeee" if dark_mode else "#111111"
    parts = ["Player Positions"]
    if game_id is not None:
        parts.append(f"Game {game_id}")
    if play_id is not None:
        parts.append(f"Play {play_id}")
    parts.append(f"Frame {frame_id}")
    ax.set_title(
        "  ·  ".join(parts),
        fontsize=13, color=title_color, pad=10, fontfamily="monospace",
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    fig.tight_layout(pad=1.2)
    return fig, ax


def plot_side_by_side_comparison(
    tracking_before: pd.DataFrame,
    tracking_after: pd.DataFrame,
    frame_id: int,
    offense_team: str,
    defense_team: str,
    title_before: str = "BEFORE",
    title_after: str = "AFTER",
    figsize: tuple[float, float] = (22, 8),
    dark_mode: bool = True,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """
    Side-by-side comparison plot (e.g. before/after normalization).

    Parameters
    ----------
    tracking_before : pd.DataFrame
        Tracking data for the left panel.
    tracking_after : pd.DataFrame
        Tracking data for the right panel.
    frame_id : int
        Frame ID to visualize in both panels.
    offense_team : str
        Offense team abbreviation.
    defense_team : str
        Defense team abbreviation.
    title_before : str, default ``'BEFORE'``
        Left panel subtitle.
    title_after : str, default ``'AFTER'``
        Right panel subtitle.
    figsize : (float, float), default (22, 8)
        Figure size in inches.
    dark_mode : bool, default True
        Dark-mode colour palette.

    Returns
    -------
    (Figure, (Axes, Axes))
    """
    bg_color = "#0e1117" if dark_mode else "#f5f5f5"
    title_color = "#eeeeee" if dark_mode else "#111111"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor=bg_color)

    def _panel(ax: Axes, df: pd.DataFrame, subtitle: str) -> None:
        frame = df[df["frameId"] == frame_id]
        off = frame[frame["team"] == offense_team]
        dfn = frame[frame["team"] == defense_team]
        bl = frame[frame["team"] == "football"]

        _configure_axes(ax, dark_mode=dark_mode)
        draw_field(ax, dark_mode=dark_mode)

        ax.scatter(off["x"], off["y"], c="#ff6b6b", s=200, marker="o",
                   edgecolors="#cc0000", linewidth=2, label=f"Offense ({offense_team})", zorder=8)
        ax.scatter(dfn["x"], dfn["y"], c="#6bc5ff", s=200, marker="s",
                   edgecolors="#005fa3", linewidth=2, label=f"Defense ({defense_team})", zorder=8)
        ax.scatter(bl["x"], bl["y"], c="#c8a96e", s=150, marker="D",
                   edgecolors="#5a3e1b", linewidth=2, label="Football", zorder=10)

        if not frame.empty:
            play_dir = str(frame["playDirection"].iloc[0])
            arrow_x = 35.0 if play_dir == "left" else 75.0
            arrow_dx = -10.0 if play_dir == "left" else 10.0
            arrow_color = "#ff9944" if play_dir == "left" else "#44ff99"
            ax.annotate(
                "", xy=(arrow_x + arrow_dx, 8.0), xytext=(arrow_x, 8.0),
                arrowprops=dict(arrowstyle="->", color=arrow_color, lw=2.5),
                zorder=11,
            )
            ax.set_title(
                f"{subtitle}\nplayDirection = \"{play_dir}\"",
                fontsize=12, color=title_color, pad=8, fontfamily="monospace",
            )
        else:
            ax.set_title(subtitle, fontsize=12, color=title_color, pad=8)

        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    _panel(ax1, tracking_before, title_before)
    _panel(ax2, tracking_after, title_after)
    fig.tight_layout(pad=1.4)
    return fig, (ax1, ax2)


def animate_play(
    tracking_df: pd.DataFrame,
    offense_team: str,
    defense_team: str,
    game_id: Optional[Union[int, str]] = None,
    play_id: Optional[Union[int, str]] = None,
    interval: int = 100,
    figsize: tuple[float, float] = (15, 8),
    dark_mode: bool = True,
    repeat: bool = True,
) -> manimation.FuncAnimation:
    """
    Simple player-position animation (no Voronoi territories).

    Retained for backward compatibility.  For Voronoi territory animations
    use :func:`animate_voronoi_play` instead.

    Parameters
    ----------
    tracking_df : pd.DataFrame
        Play tracking data (all frames).
    offense_team : str
        Offense team abbreviation.
    defense_team : str
        Defense team abbreviation.
    game_id : int or str, optional
        Game ID for the title.
    play_id : int or str, optional
        Play ID for the title.
    interval : int, default 100
        Milliseconds between frames.
    figsize : (float, float), default (15, 8)
        Figure size in inches.
    dark_mode : bool, default True
        Dark-mode colour palette.
    repeat : bool, default True
        Loop the animation.

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    frames = sorted(tracking_df["frameId"].unique())
    bg_color = "#0e1117" if dark_mode else "#f5f5f5"

    fig, ax = plt.subplots(figsize=figsize, facecolor=bg_color)
    _configure_axes(ax, dark_mode=dark_mode)
    draw_field(ax, dark_mode=dark_mode)

    off_sc = ax.scatter([], [], c="#ff6b6b", s=200, marker="o",
                        edgecolors="#cc0000", linewidth=2,
                        label=f"Offense ({offense_team})", zorder=8)
    def_sc = ax.scatter([], [], c="#6bc5ff", s=200, marker="s",
                        edgecolors="#005fa3", linewidth=2,
                        label=f"Defense ({defense_team})", zorder=8)
    ball_sc = ax.scatter([], [], c="#c8a96e", s=150, marker="D",
                         edgecolors="#5a3e1b", linewidth=2,
                         label="Football", zorder=10)

    title_color = "#eeeeee" if dark_mode else "#111111"
    title_txt = ax.text(
        0.5, 1.012, "", transform=ax.transAxes,
        ha="center", va="bottom", fontsize=12,
        color=title_color, fontfamily="monospace",
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)

    def _update(fi: int) -> tuple:
        fid = frames[fi]
        fd = tracking_df[tracking_df["frameId"] == fid]
        off = fd[fd["team"] == offense_team][["x", "y"]].to_numpy()
        dfn = fd[fd["team"] == defense_team][["x", "y"]].to_numpy()
        bl = fd[fd["team"] == "football"][["x", "y"]].to_numpy()
        off_sc.set_offsets(off if len(off) else np.empty((0, 2)))
        def_sc.set_offsets(dfn if len(dfn) else np.empty((0, 2)))
        ball_sc.set_offsets(bl if len(bl) else np.empty((0, 2)))
        parts = ["Play Animation"]
        if game_id is not None:
            parts.append(f"Game {game_id}")
        if play_id is not None:
            parts.append(f"Play {play_id}")
        parts.append(f"Frame {fid} / {frames[-1]}")
        title_txt.set_text("  ·  ".join(parts))
        return off_sc, def_sc, ball_sc, title_txt

    return manimation.FuncAnimation(
        fig, _update, frames=len(frames),
        interval=interval, blit=True, repeat=repeat,
    )


def plot_random_frame_analysis(
    tracking_df: pd.DataFrame,
    offense_team: str,
    defense_team: str,
    game_id: Optional[Union[int, str]] = None,
    play_id: Optional[Union[int, str]] = None,
    seed: Optional[int] = None,
    figsize: tuple[float, float] = (16, 9),
    dark_mode: bool = True,
) -> tuple[Figure, Axes, int, pd.DataFrame]:
    """
    Select a random frame and render a detailed annotated snapshot.

    Parameters
    ----------
    tracking_df : pd.DataFrame
        Play tracking data (all frames).
    offense_team : str
        Offense team abbreviation.
    defense_team : str
        Defense team abbreviation.
    game_id : int or str, optional
        Game ID for the title.
    play_id : int or str, optional
        Play ID for the title.
    seed : int, optional
        NumPy random seed for reproducibility.
    figsize : (float, float), default (16, 9)
    dark_mode : bool, default True

    Returns
    -------
    (Figure, Axes, int, pd.DataFrame)
        Figure, axes, selected frame ID, and frame data slice.
    """
    rng = np.random.default_rng(seed)
    available = tracking_df["frameId"].unique()
    random_frame = int(rng.choice(available))
    frame_data = tracking_df[tracking_df["frameId"] == random_frame].copy()

    offense = frame_data[frame_data["team"] == offense_team]
    defense = frame_data[frame_data["team"] == defense_team]
    ball = frame_data[frame_data["team"] == "football"]

    bg_color = "#0e1117" if dark_mode else "#f5f5f5"
    title_color = "#eeeeee" if dark_mode else "#111111"
    fig, ax = plt.subplots(figsize=figsize, facecolor=bg_color)
    _configure_axes(ax, dark_mode=dark_mode)
    draw_field(ax, dark_mode=dark_mode)

    ax.scatter(offense["x"], offense["y"], c="#ff6b6b", s=250, marker="o",
               edgecolors="#cc0000", linewidth=2.5, label=f"Offense ({offense_team})",
               zorder=8, alpha=0.9)
    ax.scatter(defense["x"], defense["y"], c="#6bc5ff", s=250, marker="s",
               edgecolors="#005fa3", linewidth=2.5, label=f"Defense ({defense_team})",
               zorder=8, alpha=0.9)
    ax.scatter(ball["x"], ball["y"], c="#c8a96e", s=200, marker="D",
               edgecolors="#5a3e1b", linewidth=3, label="Football", zorder=10)

    tc = "#ffffffcc" if dark_mode else "#111111cc"
    for _, player in pd.concat([offense, defense]).iterrows():
        jn = player.get("jerseyNumber")
        if jn is not None and not (isinstance(jn, float) and np.isnan(jn)):
            ax.text(float(player["x"]), float(player["y"]), str(int(jn)),
                    fontsize=8, ha="center", va="center",
                    color=tc, weight="bold", zorder=9)

    title_parts = ["Random Frame Snapshot — Data Quality Validation"]
    info_parts: list[str] = []
    if game_id is not None:
        info_parts.append(f"Game {game_id}")
    if play_id is not None:
        info_parts.append(f"Play {play_id}")
    info_parts.append(f"Frame {random_frame}")
    events = frame_data["event"].dropna().unique() if "event" in frame_data.columns else []
    if len(events):
        info_parts.append(f"Event: {events[0]}")
    if info_parts:
        title_parts.append("  ·  ".join(info_parts))

    ax.set_title(
        "\n".join(title_parts),
        fontsize=13, color=title_color, pad=12, fontfamily="monospace",
    )
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    fig.tight_layout(pad=1.4)
    return fig, ax, random_frame, frame_data


def print_spatial_validation_summary(
    frame_data: pd.DataFrame,
    offense_team: str,
    defense_team: str,
) -> dict:
    """
    Print a spatial validation analysis for a single tracking frame.

    Analyzes field boundary violations, formation statistics, player
    clustering, and team separation.

    Parameters
    ----------
    frame_data : pd.DataFrame
        Tracking data for a single frame.
    offense_team : str
        Offense team abbreviation.
    defense_team : str
        Defense team abbreviation.

    Returns
    -------
    dict
        Validation metrics keyed by analysis category.
    """
    print("=" * 80)
    print("SPATIAL VALIDATION ANALYSIS")
    print("=" * 80)

    offense = frame_data[frame_data["team"] == offense_team]
    defense = frame_data[frame_data["team"] == defense_team]
    ball = frame_data[frame_data["team"] == "football"]
    all_entities = pd.concat([offense, defense, ball])

    x_violations = all_entities[
        (all_entities["x"] < 0) | (all_entities["x"] > FIELD_LENGTH)
    ]
    y_violations = all_entities[
        (all_entities["y"] < 0) | (all_entities["y"] > FIELD_WIDTH)
    ]
    print("\n1. FIELD BOUNDARY CHECKS:")
    print(f"   X-coordinate violations (outside 0–{FIELD_LENGTH}): {len(x_violations)}")
    print(f"   Y-coordinate violations (outside 0–{FIELD_WIDTH}): {len(y_violations)}")
    field_valid = len(x_violations) == 0 and len(y_violations) == 0
    print("   ✓ All entities within valid field boundaries" if field_valid
          else "   ⚠  WARNING: Entities detected outside field boundaries!")

    ball_x = float(ball["x"].iloc[0]) if len(ball) else None
    ball_y = float(ball["y"].iloc[0]) if len(ball) else None
    print("\n2. BALL POSITION:")
    if ball_x is not None:
        print(f"   X = {ball_x:.2f} yd, Y = {ball_y:.2f} yd")
    else:
        print("   Ball not found in frame.")

    def _form_stats(grp: pd.DataFrame, label: str) -> tuple[float, float, float, float]:
        if grp.empty:
            print(f"\n{label}: no players found.")
            return 0.0, 0.0, 0.0, 0.0
        x_range = float(grp["x"].max() - grp["x"].min())
        y_range = float(grp["y"].max() - grp["y"].min())
        x_mean = float(grp["x"].mean())
        x_std = float(grp["x"].std(ddof=0))
        print(f"\n{label}:")
        print(f"   X spread: {x_range:.2f} yd  (min {grp['x'].min():.2f}, max {grp['x'].max():.2f})")
        print(f"   Y spread: {y_range:.2f} yd")
        print(f"   Mean X: {x_mean:.2f} ± {x_std:.2f} yd")
        return x_range, y_range, x_mean, x_std

    print()
    _, _, off_x_mean, _ = _form_stats(offense, "3. OFFENSIVE FORMATION")
    _, _, def_x_mean, _ = _form_stats(defense, "4. DEFENSIVE FORMATION")

    separation = def_x_mean - off_x_mean
    print("\n5. TEAM SEPARATION:")
    print(f"   Avg defensive depth: {separation:.2f} yd")
    if separation < -5:
        print("   ⚠  SUSPICIOUS: Defense ahead of offense.")
    elif separation > 30:
        print("   ⚠  UNUSUAL: Very deep defensive alignment.")
    else:
        print("   ✓ Reasonable defensive positioning.")

    print("\n6. PLAYER CLUSTERING (minimum pair distance):")
    for grp, name in ((offense, "Offensive"), (defense, "Defensive")):
        if len(grp) < 2:
            continue
        xy = grp[["x", "y"]].to_numpy(dtype=float)
        diffs = xy[:, np.newaxis, :] - xy[np.newaxis, :, :]
        dists = np.sqrt((diffs ** 2).sum(axis=2))
        np.fill_diagonal(dists, np.inf)
        min_d = float(dists.min())
        print(f"   {name} min separation: {min_d:.2f} yd", end="")
        if min_d < 1.0:
            print("  ⚠  extremely close")
        elif min_d < 2.0:
            print("  ⚠  tight clustering")
        else:
            print("  ✓")

    print("\n" + "=" * 80)
    return {
        "field_valid": field_valid,
        "x_violations": int(len(x_violations)),
        "y_violations": int(len(y_violations)),
        "ball_position": (ball_x, ball_y),
        "offense_x_mean": off_x_mean,
        "defense_x_mean": def_x_mean,
        "team_separation": separation,
    }
