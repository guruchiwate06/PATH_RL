"""
visualize.py
------------
Minimal development visualizer for the evacuation simulation.

This is a *development utility only* — it lives outside the core
simulation package and has no effect on simulation logic.

Usage
-----
::

    from visualize import render_grid, save_grid_snapshot

    sim = Simulation(config, ShortestPathStrategy())
    render_grid(sim, title="Step 0")

    for step_idx in range(10):
        sim.step()
        render_grid(sim, title=f"Step {sim.state.timestep}")

Requires matplotlib.  If not installed, functions degrade gracefully
with a printed warning rather than crashing the simulation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evacuation_simulation.simulation.simulation import Simulation


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_COLOUR_OPEN = "#F0F0F0"    # light grey  — traversable cell
_COLOUR_WALL = "#2C2C2C"    # near-black  — wall
_COLOUR_EXIT = "#27AE60"    # green       — exit
_COLOUR_AGENT_ACTIVE = "#2980B9"   # blue   — active agent
_COLOUR_AGENT_EVAC = "#95A5A6"     # grey   — evacuated agent


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_grid(
    sim: "Simulation",
    title: str = "",
    *,
    show: bool = True,
    save_path: str | Path | None = None,
) -> None:
    """
    Render the current simulation state as a 2D grid image.

    Parameters
    ----------
    sim : Simulation
        The simulation to visualize.
    title : str
        Optional title displayed above the grid.
    show : bool
        If True, display the figure interactively (``plt.show()``).
    save_path : str or Path, optional
        If provided, save the figure to this path instead of (or in
        addition to) displaying it.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[visualize] matplotlib not installed — skipping render.")
        return

    env = sim.environment
    rows, cols = env.rows, env.cols

    fig, ax = plt.subplots(figsize=(max(6, cols * 0.7), max(5, rows * 0.7)))
    ax.set_aspect("equal")
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.invert_yaxis()  # row 0 at top

    # Draw cells
    for r in range(rows):
        for c in range(cols):
            cell_type = env.get_cell_type(r, c)
            from evacuation_simulation.simulation.environment import CellType
            if cell_type == CellType.WALL:
                colour = _COLOUR_WALL
            elif cell_type == CellType.EXIT:
                colour = _COLOUR_EXIT
            else:
                colour = _COLOUR_OPEN

            rect = mpatches.FancyBboxPatch(
                (c, r), 1, 1,
                boxstyle="square,pad=0.02",
                facecolor=colour,
                edgecolor="#CCCCCC",
                linewidth=0.5,
            )
            ax.add_patch(rect)

    # Draw agents
    agent_positions: dict[tuple[int, int], list[str]] = {}
    for agent in sim.agents:
        pos = (agent.row, agent.col)
        agent_positions.setdefault(pos, []).append(agent.agent_id)

    for (r, c), ids in agent_positions.items():
        if not sim.agents[0].is_active:  # colour based on first at cell
            pass
        # Determine colour by checking if any agent at this cell is active
        agents_here = [a for a in sim.agents if a.position == (r, c)]
        active_here = any(a.is_active for a in agents_here)
        colour = _COLOUR_AGENT_ACTIVE if active_here else _COLOUR_AGENT_EVAC
        ax.add_patch(
            mpatches.Circle(
                (c + 0.5, r + 0.5), 0.3,
                facecolor=colour,
                edgecolor="white",
                linewidth=1.5,
                zorder=5,
            )
        )
        if len(ids) > 1:
            ax.text(c + 0.5, r + 0.5, str(len(ids)),
                    ha="center", va="center", fontsize=7, color="white",
                    fontweight="bold", zorder=6)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=_COLOUR_OPEN, edgecolor="#CCCCCC", label="Open"),
        mpatches.Patch(facecolor=_COLOUR_WALL, label="Wall"),
        mpatches.Patch(facecolor=_COLOUR_EXIT, label="Exit"),
        mpatches.Circle((0, 0), 0.1, facecolor=_COLOUR_AGENT_ACTIVE, label="Agent (active)"),
        mpatches.Circle((0, 0), 0.1, facecolor=_COLOUR_AGENT_EVAC, label="Agent (evacuated)"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        fontsize=8,
    )

    step = sim.state.timestep
    evac = len(sim.evacuated_agents)
    total = len(sim.agents)
    full_title = (
        f"{title}  |  Step {step}  |  Evacuated {evac}/{total}"
        if title
        else f"Step {step}  |  Evacuated {evac}/{total}"
    )
    ax.set_title(full_title, fontsize=11, pad=8)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.tick_params(labelsize=7)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"[visualize] Saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)


def save_grid_snapshot(
    sim: "Simulation",
    output_dir: str | Path = ".",
    prefix: str = "step",
) -> Path:
    """
    Save a PNG snapshot of the current simulation state.

    Parameters
    ----------
    sim : Simulation
        Current simulation state.
    output_dir : str or Path
        Directory to write the PNG file into.
    prefix : str
        Filename prefix.  The step number is appended automatically.

    Returns
    -------
    Path
        Absolute path of the saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"{prefix}_{sim.state.timestep:04d}.png"
    render_grid(sim, show=False, save_path=filename)
    return filename.resolve()
