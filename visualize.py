"""
visualize.py
------------
Matplotlib-based interactive visualization and simulation inspector for the
2D multi-agent evacuation simulation (Stage 4).

Features
~~~~~~~~
- 2D Floor Plan Grid: passable cells, walls, exits, and active agents.
- Agent Lifecycle States: visually distinguishes MOVING, WAITING, and EVACUATED.
- Planned Shortest Paths: overlays planned paths from strategy / nav_graph.
- Dynamic Occupancy & Congestion Heatmap: color-coded density overlay (occupancy/capacity).
- Interactive Control Panel: Play, Pause, Step, and Reset.
- Real-time Dashboard HUD: Timestep, active count, evacuated count, rate, waiting count.
- Clickable Agent Inspector: Click any agent on the grid to inspect state, path, and stats.
- Timestep Trace Log: Formatted report of movement resolutions and rejection reasons.
- Headless & Programmatic: Fully testable and scriptable without GUI windows.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from evacuation_simulation.simulation.config import GridCell, SimulationConfig
from evacuation_simulation.simulation.environment import CellType
from evacuation_simulation.simulation.simulation import Simulation
from evacuation_simulation.simulation.strategy import ShortestPathStrategy

if TYPE_CHECKING:
    from evacuation_simulation.simulation.agent import Agent
    from evacuation_simulation.simulation.strategy import MovementStrategy


# ---------------------------------------------------------------------------
# Visual Design System & Palette
# ---------------------------------------------------------------------------

COLOR_BG = "#FFFFFF"
COLOR_OPEN = "#F8F9FA"          # soft light grey (passable floor)
COLOR_GRID_LINE = "#E2E8F0"     # subtle grid lines
COLOR_WALL = "#1E293B"          # deep slate/charcoal
COLOR_EXIT = "#10B981"          # vibrant emerald green
COLOR_EXIT_TEXT = "#FFFFFF"

# Agent state colors
COLOR_AGENT_MOVING = "#2563EB"   # blue (moving normally)
COLOR_AGENT_WAITING = "#D97706"  # amber/orange (rejected/waiting)
COLOR_AGENT_SELECTED = "#F59E0B" # gold ring highlight
COLOR_AGENT_EVAC = "#94A3B8"     # slate grey

# Occupancy heatmap overlays
COLOR_OCC_LOW = "#FEF3C7"       # light amber tint (< 50%)
COLOR_OCC_MED = "#FCD34D"       # medium amber (50% - 99%)
COLOR_OCC_CONGESTED = "#F87171" # soft red / congestion (>= 100%)

# Path overlays
COLOR_PATH_DEFAULT = "#94A3B8"
COLOR_PATH_SELECTED = "#3B82F6"


# ---------------------------------------------------------------------------
# SimulationVisualizer
# ---------------------------------------------------------------------------


class SimulationVisualizer:
    """
    Interactive 2D visualizer and inspector for discrete evacuation simulations.

    Parameters
    ----------
    simulation : Simulation or None
        An active Simulation instance. If None, config must be provided.
    config : SimulationConfig or None
        Configuration used to construct or reset the simulation.
    movement_strategy_factory : Callable[[], MovementStrategy] or None
        Factory function returning a fresh movement strategy instance upon reset.
        Defaults to ``ShortestPathStrategy``.
    show_paths : bool
        Whether planned shortest paths are drawn by default. Default True.
    show_occupancy : bool
        Whether the occupancy heatmap overlay is drawn by default. Default False.
    playback_interval_ms : int
        Time interval in milliseconds between steps in Play mode. Default 250.
    """

    def __init__(
        self,
        simulation: Optional[Simulation] = None,
        config: Optional[SimulationConfig] = None,
        movement_strategy_factory: Optional[Callable[[], MovementStrategy]] = None,
        *,
        show_paths: bool = True,
        show_occupancy: bool = False,
        playback_interval_ms: int = 250,
    ) -> None:
        if simulation is None and config is None:
            raise ValueError("Either simulation or config must be provided.")

        self.config: SimulationConfig = (
            simulation.config if simulation is not None else config  # type: ignore[assignment]
        )
        self.strategy_factory: Callable[[], MovementStrategy] = (
            movement_strategy_factory or (lambda: ShortestPathStrategy())
        )

        self.sim: Simulation = (
            simulation
            if simulation is not None
            else Simulation(self.config, self.strategy_factory())
        )

        # Display toggles & inspection state
        self.show_paths: bool = show_paths
        self.show_occupancy: bool = show_occupancy
        self.selected_agent_id: Optional[str] = None
        self.is_playing: bool = False
        self.playback_interval_ms: int = playback_interval_ms

        # Matplotlib figure and widget handles
        self.fig: Optional[plt.Figure] = None
        self.ax_grid: Optional[plt.Axes] = None
        self.ax_hud: Optional[plt.Axes] = None
        self.ax_inspector: Optional[plt.Axes] = None
        self.ax_trace: Optional[plt.Axes] = None
        self._buttons: dict[str, Button] = {}
        self._timer = None

    # ------------------------------------------------------------------
    # Simulation Control API
    # ------------------------------------------------------------------

    def step(self) -> bool:
        """
        Advance the simulation by exactly one timestep and update the view.

        Returns
        -------
        bool
            True if simulation is still running, False if terminated.
        """
        running = self.sim.step()
        if not running and self.is_playing:
            self.pause()
        self.render_frame()
        return running

    def reset(self) -> None:
        """
        Reset the simulation to its initial state from configuration.
        """
        self.pause()
        self.sim = Simulation(self.config, self.strategy_factory())
        self.selected_agent_id = None
        self.render_frame()

    def play(self) -> None:
        """Start automatic advancement."""
        if self.sim.state.is_terminated:
            return
        self.is_playing = True
        self._update_play_button_label()
        if self._timer is not None:
            self._timer.start()

    def pause(self) -> None:
        """Stop automatic advancement."""
        self.is_playing = False
        self._update_play_button_label()
        if self._timer is not None:
            self._timer.stop()

    def toggle_play(self) -> None:
        """Toggle between Play and Pause."""
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def toggle_paths(self, show: Optional[bool] = None) -> None:
        """Toggle path overlay visibility."""
        self.show_paths = not self.show_paths if show is None else show
        self._update_paths_button_label()
        self.render_frame()

    def toggle_occupancy(self, show: Optional[bool] = None) -> None:
        """Toggle occupancy heatmap overlay visibility."""
        self.show_occupancy = not self.show_occupancy if show is None else show
        self._update_occupancy_button_label()
        self.render_frame()

    def select_agent(self, agent_id: Optional[str]) -> None:
        """Set the active inspected agent."""
        self.selected_agent_id = agent_id
        self.render_frame()

    # ------------------------------------------------------------------
    # Rendering & Layout
    # ------------------------------------------------------------------

    def build_figure(self) -> plt.Figure:
        """
        Create and layout the Matplotlib figure with grid, inspection panels,
        and control buttons.
        """
        rows, cols = self.sim.environment.rows, self.sim.environment.cols

        # Figure sizing proportional to grid dimensions
        grid_w = max(5.0, cols * 0.6)
        grid_h = max(5.0, rows * 0.6)
        fig_w = grid_w + 5.5
        fig_h = max(6.5, grid_h + 1.2)

        fig = plt.figure(figsize=(fig_w, fig_h), facecolor=COLOR_BG)
        self.fig = fig

        # Left / Center: Main Grid (leave space at bottom for buttons)
        grid_left = 0.06
        grid_bottom = 0.12
        grid_width = (grid_w / fig_w) * 0.85
        grid_height = (grid_h / fig_h) * 0.82
        ax_grid = fig.add_axes([grid_left, grid_bottom, grid_width, grid_height])
        self.ax_grid = ax_grid

        # Right Panels: Top HUD, Middle Inspector, Bottom Trace Log
        panel_left = grid_left + grid_width + 0.04
        panel_width = 1.0 - panel_left - 0.04

        ax_hud = fig.add_axes([panel_left, 0.72, panel_width, 0.22])
        ax_inspector = fig.add_axes([panel_left, 0.38, panel_width, 0.30])
        ax_trace = fig.add_axes([panel_left, 0.06, panel_width, 0.28])

        for ax in (ax_hud, ax_inspector, ax_trace):
            ax.set_facecolor("#F1F5F9")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#CBD5E1")
                spine.set_linewidth(1.0)

        self.ax_hud = ax_hud
        self.ax_inspector = ax_inspector
        self.ax_trace = ax_trace

        # Bottom Control Buttons under Grid
        btn_y = 0.03
        btn_h = 0.055
        btn_w = (grid_width - 0.04) / 5.0

        btn_ax_play = fig.add_axes([grid_left, btn_y, btn_w, btn_h])
        btn_ax_step = fig.add_axes([grid_left + btn_w + 0.01, btn_y, btn_w, btn_h])
        btn_ax_reset = fig.add_axes([grid_left + 2 * (btn_w + 0.01), btn_y, btn_w, btn_h])
        btn_ax_paths = fig.add_axes([grid_left + 3 * (btn_w + 0.01), btn_y, btn_w, btn_h])
        btn_ax_occ = fig.add_axes([grid_left + 4 * (btn_w + 0.01), btn_y, btn_w, btn_h])

        self._buttons["play"] = Button(btn_ax_play, "Play", color="#E2E8F0", hovercolor="#CBD5E1")
        self._buttons["step"] = Button(btn_ax_step, "Step", color="#E2E8F0", hovercolor="#CBD5E1")
        self._buttons["reset"] = Button(btn_ax_reset, "Reset", color="#E2E8F0", hovercolor="#CBD5E1")
        self._buttons["paths"] = Button(btn_ax_paths, "Paths: ON", color="#E2E8F0", hovercolor="#CBD5E1")
        self._buttons["occ"] = Button(btn_ax_occ, "Occ: OFF", color="#E2E8F0", hovercolor="#CBD5E1")

        self._buttons["play"].on_clicked(lambda _: self.toggle_play())
        self._buttons["step"].on_clicked(lambda _: self.step())
        self._buttons["reset"].on_clicked(lambda _: self.reset())
        self._buttons["paths"].on_clicked(lambda _: self.toggle_paths())
        self._buttons["occ"].on_clicked(lambda _: self.toggle_occupancy())

        for btn in self._buttons.values():
            btn.label.set_fontsize(8.5)
            btn.label.set_fontweight("bold")

        # Interactive click handler on grid for Agent Inspection
        fig.canvas.mpl_connect("button_press_event", self._on_click)

        # Setup animation timer
        if hasattr(fig.canvas, "new_timer"):
            self._timer = fig.canvas.new_timer(interval=self.playback_interval_ms)
            self._timer.add_callback(self._on_timer_tick)

        self._update_paths_button_label()
        self._update_occupancy_button_label()
        self._update_play_button_label()

        return fig

    def render_frame(self) -> None:
        """Draw the complete visual state onto all axes."""
        if self.fig is None:
            self.build_figure()

        self._draw_grid()
        self._draw_hud()
        self._draw_inspector()
        self._draw_trace()

        if self.fig is not None and self.fig.canvas is not None:
            self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Sub-component Drawing
    # ------------------------------------------------------------------

    def _draw_grid(self) -> None:
        if self.ax_grid is None:
            return
        ax = self.ax_grid
        ax.clear()

        env = self.sim.environment
        rows, cols = env.rows, env.cols

        ax.set_aspect("equal")
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.invert_yaxis()  # row 0 at top

        # 1. Base cell rectangles
        for r in range(rows):
            for c in range(cols):
                cell_type = env.get_cell_type(r, c)
                if cell_type == CellType.WALL:
                    color = COLOR_WALL
                    edge_color = COLOR_WALL
                elif cell_type == CellType.EXIT:
                    color = COLOR_EXIT
                    edge_color = "#059669"
                else:
                    color = COLOR_OPEN
                    edge_color = COLOR_GRID_LINE

                rect = mpatches.Rectangle(
                    (c, r), 1, 1,
                    facecolor=color,
                    edgecolor=edge_color,
                    linewidth=0.8,
                    zorder=1,
                )
                ax.add_patch(rect)

                # Exit marker
                if cell_type == CellType.EXIT:
                    ax.text(
                        c + 0.5, r + 0.5, "EXIT",
                        ha="center", va="center",
                        fontsize=7, fontweight="bold",
                        color=COLOR_EXIT_TEXT, zorder=2,
                    )

        # 2. Occupancy Overlay (if enabled)
        if self.show_occupancy:
            cap = self.sim.occupancy.default_capacity
            for (r, c), occ in self.sim.occupancy.get_occupied_cells().items():
                ratio = occ / cap if cap > 0 else 0.0
                if ratio >= 1.0:
                    overlay_color = COLOR_OCC_CONGESTED
                elif ratio >= 0.5:
                    overlay_color = COLOR_OCC_MED
                else:
                    overlay_color = COLOR_OCC_LOW

                occ_rect = mpatches.Rectangle(
                    (c, r), 1, 1,
                    facecolor=overlay_color,
                    alpha=0.6,
                    edgecolor="#DC2626" if ratio >= 1.0 else "#D97706",
                    linewidth=1.2,
                    zorder=2,
                )
                ax.add_patch(occ_rect)
                ax.text(
                    c + 0.85, r + 0.15, f"{occ}/{cap}",
                    ha="right", va="top",
                    fontsize=6, fontweight="bold",
                    color="#7F1D1D" if ratio >= 1.0 else "#78350F",
                    zorder=3,
                )

        # 3. Planned Paths Overlay (if enabled)
        if self.show_paths:
            for agent in self.sim.active_agents:
                path = self.sim.get_agent_planned_path(agent.agent_id)
                if path and len(path) > 1:
                    is_sel = agent.agent_id == self.selected_agent_id
                    p_color = COLOR_PATH_SELECTED if is_sel else COLOR_PATH_DEFAULT
                    p_lw = 2.4 if is_sel else 1.2
                    p_alpha = 0.9 if is_sel else 0.4
                    p_zorder = 5 if is_sel else 3

                    xs = [col + 0.5 for _, col in path]
                    ys = [row + 0.5 for row, _ in path]
                    ax.plot(
                        xs, ys,
                        color=p_color,
                        linewidth=p_lw,
                        alpha=p_alpha,
                        linestyle="-" if is_sel else "--",
                        zorder=p_zorder,
                    )

        # 4. Active Agents
        # Group by position to display multi-agent count badges
        agents_at_pos: dict[GridCell, list[Agent]] = {}
        for agent in self.sim.active_agents:
            agents_at_pos.setdefault(agent.position, []).append(agent)

        waiting_ids = self._get_current_waiting_agent_ids()

        for (r, c), agents in agents_at_pos.items():
            primary_agent = agents[0]
            is_selected = any(a.agent_id == self.selected_agent_id for a in agents)
            is_waiting = any(a.agent_id in waiting_ids for a in agents)

            agent_color = COLOR_AGENT_WAITING if is_waiting else COLOR_AGENT_MOVING

            # Selection gold highlight halo
            if is_selected:
                halo = mpatches.Circle(
                    (c + 0.5, r + 0.5), 0.38,
                    facecolor="none",
                    edgecolor=COLOR_AGENT_SELECTED,
                    linewidth=2.8,
                    zorder=7,
                )
                ax.add_patch(halo)

            # Agent body circle
            circle = mpatches.Circle(
                (c + 0.5, r + 0.5), 0.30,
                facecolor=agent_color,
                edgecolor="white",
                linewidth=1.5,
                zorder=8,
            )
            ax.add_patch(circle)

            # Agent label / density count
            if len(agents) > 1:
                label = str(len(agents))
                ax.text(
                    c + 0.5, r + 0.5, label,
                    ha="center", va="center",
                    fontsize=7.5, fontweight="bold",
                    color="white", zorder=9,
                )
            else:
                # Single agent: display short ID (e.g. "a01" or last 2 chars)
                short_id = primary_agent.agent_id[-2:]
                ax.text(
                    c + 0.5, r + 0.5, short_id,
                    ha="center", va="center",
                    fontsize=6, fontweight="bold",
                    color="white", zorder=9,
                )

        # Title and axis styling
        ax.set_title(
            f"Floor Plan  ({self.config.scenario_name})",
            fontsize=10, fontweight="bold", pad=6,
        )
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        ax.set_xticklabels(range(cols), fontsize=6.5)
        ax.set_yticklabels(range(rows), fontsize=6.5)
        ax.tick_params(length=2, color="#94A3B8")

    def _draw_hud(self) -> None:
        if self.ax_hud is None:
            return
        ax = self.ax_hud
        ax.clear()
        ax.set_facecolor("#F8FAFC")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#CBD5E1")
            spine.set_linewidth(1.0)

        step = self.sim.state.timestep
        max_step = self.config.parameters.max_timesteps
        total_ag = len(self.sim.agents)
        active_ag = len(self.sim.active_agents)
        evac_ag = len(self.sim.evacuated_agents)
        evac_rate = (evac_ag / total_ag * 100.0) if total_ag > 0 else 0.0
        waiting_ag = len(self._get_current_waiting_agent_ids())

        status_str = (
            "TERMINATED"
            if self.sim.state.is_terminated
            else ("PLAYING" if self.is_playing else "PAUSED")
        )
        status_color = (
            "#DC2626"
            if self.sim.state.is_terminated
            else ("#16A34A" if self.is_playing else "#2563EB")
        )

        ax.text(0.05, 0.85, "SIMULATION DASHBOARD", fontsize=9, fontweight="bold", color="#0F172A")
        ax.text(0.95, 0.85, status_str, fontsize=8, fontweight="bold", color=status_color, ha="right")

        lines = [
            f"Timestep      : {step} / {max_step}",
            f"Active Agents : {active_ag} / {total_ag}",
            f"Evacuated     : {evac_ag}  ({evac_rate:.1f}%)",
            f"Waiting (Now) : {waiting_ag}",
        ]
        for idx, line in enumerate(lines):
            ax.text(0.05, 0.62 - idx * 0.18, line, fontsize=8, fontfamily="monospace", color="#334155")

    def _draw_inspector(self) -> None:
        if self.ax_inspector is None:
            return
        ax = self.ax_inspector
        ax.clear()
        ax.set_facecolor("#F8FAFC")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#CBD5E1")
            spine.set_linewidth(1.0)

        ax.text(0.05, 0.88, "AGENT INSPECTOR", fontsize=9, fontweight="bold", color="#0F172A")

        if not self.selected_agent_id:
            ax.text(
                0.05, 0.50,
                "Click an agent on the grid\nto inspect state and path.",
                fontsize=8, color="#64748B", style="italic",
            )
            return

        info = self.sim.get_agent_info(self.selected_agent_id)
        if not info:
            ax.text(0.05, 0.50, f"Agent '{self.selected_agent_id}' not found.", fontsize=8, color="#DC2626")
            return

        state_color = (
            COLOR_EXIT
            if info["state"] == "EVACUATED"
            else (COLOR_AGENT_WAITING if info["state"] == "WAITING" else COLOR_AGENT_MOVING)
        )

        lines = [
            f"Agent ID     : {info['agent_id']}",
            f"Position     : {info['position']}",
            f"State        : {info['state']}",
            f"Target Exit  : {info['target_exit']}",
            f"Path Hops    : {info['path_length'] if info['path_length'] is not None else 'N/A'}",
            f"Wait Steps   : {info['waiting_steps']}",
            f"Last Result  : {info['last_result'] or '(none)'}",
        ]
        if info["last_rejection_reason"]:
            lines.append(f"Rejection    : {info['last_rejection_reason']}")

        for idx, line in enumerate(lines):
            y_pos = 0.72 - idx * 0.12
            if "State" in line:
                ax.text(0.05, y_pos, "State        : ", fontsize=7.5, fontfamily="monospace", color="#334155")
                ax.text(0.50, y_pos, info["state"], fontsize=7.5, fontfamily="monospace", fontweight="bold", color=state_color)
            else:
                ax.text(0.05, y_pos, line, fontsize=7.5, fontfamily="monospace", color="#334155")

    def _draw_trace(self) -> None:
        if self.ax_trace is None:
            return
        ax = self.ax_trace
        ax.clear()
        ax.set_facecolor("#F8FAFC")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#CBD5E1")
            spine.set_linewidth(1.0)

        ax.text(0.05, 0.88, "LAST TIMESTEP TRACE", fontsize=9, fontweight="bold", color="#0F172A")

        trace = self.sim.last_step_trace
        if trace is None:
            ax.text(0.05, 0.50, "No movement steps executed yet.", fontsize=8, color="#64748B", style="italic")
            return

        # Show summary of approved vs rejected
        approved = sum(1 for ar in trace.agent_results if ar.result == "APPROVED")
        rejected = sum(1 for ar in trace.agent_results if "REJECTED" in ar.result)
        evac = sum(1 for ar in trace.agent_results if ar.result == "EVACUATED")

        summary = f"Step {trace.timestep}: {approved} moved, {rejected} rejected, {evac} evac"
        ax.text(0.05, 0.70, summary, fontsize=7.5, fontweight="bold", fontfamily="monospace", color="#1E293B")

        # List up to 4 key events
        sample_results = trace.agent_results[:4]
        for idx, ar in enumerate(sample_results):
            res_short = ar.result
            if ar.result == "REJECTED_CAPACITY":
                res_short = "REJ (Cap)"
            elif ar.result == "REJECTED_CONFLICT":
                res_short = "REJ (Tie)"
            text = f"{ar.agent_id}: {ar.position_before}->{ar.position_after} [{res_short}]"
            ax.text(0.05, 0.50 - idx * 0.14, text, fontsize=7, fontfamily="monospace", color="#475569")

    # ------------------------------------------------------------------
    # Event Callbacks
    # ------------------------------------------------------------------

    def _on_click(self, event) -> None:
        """Handle mouse clicks on the grid to inspect agents."""
        if event.inaxes != self.ax_grid or event.xdata is None or event.ydata is None:
            return

        col = int(event.xdata)
        row = int(event.ydata)

        # Look for active agents at clicked cell
        agents_here = [
            a for a in self.sim.active_agents if a.position == (row, col)
        ]

        if agents_here:
            # Cycle through agents at cell if multiple
            if (
                self.selected_agent_id
                and any(a.agent_id == self.selected_agent_id for a in agents_here)
                and len(agents_here) > 1
            ):
                curr_idx = [a.agent_id for a in agents_here].index(self.selected_agent_id)
                next_agent = agents_here[(curr_idx + 1) % len(agents_here)]
                self.select_agent(next_agent.agent_id)
            else:
                self.select_agent(agents_here[0].agent_id)
        else:
            # Deselect if clicked on empty cell
            self.select_agent(None)

    def _on_timer_tick(self) -> None:
        """Advance one step per timer interval."""
        if self.is_playing:
            running = self.step()
            if not running:
                self.pause()

    def _update_play_button_label(self) -> None:
        if "play" in self._buttons:
            self._buttons["play"].label.set_text("Pause" if self.is_playing else "Play")

    def _update_paths_button_label(self) -> None:
        if "paths" in self._buttons:
            self._buttons["paths"].label.set_text(
                "Paths: ON" if self.show_paths else "Paths: OFF"
            )

    def _update_occupancy_button_label(self) -> None:
        if "occ" in self._buttons:
            self._buttons["occ"].label.set_text(
                "Occ: ON" if self.show_occupancy else "Occ: OFF"
            )

    def _get_current_waiting_agent_ids(self) -> set[str]:
        """Return set of agent IDs that were rejected in the last step."""
        if self.sim.last_step_trace is None:
            return set()
        return {
            ar.agent_id
            for ar in self.sim.last_step_trace.agent_results
            if "REJECTED" in ar.result
        }

    # ------------------------------------------------------------------
    # Public Display & Snapshot API
    # ------------------------------------------------------------------

    def show(self, auto_play: bool = False) -> None:
        """Display the interactive window (blocking)."""
        self.render_frame()
        if auto_play or self.is_playing:
            self.play()
        plt.show()

    def save_snapshot(self, path: str | Path) -> Path:
        """
        Save the current visual state to a PNG file.

        Parameters
        ----------
        path : str or Path
            Destination image file path.

        Returns
        -------
        Path
            Resolved absolute path to saved file.
        """
        self.render_frame()
        dest = Path(path).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        if self.fig is not None:
            self.fig.savefig(dest, dpi=120, bbox_inches="tight")
        return dest


# ---------------------------------------------------------------------------
# Standalone Utility Functions (Backward Compatible)
# ---------------------------------------------------------------------------


def render_grid(
    sim: Simulation,
    title: str = "",
    *,
    show: bool = True,
    save_path: str | Path | None = None,
    show_paths: bool = True,
    show_occupancy: bool = False,
) -> None:
    """
    Render the current simulation state as a 2D floor-plan figure.
    """
    viz = SimulationVisualizer(
        simulation=sim,
        show_paths=show_paths,
        show_occupancy=show_occupancy,
    )
    viz.build_figure()
    viz.render_frame()

    if save_path is not None:
        viz.save_snapshot(save_path)
        print(f"[visualize] Saved snapshot: {save_path}")

    if show:
        viz.show()
    elif viz.fig is not None:
        plt.close(viz.fig)


def save_grid_snapshot(
    sim: Simulation,
    output_dir: str | Path = ".",
    prefix: str = "step",
) -> Path:
    """
    Save a PNG snapshot of the current simulation step.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    filename = out / f"{prefix}_{sim.state.timestep:04d}.png"
    render_grid(sim, show=False, save_path=filename)
    return filename.resolve()
