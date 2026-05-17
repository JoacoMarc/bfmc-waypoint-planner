"""Animate a saved plan to GIF.

Renders the car traversing the lanelet_sequence at each segment's effective speed.
Output: GIF via Pillow (default) or MP4 if ffmpeg is available.
"""

from __future__ import annotations

import math
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Polygon as MplPolygon, Rectangle

from .cost_matrix import CostParams, lanelet_speed_mps
from .plan_writer import Plan
from .topology import Lanelet
from .waypoint_anchor import WaypointArea


def _path_polyline(plan: Plan, lanelets: dict[str, Lanelet]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate centerlines of lanelet_sequence into one polyline + arc lengths."""
    pts: list[np.ndarray] = []
    for lid in plan.lanelet_sequence:
        l = lanelets.get(lid)
        if l is None or l.centerline.shape[0] < 2:
            continue
        if pts:
            # Avoid duplicating join point
            pts.append(l.centerline[1:])
        else:
            pts.append(l.centerline)
    if not pts:
        return np.zeros((0, 2)), np.zeros((0,))
    poly = np.vstack(pts)
    diffs = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(diffs)])
    return poly, arc


def _frame_pose(poly: np.ndarray, arc: np.ndarray, target_arc: float) -> tuple[float, float, float]:
    """Interpolate (x, y, yaw) at given arc length along the polyline."""
    if poly.shape[0] < 2:
        return (0.0, 0.0, 0.0)
    target_arc = max(0.0, min(target_arc, float(arc[-1])))
    i = int(np.searchsorted(arc, target_arc, side="right") - 1)
    i = max(0, min(i, poly.shape[0] - 2))
    seg_len = float(arc[i + 1] - arc[i])
    t = 0.0 if seg_len <= 1e-9 else (target_arc - float(arc[i])) / seg_len
    x = float(poly[i, 0] + t * (poly[i + 1, 0] - poly[i, 0]))
    y = float(poly[i, 1] + t * (poly[i + 1, 1] - poly[i, 1]))
    yaw = math.atan2(float(poly[i + 1, 1] - poly[i, 1]), float(poly[i + 1, 0] - poly[i, 0]))
    return (x, y, yaw)


def animate_plan_gif(
    plan: Plan,
    lanelets: dict[str, Lanelet],
    waypoints: list[WaypointArea],
    start_area: WaypointArea | None,
    output_path: str,
    fps: int = 12,
    duration_scale: float = 1.0,
    use_mp4: bool = False,
) -> None:
    """Generate animation.

    duration_scale: 1.0 = real-time playback (animation duration = plan time).
                    0.5 = 2x speed; 2.0 = half speed.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    poly, arc = _path_polyline(plan, lanelets)
    total_arc = float(arc[-1]) if arc.shape[0] else 0.0
    total_time = max(plan.expected_total_time_s, 0.01)
    avg_speed = max(total_arc / total_time, 1e-3)

    duration_s = total_time * duration_scale
    n_frames = max(2, int(duration_s * fps))

    # Compute view bounds
    all_x: list[float] = []
    all_y: list[float] = []
    for l in lanelets.values():
        all_x.extend(l.centerline[:, 0].tolist())
        all_y.extend(l.centerline[:, 1].tolist())
    for w in waypoints:
        all_x.extend(w.polygon_xy[:, 0].tolist())
        all_y.extend(w.polygon_xy[:, 1].tolist())
    x_min, x_max = min(all_x) - 0.5, max(all_x) + 0.5
    y_min, y_max = min(all_y) - 0.5, max(all_y) + 0.5

    fig_w = 12.0
    fig_h = fig_w * (y_max - y_min) / max(0.1, x_max - x_min)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_facecolor("#0a0a0a")
    fig.patch.set_facecolor("#0a0a0a")

    # Background: faint lanelets + waypoints
    for l in lanelets.values():
        ax.plot(l.centerline[:, 0], l.centerline[:, 1], color="#333333", linewidth=0.4, zorder=1)
    visited = {entry["wp_id"] for entry in plan.waypoint_sequence}
    for w in waypoints:
        face = "#3a5a8c" if w.wp_id in visited else "#444444"
        ax.add_patch(MplPolygon(w.polygon_xy, closed=True, facecolor=face,
                                edgecolor="#666666", alpha=0.5, zorder=2))
    if start_area is not None:
        ax.add_patch(MplPolygon(start_area.polygon_xy, closed=True, facecolor="#22cc44",
                                edgecolor="#44ff66", alpha=0.15, zorder=2))

    # Route polyline
    if poly.shape[0] >= 2:
        ax.plot(poly[:, 0], poly[:, 1], color="#22cc44", linewidth=1.5, alpha=0.8, zorder=3)

    # Car marker (rectangle, oriented with yaw)
    car_len, car_wid = 0.30, 0.14
    car_patch = Rectangle((0, 0), car_len, car_wid, facecolor="#ff3333", edgecolor="#ffffff",
                          linewidth=1.0, zorder=10)
    ax.add_patch(car_patch)
    head_arrow = ax.plot([], [], color="#ff5555", linewidth=2.0, zorder=11)[0]

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)  # mirror Y, same as static SVG
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    title = ax.set_title("", color="white", fontsize=10)

    def init():
        car_patch.set_xy((-100, -100))
        head_arrow.set_data([], [])
        return car_patch, head_arrow, title

    def update(frame_idx: int):
        progress = frame_idx / max(1, n_frames - 1)
        plan_t = progress * total_time
        target_arc = progress * total_arc
        x, y, yaw = _frame_pose(poly, arc, target_arc)
        # Place rectangle centered at (x, y) rotated by yaw
        cx = x - 0.5 * car_len * math.cos(yaw) + 0.5 * car_wid * math.sin(yaw)
        cy = y - 0.5 * car_len * math.sin(yaw) - 0.5 * car_wid * math.cos(yaw)
        car_patch.set_xy((cx, cy))
        car_patch.angle = math.degrees(yaw)
        # heading line
        hx = x + 0.30 * math.cos(yaw)
        hy = y + 0.30 * math.sin(yaw)
        head_arrow.set_data([x, hx], [y, hy])
        title.set_text(
            f"t={plan_t:.1f}s / {total_time:.1f}s   "
            f"dist={target_arc:.1f}m / {total_arc:.1f}m   "
            f"score={int(plan.expected_score)} pts"
        )
        return car_patch, head_arrow, title

    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=n_frames, interval=1000 // fps, blit=False,
    )

    if use_mp4 and output_path.endswith(".mp4"):
        try:
            writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
            anim.save(output_path, writer=writer)
        except Exception as e:
            print(f"WARN: ffmpeg writer failed ({e}); falling back to GIF.")
            output_path = output_path.replace(".mp4", ".gif")
            anim.save(output_path, writer="pillow", fps=fps)
    else:
        anim.save(output_path, writer="pillow", fps=fps)
    plt.close(fig)
