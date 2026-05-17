"""Render the BFMC plan as an SVG via matplotlib.

Layers:
  - All 175 lanelet centerlines in light gray.
  - Lanelets in the route in green/orange.
  - 37 waypoint polygons: blue fill if visited, gray fill if skipped.
  - Start area polygon (1821) in semi-transparent green.
  - Start pose: red arrow.
  - Order labels (1, 2, 3...) on visited waypoints.
  - Header with expected score, time, distance, count.

The Y axis is inverted so the SVG matches the orientation of the BFMC official
track diagram (random start at the bottom, START/parking at the top of the SVG).
"""

from __future__ import annotations

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Polygon as MplPolygon

from .plan_writer import Plan
from .topology import Lanelet
from .waypoint_anchor import WaypointArea


def render_plan_svg(
    plan: Plan,
    lanelets: dict[str, Lanelet],
    waypoints: list[WaypointArea],
    start_area: WaypointArea | None,
    output_path: str,
    lanelet_penalties: dict[str, float] | None = None,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    # Compute view bounds
    all_x: list[float] = []
    all_y: list[float] = []
    for l in lanelets.values():
        all_x.extend(l.centerline[:, 0].tolist())
        all_y.extend(l.centerline[:, 1].tolist())
    for w in waypoints:
        all_x.extend(w.polygon_xy[:, 0].tolist())
        all_y.extend(w.polygon_xy[:, 1].tolist())
    if start_area is not None:
        all_x.extend(start_area.polygon_xy[:, 0].tolist())
        all_y.extend(start_area.polygon_xy[:, 1].tolist())
    if not all_x:
        return
    x_min, x_max = min(all_x) - 0.5, max(all_x) + 0.5
    y_min, y_max = min(all_y) - 0.5, max(all_y) + 0.5
    width = x_max - x_min
    height = y_max - y_min
    fig_w = 16.0
    fig_h = fig_w * height / max(width, 0.1)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_facecolor("#0a0a0a")
    fig.patch.set_facecolor("#0a0a0a")

    # All lanelets in gray
    for l in lanelets.values():
        ax.plot(l.centerline[:, 0], l.centerline[:, 1], color="#444444", linewidth=0.5, zorder=1)

    # Highlight route lanelets. Color encodes the per-lanelet penalty:
    #   yellow  -> penalized (positive penalty, e.g. STOP / crosswalk / zone G / parking)
    #   violet  -> rewarded  (negative penalty, e.g. highway bonus)
    #   orange  -> default highway (is_highway=True but no explicit penalty)
    #   green   -> normal urban lanelet
    pen_map = lanelet_penalties or {}
    route_lids = set(plan.lanelet_sequence)
    for lid in route_lids:
        l = lanelets.get(lid)
        if l is None:
            continue
        pen = float(pen_map.get(lid, 0.0))
        if pen > 1e-6:
            color = "#ffd400"          # yellow — penalized
        elif pen < -1e-6:
            color = "#b366ff"          # violet — rewarded
        elif l.is_highway:
            color = "#ffaa00"          # orange — highway, no extra weight
        else:
            color = "#22cc44"          # green — normal
        flag = (
            "penalized" if pen > 1e-6 else
            "rewarded" if pen < -1e-6 else
            "highway" if l.is_highway else "normal"
        )
        line, = ax.plot(l.centerline[:, 0], l.centerline[:, 1], color=color, linewidth=1.8, zorder=2)
        line.set_gid(f"lanelet-{lid}")
        line.set_label(f"lanelet {lid} | flag={flag} | penalty={pen:+.2f}s | length={l.length_m:.2f}m")
        if l.centerline.shape[0] >= 2:
            mid_idx = l.centerline.shape[0] // 2
            if mid_idx + 1 < l.centerline.shape[0]:
                p0 = l.centerline[mid_idx]
                p1 = l.centerline[mid_idx + 1]
                dx = p1[0] - p0[0]
                dy = p1[1] - p0[1]
                norm = np.hypot(dx, dy)
                if norm > 1e-6:
                    dx /= norm
                    dy /= norm
                    ax.add_patch(
                        FancyArrow(
                            p0[0], p0[1], dx * 0.10, dy * 0.10,
                            width=0.025, head_width=0.10, head_length=0.10,
                            facecolor=color, edgecolor=color, alpha=0.9, zorder=3,
                        )
                    )

    # Random start area (semi-transparent green) — only when a start_area is provided.
    if start_area is not None:
        ax.add_patch(
            MplPolygon(
                start_area.polygon_xy, closed=True,
                facecolor="#22cc44", edgecolor="#44ff66", alpha=0.20, linewidth=2.0, zorder=2,
            )
        )
        ax.text(
            start_area.centroid[0], start_area.centroid[1] + 0.3,
            "RANDOM START", color="#44ff66", fontsize=8, ha="center", weight="bold", zorder=5,
        )

    # Waypoints
    visited_wp_ids = {entry["wp_id"] for entry in plan.waypoint_sequence}
    order_by_wp_id: dict[str, int] = {}
    for entry in plan.waypoint_sequence:
        wid = entry["wp_id"]
        if wid not in order_by_wp_id:
            order_by_wp_id[wid] = len(order_by_wp_id) + 1
    # Build a lookup of waypoint details for tooltips.
    wp_details: dict[str, dict] = {}
    for entry in plan.waypoint_sequence:
        wp_details[entry["wp_id"]] = entry

    for w in waypoints:
        visited = w.wp_id in visited_wp_ids
        face = "#4488ff" if visited else "#666666"
        edge = "#66aaff" if visited else "#888888"
        patch = MplPolygon(
            w.polygon_xy, closed=True,
            facecolor=face, edgecolor=edge, alpha=0.7 if visited else 0.35,
            linewidth=1.0, zorder=3,
        )
        patch.set_gid(f"wp-{w.wp_id}")
        if visited:
            d = wp_details.get(w.wp_id, {})
            patch.set_label(
                f"wp {w.wp_id} | order={d.get('order','?')} | "
                f"t={d.get('cumulative_time_s','?')}s | dist={d.get('edge_distance_m','?')}m | "
                f"+{d.get('score','?')}pts"
            )
        else:
            patch.set_label(f"wp {w.wp_id} | SKIPPED")
        ax.add_patch(patch)
        if visited:
            order = order_by_wp_id.get(w.wp_id, 0)
            ax.text(
                w.centroid[0], w.centroid[1],
                str(order),
                color="white", fontsize=9, ha="center", va="center", weight="bold", zorder=6,
            )

    # Start pose arrow
    sx, sy, syaw = plan.start_pose
    arrow_len = 0.3
    ax.add_patch(
        FancyArrow(
            sx, sy,
            arrow_len * np.cos(syaw), arrow_len * np.sin(syaw),
            width=0.05, head_width=0.18, head_length=0.18,
            facecolor="#ff3333", edgecolor="#ff5555", linewidth=1.2, zorder=7,
        )
    )

    # Header text
    header = (
        f"BFMC Plan: score={int(plan.expected_score)}  "
        f"time={plan.expected_total_time_s:.1f}s / {plan.budget_s:.0f}s  "
        f"distance={plan.expected_distance_m:.1f}m  "
        f"waypoints={plan.expected_waypoints_visited}/{len(waypoints)}"
    )
    ax.set_title(header, color="white", fontsize=11, pad=10)

    # Legend for route colors (only relevant when penalties exist).
    if pen_map:
        from matplotlib.lines import Line2D
        legend_items = [
            Line2D([0], [0], color="#22cc44", lw=3, label="Normal route"),
            Line2D([0], [0], color="#ffaa00", lw=3, label="Highway (no extra weight)"),
            Line2D([0], [0], color="#ffd400", lw=3, label="Penalized lanelet"),
            Line2D([0], [0], color="#b366ff", lw=3, label="Rewarded lanelet (bonus)"),
        ]
        leg = ax.legend(
            handles=legend_items, loc="lower right", facecolor="#1a1a1a",
            edgecolor="#444444", labelcolor="white", fontsize=8, framealpha=0.9,
        )
        leg.set_zorder(20)

    ax.set_xlim(x_min, x_max)
    # Y axis is mirrored so the SVG matches the BFMC official track image
    # (random start at the bottom of the SVG, START / parking lots at the top).
    ax.set_ylim(y_max, y_min)
    ax.set_aspect("equal")
    ax.tick_params(colors="#888888")
    for spine in ax.spines.values():
        spine.set_color("#444444")
    ax.set_xlabel("X (m)", color="#888888")
    ax.set_ylabel("Y (m)", color="#888888")
    plt.tight_layout()
    fig.savefig(output_path, format="svg", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    _inject_svg_tooltips(output_path)


def _inject_svg_tooltips(svg_path: str) -> None:
    """Post-process the matplotlib SVG: inject <title> children into <g> elements
    whose id matches `lanelet-*` or `wp-*`. The text comes from the artist's label
    which matplotlib embeds as a sibling <desc> or attribute, but we rebuild it
    from gid + a tiny lookup pattern (gid format encodes lanelet/wp IDs)."""
    import re
    import xml.etree.ElementTree as ET

    ns = "http://www.w3.org/2000/svg"
    ET.register_namespace("", ns)
    try:
        tree = ET.parse(svg_path)
    except ET.ParseError:
        return
    root = tree.getroot()
    # Collect labels stored by matplotlib as <metadata> isn't reliable. We use
    # the simpler approach: matplotlib stores set_label() text inside a <desc>
    # tag child of the <g id=...>. We rely on `id` only and rebuild the title
    # text from the id (the SVG looks cleaner this way).

    changed = False
    for el in root.iter():
        eid = el.get("id") or ""
        if eid.startswith("lanelet-"):
            title_text = f"Lanelet {eid[len('lanelet-') :]} (in route)"
            existing = el.find(f"{{{ns}}}title")
            if existing is None:
                title = ET.Element(f"{{{ns}}}title")
                title.text = title_text
                el.insert(0, title)
                changed = True
        elif eid.startswith("wp-"):
            title_text = f"Waypoint {eid[len('wp-') :]}"
            existing = el.find(f"{{{ns}}}title")
            if existing is None:
                title = ET.Element(f"{{{ns}}}title")
                title.text = title_text
                el.insert(0, title)
                changed = True
    if changed:
        tree.write(svg_path, xml_declaration=True, encoding="utf-8")
