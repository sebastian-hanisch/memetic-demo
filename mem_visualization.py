"""Plotly-Abbildungen der Memetic-Algorithmus-Demo: Karte der bisher besten Tour, Fitness-/Diversitätsverlauf,
Kopfexperiment (Qualität + Kosten), eigener Regler (p_LS) und Sweep. Achsen sind gesperrt (fixedrange)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import mem_constants as C

TOUR_COLOR = "#4c78a8"
BEST_COLOR = "#54a24b"
POP_COLOR = "#9ecae9"
GA_COLOR = "#e45756"
MEM_COLOR = "#54a24b"
REF_COLOR = "#7f7f7f"


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.1), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _map_layout(fig, height=430):
    fig.update_xaxes(range=[-3, C.AREA + 3], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(range=[-3, C.AREA + 3], showgrid=False, zeroline=False, showticklabels=False)
    return _base(fig, height)


def _tour_edges_list(tour):
    t = np.asarray(tour)
    return list(zip(t.tolist(), np.roll(t, -1).tolist()))


def build_route(xy, tour, title=None):
    fig = go.Figure()
    edges = _tour_edges_list(tour)
    x, y = [], []
    for a, b in edges:
        x += [xy[a, 0], xy[b, 0], None]
        y += [xy[a, 1], xy[b, 1], None]
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=TOUR_COLOR, width=2.5), name="beste Tour bisher"))
    fig.add_trace(go.Scatter(x=xy[1:, 0], y=xy[1:, 1], mode="markers", marker=dict(size=7, color=TOUR_COLOR, line=dict(width=1, color="white")), name="Stopps"))
    fig.add_trace(go.Scatter(x=[xy[0, 0]], y=[xy[0, 1]], mode="markers", marker=dict(size=14, symbol="star", color="#f58518", line=dict(width=1, color="white")), name="Depot"))
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=13), x=0.02, y=0.98))
    return _map_layout(fig)


def build_fitness_curve(best_hist, mean_hist, reference=None):
    xs = list(range(len(best_hist)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=mean_hist, mode="lines", line=dict(color=POP_COLOR, width=2), name="Mittelwert"))
    fig.add_trace(go.Scatter(x=xs, y=best_hist, mode="lines", line=dict(color=BEST_COLOR, width=2.5), name="bester Wert"))
    if reference is not None and np.isfinite(reference):
        fig.add_hline(y=reference, line=dict(color=REF_COLOR, dash="dot"), annotation_text="Brute-Force-Optimum", annotation_position="bottom right")
    fig.update_xaxes(title_text="Generation")
    fig.update_yaxes(title_text="Tourlänge")
    return _base(fig, 300)


def build_diversity_curve(div_hist):
    xs = list(range(len(div_hist)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=div_hist, mode="lines", line=dict(color=TOUR_COLOR, width=2.5), name="Diversität"))
    fig.update_xaxes(title_text="Generation")
    fig.update_yaxes(title_text="Diversität")
    fig.update_layout(showlegend=False)
    return _base(fig, 260)


def build_head_comparison(report):
    """Reines GA gegen voll memetisch: Tourlänge (niedriger besser) und Rechenkosten (Bewertungen der lokalen Suche)."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Gefundene Tourlänge (Median)", "Bewertungen der lokalen Suche (Mittel)"), horizontal_spacing=0.12)
    labels = ["Reines GA<br>(p_LS=0)", "Voll memetisch<br>(p_LS=1)"]
    fig.add_trace(go.Bar(x=labels, y=[report["ga"]["best_median"], report["memetic"]["best_median"]], marker_color=[GA_COLOR, MEM_COLOR], showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=[report["ga"]["ls_evaluations_mean"], report["memetic"]["ls_evaluations_mean"]], marker_color=[GA_COLOR, MEM_COLOR], showlegend=False), row=1, col=2)
    return _base(fig, 340)


def build_p_ls_experiment(rows):
    xs = [r["p_ls"] for r in rows]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Gefundene Tourlänge (Median)", "Bewertungen der lokalen Suche (Mittel)"), horizontal_spacing=0.12)
    fig.add_trace(go.Scatter(x=xs, y=[r["best_median"] for r in rows], mode="lines+markers", line=dict(color=MEM_COLOR, width=2.5), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=[r["ls_evaluations_mean"] for r in rows], mode="lines+markers", line=dict(color=TOUR_COLOR, width=2.5), showlegend=False), row=1, col=2)
    fig.update_xaxes(title_text="Polierwahrscheinlichkeit p_LS")
    return _base(fig, 340)


def build_sweep(rows, param_label):
    xs = [r["value"] for r in rows]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Abstand zum Brute-Force-Optimum (%)", "Bewertungen der lokalen Suche (Mittel)"), horizontal_spacing=0.12)
    fig.add_trace(go.Scatter(x=xs, y=[r["gap"] for r in rows], mode="lines+markers", line=dict(color=TOUR_COLOR, width=2.5), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=[r["ls_evaluations"] for r in rows], mode="lines+markers", line=dict(color=MEM_COLOR, width=2.5), showlegend=False), row=1, col=2)
    fig.update_xaxes(title_text=param_label)
    return _base(fig, 300)
