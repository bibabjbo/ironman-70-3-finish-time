"""
Neubrutalism + Sports Analytics Editorial UI Components for IRONMAN 70.3 Demo.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from inference import format_seconds_hms

PALETTE = {
    "cream": "#FFFDF5",
    "black": "#000000",
    "white": "#FFFFFF",
    "C0": "#FFD23F",
    "C1": "#74B9FF",
    "C2": "#88D498",
    "accent": "#FF6B6B",
}

CHECKPOINT_LABELS = {
    "C0": ("C0 — BEFORE RACE", "Pre-race demographics & event info only"),
    "C1": ("C1 — AFTER SWIM", "Pre-race info + actual 1.9 km SwimTime"),
    "C2": ("C2 — AFTER BIKE", "Pre-race info + SwimTime + actual 90 km BikeTime"),
}


def render_header_banner(active_mode: str, active_split: str, active_cp: str) -> None:
    """Render top branding, main title, short explanation, and offline status indicator."""
    split_desc = (
        "EVALUATION II (Fit 2004–2018 · Test 2021–2023)"
        if active_split == "II"
        else "EVALUATION I (Fit 2004–2016 · Test 2019–2020)"
    )
    cp_badge_class = f"nb-badge-{active_cp.lower()}"
    html = f"""
    <div class="nb-header-box">
        <div>
            <span class="nb-badge nb-badge-black">IRONMAN 70.3 SPORTS ANALYTICS · GROUP 6</span>
            <span class="nb-badge nb-badge-c2">● 100% OFFLINE LOCAL ENGINE (READ-ONLY)</span>
            <span class="nb-badge">{split_desc}</span>
            <span class="nb-badge {cp_badge_class}">ACTIVE CHECKPOINT: {active_cp}</span>
        </div>
        <div class="nb-title-main">
            PROGRESSIVE FINISH TIME PREDICTION ACROSS RACE CHECKPOINTS
        </div>
        <p class="nb-subtitle">
            Prospective, leakage-free finish-time forecasting at <b>C0 (Before Race)</b>,
            <b>C1 (After 1.9 km Swim)</b>, and <b>C2 (After 90 km Bike)</b> using saved CatBoost reference models
            (<code>models/*.cbm</code>) and non-parametric conditional remainder baselines (<code>B0, B1, B2</code>).
        </p>
    </div>
    """
    cleaned_html = "\n".join(line.lstrip() for line in html.splitlines() if line.strip())
    st.markdown(cleaned_html, unsafe_allow_html=True)


def render_checkpoint_stepper(active_cp: str) -> None:
    """Render visual progression strip: BEFORE RACE -> AFTER SWIM -> AFTER BIKE."""
    c0_cls = "nb-step-box nb-step-active-C0" if active_cp == "C0" else "nb-step-box"
    c1_cls = "nb-step-box nb-step-active-C1" if active_cp == "C1" else "nb-step-box"
    c2_cls = "nb-step-box nb-step-active-C2" if active_cp == "C2" else "nb-step-box"

    c0_state = "● ACTIVE CHECKPOINT" if active_cp == "C0" else "✓ AVAILABLE"
    c1_state = (
        "● ACTIVE CHECKPOINT"
        if active_cp == "C1"
        else ("✓ COMPLETED" if active_cp == "C2" else "○ STEP 2")
    )
    c2_state = "● ACTIVE CHECKPOINT" if active_cp == "C2" else "○ STEP 3"

    html = f"""
    <div class="nb-stepper-grid">
        <div class="{c0_cls}">
            <div class="nb-step-code">{c0_state} · CHECKPOINT C0</div>
            <div class="nb-step-title">BEFORE RACE</div>
            <div class="nb-step-desc">Features (5): gender, ageband, countryiso, eventlocation, eventyear</div>
        </div>
        <div class="{c1_cls}">
            <div class="nb-step-code">{c1_state} · CHECKPOINT C1</div>
            <div class="nb-step-title">→ AFTER SWIM (1.9 KM)</div>
            <div class="nb-step-desc">Features (6): C0 features + actual swimtime (seconds)</div>
        </div>
        <div class="{c2_cls}">
            <div class="nb-step-code">{c2_state} · CHECKPOINT C2</div>
            <div class="nb-step-title">→ AFTER BIKE (90 KM)</div>
            <div class="nb-step-desc">Features (7): C1 features + actual biketime (seconds)</div>
        </div>
    </div>
    """
    cleaned_html = "\n".join(line.lstrip() for line in html.splitlines() if line.strip())
    st.markdown(cleaned_html, unsafe_allow_html=True)


def render_hero_prediction_card(
    pred_result: Dict[str, Any],
    c0_pred_seconds: Optional[float] = None,
    c1_pred_seconds: Optional[float] = None,
) -> None:
    """
    Render the main FinishTime prediction card (7px 7px 0px #000000 hard shadow).
    """
    cp = pred_result["checkpoint"]
    split = pred_result["split"]
    cb_sec = float(pred_result["CatBoost_ref"])
    b2_sec = float(pred_result["B2"])
    b0_sec = float(pred_result["B0"])

    cp_title, cp_subtitle = CHECKPOINT_LABELS[cp]

    deltas_html = ""
    if cp == "C1" and c0_pred_seconds is not None:
        diff_c0 = cb_sec - c0_pred_seconds
        deltas_html += (
            f'<span class="nb-badge">Δ VS C0 PREDICTION: {format_seconds_hms(diff_c0, signed=True)} '
            f'({diff_c0/60:+.2f} min)</span>'
        )
    elif cp == "C2":
        if c0_pred_seconds is not None:
            diff_c0 = cb_sec - c0_pred_seconds
            deltas_html += (
                f'<span class="nb-badge">Δ VS C0: {format_seconds_hms(diff_c0, signed=True)} '
                f'({diff_c0/60:+.2f} min)</span>'
            )
        if c1_pred_seconds is not None:
            diff_c1 = cb_sec - c1_pred_seconds
            deltas_html += (
                f'<span class="nb-badge">Δ VS C1: {format_seconds_hms(diff_c1, signed=True)} '
                f'({diff_c1/60:+.2f} min)</span>'
            )

    diff_b2 = cb_sec - b2_sec
    deltas_html += (
        f'<span class="nb-badge">Δ VS BASELINE B2: {format_seconds_hms(diff_b2, signed=True)} '
        f'({diff_b2/60:+.2f} min)</span>'
    )
    diff_b0 = cb_sec - b0_sec
    deltas_html += (
        f'<span class="nb-badge">Δ VS GLOBAL B0: {format_seconds_hms(diff_b0, signed=True)}</span>'
    )

    html = f"""
    <div class="nb-hero-card nb-hero-{cp}">
        <div class="nb-pred-label">
            PREDICTED TOTAL FINISH TIME · {cp_title} · MODEL: CatBoost_ref ({pred_result["model_file"]}) · EVALUATION {split}
        </div>
        <div class="nb-pred-time">{format_seconds_hms(cb_sec)}</div>
        <div style="margin-bottom: 0.85rem;">
            <span class="nb-pred-seconds">EXACT OUTPUT: {cb_sec:,.2f} SECONDS ({cb_sec/60:,.2f} MINUTES)</span>
        </div>
        <div>{deltas_html}</div>
    </div>
    """
    cleaned_html = "\n".join(line.lstrip() for line in html.splitlines() if line.strip())
    st.markdown(cleaned_html, unsafe_allow_html=True)


def render_race_progress_card(
    checkpoint: str,
    pred_finish_seconds: float,
    swimtime_seconds: Optional[float] = None,
    biketime_seconds: Optional[float] = None,
) -> None:
    """
    Render completed vs. remaining race stages (Swim 1.9 km -> Bike 90 km -> Run 21.1 km).
    """
    if checkpoint == "C0":
        swim_status = "REMAINING STAGE 1"
        swim_cls = "nb-stage-item nb-stage-pending"
        swim_val = "Not started (Pre-race)"

        bike_status = "REMAINING STAGE 2"
        bike_cls = "nb-stage-item nb-stage-pending"
        bike_val = "Not started (Pre-race)"

        run_status = "REMAINING STAGE 3"
        run_cls = "nb-stage-item nb-stage-pending"
        run_val = f"Full 70.3 Course Remaining ({format_seconds_hms(pred_finish_seconds)})"
    elif checkpoint == "C1":
        s_sec = float(swimtime_seconds or 0.0)
        rem_sec = pred_finish_seconds - s_sec
        swim_status = "✓ COMPLETED STAGE 1"
        swim_cls = "nb-stage-item nb-stage-done-swim"
        swim_val = f"{format_seconds_hms(s_sec)} ({s_sec:,.0f} s)"

        bike_status = "REMAINING STAGE 2"
        bike_cls = "nb-stage-item nb-stage-pending"
        bike_val = "In progress / upcoming (90 km Bike)"

        run_status = "REMAINING STAGE 2 + 3 (BIKE + RUN + TRANSITIONS)"
        run_cls = "nb-stage-item nb-stage-pending"
        run_val = f"Est. Remaining: {format_seconds_hms(rem_sec)} ({rem_sec/60:,.1f} min)"
    else:
        s_sec = float(swimtime_seconds or 0.0)
        b_sec = float(biketime_seconds or 0.0)
        rem_sec = pred_finish_seconds - (s_sec + b_sec)
        swim_status = "✓ COMPLETED STAGE 1"
        swim_cls = "nb-stage-item nb-stage-done-swim"
        swim_val = f"{format_seconds_hms(s_sec)} ({s_sec:,.0f} s)"

        bike_status = "✓ COMPLETED STAGE 2"
        bike_cls = "nb-stage-item nb-stage-done-bike"
        bike_val = f"{format_seconds_hms(b_sec)} ({b_sec:,.0f} s)"

        run_status = "→ REMAINING STAGE: 21.1 KM RUN (+ TRANSITIONS)"
        run_cls = "nb-stage-item"
        run_val = (
            f"Est. Remaining: {format_seconds_hms(rem_sec)} ({rem_sec/60:,.1f} min)"
            if rem_sec > 0
            else f"Warning: Splits exceed predicted finish ({format_seconds_hms(rem_sec, signed=True)})"
        )

    html = f"""
    <div class="nb-card">
        <div class="nb-section-header">
            <span>RACE STAGE PROGRESSION (COMPLETED VS. REMAINING)</span>
            <span class="nb-badge">CHECKPOINT {checkpoint}</span>
        </div>
        <div class="nb-stage-grid">
            <div class="{swim_cls}">
                <div class="nb-step-code">{swim_status}</div>
                <div class="nb-step-title">1.9 KM SWIM</div>
                <div class="nb-step-desc"><b>{swim_val}</b></div>
            </div>
            <div class="{bike_cls}">
                <div class="nb-step-code">{bike_status}</div>
                <div class="nb-step-title">90 KM BIKE</div>
                <div class="nb-step-desc"><b>{bike_val}</b></div>
            </div>
            <div class="{run_cls}">
                <div class="nb-step-code">{run_status}</div>
                <div class="nb-step-title">21.1 KM HALF-MARATHON RUN</div>
                <div class="nb-step-desc"><b>{run_val}</b></div>
            </div>
        </div>
    </div>
    """
    cleaned_html = "\n".join(line.lstrip() for line in html.splitlines() if line.strip())
    st.markdown(cleaned_html, unsafe_allow_html=True)


def make_live_comparison_figure(
    checkpointResults: Dict[str, Dict[str, Any]],
    active_cp: str,
) -> plt.Figure:
    """
    Create a clean Neubrutalist Matplotlib horizontal bar chart comparing live predictions
    (B0, B1/B2, CatBoost_ref) across available checkpoints.
    """
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
    fig, ax = plt.subplots(figsize=(8.2, 3.8), dpi=140)
    fig.patch.set_facecolor(PALETTE["white"])
    ax.set_facecolor(PALETTE["cream"])

    res = checkpointResults[active_cp]
    labels: List[str] = []
    values_min: List[float] = []
    colors: List[str] = []
    hms_labels: List[str] = []

    # Always include B0
    labels.append("B0 (Global Fit Median)")
    values_min.append(res["B0"] / 60.0)
    colors.append("#E2E0D8")
    hms_labels.append(format_seconds_hms(res["B0"]))

    if active_cp == "C0" and res.get("B1") is not None:
        labels.append("B1 (Pre-Race Gender×Age Median)")
        values_min.append(res["B1"] / 60.0)
        colors.append("#D0CEB8")
        hms_labels.append(format_seconds_hms(res["B1"]))

    labels.append(f"B2 ({active_cp} Known Splits + Median Rem.)")
    values_min.append(res["B2"] / 60.0)
    colors.append("#B8B5A0")
    hms_labels.append(format_seconds_hms(res["B2"]))

    # Include prior checkpoint CatBoost predictions if available for progressive comparison
    for prior_cp in ["C0", "C1", "C2"]:
        if prior_cp in checkpointResults:
            val_sec = checkpointResults[prior_cp]["CatBoost_ref"]
            is_curr = prior_cp == active_cp
            lbl = (
                f"★ CatBoost_ref @ {prior_cp} (ACTIVE)"
                if is_curr
                else f"CatBoost_ref @ {prior_cp} (Earlier Checkpoint)"
            )
            labels.append(lbl)
            values_min.append(val_sec / 60.0)
            colors.append(PALETTE[prior_cp])
            hms_labels.append(format_seconds_hms(val_sec))

    y_pos = np.arange(len(labels))
    bars = ax.barh(
        y_pos,
        values_min,
        color=colors,
        edgecolor=PALETTE["black"],
        linewidth=2.0,
        height=0.62,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9.5, fontweight="bold", color=PALETTE["black"])
    ax.invert_yaxis()
    ax.set_xlabel(
        "Predicted Total Finish Time (Minutes & HH:MM:SS)",
        fontsize=9.5,
        fontweight="bold",
        color=PALETTE["black"],
    )
    ax.set_title(
        f"LIVE INFERENCE COMPARISON AT CHECKPOINT {active_cp} (EVALUATION {res['split']})",
        fontsize=10.5,
        fontweight="bold",
        color=PALETTE["black"],
        pad=10,
    )

    max_v = max(values_min) if values_min else 360.0
    min_v = min(values_min) if values_min else 200.0
    ax.set_xlim(max(0, min_v - 45), max_v + 55)

    for bar, hms, vmin in zip(bars, hms_labels, values_min):
        ax.text(
            bar.get_width() + 2.5,
            bar.get_y() + bar.get_height() / 2.0,
            f"{hms} ({vmin:.1f}m)",
            va="center",
            ha="left",
            fontsize=8.8,
            fontweight="bold",
            color=PALETTE["black"],
        )

    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(PALETTE["black"])
        ax.spines[spine].set_linewidth(2.0)

    ax.grid(axis="x", linestyle="--", alpha=0.35, color="#000000")
    fig.tight_layout()
    return fig


def make_replay_chart(
    athlete_df: pd.DataFrame,
    active_cp: str,
    reveal_actual: bool,
) -> plt.Figure:
    """
    Create a Neubrutalist comparison chart for Real Test Replay (metrics/example.csv).
    Displays saved predictions for B0, B1 (at C0), B2, LR, RF, and CatBoost_ref.
    Only plots Actual Finish Time when reveal_actual is True.
    """
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
    fig, ax = plt.subplots(figsize=(8.4, 4.0), dpi=140)
    fig.patch.set_facecolor(PALETTE["white"])
    ax.set_facecolor(PALETTE["cream"])

    sub = athlete_df.loc[athlete_df["checkpoint"] == active_cp].copy()
    model_order = ["B0", "B1", "B2", "LR", "RF", "CatBoost_ref"]
    sub["order"] = sub["model"].apply(lambda m: model_order.index(m) if m in model_order else 99)
    sub = sub.sort_values("order")

    labels = []
    vals_min = []
    colors = []
    hms_list = []

    for _, row in sub.iterrows():
        m = str(row["model"])
        p_sec = float(row["pred_finish_seconds"])
        labels.append(f"{m} (seed={int(row['seed'])})")
        vals_min.append(p_sec / 60.0)
        hms_list.append(format_seconds_hms(p_sec))
        if m == "CatBoost_ref":
            colors.append(PALETTE[active_cp])
        elif m in {"LR", "RF"}:
            colors.append("#FFFFFF")
        else:
            colors.append("#E5E2D6")

    actual_sec = float(sub["actual_finish_seconds"].iloc[0])
    if reveal_actual:
        labels.append("★ ACTUAL FINISH TIME")
        vals_min.append(actual_sec / 60.0)
        hms_list.append(format_seconds_hms(actual_sec))
        colors.append(PALETTE["accent"])

    y_pos = np.arange(len(labels))
    bars = ax.barh(
        y_pos,
        vals_min,
        color=colors,
        edgecolor=PALETTE["black"],
        linewidth=2.0,
        height=0.62,
    )

    if reveal_actual:
        ax.axvline(
            actual_sec / 60.0,
            color=PALETTE["accent"],
            linestyle="--",
            linewidth=2.2,
            label=f"Actual ({format_seconds_hms(actual_sec)})",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9.5, fontweight="bold", color=PALETTE["black"])
    ax.invert_yaxis()
    ax.set_xlabel(
        "Finish Time (Minutes & HH:MM:SS)",
        fontsize=9.5,
        fontweight="bold",
        color=PALETTE["black"],
    )
    rid = int(sub["resultid"].iloc[0])
    ax.set_title(
        f"SAVED TEST PREDICTIONS · RESULTID #{rid} · CHECKPOINT {active_cp} (EVALUATION II)",
        fontsize=10.5,
        fontweight="bold",
        color=PALETTE["black"],
        pad=10,
    )

    max_v = max(vals_min) if vals_min else 360.0
    min_v = min(vals_min) if vals_min else 200.0
    ax.set_xlim(max(0, min_v - 45), max_v + 60)

    for bar, hms, vmin in zip(bars, hms_list, vals_min):
        ax.text(
            bar.get_width() + 2.5,
            bar.get_y() + bar.get_height() / 2.0,
            f"{hms} ({vmin:.1f}m)",
            va="center",
            ha="left",
            fontsize=8.8,
            fontweight="bold",
            color=PALETTE["black"],
        )

    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(PALETTE["black"])
        ax.spines[spine].set_linewidth(2.0)

    ax.grid(axis="x", linestyle="--", alpha=0.35, color="#000000")
    fig.tight_layout()
    return fig


def make_global_mae_chart(all_models_df: pd.DataFrame, split: str) -> plt.Figure:
    """
    Plot official MAE (minutes) across C0, C1, C2 from metrics/all_models.csv.
    """
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
    fig, ax = plt.subplots(figsize=(8.4, 3.8), dpi=140)
    fig.patch.set_facecolor(PALETTE["white"])
    ax.set_facecolor(PALETTE["cream"])

    sub = all_models_df.loc[all_models_df["split"] == split].copy()
    checkpoints = ["C0", "C1", "C2"]
    models_to_plot = [
        ("B0", "B0 (Global Median)", "#CCCCCC"),
        ("B2", "B2 (Split + Group Median)", "#8E8D8A"),
        ("LR", "Linear Regression (LR)", "#FFFFFF"),
        ("RF", "Random Forest (RF)", "#74B9FF"),
        ("CatBoost_ref", "CatBoost_ref (.cbm)", "#FFD23F"),
    ]

    x = np.arange(len(checkpoints))
    width = 0.15

    for idx, (m_code, m_label, color) in enumerate(models_to_plot):
        m_sub = sub.loc[sub["model"] == m_code].set_index("checkpoint")
        y_vals = [float(m_sub.loc[cp, "mae_minutes"]) for cp in checkpoints]
        offset = (idx - 2) * width
        bars = ax.bar(
            x + offset,
            y_vals,
            width=width,
            label=m_label,
            color=color,
            edgecolor=PALETTE["black"],
            linewidth=1.6,
        )
        for bar, val in zip(bars, y_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.6,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
                color=PALETTE["black"],
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["C0 — BEFORE RACE", "C1 — AFTER SWIM", "C2 — AFTER BIKE"],
        fontsize=9.5,
        fontweight="bold",
        color=PALETTE["black"],
    )
    ax.set_ylabel("Mean Absolute Error (Minutes)", fontsize=9.5, fontweight="bold")
    n_fit = int(sub["n_fit"].iloc[0])
    n_test = int(sub["n_test"].iloc[0])
    n_races = int(sub["races"].iloc[0])
    ax.set_title(
        f"OFFICIAL MAE (MINUTES) BY CHECKPOINT · EVALUATION {split} "
        f"(n_fit={n_fit:,} · n_test={n_test:,} · races={n_races})",
        fontsize=10.0,
        fontweight="bold",
        pad=10,
    )
    ax.set_ylim(0, max(sub["mae_minutes"]) + 8)
    ax.legend(
        frameon=True,
        facecolor=PALETTE["white"],
        edgecolor=PALETTE["black"],
        fontsize=8.0,
        loc="upper right",
    )

    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(PALETTE["black"])
        ax.spines[spine].set_linewidth(2.0)

    ax.grid(axis="y", linestyle="--", alpha=0.35, color="#000000")
    fig.tight_layout()
    return fig
