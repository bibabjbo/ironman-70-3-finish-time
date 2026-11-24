"""
IRONMAN 70.3 Finish Time Prediction — Offline Exhibition Demo Application.

Design System: NEUBRUTALISM + SPORTS ANALYTICS EDITORIAL
Architecture: 100% Local Offline Execution (Read-Only Access to Repository Assets)

Modes:
1. Demo Mode A — Live Simulation (C0 -> C1 -> C2 progressive race inference)
2. Demo Mode B — Real Test Replay (10 saved test examples from metrics/example.csv)

Launch command:
    streamlit run bibi_demo_poster/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import streamlit as st

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from components import (
    make_global_mae_chart,
    make_live_comparison_figure,
    make_replay_chart,
    render_checkpoint_stepper,
    render_header_banner,
    render_hero_prediction_card,
    render_race_progress_card,
)
from inference import (
    ACTIVE_COUNTRIES,
    ACTIVE_EVENT_LOCATIONS,
    IronmanInferenceEngine,
    format_seconds_hms,
    hms_to_seconds,
    validate_athlete_inputs,
)
from styles import NEUBRUTAL_CSS

st.set_page_config(
    page_title="IRONMAN 70.3 · Finish Time Prediction Demo",
    page_icon="⏱️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Apply Neubrutalist Editorial CSS
st.markdown(NEUBRUTAL_CSS, unsafe_allow_html=True)


def render_html(html_str: str) -> None:
    """
    Strip leading indentation from each line so Markdown never misinterprets
    nested HTML strings as 4-space indented `<pre><code>` blocks.
    """
    cleaned = "\n".join(line.lstrip() for line in html_str.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_engine() -> IronmanInferenceEngine:
    """Load the read-only inference engine once per session."""
    return IronmanInferenceEngine(REPO_ROOT)


engine = get_engine()


def init_session_state() -> None:
    """Initialize session state with clean defaults."""
    defaults: Dict[str, Any] = {
        "demo_mode": "LIVE_SIMULATION",  # "LIVE_SIMULATION" or "REAL_TEST_REPLAY"
        "eval_split": "II",  # "II" (Fit 2004-2018, Test 2021-2023) or "I" (Fit 2004-2016, Test 2019-2020)
        "live_cp": "C0",  # "C0", "C1", or "C2"
        "gender": "F",
        "agegroup": "35-39",
        "countryiso": "US",
        "eventlocation": "IRONMAN 70.3 Florida",
        "custom_location": "",
        "eventyear": 2022,
        "swim_h": 0,
        "swim_m": 35,
        "swim_s": 0,
        "bike_h": 2,
        "bike_m": 40,
        "bike_s": 0,
        "replay_cp": "C0",
        "replay_resultid": 840060,
        "replay_reveal_actual": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_demo_state() -> None:
    """Reset all interactive states to default exhibition values."""
    keys_to_clear = [
        "demo_mode",
        "eval_split",
        "live_cp",
        "gender",
        "agegroup",
        "countryiso",
        "eventlocation",
        "custom_location",
        "eventyear",
        "swim_h",
        "swim_m",
        "swim_s",
        "bike_h",
        "bike_m",
        "bike_s",
        "replay_cp",
        "replay_resultid",
        "replay_reveal_actual",
    ]
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]
    init_session_state()


init_session_state()

active_cp_for_header = (
    st.session_state["live_cp"]
    if st.session_state["demo_mode"] == "LIVE_SIMULATION"
    else st.session_state["replay_cp"]
)

# 1–3. Header, Branding, Title, Explanation, and Offline Mode Indicator
render_header_banner(
    active_mode=st.session_state["demo_mode"],
    active_split=st.session_state["eval_split"],
    active_cp=active_cp_for_header,
)

# 4. Top Navigation Bar: Mode Switcher + Evaluation Cohort Selector + Reset Demo
nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1.3, 1.3, 1.7, 0.8])

with nav_col1:
    mode_a_label = (
        "● MODE A: LIVE SIMULATION"
        if st.session_state["demo_mode"] == "LIVE_SIMULATION"
        else "○ MODE A: LIVE SIMULATION"
    )
    if st.button(mode_a_label, key="btn_mode_a", use_container_width=True):
        st.session_state["demo_mode"] = "LIVE_SIMULATION"
        st.rerun()

with nav_col2:
    mode_b_label = (
        "● MODE B: REAL TEST REPLAY (10)"
        if st.session_state["demo_mode"] == "REAL_TEST_REPLAY"
        else "○ MODE B: REAL TEST REPLAY (10)"
    )
    if st.button(mode_b_label, key="btn_mode_b", use_container_width=True):
        st.session_state["demo_mode"] = "REAL_TEST_REPLAY"
        st.rerun()

with nav_col3:
    split_options = {
        "II": "EVAL II · Fit 2004–18 / Test 2021–23 (II_*.cbm)",
        "I": "EVAL I · Fit 2004–16 / Test 2019–20 (I_*.cbm)",
    }
    selected_split = st.selectbox(
        "EVALUATION COHORT & SAVED MODEL GROUP",
        options=["II", "I"],
        index=0 if st.session_state["eval_split"] == "II" else 1,
        format_func=lambda s: split_options[s],
        disabled=(st.session_state["demo_mode"] == "REAL_TEST_REPLAY"),
        help="Selects which temporal evaluation split's CatBoost models (I_C*.cbm or II_C*.cbm) and baseline tables are used. Real Test Replay uses Evaluation II as stored in metrics/example.csv.",
    )
    if (
        st.session_state["demo_mode"] == "LIVE_SIMULATION"
        and selected_split != st.session_state["eval_split"]
    ):
        st.session_state["eval_split"] = selected_split
        st.session_state["eventyear"] = 2022 if selected_split == "II" else 2019
        st.rerun()

with nav_col4:
    st.markdown("<div style='height: 1.68rem;'></div>", unsafe_allow_html=True)
    if st.button("↺ RESET DEMO", key="btn_reset_demo", type="primary", use_container_width=True):
        reset_demo_state()
        st.rerun()


# =============================================================================
# MODE A: LIVE SIMULATION (C0 -> C1 -> C2)
# =============================================================================
if st.session_state["demo_mode"] == "LIVE_SIMULATION":
    current_cp = st.session_state["live_cp"]
    current_split = st.session_state["eval_split"]

    # 5. C0 / C1 / C2 Checkpoint Navigation
    render_checkpoint_stepper(current_cp)

    cp_btn1, cp_btn2, cp_btn3 = st.columns(3)
    with cp_btn1:
        b_c0 = "● SELECT C0 — BEFORE RACE" if current_cp == "C0" else "SELECT C0 — BEFORE RACE"
        if st.button(b_c0, key="btn_live_c0", use_container_width=True):
            st.session_state["live_cp"] = "C0"
            st.rerun()
    with cp_btn2:
        b_c1 = "● SELECT C1 — AFTER SWIM" if current_cp == "C1" else "SELECT C1 — AFTER SWIM"
        if st.button(b_c1, key="btn_live_c1", use_container_width=True):
            st.session_state["live_cp"] = "C1"
            st.rerun()
    with cp_btn3:
        b_c2 = "● SELECT C2 — AFTER BIKE" if current_cp == "C2" else "SELECT C2 — AFTER BIKE"
        if st.button(b_c2, key="btn_live_c2", use_container_width=True):
            st.session_state["live_cp"] = "C2"
            st.rerun()

    # 6 & 7. Left Column: Athlete Information & Input Section | Right Column: Main Prediction Card & Race Progress
    left_col, right_col = st.columns([1.05, 1.35], gap="large")

    with left_col:
        render_html(
            f"""
            <div class="nb-card">
                <div class="nb-section-header">
                    <span>ATHLETE & RACE INPUTS ({current_cp})</span>
                    <span class="nb-badge nb-badge-{current_cp.lower()}">MODEL: {current_split}_{current_cp}.cbm</span>
                </div>
                <p style="font-size: 0.88rem; margin-bottom: 0.5rem;">
                    Only features genuinely required by <code>models/{current_split}_{current_cp}.cbm</code> are active below.
                    Actual FinishTime is never requested or used as an input.
                </p>
            </div>
            """
        )

        # Quick Athlete Presets for Exhibition Visitors
        pr_c1, pr_c2, pr_c3 = st.columns(3)
        with pr_c1:
            if st.button("P1: F 35-39 US", key="pre_1", use_container_width=True):
                st.session_state["gender"] = "F"
                st.session_state["agegroup"] = "35-39"
                st.session_state["countryiso"] = "US"
                st.session_state["eventlocation"] = "IRONMAN 70.3 Florida"
                st.session_state["eventyear"] = 2022 if current_split == "II" else 2019
                st.session_state["swim_h"], st.session_state["swim_m"], st.session_state["swim_s"] = 0, 35, 0
                st.session_state["bike_h"], st.session_state["bike_m"], st.session_state["bike_s"] = 2, 42, 0
                st.rerun()
        with pr_c2:
            if st.button("P2: M 30-34 VN", key="pre_2", use_container_width=True):
                st.session_state["gender"] = "M"
                st.session_state["agegroup"] = "30-34"
                st.session_state["countryiso"] = "VN"
                st.session_state["eventlocation"] = "IRONMAN 70.3 Vietnam"
                st.session_state["eventyear"] = 2023 if current_split == "II" else 2019
                st.session_state["swim_h"], st.session_state["swim_m"], st.session_state["swim_s"] = 0, 32, 30
                st.session_state["bike_h"], st.session_state["bike_m"], st.session_state["bike_s"] = 2, 28, 0
                st.rerun()
        with pr_c3:
            if st.button("P3: M 50-54 AT", key="pre_3", use_container_width=True):
                st.session_state["gender"] = "M"
                st.session_state["agegroup"] = "50-54"
                st.session_state["countryiso"] = "AT"
                st.session_state["eventlocation"] = "IRONMAN 70.3 Zell am See-Kaprun"
                st.session_state["eventyear"] = 2022 if current_split == "II" else 2019
                st.session_state["swim_h"], st.session_state["swim_m"], st.session_state["swim_s"] = 0, 38, 15
                st.session_state["bike_h"], st.session_state["bike_m"], st.session_state["bike_s"] = 2, 51, 30
                st.rerun()

        # Pre-race C0 Features (Always Required at C0, C1, C2)
        in_c1, in_c2 = st.columns(2)
        with in_c1:
            gender_val = st.selectbox(
                "1. GENDER (gender)",
                options=["F", "M"],
                index=0 if st.session_state["gender"] == "F" else 1,
                format_func=lambda g: "F — Female (Age-Group)" if g == "F" else "M — Male (Age-Group)",
            )
            st.session_state["gender"] = gender_val

        with in_c2:
            ag_list = engine.agegroups
            ag_idx = ag_list.index(st.session_state["agegroup"]) if st.session_state["agegroup"] in ag_list else 3
            agegroup_val = st.selectbox(
                "2. AGE GROUP → AGEBAND (ageband)",
                options=ag_list,
                index=ag_idx,
                format_func=lambda ag: f"{ag} (ageband = {ag.split('-')[0]})",
            )
            st.session_state["agegroup"] = agegroup_val

        country_codes = [c[0] for c in ACTIVE_COUNTRIES]
        country_labels = {c[0]: c[1] for c in ACTIVE_COUNTRIES}
        c_idx = country_codes.index(st.session_state["countryiso"]) if st.session_state["countryiso"] in country_codes else 0
        country_val = st.selectbox(
            "3. ATHLETE COUNTRY ISO2 (countryiso)",
            options=country_codes,
            index=c_idx,
            format_func=lambda code: country_labels.get(code, code),
        )
        st.session_state["countryiso"] = country_val

        loc_options = ACTIVE_EVENT_LOCATIONS + ["[Custom / Other Event Location...]"]
        loc_idx = (
            loc_options.index(st.session_state["eventlocation"])
            if st.session_state["eventlocation"] in loc_options
            else 0
        )
        selected_loc = st.selectbox(
            "4. EVENT LOCATION (eventlocation)",
            options=loc_options,
            index=loc_idx,
        )
        if selected_loc == "[Custom / Other Event Location...]":
            custom_loc = st.text_input(
                "ENTER CUSTOM EVENT LOCATION (Whitespace trimmed; empty → __MISSING__)",
                value=st.session_state["custom_location"],
                placeholder="e.g. IRONMAN 70.3 Tasmania",
            )
            st.session_state["custom_location"] = custom_loc
            final_location = custom_loc
        else:
            st.session_state["eventlocation"] = selected_loc
            final_location = selected_loc

        year_val = st.number_input(
            "5. EVENT YEAR (eventyear)",
            min_value=2004,
            max_value=2030,
            value=int(st.session_state["eventyear"]),
            step=1,
            help="Evaluation I fit: 2004–2016 (test 2019–2020). Evaluation II fit: 2004–2018 (test 2021–2023).",
        )
        st.session_state["eventyear"] = int(year_val)

        # Conditional Stage Split Inputs: C1 (SwimTime) and C2 (SwimTime + BikeTime)
        swim_seconds_val = None
        bike_seconds_val = None

        if current_cp in {"C1", "C2"}:
            render_html(
                """
                <div class="nb-card" style="background-color: #74B9FF; padding: 0.75rem 1rem; margin-top: 0.75rem; margin-bottom: 0.5rem;">
                    <span style="font-family: var(--font-mono); font-weight: 700; font-size: 0.86rem;">
                        CHECKPOINT C1 UNLOCKED · ENTER ACTUAL 1.9 KM SWIM TIME (HH:MM:SS)
                    </span>
                </div>
                """
            )
            sw_c1, sw_c2, sw_c3 = st.columns(3)
            with sw_c1:
                sh = st.number_input("SWIM HOURS", min_value=0, max_value=3, value=int(st.session_state["swim_h"]), step=1)
                st.session_state["swim_h"] = int(sh)
            with sw_c2:
                sm = st.number_input("SWIM MINUTES", min_value=0, max_value=59, value=int(st.session_state["swim_m"]), step=1)
                st.session_state["swim_m"] = int(sm)
            with sw_c3:
                ss = st.number_input("SWIM SECONDS", min_value=0, max_value=59, value=int(st.session_state["swim_s"]), step=1)
                st.session_state["swim_s"] = int(ss)

            swim_seconds_val = hms_to_seconds(sh, sm, ss)
            st.caption(
                f"Entered SwimTime: **{format_seconds_hms(swim_seconds_val)}** = **`{swim_seconds_val:,.0f}` seconds** "
                f"({swim_seconds_val/60:,.2f} min)"
            )

        if current_cp == "C2":
            render_html(
                """
                <div class="nb-card" style="background-color: #88D498; padding: 0.75rem 1rem; margin-top: 0.75rem; margin-bottom: 0.5rem;">
                    <span style="font-family: var(--font-mono); font-weight: 700; font-size: 0.86rem;">
                        CHECKPOINT C2 UNLOCKED · ENTER ACTUAL 90 KM BIKE TIME (HH:MM:SS)
                    </span>
                </div>
                """
            )
            bk_c1, bk_c2, bk_c3 = st.columns(3)
            with bk_c1:
                bh = st.number_input("BIKE HOURS", min_value=0, max_value=7, value=int(st.session_state["bike_h"]), step=1)
                st.session_state["bike_h"] = int(bh)
            with bk_c2:
                bm = st.number_input("BIKE MINUTES", min_value=0, max_value=59, value=int(st.session_state["bike_m"]), step=1)
                st.session_state["bike_m"] = int(bm)
            with bk_c3:
                bs = st.number_input("BIKE SECONDS", min_value=0, max_value=59, value=int(st.session_state["bike_s"]), step=1)
                st.session_state["bike_s"] = int(bs)

            bike_seconds_val = hms_to_seconds(bh, bm, bs)
            st.caption(
                f"Entered BikeTime: **{format_seconds_hms(bike_seconds_val)}** = **`{bike_seconds_val:,.0f}` seconds** "
                f"({bike_seconds_val/60:,.2f} min)"
            )

        # Quick Advance Checkpoint Button
        if current_cp == "C0":
            if st.button("→ COMPLETE SWIM & ADVANCE TO C1 (AFTER SWIM)", key="adv_c1", use_container_width=True):
                st.session_state["live_cp"] = "C1"
                st.rerun()
        elif current_cp == "C1":
            if st.button("→ COMPLETE BIKE & ADVANCE TO C2 (AFTER BIKE)", key="adv_c2", use_container_width=True):
                st.session_state["live_cp"] = "C2"
                st.rerun()

    # Right Column: Validation, Main Prediction Card, Race Progress, and Model Comparison
    with right_col:
        is_valid, validation_errors = validate_athlete_inputs(
            checkpoint=current_cp,
            gender=st.session_state["gender"],
            agegroup=st.session_state["agegroup"],
            countryiso=st.session_state["countryiso"],
            eventlocation=final_location,
            eventyear=st.session_state["eventyear"],
            valid_agegroups=engine.agegroups,
            swimtime_seconds=swim_seconds_val,
            biketime_seconds=bike_seconds_val,
        )

        if not is_valid:
            err_items = "".join(f"<li>{e}</li>" for e in validation_errors)
            render_html(
                f"""
                <div class="nb-card" style="background-color: #FF6B6B;">
                    <div class="nb-section-header">INPUT VALIDATION ERROR</div>
                    <ul style="margin: 0; padding-left: 1.25rem; font-weight: 700;">{err_items}</ul>
                </div>
                """
            )
        else:
            # Compute progressive predictions up to current_cp
            results_by_cp: Dict[str, Dict[str, Any]] = {}
            c0_res = engine.predict_checkpoint(
                split=current_split,
                checkpoint="C0",
                gender=st.session_state["gender"],
                agegroup=st.session_state["agegroup"],
                countryiso=st.session_state["countryiso"],
                eventlocation=final_location,
                eventyear=st.session_state["eventyear"],
            )
            results_by_cp["C0"] = c0_res

            c1_res = None
            if current_cp in {"C1", "C2"} and swim_seconds_val is not None:
                c1_res = engine.predict_checkpoint(
                    split=current_split,
                    checkpoint="C1",
                    gender=st.session_state["gender"],
                    agegroup=st.session_state["agegroup"],
                    countryiso=st.session_state["countryiso"],
                    eventlocation=final_location,
                    eventyear=st.session_state["eventyear"],
                    swimtime_seconds=swim_seconds_val,
                )
                results_by_cp["C1"] = c1_res

            c2_res = None
            if current_cp == "C2" and swim_seconds_val is not None and bike_seconds_val is not None:
                c2_res = engine.predict_checkpoint(
                    split=current_split,
                    checkpoint="C2",
                    gender=st.session_state["gender"],
                    agegroup=st.session_state["agegroup"],
                    countryiso=st.session_state["countryiso"],
                    eventlocation=final_location,
                    eventyear=st.session_state["eventyear"],
                    swimtime_seconds=swim_seconds_val,
                    biketime_seconds=bike_seconds_val,
                )
                results_by_cp["C2"] = c2_res

            active_res = results_by_cp[current_cp]

            # 7. Main FinishTime Prediction Card
            render_hero_prediction_card(
                pred_result=active_res,
                c0_pred_seconds=c0_res["CatBoost_ref"] if current_cp in {"C1", "C2"} else None,
                c1_pred_seconds=c1_res["CatBoost_ref"] if (current_cp == "C2" and c1_res is not None) else None,
            )

            # 8. Race Progress Visualization
            render_race_progress_card(
                checkpoint=current_cp,
                pred_finish_seconds=active_res["CatBoost_ref"],
                swimtime_seconds=swim_seconds_val,
                biketime_seconds=bike_seconds_val,
            )

            # Exact Feature Vector Passed to CatBoost Model
            feat_badges = "".join(
                f'<span class="nb-badge">{k} = <b>{v}</b></span>'
                for k, v in active_res["feature_values"].items()
            )
            render_html(
                f"""
                <div class="nb-card">
                    <div class="nb-section-header">
                        <span>EXACT FEATURE VECTOR PASSED TO {active_res["model_file"]}</span>
                        <span class="nb-badge">{len(active_res["features_used"])} FEATURES</span>
                    </div>
                    <div>{feat_badges}</div>
                </div>
                """
            )

    # 9. Model Comparison Section (Live Inference vs Baselines + Saved Historical Reference)
    if is_valid:
        rows_html = ""
        b0_s = active_res["B0"]
        rows_html += (
            f"<tr>"
            f"<td>B0 (Global Fit Median)</td>"
            f"<td>{current_cp}</td>"
            f"<td>{active_res['baseline_file']}</td>"
            f"<td>{format_seconds_hms(b0_s)}</td>"
            f"<td>{b0_s:,.1f} s</td>"
            f"<td>{format_seconds_hms(active_res['CatBoost_ref'] - b0_s, signed=True)}</td>"
            f"</tr>"
        )
        if current_cp == "C0" and active_res.get("B1") is not None:
            b1_s = active_res["B1"]
            rows_html += (
                f"<tr>"
                f"<td>B1 (Pre-Race Gender×Age Median)</td>"
                f"<td>C0</td>"
                f"<td>{active_res['matched_level']} (n={active_res['matched_group_size']:,})</td>"
                f"<td>{format_seconds_hms(b1_s)}</td>"
                f"<td>{b1_s:,.1f} s</td>"
                f"<td>{format_seconds_hms(active_res['CatBoost_ref'] - b1_s, signed=True)}</td>"
                f"</tr>"
            )
        b2_s = active_res["B2"]
        rows_html += (
            f"<tr>"
            f"<td>B2 (Known Splits + Group Median)</td>"
            f"<td>{current_cp}</td>"
            f"<td>{active_res['matched_level']} (n={active_res['matched_group_size']:,})</td>"
            f"<td>{format_seconds_hms(b2_s)}</td>"
            f"<td>{b2_s:,.1f} s</td>"
            f"<td>{format_seconds_hms(active_res['CatBoost_ref'] - b2_s, signed=True)}</td>"
            f"</tr>"
        )

        for cp_key in ["C0", "C1", "C2"]:
            if cp_key in results_by_cp:
                r_cp = results_by_cp[cp_key]
                cb_val = r_cp["CatBoost_ref"]
                row_cls = f"nb-row-highlight-{cp_key}" if cp_key == current_cp else ""
                delta_str = (
                    "ACTIVE REFERENCE"
                    if cp_key == current_cp
                    else format_seconds_hms(active_res["CatBoost_ref"] - cb_val, signed=True)
                )
                rows_html += (
                    f'<tr class="{row_cls}">'
                    f"<td>★ CatBoost_ref ({r_cp['model_file']})</td>"
                    f"<td>{cp_key}</td>"
                    f"<td>Live CatBoost Inference (seed=42)</td>"
                    f"<td>{format_seconds_hms(cb_val)}</td>"
                    f"<td>{cb_val:,.2f} s</td>"
                    f"<td>{delta_str}</td>"
                    f"</tr>"
                )

        render_html(
            f"""
            <div class="nb-card">
                <div class="nb-section-header">
                    <span>LIVE INFERENCE COMPARISON (CATBOOST_REF VS. EMPIRICAL BASELINES)</span>
                    <span class="nb-badge nb-badge-black">LIVE OFFLINE INFERENCE</span>
                </div>
                <div class="nb-table-wrap">
                    <table class="nb-table">
                        <thead>
                            <tr>
                                <th style="width: 26%;">Method / Artifact</th>
                                <th style="width: 8%;">CP</th>
                                <th style="width: 24%;">Inference Source</th>
                                <th style="width: 14%;">Predicted (HH:MM:SS)</th>
                                <th style="width: 14%;">Exact Seconds</th>
                                <th style="width: 14%;">Active CB Δ</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rows_html}
                        </tbody>
                    </table>
                </div>
                <p style="font-size: 0.82rem; font-family: var(--font-mono); margin: 0;">
                    <b>Scientific Integrity Note:</b> Only the 6 CatBoost models (<code>models/*.cbm</code>) and 6 baseline lookup tables
                    (<code>models/*_baseline.csv</code>) are persisted in the repository for live inference. Linear Regression (LR) and
                    Random Forest (RF) do not have saved model weights in <code>models/</code>, so live LR/RF predictions are never fabricated.
                    To inspect saved LR and RF predictions on real test athletes, switch to <b>Mode B: Real Test Replay</b>.
                </p>
            </div>
            """
        )

        fig_live = make_live_comparison_figure(results_by_cp, current_cp)
        st.pyplot(fig_live, use_container_width=True)
        plt.close(fig_live)


# =============================================================================
# MODE B: REAL TEST REPLAY (10 SAVED EXAMPLES FROM metrics/example.csv)
# =============================================================================
else:
    ex_df = engine.load_example_predictions()
    replay_cp = st.session_state["replay_cp"]
    replay_rid = st.session_state["replay_resultid"]
    reveal_actual = bool(st.session_state["replay_reveal_actual"])

    # 5. Checkpoint Navigation for Replay Mode
    render_checkpoint_stepper(replay_cp)

    rcp_1, rcp_2, rcp_3 = st.columns(3)
    with rcp_1:
        lbl0 = "● REPLAY C0 — BEFORE RACE" if replay_cp == "C0" else "REPLAY C0 — BEFORE RACE"
        if st.button(lbl0, key="btn_rep_c0", use_container_width=True):
            st.session_state["replay_cp"] = "C0"
            st.rerun()
    with rcp_2:
        lbl1 = "● REPLAY C1 — AFTER SWIM" if replay_cp == "C1" else "REPLAY C1 — AFTER SWIM"
        if st.button(lbl1, key="btn_rep_c1", use_container_width=True):
            st.session_state["replay_cp"] = "C1"
            st.rerun()
    with rcp_3:
        lbl2 = "● REPLAY C2 — AFTER BIKE" if replay_cp == "C2" else "REPLAY C2 — AFTER BIKE"
        if st.button(lbl2, key="btn_rep_c2", use_container_width=True):
            st.session_state["replay_cp"] = "C2"
            st.rerun()

    rep_left, rep_right = st.columns([0.95, 1.45], gap="large")

    with rep_left:
        render_html(
            """
            <div class="nb-card">
                <div class="nb-section-header">
                    <span>REAL TEST ATHLETE SELECTION</span>
                    <span class="nb-badge nb-badge-black">metrics/example.csv</span>
                </div>
                <p style="font-size: 0.88rem; margin-bottom: 0.5rem;">
                    Displays the 10 authentic test records from <code>metrics/example.csv</code> (Evaluation II, EventYear 2021, RaceID 774).
                    All predictions (<code>B0, B1, B2, LR, RF, CatBoost_ref</code>) are read directly from the saved file without reconstruction or modification.
                </p>
            </div>
            """
        )

        available_rids = sorted(ex_df["resultid"].unique().tolist())
        rid_idx = available_rids.index(replay_rid) if replay_rid in available_rids else 0
        chosen_rid = st.selectbox(
            "SELECT SAVED TEST RECORD (resultid)",
            options=available_rids,
            index=rid_idx,
            format_func=lambda r: f"Test Athlete #{available_rids.index(r)+1:02d} · resultid={r} (Race 774 · 2021)",
        )
        if chosen_rid != replay_rid:
            st.session_state["replay_resultid"] = int(chosen_rid)
            st.session_state["replay_reveal_actual"] = False
            st.rerun()

        athlete_rows = ex_df.loc[ex_df["resultid"] == st.session_state["replay_resultid"]].copy()
        race_id_val = int(athlete_rows["raceid"].iloc[0])
        event_yr_val = int(athlete_rows["eventyear"].iloc[0])
        actual_sec_val = float(athlete_rows["actual_finish_seconds"].iloc[0])

        # Explicit Reveal Actual Finish Time Control
        render_html(
            """
            <div class="nb-card" style="margin-top: 0.75rem;">
                <div class="nb-section-header">
                    <span>ACTUAL FINISH TIME CONTROL</span>
                    <span class="nb-badge nb-badge-accent">FINAL STEP REVEAL</span>
                </div>
                <p style="font-size: 0.86rem; margin-bottom: 0.5rem;">
                    Per experimental demo guidelines (<i>"Khi làm demo chỉ mở actual ở bước cuối"</i>),
                    the athlete's actual finish time remains hidden until you explicitly click the reveal button below.
                </p>
            </div>
            """
        )

        if not reveal_actual:
            if st.button("REVEAL ACTUAL FINISH TIME", key="btn_reveal_actual", type="primary", use_container_width=True):
                st.session_state["replay_reveal_actual"] = True
                st.rerun()
        else:
            if st.button("HIDE ACTUAL FINISH TIME", key="btn_hide_actual", use_container_width=True):
                st.session_state["replay_reveal_actual"] = False
                st.rerun()

        if reveal_actual:
            render_html(
                f"""
                <div class="nb-hero-card" style="background-color: #FF6B6B; padding: 1.1rem 1.4rem; margin-top: 0.75rem;">
                    <div class="nb-pred-label">★ REVEALED ACTUAL FINISH TIME · RESULTID #{chosen_rid}</div>
                    <div style="font-family: var(--font-mono); font-size: 2.6rem; font-weight: 700;">{format_seconds_hms(actual_sec_val)}</div>
                    <span class="nb-pred-seconds">EXACT ACTUAL: {actual_sec_val:,.1f} SECONDS ({actual_sec_val/60:,.2f} MINUTES)</span>
                </div>
                """
            )
        else:
            render_html(
                """
                <div class="nb-card" style="background-color: #FFFDF5; border-style: dashed; margin-top: 0.75rem;">
                    <span style="font-family: var(--font-mono); font-weight: 700;">🔒 ACTUAL FINISH TIME: LOCKED / HIDDEN (Click "REVEAL ACTUAL FINISH TIME" above to unlock)</span>
                </div>
                """
            )

    with rep_right:
        cp_rows = athlete_rows.loc[athlete_rows["checkpoint"] == replay_cp].copy()
        cb_row = cp_rows.loc[cp_rows["model"] == "CatBoost_ref"].iloc[0]
        cb_pred_sec = float(cb_row["pred_finish_seconds"])

        c0_cb_sec = float(
            athlete_rows.loc[
                (athlete_rows["checkpoint"] == "C0") & (athlete_rows["model"] == "CatBoost_ref"),
                "pred_finish_seconds",
            ].iloc[0]
        )
        c1_cb_sec = float(
            athlete_rows.loc[
                (athlete_rows["checkpoint"] == "C1") & (athlete_rows["model"] == "CatBoost_ref"),
                "pred_finish_seconds",
            ].iloc[0]
        )
        c2_cb_sec = float(
            athlete_rows.loc[
                (athlete_rows["checkpoint"] == "C2") & (athlete_rows["model"] == "CatBoost_ref"),
                "pred_finish_seconds",
            ].iloc[0]
        )

        err_badge = ""
        if reveal_actual:
            err_sec = cb_pred_sec - actual_sec_val
            err_badge = (
                f'<span class="nb-badge nb-badge-accent">'
                f'ERROR VS ACTUAL: {format_seconds_hms(err_sec, signed=True)} ({err_sec/60:+.2f} min | |err|={abs(err_sec)/60:.2f} min)'
                f'</span>'
            )

        # Main Hero Card for Saved Replay Record
        render_html(
            f"""
            <div class="nb-hero-card nb-hero-{replay_cp}">
                <div class="nb-pred-label">
                    SAVED TEST PREDICTION · RESULTID #{chosen_rid} · RACEID #{race_id_val} ({event_yr_val}) · CHECKPOINT {replay_cp} · CatBoost_ref (seed=42)
                </div>
                <div class="nb-pred-time">{format_seconds_hms(cb_pred_sec)}</div>
                <div style="margin-bottom: 0.75rem;">
                    <span class="nb-pred-seconds">SAVED VALUE: {cb_pred_sec:,.8f} SECONDS ({cb_pred_sec/60:,.2f} MINUTES)</span>
                </div>
                <div>
                    <span class="nb-badge">C0 CatBoost_ref: {format_seconds_hms(c0_cb_sec)}</span>
                    <span class="nb-badge">C1 CatBoost_ref: {format_seconds_hms(c1_cb_sec)}</span>
                    <span class="nb-badge">C2 CatBoost_ref: {format_seconds_hms(c2_cb_sec)}</span>
                    {err_badge}
                </div>
            </div>
            """
        )

        fig_rep = make_replay_chart(athlete_rows, replay_cp, reveal_actual)
        st.pyplot(fig_rep, use_container_width=True)
        plt.close(fig_rep)

    # Full-width card for the 7-column SAVED PREDICTIONS AT CHECKPOINT table
    model_order = ["B0", "B1", "B2", "LR", "RF", "CatBoost_ref"]
    cp_rows["order"] = cp_rows["model"].apply(lambda m: model_order.index(m) if m in model_order else 99)
    cp_rows = cp_rows.sort_values("order")

    t_rows_html = ""
    for _, r in cp_rows.iterrows():
        m_name = str(r["model"])
        m_seed = int(r["seed"])
        p_sec = float(r["pred_finish_seconds"])
        row_cls = f"nb-row-highlight-{replay_cp}" if m_name == "CatBoost_ref" else ""

        if reveal_actual:
            diff_s = p_sec - actual_sec_val
            actual_col = format_seconds_hms(actual_sec_val)
            err_col = f"{format_seconds_hms(diff_s, signed=True)} ({abs(diff_s)/60:.2f}m)"
        else:
            actual_col = "🔒 HIDDEN"
            err_col = "🔒 HIDDEN"

        t_rows_html += (
            f'<tr class="{row_cls}">'
            f"<td>{m_name}</td>"
            f"<td>{m_seed}</td>"
            f"<td>{replay_cp}</td>"
            f"<td>{format_seconds_hms(p_sec)}</td>"
            f"<td>{p_sec:,.2f} s</td>"
            f"<td>{actual_col}</td>"
            f"<td>{err_col}</td>"
            f"</tr>"
        )

    render_html(
        f"""
        <div class="nb-card">
            <div class="nb-section-header">
                <span>SAVED PREDICTIONS AT CHECKPOINT {replay_cp} (RESULTID #{chosen_rid})</span>
                <span class="nb-badge">{len(cp_rows)} SAVED METHODS</span>
            </div>
            <div class="nb-table-wrap">
                <table class="nb-table">
                    <thead>
                        <tr>
                            <th style="width: 18%;">Model</th>
                            <th style="width: 8%;">Seed</th>
                            <th style="width: 8%;">CP</th>
                            <th style="width: 16%;">Predicted</th>
                            <th style="width: 16%;">Seconds</th>
                            <th style="width: 14%;">Actual</th>
                            <th style="width: 20%;">Error (|MAE|)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {t_rows_html}
                    </tbody>
                </table>
            </div>
        </div>
        """
    )

    # Full-width card for C0 -> C1 -> C2 Progression Matrix
    pivot_rows_html = ""
    for m_name in ["B0", "B1", "B2", "LR", "RF", "CatBoost_ref"]:
        m_sub = athlete_rows.loc[athlete_rows["model"] == m_name].set_index("checkpoint")
        if m_sub.empty:
            continue
        c0_v = format_seconds_hms(float(m_sub.loc["C0", "pred_finish_seconds"])) if "C0" in m_sub.index else "N/A (C0 only)"
        c1_v = format_seconds_hms(float(m_sub.loc["C1", "pred_finish_seconds"])) if "C1" in m_sub.index else "—"
        c2_v = format_seconds_hms(float(m_sub.loc["C2", "pred_finish_seconds"])) if "C2" in m_sub.index else "—"
        hl = f"nb-row-highlight-{replay_cp}" if m_name == "CatBoost_ref" else ""
        if reveal_actual and "C2" in m_sub.index:
            c2_err = abs(float(m_sub.loc["C2", "pred_finish_seconds"]) - actual_sec_val) / 60.0
            c2_err_str = f"{c2_err:.2f} min"
        elif reveal_actual and "C0" in m_sub.index:
            c0_err = abs(float(m_sub.loc["C0", "pred_finish_seconds"]) - actual_sec_val) / 60.0
            c2_err_str = f"{c0_err:.2f} min (C0)"
        else:
            c2_err_str = "🔒 HIDDEN"

        pivot_rows_html += (
            f'<tr class="{hl}">'
            f"<td>{m_name}</td>"
            f"<td>{c0_v}</td>"
            f"<td>{c1_v}</td>"
            f"<td>{c2_v}</td>"
            f"<td>{c2_err_str}</td>"
            f"</tr>"
        )

    render_html(
        f"""
        <div class="nb-card">
            <div class="nb-section-header">
                <span>C0 → C1 → C2 PROGRESSION MATRIX (RESULTID #{chosen_rid})</span>
                <span class="nb-badge">EVALUATION II</span>
            </div>
            <div class="nb-table-wrap">
                <table class="nb-table">
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>C0 (Before Race)</th>
                            <th>C1 (After Swim)</th>
                            <th>C2 (After Bike)</th>
                            <th>Final |Error|</th>
                        </tr>
                    </thead>
                    <tbody>
                        {pivot_rows_html}
                    </tbody>
                </table>
            </div>
            <p style="font-size: 0.80rem; font-family: var(--font-mono); margin: 0;">
                Note: In the original experimental design (<code>scripts/run.py</code>), <code>B1</code> is defined specifically
                at pre-race checkpoint <code>C0</code> (where <code>B1 == B2</code>). At <code>C1</code> and <code>C2</code>,
                the demographic remainder baseline is reported as <code>B2</code>.
            </p>
        </div>
        """
    )


# =============================================================================
# 10. SUPPORTING METHODOLOGY & OFFICIAL BENCHMARK METRICS (metrics/all_models.csv)
# =============================================================================
st.markdown("<hr style='border: 2px solid #000000; margin: 1.5rem 0;'>", unsafe_allow_html=True)

with st.expander("📘 OFFICIAL EXPERIMENTAL BENCHMARK RESULTS & RESEARCH METHODOLOGY (metrics/all_models.csv)", expanded=False):
    all_models_df = engine.load_all_models_metrics()
    cohort_df = engine.load_cohort_metrics()
    snap_info = engine.load_snapshot_comparison()

    m_col1, m_col2 = st.columns(2, gap="large")
    with m_col1:
        fig_m2 = make_global_mae_chart(all_models_df, "II")
        st.pyplot(fig_m2, use_container_width=True)
        plt.close(fig_m2)
    with m_col2:
        fig_m1 = make_global_mae_chart(all_models_df, "I")
        st.pyplot(fig_m1, use_container_width=True)
        plt.close(fig_m1)

    disp_df = all_models_df.copy()
    disp_df["mae_minutes"] = disp_df["mae_minutes"].map(lambda x: f"{x:.2f} min")
    disp_df["rmse_minutes"] = disp_df["rmse_minutes"].map(lambda x: f"{x:.2f} min")
    disp_df["bias_minutes"] = disp_df["bias_minutes"].map(lambda x: f"{x:+.2f} min")
    disp_df["p90_abs_minutes"] = disp_df["p90_abs_minutes"].map(lambda x: f"{x:.2f} min")

    render_html(
        f"""
        <div class="nb-card">
            <div class="nb-section-header">
                <span>VERIFIED RESEARCH METHODOLOGY SUMMARY</span>
                <span class="nb-badge nb-badge-c2">1,152,658 ELIGIBLE RECORDS</span>
            </div>
            <ul style="font-size: 0.9rem; line-height: 1.5; margin-bottom: 0;">
                <li><b>Unified Cohort & Snapshot Verification:</b> {snap_info['drive_eligible']:,} eligible Age-Group records across {int(cohort_df['races'].sum()):,} race editions (2004–2023), verified with 0 discrepancies between Google Drive CSVs and Supabase view <code>vw_valid_c1c2</code>.</li>
                <li><b>Temporal Evaluation Splits:</b> <b>Evaluation I</b> fits on 2004–2016 (n = 495,489) and tests on 2019–2020 (n = 125,912 across 127 races). <b>Evaluation II</b> fits on 2004–2018 (n = 697,267) and tests on 2021–2023 (n = 329,366 post-QC across 258 races; 113 records with <code>FinishTime &lt;= SwimTime + BikeTime</code> excluded in final QC).</li>
                <li><b>Baselines:</b> <code>B0</code> = global fit median FinishTime; <code>B1</code> = pre-race median by <code>gender × agegroup</code>; <code>B2</code> = known split sum + median remaining time by <code>gender × agegroup</code> (fallback to <code>gender</code> then <code>all</code> if group size &lt; 30).</li>
                <li><b>CatBoost Reference Configuration (<code>CatBoost_ref</code>):</b> 120 iterations, depth 6, learning_rate 0.08, L2 leaf reg 3, RMSE loss, seed 42.</li>
            </ul>
        </div>
        """
    )
    st.dataframe(disp_df, use_container_width=True, hide_index=True)
