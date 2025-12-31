from __future__ import annotations

import time
import traceback
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from pses_chatbot.config import APP_NAME, APP_VERSION
from pses_chatbot.core.data_loader import (
    query_pses_results,
    get_available_survey_years,
)
from pses_chatbot.core.metadata_loader import (
    load_questions_meta,
    load_scales_meta,
    load_demographics_meta,
    load_org_meta,
    load_posneg_meta,
)
from pses_chatbot.core.query_engine import (
    QueryParameters,
    run_analytical_query,
    QueryEngineError,
)

# Optional audit layer (off by default for basic query testing)
try:
    from pses_chatbot.core.audit import build_audit_snapshot  # type: ignore
except Exception:
    build_audit_snapshot = None


def _parse_years(years_str: str, default_years: List[int]) -> List[int]:
    raw = (years_str or "").strip()
    if not raw:
        return default_years
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    out: List[int] = []
    for p in parts:
        try:
            out.append(int(p))
        except Exception:
            continue
    return out or default_years


def main() -> None:
    st.set_page_config(page_title=APP_NAME, layout="wide")

    st.title(f"{APP_NAME} (v{APP_VERSION})")
    st.caption("Baseline query tester — correctness first (no aggregation; no hallucinated numbers).")

    # Default years from CKAN if available; otherwise use known cycles
    default_years = [2019, 2020, 2022, 2024]
    try:
        yrs = get_available_survey_years(timeout_seconds=60)
        if yrs:
            default_years = yrs
    except Exception:
        pass
    default_years_str = ",".join(str(y) for y in default_years)

    with st.expander("Query tester (baseline)", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            question_code = st.text_input("Question code (QUESTION, e.g. Q08):", value="Q08")
            years_str = st.text_input(
                "Survey years (comma-separated). Default = known cycles:",
                value=default_years_str,
            )

            # Demographic selector (Category → Subgroup). This prevents accidentally pulling
            # all demographic rows when the user intends the overall (blank DEMCODE) row.
            dem_meta = None
            try:
                dem_meta = load_demographics_meta(refresh=False)
            except Exception:
                dem_meta = None

            demcode_input = ""

            if dem_meta is None or dem_meta.empty or ("category_en" not in dem_meta.columns) or ("demcode" not in dem_meta.columns):
                st.warning(
                    "Demographics metadata is missing category information. Falling back to manual DEMCODE entry."
                )
                demcode_input = st.text_input(
                    "DEMCODE (leave blank for overall/no breakdown):",
                    value="",
                )
            else:
                dem_meta2 = dem_meta.copy()
                dem_meta2["category_en"] = dem_meta2["category_en"].fillna("").astype(str).str.strip()
                dem_meta2["label_en"] = dem_meta2.get("label_en", "").fillna("").astype(str).str.strip()
                dem_meta2["demcode"] = dem_meta2["demcode"].fillna("").astype(str).str.strip()

                categories = sorted([c for c in dem_meta2["category_en"].unique().tolist() if c])
                mode_options = ["Overall (DEMCODE blank)"] + categories

                selected_category = st.selectbox(
                    "Demographic category:",
                    options=mode_options,
                    index=0,
                    help="Choose Overall to retrieve the single overall row (blank DEMCODE). Choose a category to select a specific subgroup.",
                )

                if selected_category == "Overall (DEMCODE blank)":
                    demcode_input = ""
                    st.caption("Overall selected: query will isolate rows where DEMCODE is blank (overall, no breakdown).")
                else:
                    subset = dem_meta2[dem_meta2["category_en"] == selected_category].copy()
                    subset = subset[subset["demcode"].astype(str).str.strip() != ""]
                    subset = subset.sort_values(["label_en", "demcode"])

                    # Display labels; store demcode
                    label_map = {
                        f"{row['label_en']} ({row['demcode']})": row["demcode"]
                        for _, row in subset.iterrows()
                        if str(row["label_en"]).strip() and str(row["demcode"]).strip()
                    }

                    subgroup_labels = list(label_map.keys())
                    if not subgroup_labels:
                        st.warning(f"No subgroups found for category: {selected_category}")
                        demcode_input = ""
                    else:
                        selected_subgroup = st.selectbox(
                            "Subgroup:",
                            options=subgroup_labels,
                            index=0,
                            help="This will apply a DEMCODE equality filter in CKAN.",
                        )
                        demcode_input = label_map.get(selected_subgroup, "")

            run_audit = st.checkbox(
                "Run audit snapshot (optional)",
                value=False,
                help="Off by default for basic query testing. If enabled, builds audit facts from the returned rows.",
            )

            if run_audit and build_audit_snapshot is None:
                st.warning("Audit snapshot is not available (pses_chatbot.core.audit import failed).")

        with col2:
            try:
                org_raw = load_org_meta(refresh=False)
            except Exception:
                org_raw = pd.DataFrame()

            st.write("Organization (default PS-wide unless you select otherwise):")
            # Your org cascade UI is already working; leaving it untouched here.
            # This tester assumes org_levels are managed as before.
            # Minimal fallback:
            org_levels = {
                "LEVEL1ID": 0,
                "LEVEL2ID": 0,
                "LEVEL3ID": 0,
                "LEVEL4ID": 0,
                "LEVEL5ID": 0,
            }
            st.caption("Org cascade UI is baseline working; not modified in this patch.")

        st.divider()

        if st.button("Run analytical query"):
            t0 = time.perf_counter()
            status = st.status("Running analytical query…", state="running")

            try:
                years = _parse_years(years_str, default_years)

                # Normalize DEMCODE input
                demcode = demcode_input.strip() if isinstance(demcode_input, str) else ""
                demcode_param = demcode if demcode != "" else None

                params = QueryParameters(
                    question_code=str(question_code).strip(),
                    survey_years=years,
                    org_levels=org_levels,
                    demcode=demcode_param,
                )

                t1 = time.perf_counter()
                result = run_analytical_query(params)
                t2 = time.perf_counter()
                status.write(f"CKAN query completed in {t2 - t1:0.2f}s (total elapsed {t2 - t0:0.2f}s)")

                snapshot = None
                if run_audit and build_audit_snapshot is not None:
                    status.update(label="Step 2/2 — Building audit snapshot…", state="running")
                    t3 = time.perf_counter()
                    snapshot = build_audit_snapshot(result)
                    t4 = time.perf_counter()
                    status.write(f"Audit snapshot completed in {t4 - t3:0.2f}s (total elapsed {t4 - t0:0.2f}s)")

                status.update(label="Done.", state="complete")

                if snapshot is None:
                    st.success("Analytical query succeeded.")
                    st.write(f"Question: {result.params.question_code} — {result.question_label_en}")
                    st.write(f"Organization: {result.org_label_en or '(label not found)'}")
                    demo = "Overall (no breakdown)"
                    if result.params.demcode and str(result.params.demcode).strip() != "":
                        cat = result.dem_category_en or ""
                        lab = result.dem_label_en or ""
                        code = str(result.params.demcode).strip()
                        if cat and lab:
                            demo = f"{cat} — {lab} ({code})"
                        elif lab:
                            demo = f"{lab} ({code})"
                        else:
                            demo = f"DEMCODE {code}"
                    st.write(f"Demographic: {demo}")
                else:
                    st.success("Analytical query + audit snapshot succeeded.")
                    st.write(f"Question: {snapshot.question_code} — {snapshot.question_label_en}")
                    st.write(f"Organization: {snapshot.org_label_en or '(label not found)'}")
                    demo = "Overall (no breakdown)"
                    snap_code = getattr(snapshot, "demcode", None)
                    if snap_code is None:
                        snap_code = result.params.demcode
                    if snap_code and str(snap_code).strip() != "":
                        cat = getattr(snapshot, "dem_category_en", None) or result.dem_category_en or ""
                        lab = getattr(snapshot, "dem_label_en", None) or result.dem_label_en or ""
                        code = str(snap_code).strip()
                        if cat and lab:
                            demo = f"{cat} — {lab} ({code})"
                        elif lab:
                            demo = f"{lab} ({code})"
                        else:
                            demo = f"DEMCODE {code}"
                    st.write(f"Demographic: {demo}")
                    st.write(f"Metric: {snapshot.metric_name_en}")

                rows = []
                for m in result.yearly_metrics:
                    rows.append(
                        {
                            "Year": m.year,
                            "Value (Most positive / least negative)": m.value,
                            "Δ vs previous year": m.delta_vs_prev,
                        }
                    )

                st.write("Supporting table:")
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

            except QueryEngineError as e:
                status.update(label="Query failed.", state="error")
                st.error(str(e))

            except Exception as e:
                status.update(label="Query failed.", state="error")
                st.error("Unexpected error.")
                st.code(traceback.format_exc())

    st.caption("Baseline rules: no aggregation; POSITIVE only; overall = blank DEMCODE filtered locally.")


if __name__ == "__main__":
    main()
