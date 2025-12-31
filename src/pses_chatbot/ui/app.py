from __future__ import annotations

import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

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
    build_audit_snapshot = None  # type: ignore


LEVEL_COLS = ["LEVEL1ID", "LEVEL2ID", "LEVEL3ID", "LEVEL4ID", "LEVEL5ID"]

ORG_LABEL_EN_COL = "org_name_en"
ORG_LABEL_FR_COL = "org_name_fr"
ORG_DEPT_CODE_COL = "dept_code"
ORG_UNITID_COL = "UNITID"

# Default years (confirmed cycles)
DEFAULT_SURVEY_YEARS = [2019, 2020, 2022, 2024]


def _normalize_text(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    s = str(x).replace("\u00A0", " ").strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def _coerce_int_series(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)


def _coerce_org_df(org_df: pd.DataFrame) -> pd.DataFrame:
    df = org_df.copy()
    for c in LEVEL_COLS:
        _coerce_int_series(df, c)
    _coerce_int_series(df, ORG_UNITID_COL)
    for c in [ORG_LABEL_EN_COL, ORG_LABEL_FR_COL, ORG_DEPT_CODE_COL]:
        if c in df.columns:
            df[c] = df[c].apply(_normalize_text)
    return df


def _label_org_row(row: pd.Series, lang: str = "EN", show_ids: bool = True) -> str:
    en = _normalize_text(row.get(ORG_LABEL_EN_COL, ""))
    fr = _normalize_text(row.get(ORG_LABEL_FR_COL, ""))
    dept = _normalize_text(row.get(ORG_DEPT_CODE_COL, ""))
    unitid = int(row.get(ORG_UNITID_COL, 0)) if ORG_UNITID_COL in row.index else 0

    name = (fr or en) if lang.upper() == "FR" else (en or fr)
    if not name:
        name = "Unit"

    if not show_ids:
        return name

    ids = ".".join(str(int(row.get(c, 0))) for c in LEVEL_COLS)
    dept_suffix = f", dept_code={dept}" if dept else ""
    unit_suffix = f", UNITID={unitid}" if unitid else ""
    return f"{name}  (IDs {ids}{unit_suffix}{dept_suffix})"


def _level_candidates(df: pd.DataFrame, level_idx: int, selected: Dict[str, int]) -> pd.DataFrame:
    work = df.copy()
    for i in range(level_idx):
        col = LEVEL_COLS[i]
        work = work[work[col] == int(selected.get(col, 0))]
    cur_col = LEVEL_COLS[level_idx]
    work = work[work[cur_col] > 0]
    for j in range(level_idx + 1, len(LEVEL_COLS)):
        dcol = LEVEL_COLS[j]
        work = work[work[dcol] == 0]
    return work.sort_values([cur_col])


def _find_rollup_name(df: pd.DataFrame, selected: Dict[str, int], lang: str) -> str:
    work = df.copy()
    for c in LEVEL_COLS:
        work = work[work[c] == int(selected.get(c, 0))]
    if work.empty:
        ids = ".".join(str(selected.get(c, 0)) for c in LEVEL_COLS)
        return f"(IDs {ids})"
    return _label_org_row(work.iloc[0], lang=lang, show_ids=False)


def render_org_cascade(org_df_raw: pd.DataFrame, lang: str = "EN") -> Dict[str, int]:
    if org_df_raw is None or len(org_df_raw) == 0:
        st.warning("Org metadata is empty; defaulting to PS-wide (LEVEL1–5 = 0).")
        return {c: 0 for c in LEVEL_COLS}

    df = _coerce_org_df(org_df_raw)

    required = set(LEVEL_COLS + [ORG_LABEL_EN_COL, ORG_LABEL_FR_COL])
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Org metadata missing required columns: {missing}")
        return {c: 0 for c in LEVEL_COLS}

    sel: Dict[str, int] = {c: 0 for c in LEVEL_COLS}

    lvl1 = df.copy()
    for c in LEVEL_COLS[1:]:
        lvl1 = lvl1[lvl1[c] == 0]
    lvl1 = lvl1.sort_values(["LEVEL1ID"])

    lvl1_map: Dict[int, str] = {0: "Public Service-wide (roll-up) — LEVEL1ID=0"}
    for _, row in lvl1.iterrows():
        l1 = int(row.get("LEVEL1ID", 0))
        if l1 <= 0:
            continue
        if l1 not in lvl1_map:
            lvl1_map[l1] = _label_org_row(row, lang=lang)

    l1_choice = st.selectbox(
        "Organization – Level 1",
        options=list(lvl1_map.keys()),
        format_func=lambda k: lvl1_map.get(k, str(k)),
        key="org_lvl1",
    )
    sel["LEVEL1ID"] = int(l1_choice)

    if sel["LEVEL1ID"] == 0:
        return sel

    for level_idx in range(1, 5):
        col = LEVEL_COLS[level_idx]

        parent_for_label = {**sel}
        for deeper in LEVEL_COLS[level_idx:]:
            parent_for_label[deeper] = 0

        rollup_name = _find_rollup_name(df, parent_for_label, lang=lang)
        rollup_label = f"All respondents (roll-up at {col}=0) — {rollup_name}"

        candidates = _level_candidates(df, level_idx=level_idx, selected=sel)

        options_map: Dict[int, str] = {0: rollup_label}
        for _, row in candidates.iterrows():
            lv = int(row.get(col, 0))
            if lv <= 0:
                continue
            if lv not in options_map:
                options_map[lv] = _label_org_row(row, lang=lang)

        if len(options_map) == 1:
            sel[col] = 0
            break

        choice = st.selectbox(
            f"Organization – {col}",
            options=list(options_map.keys()),
            format_func=lambda k, m=options_map: m.get(k, str(k)),
            key=f"org_{col}",
        )
        sel[col] = int(choice)

        if sel[col] == 0:
            break

        for deeper in LEVEL_COLS[level_idx + 1:]:
            sel[deeper] = 0

    return sel


def _render_chat_area() -> None:
    st.subheader("Prototype chat area")
    st.write("Conversational engine not wired yet. Use the developer panels below.")


def _render_backend_status() -> None:
    with st.expander("Backend status (developer view)", expanded=False):
        if st.button("Run test query (max 200 rows)"):
            try:
                with st.spinner("Querying PSES DataStore..."):
                    df = query_pses_results(
                        filters=None,
                        fields=None,
                        sort=None,
                        max_rows=200,
                        page_size=200,
                        timeout_seconds=90,
                        include_total=False,
                    )
                st.success("PSES DataStore query succeeded.")
                st.write(f"Returned: {df.shape[0]} rows × {df.shape[1]} columns")
            except Exception:
                st.error("Error while querying the PSES DataStore.")
                st.text_area("Traceback", value=traceback.format_exc(), height=220)


def _render_metadata_status() -> None:
    with st.expander("Metadata status (developer view)", expanded=True):
        colA, colB = st.columns([1, 2])
        with colA:
            refresh = st.checkbox("Force refresh metadata loaders", value=False)
        with colB:
            discover = st.button("Discover available survey years (SQL)")

        if st.button("Load metadata and show summary"):
            try:
                with st.spinner("Loading questions..."):
                    q = load_questions_meta(refresh=bool(refresh))
                with st.spinner("Loading scales..."):
                    s = load_scales_meta(refresh=bool(refresh))
                with st.spinner("Loading demographics..."):
                    d = load_demographics_meta(refresh=bool(refresh))
                with st.spinner("Loading org hierarchy..."):
                    o = load_org_meta(refresh=bool(refresh))
                with st.spinner("Loading pos/neg/agree..."):
                    p = load_posneg_meta(refresh=bool(refresh))

                st.success("Metadata loaded successfully.")
                st.write(f"Questions: {len(q)}")
                st.write(f"Scales: {len(s)}")
                st.write(f"Demographics: {len(d)}")
                st.write(f"Org rows: {len(o)}")
                st.write(f"Pos/Neg mappings: {len(p)}")
                st.write(f"Default survey years: {DEFAULT_SURVEY_YEARS}")

            except Exception:
                st.error("Error while loading metadata.")
                st.text_area("Traceback", value=traceback.format_exc(), height=260)

        if discover:
            try:
                with st.spinner("Discovering years via SQL..."):
                    years = get_available_survey_years(timeout_seconds=30)
                st.success(f"Available years (SQL): {years}")
            except Exception as e:
                st.warning("Year discovery failed.")
                st.code(repr(e))


def _dem_ui_resolve_demcode_and_labels(demo_df: Optional[pd.DataFrame]) -> Tuple[Optional[str], str]:
    """
    Returns:
      (demcode_value, ui_label)

    demcode_value:
      - None => overall (blank DEMCODE rows)
      - "1234" => subgroup demcode
    ui_label:
      - user-friendly string for the selected demographic
    """
    # Fallback: manual DEMCODE entry
    if demo_df is None or demo_df.empty:
        demcode_input = st.text_input("DEMCODE (leave blank for overall/no breakdown):", value="")
        demcode = demcode_input.strip()
        demcode_value = None if demcode == "" else demcode
        ui_label = "Overall (no breakdown)" if demcode_value is None else f"DEMCODE {demcode_value}"
        return demcode_value, ui_label

    df = demo_df.copy()

    # Accept either normalized columns (preferred) or raw sheet columns.
    # Normalized (new): demcode, label_en, category_en
    # Raw (legacy): 'DEMCODE 2024', 'DESCRIP_E', 'Category_E'
    if "demcode" in df.columns:
        code_col = "demcode"
    elif "DEMCODE 2024" in df.columns:
        code_col = "DEMCODE 2024"
    else:
        code_col = ""

    if "label_en" in df.columns:
        label_col = "label_en"
    elif "DESCRIP_E" in df.columns:
        label_col = "DESCRIP_E"
    else:
        label_col = ""

    if "category_en" in df.columns:
        cat_col = "category_en"
    elif "Category_E" in df.columns:
        cat_col = "Category_E"
    else:
        cat_col = ""

    # If we can’t find the necessary columns, fall back to manual
    if not code_col or not label_col or not cat_col:
        demcode_input = st.text_input("DEMCODE (leave blank for overall/no breakdown):", value="")
        demcode = demcode_input.strip()
        demcode_value = None if demcode == "" else demcode
        ui_label = "Overall (no breakdown)" if demcode_value is None else f"DEMCODE {demcode_value}"
        return demcode_value, ui_label

    df[cat_col] = df[cat_col].apply(_normalize_text)
    df[label_col] = df[label_col].apply(_normalize_text)
    df[code_col] = df[code_col].apply(_normalize_text)

    categories = sorted([c for c in df[cat_col].unique().tolist() if c])

    options = ["Overall (DEMCODE blank)"] + categories
    selected_category = st.selectbox("Demographic category", options=options, index=0)

    if selected_category == "Overall (DEMCODE blank)":
        return None, "Overall (no breakdown)"

    subset = df[df[cat_col] == selected_category].copy()
    subset = subset[subset[code_col] != ""]
    subset = subset.sort_values([label_col, code_col])

    # show: Label (code)
    display = [f"{row[label_col]} ({row[code_col]})" for _, row in subset.iterrows() if row[label_col] and row[code_col]]
    if not display:
        st.warning(f"No subgroups found for category: {selected_category}")
        return None, "Overall (no breakdown)"

    selected_subgroup = st.selectbox("Subgroup", options=display, index=0)

    # Extract code from "... (####)"
    code = selected_subgroup.rsplit("(", 1)[-1].rstrip(")").strip()
    if code == "":
        return None, "Overall (no breakdown)"

    return code, f"{selected_category} — {selected_subgroup}"


def _render_analytical_query_tester() -> None:
    with st.expander("Analytical query test (developer view)", expanded=True):
        default_years_str = ",".join(str(y) for y in DEFAULT_SURVEY_YEARS)

        col1, col2 = st.columns(2)

        with col1:
            question_code = st.text_input("Question code (QUESTION, e.g. Q08):", value="Q08")
            years_str = st.text_input(
                "Survey years (comma-separated). Default = known cycles:",
                value=default_years_str,
            )

            # ✅ APPROVED CHANGE: Category → Subgroup selector (with safe fallback to manual DEMCODE)
            try:
                demo_df = load_demographics_meta(refresh=False)
            except Exception:
                demo_df = None

            demcode_value, dem_ui_label = _dem_ui_resolve_demcode_and_labels(demo_df)

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
            org_levels = render_org_cascade(org_raw, lang="EN")

        if st.button("Run analytical query", key="run_analytical_query_btn"):
            status = st.status("Preparing query…", expanded=True)
            t0 = time.perf_counter()

            try:
                years: List[int] = []
                if years_str.strip():
                    for part in years_str.split(","):
                        p = part.strip()
                        if p:
                            years.append(int(p))
                else:
                    years = list(DEFAULT_SURVEY_YEARS)

                if not years:
                    raise QueryEngineError("No survey years specified.")

                # Canonical representation:
                #   None => overall (DEMCODE IS NULL OR '')
                #   "1234" => subgroup
                params = QueryParameters(
                    survey_years=sorted(years),
                    question_code=question_code.strip(),
                    demcode=demcode_value,
                    org_levels=org_levels,
                )

                status.write(
                    f"Resolved params: years={params.survey_years}, "
                    f"question={params.question_code}, demcode={params.demcode!r}, org={params.org_levels}"
                )

                status.update(label="Step 1/2 — Running CKAN analytical query…", state="running")

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
                    # ✅ show the selector label (Category — Subgroup (code)) when subgroup selected
                    if result.params.demcode is None:
                        st.write("Demographic: Overall (no breakdown)")
                    else:
                        st.write(f"Demographic: {dem_ui_label}")
                else:
                    st.success("Analytical query + audit snapshot succeeded.")
                    st.write(f"Question: {snapshot.question_code} — {snapshot.question_label_en}")
                    st.write(f"Organization: {snapshot.org_label_en or '(label not found)'}")
                    if result.params.demcode is None:
                        st.write("Demographic: Overall (no breakdown)")
                    else:
                        st.write(f"Demographic: {dem_ui_label}")
                    st.write(f"Metric: {snapshot.metric_name_en}")

                rows = []
                for m in result.yearly_metrics:
                    rows.append(
                        {
                            "Year": m.year,
                            "Value (Most positive / least negative)": m.value,
                            "Δ vs previous year": m.delta_vs_prev,
                            "N (ANSCOUNT)": m.n,
                        }
                    )
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

                st.write("Supporting table (raw slice):")
                st.dataframe(result.raw_df, use_container_width=True)

            except QueryEngineError as qerr:
                status.update(label="Analytical query failed.", state="error")
                st.error(f"Analytical query failed: {qerr}")
                st.text_area("Traceback", value=traceback.format_exc(), height=280)

            except Exception as e:
                status.update(label="Unexpected error.", state="error")
                st.error("Unexpected error while running analytical query.")
                st.code(repr(e))
                st.text_area("Traceback", value=traceback.format_exc(), height=280)


def run_app() -> None:
    st.set_page_config(page_title=APP_NAME, page_icon="📊", layout="wide")
    st.title(APP_NAME)
    st.caption(f"Prototype version {APP_VERSION}")

    _render_chat_area()
    _render_backend_status()
    _render_metadata_status()
    _render_analytical_query_tester()
