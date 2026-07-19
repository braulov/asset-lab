from __future__ import annotations

from pathlib import Path
import re

import streamlit as st


st.set_page_config(page_title="Asset Lab v6", page_icon="📈", layout="wide")

INTERVAL_OPTIONS = ("1 day", "1 hour")


def run_asset_lab() -> None:
    selected_interval = st.session_state.get("analysis_interval", INTERVAL_OPTIONS[0])
    if selected_interval not in INTERVAL_OPTIONS:
        selected_interval = INTERVAL_OPTIONS[0]

    page_path = Path(__file__).parent / (
        "pages/1_Daily_Lab.py"
        if selected_interval == "1 day"
        else "pages/hourly_moments.py"
    )
    source = page_path.read_text(encoding="utf-8")
    source = re.sub(
        r'^st\.set_page_config\([^\n]*\)\n',
        "",
        source,
        count=1,
        flags=re.MULTILINE,
    )

    interval_control = '''    interval_label = st.selectbox(
        "Interval",
        ["1 day", "1 hour"],
        key="analysis_interval",
    )
'''

    if selected_interval == "1 day":
        sidebar_marker = 'with st.sidebar:\n    st.header("Data")\n'
        source = source.replace(sidebar_marker, sidebar_marker + interval_control, 1)
        source = source.replace(
            '    interval_label = st.selectbox("Interval", list(INTERVALS), index=0)\n',
            "",
            1,
        )
    else:
        sidebar_marker = 'with st.sidebar:\n    st.header("Hourly data")\n'
        source = source.replace(
            sidebar_marker,
            'with st.sidebar:\n    st.header("Data")\n' + interval_control,
            1,
        )

    if 'key="analysis_interval"' not in source:
        raise RuntimeError(
            f"Could not insert the shared interval selector into {page_path.name}."
        )

    namespace = {
        "__name__": "__main__",
        "__file__": str(page_path),
        "__package__": None,
    }
    exec(compile(source, str(page_path), "exec"), namespace)


navigation = st.navigation(
    [st.Page(run_asset_lab, title="Asset Lab", default=True)],
    position="hidden",
)
navigation.run()
