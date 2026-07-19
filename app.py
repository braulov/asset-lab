from __future__ import annotations

import streamlit as st


navigation = st.navigation(
    [
        st.Page(
            "pages/1_Daily_Lab.py",
            title="Daily Lab",
            icon=":material/calendar_view_day:",
            default=True,
        ),
        st.Page(
            "pages/hourly_moments.py",
            title="Hourly Moments",
            icon=":material/schedule:",
        ),
    ]
)
navigation.run()
