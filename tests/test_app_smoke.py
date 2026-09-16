"""Streamlit UI smoke test.

Not a test of prediction accuracy - checks that the app itself starts
without crashing, which is the minimum bar for "does the UI work" and
the thing most likely to silently break after a dependency update.

Uses Streamlit's built-in AppTest framework (no browser needed).
Requires streamlit>=1.28. Run from the project root:

    pytest tests/test_app_smoke.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

APP_PATH = "app/app.py"


def _app_exists() -> bool:
    return Path(APP_PATH).exists()


@pytest.mark.skipif(not _app_exists(), reason="app/app.py not found")
class TestAppSmoke:
    def test_app_starts_without_exception(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(APP_PATH, default_timeout=60)
        at.run()

        assert not at.exception, (
            f"app/app.py raised an exception on startup: {at.exception}"
        )

    def test_app_renders_at_least_one_element(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(APP_PATH, default_timeout=60)
        at.run()

        total_elements = (
            len(at.title) + len(at.header) + len(at.markdown)
            + len(at.text) + len(at.button) + len(at.text_input)
            + len(at.text_area) + len(at.radio) + len(at.selectbox)
        )
        assert total_elements > 0, (
            "app produced no visible elements on first run - "
            "it may have crashed silently before rendering anything"
        )
