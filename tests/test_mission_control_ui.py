import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_index_has_mission_control_modal_and_entries():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    for el in ('id="mission-control-modal"', 'id="rail-mission-control"',
               'id="tool-mission-control-btn"', '/static/js/missionControl.js'):
        assert el in html, f"missing {el}"


def test_mission_control_js_registers_with_modal_manager():
    src = (ROOT / "static" / "js" / "missionControl.js").read_text(encoding="utf-8")
    assert "Modals.register" in src
    assert "'mission-control-modal'" in src or '"mission-control-modal"' in src


def test_mission_control_js_fetches_models():
    src = (ROOT / "static" / "js" / "missionControl.js").read_text(encoding="utf-8")
    assert "/api/models" in src


def test_mission_control_js_syntax():
    import subprocess
    result = subprocess.run(
        ["node", "--check", str(ROOT / "static" / "js" / "missionControl.js")],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_mission_control_has_hardware_widget():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    assert 'data-widget="hardware"' in html
    assert 'id="mc-body-hardware"' in html

    src = (ROOT / "static" / "js" / "missionControl.js").read_text(encoding="utf-8")
    assert "loadHardwareWidget" in src
    assert "/api/hwfit/usage" in src


def test_mission_control_has_tasks_widget():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    assert 'data-widget="tasks"' in html
    assert 'id="mc-body-tasks"' in html

    src = (ROOT / "static" / "js" / "missionControl.js").read_text(encoding="utf-8")
    assert "loadTasksWidget" in src
    assert "/api/tasks" in src
