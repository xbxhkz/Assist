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


def test_mission_control_has_memory_widget():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    assert 'data-widget="memory"' in html
    assert 'id="mc-body-memory"' in html

    src = (ROOT / "static" / "js" / "missionControl.js").read_text(encoding="utf-8")
    assert "loadMemoryWidget" in src
    assert "/api/memory/timeline" in src


def test_mission_control_has_integrations_widget():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    assert 'data-widget="integrations"' in html
    assert 'id="mc-body-integrations"' in html

    src = (ROOT / "static" / "js" / "missionControl.js").read_text(encoding="utf-8")
    assert "loadIntegrationsWidget" in src
    assert "/api/auth/integrations" in src


def test_mission_control_loadAllWidgets_wires_all_loaders():
    """Verify that loadAllWidgets() actually calls all 5 loader functions.

    This guards against a future change that removes a loader call but
    forgets to update loadAllWidgets(), which would leave that widget
    silently at "Loading..." forever. The existing per-widget tests
    only check that the function NAME appears somewhere in the file,
    not that it's actually wired into loadAllWidgets().
    """
    src = (ROOT / "static" / "js" / "missionControl.js").read_text(encoding="utf-8")

    # Extract the loadAllWidgets function body using regex.
    # Pattern: from "function loadAllWidgets()" to the closing "}"
    import re
    match = re.search(
        r'function\s+loadAllWidgets\s*\(\)\s*\{(.*?)\n\}',
        src,
        re.DOTALL
    )
    assert match, "Could not find loadAllWidgets() function"

    body = match.group(1)

    # All 5 loaders must be called within the function body
    expected_calls = [
        'loadModelsWidget()',
        'loadHardwareWidget()',
        'loadTasksWidget()',
        'loadMemoryWidget()',
        'loadIntegrationsWidget()',
    ]
    for call in expected_calls:
        assert call in body, f"loadAllWidgets() does not call {call}"


def test_mission_control_link_targets_exist_in_html():
    """Verify that all mc-open-* link targets exist in index.html.

    The init() function's event listeners reference element IDs that
    must exist. This guards against a future rename of those IDs that
    forgets to update the hardcoded references in missionControl.js.
    """
    js_src = (ROOT / "static" / "js" / "missionControl.js").read_text(encoding="utf-8")
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")

    # Map of mc-open-* link id to its target element id.
    # These come from the init() function's hardcoded lookups:
    # mc-open-models -> model-picker-btn
    # mc-open-hardware -> hwmon
    # mc-open-tasks -> rail-tasks
    # mc-open-memory -> tool-memory-btn
    # mc-open-integrations -> tool-plugins-btn
    targets = {
        'model-picker-btn': 'models',
        'hwmon': 'hardware',
        'rail-tasks': 'tasks',
        'tool-memory-btn': 'memory',
        'tool-plugins-btn': 'integrations',
    }

    for target_id, widget_name in targets.items():
        # Verify the mc-open-* link exists in the JS
        open_id = f'mc-open-{widget_name}'
        assert f"'{open_id}'" in js_src or f'"{open_id}"' in js_src, \
            f"JS does not reference link id '{open_id}'"

        # Verify the target element exists in HTML
        assert f'id="{target_id}"' in html, \
            f"HTML does not contain element with id '{target_id}' (target of mc-open-{widget_name})"
