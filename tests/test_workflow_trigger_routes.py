import pytest
from fastapi import HTTPException

import routes.task_routes as tr


def test_missing_action_is_400():
    with pytest.raises(HTTPException) as ei:
        tr.validate_workflow_task_create(None, None, True)
    assert ei.value.status_code == 400


def test_non_admin_is_403():
    with pytest.raises(HTTPException) as ei:
        tr.validate_workflow_task_create("flow1", None, False)
    assert ei.value.status_code == 403


def test_non_object_prompt_is_400():
    for bad in ("[1,2]", "not json", '"a string"', "5"):
        with pytest.raises(HTTPException) as ei:
            tr.validate_workflow_task_create("flow1", bad, True)
        assert ei.value.status_code == 400


def test_valid_workflow_task_passes():
    # admin, valid id, JSON-object prompt (and empty prompt) → no raise
    assert tr.validate_workflow_task_create("flow1", '{"topic": "AI"}', True) is None
    assert tr.validate_workflow_task_create("flow1", None, True) is None
    assert tr.validate_workflow_task_create("flow1", "", True) is None
