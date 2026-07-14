import pytest
from src.shell_exec.gate import require_shell_consent


def test_gate_blocks_when_off():
    with pytest.raises(PermissionError):
        require_shell_consent(lambda k, d=None: False)


def test_gate_passes_when_on():
    require_shell_consent(lambda k, d=None: True)  # must not raise


def test_gate_reads_the_right_key():
    seen = {}
    def get(k, d=None):
        seen["key"] = k
        return True
    require_shell_consent(get)
    assert seen["key"] == "shell_exec_enabled"
