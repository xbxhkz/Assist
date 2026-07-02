"""Env-var compatibility shim: ASSIST_* <-> ODYSSEUS_* mirroring."""
import os

import src.brand_compat as bc


def test_mirror_assist_to_odysseus():
    env = {"ASSIST_PORT": "9000"}
    bc.mirror_brand_env(env)
    assert env["ODYSSEUS_PORT"] == "9000"


def test_mirror_odysseus_to_assist():
    env = {"ODYSSEUS_DATA_DIR": "/x"}
    bc.mirror_brand_env(env)
    assert env["ASSIST_DATA_DIR"] == "/x"


def test_mirror_no_clobber_when_both_set():
    env = {"ASSIST_PORT": "1", "ODYSSEUS_PORT": "2"}
    bc.mirror_brand_env(env)
    assert env["ASSIST_PORT"] == "1"
    assert env["ODYSSEUS_PORT"] == "2"


def test_mirror_ignores_non_brand_keys():
    env = {"PATH": "/usr/bin", "HF_TOKEN": "x"}
    bc.mirror_brand_env(env)
    assert set(env) == {"PATH", "HF_TOKEN"}


def test_mirror_bridges_real_os_environ(monkeypatch):
    monkeypatch.delenv("ODYSSEUS_SCRIPT_HOST", raising=False)
    monkeypatch.setenv("ASSIST_SCRIPT_HOST", "myhost")
    bc.mirror_brand_env()  # defaults to os.environ
    assert os.getenv("ODYSSEUS_SCRIPT_HOST") == "myhost"


def test_mirror_is_idempotent_and_bridges_late_values():
    env = {"ASSIST_LATE": "v1"}       # value that appeared "after" first mirror (e.g. from .env)
    bc.mirror_brand_env(env)
    assert env["ODYSSEUS_LATE"] == "v1"
    env["ASSIST_LATER"] = "v2"         # another late arrival
    bc.mirror_brand_env(env)           # second call
    assert env["ODYSSEUS_LATER"] == "v2"
    assert env["ODYSSEUS_LATE"] == "v1"   # earlier bridge still intact
