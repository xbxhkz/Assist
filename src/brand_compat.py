"""Brand env-var compatibility (Phase 4).

The product's documented env vars are ASSIST_*; the legacy names are
ODYSSEUS_*. Instead of rewriting ~1000 getenv call sites, mirror the two
prefixes into each other at process start so either name resolves. Import
this module (and/or call mirror_brand_env()) BEFORE any module that reads env
at import time (src.constants / core.constants).
"""
import os


def mirror_brand_env(environ=None) -> None:
    """Mirror ASSIST_* <-> ODYSSEUS_* env vars. setdefault never clobbers a
    value the caller set explicitly under both prefixes."""
    env = os.environ if environ is None else environ
    for k, v in list(env.items()):
        if k.startswith("ASSIST_"):
            env.setdefault("ODYSSEUS_" + k[len("ASSIST_"):], v)
        elif k.startswith("ODYSSEUS_"):
            env.setdefault("ASSIST_" + k[len("ODYSSEUS_"):], v)


# Run once on import so a bare `import src.brand_compat` is enough to bridge.
mirror_brand_env()
