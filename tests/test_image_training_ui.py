import pathlib, subprocess
ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_index_has_image_training_modal_and_entries():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    for el in ('id="image-training-modal"', 'id="rail-imagetraining"', 'id="tool-imagetraining-btn"',
               '/static/js/imageTraining.js',
               'id="imgtrain-env-status"', 'id="imgtrain-env-setup"',
               'id="imgtrain-dataset"', 'id="imgtrain-dataset-suggestions"',
               'id="imgtrain-output-name"', 'id="imgtrain-rank"', 'id="imgtrain-alpha"',
               'id="imgtrain-lr"', 'id="imgtrain-steps"', 'id="imgtrain-resolution"',
               'id="imgtrain-start"', 'id="imgtrain-stop"', 'id="imgtrain-progress"',
               'id="imgtrain-close"', 'id="imgtrain-run-card"', 'id="imgtrain-env-progress"'):
        assert el in html, f"{el} missing from index.html"


def test_image_training_js_wires_admin_and_routes():
    src = (ROOT / "static" / "js" / "imageTraining.js").read_text(encoding="utf-8")
    for s in ('rail-imagetraining', 'tool-imagetraining-btn', 'isAdmin', 'Modals.register',
              "api('/api/image-training/env')", '/api/image-training/env/setup',
              "api('/api/image-training/runs',", '/api/image-training/runs/current',
              '/api/image-training/runs/stop', '/api/image-datasets',
              "from './imageTrainingCore.js'", 'formToConfig('):
        assert s in src, f"{s} missing from imageTraining.js"


def test_image_training_js_syntax(tmp_path):
    src = (ROOT / "static" / "js" / "imageTraining.js").read_text(encoding="utf-8")
    mjs = tmp_path / "imageTraining.mjs"; mjs.write_text(src, encoding="utf-8")
    p = subprocess.run(["node", "--check", str(mjs)], capture_output=True, text=True, encoding="utf-8")
    assert p.returncode == 0, p.stderr


def test_help_manual_has_image_training_section():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    assert "train an SDXL LoRA from a prepared image dataset" in html
