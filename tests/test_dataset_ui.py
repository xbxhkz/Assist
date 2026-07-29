import pathlib, subprocess
ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_index_has_dataset_modal_and_entries():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    for el in ('id="dataset-modal"', 'id="rail-dataset"', 'id="tool-dataset-btn"',
               '/static/js/dataset.js', '/static/js/datasetCore.js',
               'id="dataset-suggestions"', 'list="dataset-suggestions"'):
        assert el in html, f"{el} missing from index.html"


def test_dataset_js_wires_admin_and_routes():
    src = (ROOT / "static" / "js" / "dataset.js").read_text(encoding="utf-8")
    for s in ('rail-dataset', 'tool-dataset-btn', '/api/datasets/validate', '/api/datasets',
              'isAdmin', 'Modals.register'):
        assert s in src, f"{s} missing from dataset.js"


def test_dataset_js_syntax(tmp_path):
    src = (ROOT / "static" / "js" / "dataset.js").read_text(encoding="utf-8")
    mjs = tmp_path / "dataset.mjs"; mjs.write_text(src, encoding="utf-8")
    p = subprocess.run(["node", "--check", str(mjs)], capture_output=True, text=True, encoding="utf-8")
    assert p.returncode == 0, p.stderr


def test_training_js_populates_dataset_datalist():
    src = (ROOT / "static" / "js" / "training.js").read_text(encoding="utf-8")
    assert "/api/datasets" in src and "dataset-suggestions" in src


def test_help_manual_has_dataset_section():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    assert "build a training dataset without writing JSON" in html


def test_index_has_generate_card():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    for el in ('id="dataset-gen-brief"', 'id="dataset-gen-count"',
               'id="dataset-generate"', 'id="dataset-gen-staging"'):
        assert el in html, f"{el} missing from index.html"


def test_dataset_js_wires_generate():
    src = (ROOT / "static" / "js" / "dataset.js").read_text(encoding="utf-8")
    for s in ('/api/datasets/generate', 'dataset-generate', 'renderStaging', 'addGenerated', 'staged'):
        assert s in src, f"{s} missing from dataset.js"
