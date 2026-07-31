import pathlib, subprocess
ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_index_has_image_dataset_modal_and_entries():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    for el in ('id="imagedataset-modal"', 'id="rail-imagedataset"', 'id="tool-imagedataset-btn"',
               '/static/js/imageDataset.js',
               'id="imgds-source"', 'id="imgds-file"', 'id="imgds-gallery-pick"',
               'id="imgds-grid"', 'id="imgds-caption-all"', 'id="imgds-trigger"',
               'id="imgds-validate"', 'id="imgds-report"', 'id="imgds-name"',
               'id="imgds-save"', 'id="imgds-saved"'):
        assert el in html, f"{el} missing from index.html"


def test_image_dataset_js_wires_admin_and_routes():
    src = (ROOT / "static" / "js" / "imageDataset.js").read_text(encoding="utf-8")
    for s in ('rail-imagedataset', 'tool-imagedataset-btn', 'isAdmin', 'Modals.register',
              '/api/image-datasets/upload', '/api/image-datasets/from-gallery',
              '/api/image-datasets/caption', '/api/image-datasets/validate',
              "api('/api/image-datasets'"):
        assert s in src, f"{s} missing from imageDataset.js"


def test_image_dataset_js_syntax(tmp_path):
    src = (ROOT / "static" / "js" / "imageDataset.js").read_text(encoding="utf-8")
    mjs = tmp_path / "imageDataset.mjs"; mjs.write_text(src, encoding="utf-8")
    p = subprocess.run(["node", "--check", str(mjs)], capture_output=True, text=True, encoding="utf-8")
    assert p.returncode == 0, p.stderr
