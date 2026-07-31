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
