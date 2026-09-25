"""Regression test for the 2026-07-29 content-hash migration in
scripts/pdf_ingest_router.py: Step 9 used to skip re-processing a PDF based
on (filename, size, mtime) - changed to SHA-256 content hash (mirroring the
same fix already made in src/retrieval/text_splitter.py's chunk_manifest.csv)
so a PDF restored/re-synced with identical bytes but a different mtime isn't
wrongly treated as changed, and a genuinely different PDF under the same
filename IS caught even if size/mtime happen to collide.

Uses real PDFs (via reportlab) and the real pdfinfo/pdftotext CLI tools -
no mocking - since this script's whole job is orchestrating those tools.
"""
import importlib
import sys

import pytest

reportlab_canvas = pytest.importorskip("reportlab.pdfgen.canvas")

import scripts.pdf_ingest_router as router


def _make_pdf(path, text, repeat=20):
    c = reportlab_canvas.Canvas(str(path))
    c.drawString(72, 720, text * repeat)
    c.save()


def _run(in_dir, out_dir, force=False, monkeypatch=None):
    argv = ["pdf_ingest_router.py", "--input", str(in_dir), "--output", str(out_dir)]
    if force:
        argv.append("--force")
    old_argv = sys.argv
    sys.argv = argv
    try:
        router.main()
    finally:
        sys.argv = old_argv


@pytest.fixture
def dirs(tmp_path):
    in_dir = tmp_path / "raw_pdfs"
    out_dir = tmp_path / "processed"
    in_dir.mkdir()
    return in_dir, out_dir


def _manifest_rows(out_dir):
    import csv
    with (out_dir / "ingest_manifest.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_rerun_with_no_changes_skips_via_hash(dirs, capsys):
    in_dir, out_dir = dirs
    _make_pdf(in_dir / "VNM_baocaothuongnien_2024.pdf", "Doanh thu tang truong manh. ")

    _run(in_dir, out_dir)
    out1 = capsys.readouterr().out
    assert "Processing VNM" in out1

    _run(in_dir, out_dir)
    out2 = capsys.readouterr().out
    assert "Skip (unchanged since last run): VNM_baocaothuongnien_2024.pdf" in out2


def test_content_change_same_filename_is_reprocessed(dirs, capsys):
    in_dir, out_dir = dirs
    pdf_path = in_dir / "VNM_baocaothuongnien_2024.pdf"
    _make_pdf(pdf_path, "Doanh thu tang truong manh. ")
    _run(in_dir, out_dir)
    capsys.readouterr()

    _make_pdf(pdf_path, "NOI DUNG DA DUOC DINH CHINH LAI HOAN TOAN KHAC. ")
    _run(in_dir, out_dir)
    out = capsys.readouterr().out
    assert "Processing VNM_baocaothuongnien_2024.pdf" in out
    assert "Skip" not in out


def test_force_reprocesses_even_if_unchanged(dirs, capsys):
    in_dir, out_dir = dirs
    _make_pdf(in_dir / "VNM_baocaothuongnien_2024.pdf", "Doanh thu tang truong manh. ")
    _run(in_dir, out_dir)
    capsys.readouterr()

    _run(in_dir, out_dir, force=True)
    out = capsys.readouterr().out
    assert "Processing VNM_baocaothuongnien_2024.pdf" in out


def test_ingest_version_bump_auto_invalidates_without_force(dirs, capsys, monkeypatch):
    in_dir, out_dir = dirs
    _make_pdf(in_dir / "VNM_baocaothuongnien_2024.pdf", "Doanh thu tang truong manh. ")
    _run(in_dir, out_dir)
    capsys.readouterr()

    monkeypatch.setattr(router, "CHAR_COUNT_MIN", 999999)
    monkeypatch.setattr(router, "INGEST_VERSION",
                         f"charmin999999_legmin{router.LEGIBILITY_MIN}_v1")
    _run(in_dir, out_dir)
    out = capsys.readouterr().out
    assert "Processing VNM_baocaothuongnien_2024.pdf" in out


def test_old_schema_manifest_without_pdf_hash_reprocesses_once_without_crashing(dirs, capsys):
    """A manifest written before 2026-07-29 has file_size/file_mtime columns,
    no pdf_hash/ingest_version at all - must not crash, and must safely
    treat that PDF as needing reprocessing exactly once."""
    in_dir, out_dir = dirs
    out_dir.mkdir()
    _make_pdf(in_dir / "AAA_baocaotaichinh_2024.pdf", "test content ")
    old_manifest = out_dir / "ingest_manifest.csv"
    old_manifest.write_text(
        "file,file_size,file_mtime,ticker,doctype,page,n_pages_total,char_count,"
        "legibility_ratio,status\n"
        "AAA_baocaotaichinh_2024.pdf,12345,1234567890,AAA,financial_statement,1,1,"
        "500,0.99,text_ok\n",
        encoding="utf-8",
    )

    _run(in_dir, out_dir)
    out = capsys.readouterr().out
    assert "Processing AAA_baocaotaichinh_2024.pdf" in out
