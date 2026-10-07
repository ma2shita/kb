"""Tests for FTS-only CJK layout whitespace normalization."""

from unittest.mock import MagicMock, patch

import pytest

from kb.api import fts_core, search_core
from kb.db import connect, fts_path
from kb.embed import serialize_f32
from kb.fts import normalize_text_for_fts
from kb.ingest import md5_hash
from kb.search import run_fts_query


@pytest.mark.parametrize(
    "text, expected",
    [
        ("車両やロボットの製\n造プロセス", "車両やロボットの製造プロセス"),
        ("製   造プロセス", "製造プロセス"),
        ("製\r\n造プロセス", "製造プロセス"),
        ("製\t\r\n造プロセス", "製造プロセス"),
        ("製\u3000造プロセス", "製造プロセス"),
        ("で操作でき\nます", "で操作できます"),
        ("ライフ\nサイクル", "ライフサイクル"),
        ("ﾈｯﾄ\nﾜｰｸ", "ﾈｯﾄﾜｰｸ"),
        ("㐀\n造", "㐀造"),
        ("𠮷 \n野", "𠮷野"),
        ("네트워크 연결", "네트워크 연결"),
        ("製\n造・物流", "製造・物流"),
        ("製造。\n物流", "製造。\n物流"),
        ("製造・\n物流", "製造・\n物流"),
        ("AWS IoT Core", "AWS IoT Core"),
        ("AWS\nIoT\tCore", "AWS\nIoT\tCore"),
        ("AWS 環境で利用する", "AWS 環境で利用する"),
        ("フィジカル AI", "フィジカル AI"),
        ("IoT デバイス", "IoT デバイス"),
        ("5G ネットワーク", "5G ネットワーク"),
        ("第 5 世代", "第 5 世代"),
        (" \n製\n造 \t", " \n製造 \t"),
        ("", ""),
    ],
)
def test_normalize_text_for_fts(text, expected):
    assert normalize_text_for_fts(text) == expected
    assert normalize_text_for_fts(expected) == expected


@pytest.fixture
def layout_index(tmp_config):
    cfg = tmp_config
    cfg.embed_dims = 4
    cfg.fts_tokenizer = "trigram"
    cfg.hyde_enabled = False
    raw = "車両やロボットの製\n造プロセス・ライフサイクル\nAWS IoT Core"
    path = "project/docs/layout.pdf"
    conn = connect(cfg)
    conn.execute(
        "INSERT INTO documents (path, type, content_hash, tags) VALUES (?, 'pdf', ?, 'layout')",
        (path, md5_hash(raw)),
    )
    doc_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute(
        "INSERT INTO chunks (doc_id, chunk_index, text, content_hash, doc_path, fts_path) "
        "VALUES (?, 0, ?, ?, ?, ?)",
        (doc_id, raw, md5_hash(raw), path, fts_path(path)),
    )
    chunk_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute(
        "INSERT INTO vec_chunks (chunk_id, embedding, chunk_text, doc_path, heading) "
        "VALUES (?, ?, ?, ?, '')",
        (chunk_id, serialize_f32([1.0, 0.0, 0.0, 0.0]), raw, path),
    )
    conn.commit()
    yield conn, cfg, chunk_id, raw
    conn.close()


def test_fts_indexes_normalized_text_and_displays_raw_text(layout_index):
    conn, cfg, chunk_id, raw = layout_index
    row = conn.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
    assert row["text"] == raw
    assert row["content_hash"] == md5_hash(raw)
    assert row["fts_text"] == raw.replace("製\n造", "製造")
    assert conn.execute("SELECT text FROM fts_chunks").fetchone()[0] == row["fts_text"]

    result = fts_core("製造プロセス", cfg)

    assert result["results"][0]["text"] == raw
    assert result["results"][0]["doc_path"] == "project/docs/layout.pdf"
    assert fts_core("AWS IoT Core", cfg)["results"][0]["text"] == raw


def test_hybrid_result_has_vec_and_fts_sources_with_raw_text(layout_index):
    conn, cfg, chunk_id, raw = layout_index
    with (
        patch("kb.api.OpenAI", return_value=MagicMock()),
        patch("kb.api.embed_batch", return_value=[[1.0, 0.0, 0.0, 0.0]]),
    ):
        result = search_core("製造プロセス", cfg)

    assert result["results"][0]["sources"] == ["vec", "fts"]
    assert result["results"][0]["text"] == raw
    assert (
        conn.execute(
            "SELECT chunk_text FROM vec_chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()[0]
        == raw
    )


def test_normalized_fts_stays_synced_on_update_and_delete(layout_index):
    conn, cfg, chunk_id, raw = layout_index
    conn.execute("PRAGMA recursive_triggers = ON")
    updated = "物流シ\r\nステム"
    conn.execute("UPDATE chunks SET text = ? WHERE id = ?", (updated, chunk_id))

    row = conn.execute(
        "SELECT text, fts_text FROM chunks WHERE id = ?", (chunk_id,)
    ).fetchone()
    assert row["text"] == updated
    assert row["fts_text"] == "物流システム"
    assert run_fts_query(conn, "製造プロセス", 5) == []
    assert run_fts_query(conn, "物流システム", 5)[0][0] == chunk_id

    conn.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))

    assert run_fts_query(conn, "物流システム", 5) == []


@pytest.mark.parametrize("tokenizer", ["porter unicode61", "unicode61", "trigram"])
def test_normalization_is_independent_of_tokenizer(layout_index, tokenizer):
    conn, cfg, chunk_id, raw = layout_index
    before = {
        table: [tuple(row) for row in conn.execute(f"SELECT * FROM {table}")]
        for table in ("documents", "chunks", "vec_chunks")
    }
    cfg.fts_tokenizer = tokenizer
    reopened = connect(cfg)
    try:
        after = {
            table: [tuple(row) for row in reopened.execute(f"SELECT * FROM {table}")]
            for table in before
        }
        assert after == before
        assert reopened.execute(
            "SELECT fts_text FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()[0] == raw.replace("製\n造", "製造")
    finally:
        reopened.close()


def test_builtin_fts_rebuild_uses_normalized_content(layout_index):
    conn, cfg, chunk_id, raw = layout_index
    conn.execute("INSERT INTO fts_chunks(fts_chunks) VALUES('rebuild')")

    assert run_fts_query(conn, "製造プロセス", 5)[0][0] == chunk_id
    assert run_fts_query(conn, "project", 5) == []
    # Verify that the FTS index and its external-content representation agree.
    conn.execute(
        "INSERT INTO fts_chunks(fts_chunks, rank) VALUES('integrity-check', 1)"
    )


def test_normalization_version_change_rebuilds_only_fts(layout_index):
    conn, cfg, chunk_id, raw = layout_index
    vectors = [tuple(row) for row in conn.execute("SELECT * FROM vec_chunks")]
    conn.execute("UPDATE chunks SET fts_text = text")
    conn.execute("INSERT INTO fts_chunks(fts_chunks) VALUES('rebuild')")
    conn.execute("UPDATE meta SET value = '0' WHERE key = 'fts_normalization_version'")
    conn.commit()
    assert run_fts_query(conn, "製造プロセス", 5) == []

    reopened = connect(cfg)
    try:
        assert run_fts_query(reopened, "製造プロセス", 5)[0][0] == chunk_id
        assert [
            tuple(row) for row in reopened.execute("SELECT * FROM vec_chunks")
        ] == vectors
        assert reopened.execute("SELECT text FROM chunks").fetchone()[0] == raw
        assert reopened.execute("SELECT content_hash FROM chunks").fetchone()[
            0
        ] == md5_hash(raw)
    finally:
        reopened.close()
