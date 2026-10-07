"""Database schema and connection management."""

import sqlite3
from pathlib import Path

import sqlite_vec

from .config import SCHEMA_VERSION, Config
from .fts import FTS_NORMALIZATION_VERSION, normalize_text_for_fts

_FTS_TOKENIZERS = {"porter unicode61", "unicode61", "trigram"}


def _validate_fts_tokenizer(tokenizer: str) -> str:
    if not isinstance(tokenizer, str) or tokenizer not in _FTS_TOKENIZERS:
        raise ValueError(
            f"Unsupported FTS tokenizer: {tokenizer!r}. "
            f"Supported values: {', '.join(sorted(_FTS_TOKENIZERS))}"
        )
    return tokenizer


def fts_path(doc_path: str) -> str:
    """Return last 2 path components for FTS indexing.

    Reduces IDF collapse from common prefixes (e.g. project name in every path).
    'openclaw-config/agents/AGENTS.md' -> 'agents/AGENTS.md'
    'notes/guide.md' -> 'notes/guide.md'
    'file.md' -> 'file.md'
    """
    parts = Path(doc_path).parts
    if len(parts) <= 2:
        return doc_path
    return str(Path(*parts[-2:]))


def connect(cfg: Config) -> sqlite3.Connection:
    """Open DB, load sqlite-vec, ensure schema is current."""
    tokenizer = _validate_fts_tokenizer(cfg.fts_tokenizer)
    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(cfg.db_path))
    conn.row_factory = sqlite3.Row
    conn.create_function(
        "normalize_text_for_fts", 1, normalize_text_for_fts, deterministic=True
    )
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)

    conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
    row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    current = int(row[0]) if row else 0
    needs_fts_rebuild = False

    if current < SCHEMA_VERSION:
        if current == 9:
            # Non-destructive: add an FTS-only representation and rebuild FTS below.
            needs_fts_rebuild = True
        elif current == 8:
            # Vec0 used L2 distance — switch to cosine. Must drop+recreate vec_chunks.
            print(
                f"Schema upgrade v{current} -> v{SCHEMA_VERSION}, switching vec0 to cosine distance..."
            )
            conn.execute("DROP TABLE IF EXISTS vec_chunks")
            print(
                "  Dropped vec_chunks (L2). Run 'kb index' to reindex with cosine distance."
            )
        elif current == 7:
            # Non-destructive: add fts_path to chunks, rebuild FTS using truncated paths
            print(
                f"Schema upgrade v{current} -> v{SCHEMA_VERSION}, truncating FTS paths..."
            )
            for trigger in ("fts_ai", "fts_ad", "fts_au"):
                conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
            conn.execute("DROP TABLE IF EXISTS fts_chunks")
            try:
                conn.execute("ALTER TABLE chunks ADD COLUMN fts_path TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass  # column already exists
            # Populate fts_path from doc_path (last 2 components)
            rows = conn.execute("SELECT id, doc_path FROM chunks").fetchall()
            for row in rows:
                conn.execute(
                    "UPDATE chunks SET fts_path = ? WHERE id = ?",
                    (fts_path(row["doc_path"]), row["id"]),
                )
            needs_fts_rebuild = True
        elif current == 6:
            # Non-destructive: add doc_path + fts_path to chunks, rebuild FTS
            print(
                f"Schema upgrade v{current} -> v{SCHEMA_VERSION}, adding doc_path to FTS..."
            )
            for trigger in ("fts_ai", "fts_ad", "fts_au"):
                conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
            conn.execute("DROP TABLE IF EXISTS fts_chunks")
            try:
                conn.execute("ALTER TABLE chunks ADD COLUMN doc_path TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE chunks ADD COLUMN fts_path TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            conn.execute(
                "UPDATE chunks SET doc_path = "
                "(SELECT path FROM documents WHERE id = chunks.doc_id)"
            )
            rows = conn.execute("SELECT id, doc_path FROM chunks").fetchall()
            for row in rows:
                conn.execute(
                    "UPDATE chunks SET fts_path = ? WHERE id = ?",
                    (fts_path(row["doc_path"]), row["id"]),
                )
            needs_fts_rebuild = True
        elif current == 5:
            # Non-destructive: rebuild FTS with configured tokenizer + doc_path + fts_path
            print(
                f"Schema upgrade v{current} -> v{SCHEMA_VERSION}, rebuilding FTS with {tokenizer} tokenizer..."
            )
            for trigger in ("fts_ai", "fts_ad", "fts_au"):
                conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
            conn.execute("DROP TABLE IF EXISTS fts_chunks")
            try:
                conn.execute("ALTER TABLE chunks ADD COLUMN doc_path TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE chunks ADD COLUMN fts_path TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            conn.execute(
                "UPDATE chunks SET doc_path = "
                "(SELECT path FROM documents WHERE id = chunks.doc_id)"
            )
            rows = conn.execute("SELECT id, doc_path FROM chunks").fetchall()
            for row in rows:
                conn.execute(
                    "UPDATE chunks SET fts_path = ? WHERE id = ?",
                    (fts_path(row["doc_path"]), row["id"]),
                )
            needs_fts_rebuild = True
        elif current == 4:
            # Non-destructive: rebuild FTS with triggers + configured tokenizer + fts_path
            print(
                f"Schema upgrade v{current} -> v{SCHEMA_VERSION}, rebuilding FTS with triggers..."
            )
            for trigger in ("fts_ai", "fts_ad", "fts_au"):
                conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
            conn.execute("DROP TABLE IF EXISTS fts_chunks")
            try:
                conn.execute("ALTER TABLE chunks ADD COLUMN doc_path TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE chunks ADD COLUMN fts_path TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            conn.execute(
                "UPDATE chunks SET doc_path = "
                "(SELECT path FROM documents WHERE id = chunks.doc_id)"
            )
            rows = conn.execute("SELECT id, doc_path FROM chunks").fetchall()
            for row in rows:
                conn.execute(
                    "UPDATE chunks SET fts_path = ? WHERE id = ?",
                    (fts_path(row["doc_path"]), row["id"]),
                )
            needs_fts_rebuild = True
        elif current == 3:
            # Non-destructive migration: add tags column + doc_path + fts_path, rebuild FTS
            print(
                f"Schema upgrade v{current} -> v{SCHEMA_VERSION}, adding tags column + FTS triggers..."
            )
            for trigger in ("fts_ai", "fts_ad", "fts_au"):
                conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
            conn.execute("DROP TABLE IF EXISTS fts_chunks")
            try:
                conn.execute("ALTER TABLE documents ADD COLUMN tags TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE chunks ADD COLUMN doc_path TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE chunks ADD COLUMN fts_path TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            conn.execute(
                "UPDATE chunks SET doc_path = "
                "(SELECT path FROM documents WHERE id = chunks.doc_id)"
            )
            rows = conn.execute("SELECT id, doc_path FROM chunks").fetchall()
            for row in rows:
                conn.execute(
                    "UPDATE chunks SET fts_path = ? WHERE id = ?",
                    (fts_path(row["doc_path"]), row["id"]),
                )
            needs_fts_rebuild = True
        else:
            print(
                f"Schema upgrade v{current} -> v{SCHEMA_VERSION}, rebuilding tables..."
            )
            for table in ["vec_chunks", "fts_chunks", "chunks", "documents"]:
                conn.execute(f"DROP TABLE IF EXISTS {table}")
        if current >= 3:
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(chunks)")}
            if "fts_text" not in columns:
                conn.execute("ALTER TABLE chunks ADD COLUMN fts_text TEXT DEFAULT ''")
            needs_fts_rebuild = True
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        conn.commit()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE NOT NULL,
            title TEXT,
            type TEXT,
            size_bytes INTEGER,
            content_hash TEXT,
            indexed_at TEXT DEFAULT (datetime('now')),
            chunk_count INTEGER DEFAULT 0,
            tags TEXT DEFAULT ''
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            fts_text TEXT DEFAULT '',
            heading TEXT,
            heading_ancestry TEXT,
            char_count INTEGER,
            content_hash TEXT,
            doc_path TEXT DEFAULT '',
            fts_path TEXT DEFAULT ''
        )
    """)
    # Auto-detect local model dims to avoid mismatch (e.g. default 1536 vs model's 768).
    # Only load model when vec_chunks doesn't exist yet (table creation needs correct dims).
    dims = cfg.embed_dims
    if cfg.embed_method == "local":
        vec_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='vec_chunks'"
        ).fetchone()
        if not vec_exists:
            from .embed import local_embed_dims

            dims = local_embed_dims(cfg)
    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
            chunk_id INTEGER PRIMARY KEY,
            embedding float[{dims}] distance_metric=cosine,
            +chunk_text TEXT,
            +doc_path TEXT,
            +heading TEXT
        )
    """)
    try:
        _ensure_fts_index(conn, tokenizer, needs_rebuild=needs_fts_rebuild)
    except Exception:
        conn.close()
        raise
    return conn


def _ensure_fts_index(
    conn: sqlite3.Connection, tokenizer: str, *, needs_rebuild: bool = False
) -> None:
    """Create or rebuild FTS atomically, preserving chunks and vector embeddings."""
    with conn:
        # SQLite DDL needs an explicit transaction so a failed rebuild rolls back.
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'fts_tokenizer'"
        ).fetchone()
        normalization_row = conn.execute(
            "SELECT value FROM meta WHERE key = 'fts_normalization_version'"
        ).fetchone()
        fts_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='fts_chunks'"
        ).fetchone()
        if (
            needs_rebuild
            or not fts_exists
            or row is None
            or row["value"] != tokenizer
            or normalization_row is None
            or normalization_row["value"] != FTS_NORMALIZATION_VERSION
        ):
            # Drop old triggers before backfilling derived text; old triggers may
            # index raw text or fire on updates to the new FTS-only column.
            for trigger in ("fts_ai", "fts_ad", "fts_au"):
                conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
            conn.execute("DROP TABLE IF EXISTS fts_chunks")
            conn.execute("UPDATE chunks SET fts_text = normalize_text_for_fts(text)")
            needs_rebuild = True

        # External-content columns must map to the same text/path used by FTS.
        conn.execute("""
            CREATE VIEW IF NOT EXISTS fts_chunks_content AS
            SELECT id, fts_path AS doc_path, heading, fts_text AS text FROM chunks
        """)
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
                doc_path,
                heading,
                text,
                content='fts_chunks_content',
                content_rowid='id',
                tokenize='{tokenizer}'
            )
        """)
        # Weighted BM25: truncated path=10x, heading=2x, text=1x.
        conn.execute("""
            INSERT INTO fts_chunks(fts_chunks, rank)
            VALUES('rank', 'bm25(10.0, 2.0, 1.0)')
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS fts_ai AFTER INSERT ON chunks BEGIN
                UPDATE chunks SET fts_text = normalize_text_for_fts(new.text)
                WHERE id = new.id;
                INSERT INTO fts_chunks(rowid, doc_path, heading, text)
                SELECT id, fts_path, heading, fts_text FROM chunks WHERE id = new.id;
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS fts_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO fts_chunks(fts_chunks, rowid, doc_path, heading, text)
                VALUES ('delete', old.id, old.fts_path, old.heading, old.fts_text);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS fts_au
            AFTER UPDATE OF text, heading, fts_path ON chunks BEGIN
                INSERT INTO fts_chunks(fts_chunks, rowid, doc_path, heading, text)
                VALUES ('delete', old.id, old.fts_path, old.heading, old.fts_text);
                UPDATE chunks SET fts_text = normalize_text_for_fts(new.text)
                WHERE id = new.id;
                INSERT INTO fts_chunks(rowid, doc_path, heading, text)
                SELECT id, fts_path, heading, fts_text FROM chunks WHERE id = new.id;
            END
        """)
        if needs_rebuild:
            # Reuse the FTS-only rebuild path, preserving fts_path and fts_text.
            conn.execute("INSERT INTO fts_chunks(fts_chunks) VALUES('delete-all')")
            conn.execute(
                "INSERT INTO fts_chunks(rowid, doc_path, heading, text) "
                "SELECT id, fts_path, heading, fts_text FROM chunks"
            )
        conn.executemany(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            [
                ("fts_tokenizer", tokenizer),
                ("fts_normalization_version", FTS_NORMALIZATION_VERSION),
            ],
        )


def reset(db_path: Path):
    if db_path.exists():
        db_path.unlink()
        # Clean up WAL sidecar files
        for suffix in ("-shm", "-wal"):
            sidecar = db_path.parent / (db_path.name + suffix)
            if sidecar.exists():
                sidecar.unlink()
        print(f"Deleted {db_path}")
    else:
        print("No database to reset.")
