import os
import sqlite3
import datetime
from typing import List, Dict, Any, Optional

DB_FILE = os.getenv("SQLITE_DB_PATH", os.path.join("outputs", "predictions.db"))


def get_db_connection():
    # Establish Connection To SQLite Or PostgreSQL Depending On Environment
    db_url = os.getenv("DATABASE_URL")
    if db_url and db_url.startswith("postgres"):
        try:
            import psycopg2
            import psycopg2.extras
            conn = psycopg2.connect(db_url, cursor_factory=psycopg2.extras.RealDictCursor)
            return conn, "postgres"
        except Exception as e:
            print(f"Warning: PostgreSQL connection failed ({e}), falling back to SQLite.")

    os.makedirs(os.path.dirname(os.path.abspath(DB_FILE)), exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn, "sqlite"


def init_db():
    # Initialize Predictions Table If Not Exists
    conn, engine = get_db_connection()
    try:
        cursor = conn.cursor()
        if engine == "postgres":
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    filename TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    raw_score REAL NOT NULL,
                    threshold REAL NOT NULL,
                    model_mode TEXT NOT NULL,
                    latency_ms REAL NOT NULL
                );
                """
            )
        else:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT DEFAULT (datetime('now', 'utc')),
                    filename TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    raw_score REAL NOT NULL,
                    threshold REAL NOT NULL,
                    model_mode TEXT NOT NULL,
                    latency_ms REAL NOT NULL
                );
                """
            )
        conn.commit()
    finally:
        conn.close()


def insert_prediction(
    filename: str,
    file_hash: str,
    verdict: str,
    confidence: float,
    raw_score: float,
    threshold: float,
    model_mode: str,
    latency_ms: float,
) -> int:
    # Record An Inference Audit Trail In Database
    conn, engine = get_db_connection()
    try:
        cursor = conn.cursor()
        if engine == "postgres":
            cursor.execute(
                """
                INSERT INTO predictions (filename, file_hash, verdict, confidence, raw_score, threshold, model_mode, latency_ms)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
                """,
                (filename, file_hash, verdict, confidence, raw_score, threshold, model_mode, latency_ms),
            )
            inserted_id = cursor.fetchone()["id"]
        else:
            now_iso = datetime.datetime.utcnow().isoformat()
            cursor.execute(
                """
                INSERT INTO predictions (timestamp, filename, file_hash, verdict, confidence, raw_score, threshold, model_mode, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (now_iso, filename, file_hash, verdict, confidence, raw_score, threshold, model_mode, latency_ms),
            )
            inserted_id = cursor.lastrowid
        conn.commit()
        return inserted_id
    finally:
        conn.close()


def get_recent_predictions(limit: int = 20) -> List[Dict[str, Any]]:
    # Fetch Recent Prediction Records In Reverse Chronological Order
    conn, engine = get_db_connection()
    try:
        cursor = conn.cursor()
        if engine == "postgres":
            cursor.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT %s;", (limit,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
        else:
            cursor.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?;", (limit,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
    except Exception as e:
        print(f"Error reading prediction history: {e}")
        return []
    finally:
        conn.close()


def get_db_health() -> Dict[str, Any]:
    # Query Database Connection And Record Count
    try:
        conn, engine = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM predictions;")
        row = cursor.fetchone()
        count = row[0] if isinstance(row, tuple) else row["count"]
        conn.close()
        return {"status": "healthy", "engine": engine, "total_records": count}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


# Ensure Initialized On Import
init_db()
