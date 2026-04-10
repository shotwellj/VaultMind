"""
VaultMind Feedback Loop System
Phase 4 -- Learn from every interaction to get smarter over time.

Users give thumbs-up or thumbs-down on responses. This module:

  1. Stores feedback with full context (question, intent, model, quality score)
  2. Tracks success rates per model/intent/template combination
  3. Recommends routing adjustments based on accumulated feedback
  4. Exports fine-tuning training data for custom model training
  5. Provides analytics on what works and what doesn't

Storage: SQLite database at ~/.vaultmind/feedback/feedback.db
No cloud. No telemetry. Everything stays local.

CROSS-PRODUCT NOTE:
  Maps to AIR Blackbox continuous improvement workflows.
  The same feedback pattern can audit agent decision quality over time.
"""

import os
import json
import sqlite3
import time
from dataclasses import dataclass, asdict
from typing import Optional
from datetime import datetime, timedelta


# ── Config ────────────────────────────────────────────────────

FEEDBACK_DIR = os.path.expanduser("~/.vaultmind/feedback")
DB_PATH = os.path.join(FEEDBACK_DIR, "feedback.db")


def _ensure_db():
    """Create the feedback database and tables if they don't exist."""
    os.makedirs(FEEDBACK_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            conversation_id TEXT,
            question TEXT NOT NULL,
            response TEXT NOT NULL,
            rating INTEGER NOT NULL,
            intent TEXT,
            complexity TEXT,
            model TEXT,
            prompt_template TEXT,
            quality_score REAL,
            quality_confidence TEXT,
            mode TEXT,
            sources_used INTEGER DEFAULT 0,
            response_length INTEGER DEFAULT 0,
            user_comment TEXT DEFAULT '',
            tags TEXT DEFAULT '[]'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS route_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            intent TEXT NOT NULL,
            model TEXT NOT NULL,
            template TEXT NOT NULL,
            total_ratings INTEGER DEFAULT 0,
            positive_ratings INTEGER DEFAULT 0,
            negative_ratings INTEGER DEFAULT 0,
            avg_quality_score REAL DEFAULT 0.0,
            last_updated TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS learning_insights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            insight_type TEXT NOT NULL,
            description TEXT NOT NULL,
            data TEXT DEFAULT '{}'
        )
    """)
    # Indexes for fast lookups
    conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_intent ON feedback(intent)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_model ON feedback(model)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback(rating)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback(timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_route_perf ON route_performance(intent, model)")
    conn.commit()
    conn.close()


# ── Data Classes ──────────────────────────────────────────────

@dataclass
class FeedbackEntry:
    question: str
    response: str
    rating: int  # 1 = thumbs up, -1 = thumbs down, 0 = neutral/skip
    conversation_id: str = ""
    intent: str = ""
    complexity: str = ""
    model: str = ""
    prompt_template: str = ""
    quality_score: float = 0.0
    quality_confidence: str = ""
    mode: str = ""  # vault, web, hybrid
    sources_used: int = 0
    response_length: int = 0
    user_comment: str = ""
    tags: list = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if not self.response_length:
            self.response_length = len(self.response.split())


@dataclass
class RouteStats:
    intent: str
    model: str
    template: str
    total: int
    positive: int
    negative: int
    success_rate: float
    avg_quality: float


# ── Core Functions ────────────────────────────────────────────

def store_feedback(entry: FeedbackEntry) -> int:
    """Store a feedback entry and update route performance.

    Returns the feedback ID.
    """
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)
    now = datetime.utcnow().isoformat()

    cursor = conn.execute("""
        INSERT INTO feedback (
            timestamp, conversation_id, question, response, rating,
            intent, complexity, model, prompt_template,
            quality_score, quality_confidence, mode,
            sources_used, response_length, user_comment, tags
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        now, entry.conversation_id, entry.question, entry.response, entry.rating,
        entry.intent, entry.complexity, entry.model, entry.prompt_template,
        entry.quality_score, entry.quality_confidence, entry.mode,
        entry.sources_used, entry.response_length, entry.user_comment,
        json.dumps(entry.tags),
    ))
    feedback_id = cursor.lastrowid

    # Update route performance aggregates
    _update_route_performance(conn, entry, now)

    conn.commit()
    conn.close()

    # Check if we should generate new insights
    _maybe_generate_insights()

    return feedback_id


def _update_route_performance(conn, entry: FeedbackEntry, now: str):
    """Update the aggregate performance stats for this route."""
    if not entry.intent or not entry.model:
        return

    template = entry.prompt_template or "default"

    # Check if route exists
    row = conn.execute("""
        SELECT id, total_ratings, positive_ratings, negative_ratings, avg_quality_score
        FROM route_performance
        WHERE intent = ? AND model = ? AND template = ?
    """, (entry.intent, entry.model, template)).fetchone()

    if row:
        rid, total, pos, neg, avg_q = row
        total += 1
        if entry.rating > 0:
            pos += 1
        elif entry.rating < 0:
            neg += 1
        # Running average for quality score
        if entry.quality_score > 0:
            avg_q = (avg_q * (total - 1) + entry.quality_score) / total

        conn.execute("""
            UPDATE route_performance
            SET total_ratings = ?, positive_ratings = ?, negative_ratings = ?,
                avg_quality_score = ?, last_updated = ?
            WHERE id = ?
        """, (total, pos, neg, avg_q, now, rid))
    else:
        pos = 1 if entry.rating > 0 else 0
        neg = 1 if entry.rating < 0 else 0
        conn.execute("""
            INSERT INTO route_performance
            (intent, model, template, total_ratings, positive_ratings, negative_ratings, avg_quality_score, last_updated)
            VALUES (?, ?, ?, 1, ?, ?, ?, ?)
        """, (entry.intent, entry.model, template, pos, neg, entry.quality_score, now))


# ── Analytics ─────────────────────────────────────────────────

def get_route_stats(min_ratings: int = 3) -> list:
    """Get performance stats for all routes with enough data.

    Returns sorted by success rate (best first).
    """
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT intent, model, template, total_ratings, positive_ratings,
               negative_ratings, avg_quality_score
        FROM route_performance
        WHERE total_ratings >= ?
        ORDER BY CAST(positive_ratings AS REAL) / total_ratings DESC
    """, (min_ratings,)).fetchall()
    conn.close()

    stats = []
    for row in rows:
        intent, model, template, total, pos, neg, avg_q = row
        success_rate = pos / total if total > 0 else 0.0
        stats.append(RouteStats(
            intent=intent, model=model, template=template,
            total=total, positive=pos, negative=neg,
            success_rate=success_rate, avg_quality=avg_q,
        ))
    return stats


def get_best_route(intent: str, min_ratings: int = 3) -> Optional[dict]:
    """Find the best performing model/template for a given intent.

    Returns None if not enough data yet.
    """
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("""
        SELECT model, template, total_ratings, positive_ratings, avg_quality_score
        FROM route_performance
        WHERE intent = ? AND total_ratings >= ?
        ORDER BY CAST(positive_ratings AS REAL) / total_ratings DESC, avg_quality_score DESC
        LIMIT 1
    """, (intent, min_ratings)).fetchone()
    conn.close()

    if row:
        model, template, total, pos, avg_q = row
        return {
            "model": model,
            "template": template,
            "success_rate": pos / total if total > 0 else 0.0,
            "avg_quality": avg_q,
            "sample_size": total,
        }
    return None


def get_feedback_summary(days: int = 30) -> dict:
    """Get a summary of feedback over the last N days."""
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

    total = conn.execute(
        "SELECT COUNT(*) FROM feedback WHERE timestamp > ?", (cutoff,)
    ).fetchone()[0]

    positive = conn.execute(
        "SELECT COUNT(*) FROM feedback WHERE timestamp > ? AND rating > 0", (cutoff,)
    ).fetchone()[0]

    negative = conn.execute(
        "SELECT COUNT(*) FROM feedback WHERE timestamp > ? AND rating < 0", (cutoff,)
    ).fetchone()[0]

    # Most common issues (from negative feedback)
    neg_intents = conn.execute("""
        SELECT intent, COUNT(*) as cnt
        FROM feedback
        WHERE timestamp > ? AND rating < 0
        GROUP BY intent
        ORDER BY cnt DESC
        LIMIT 5
    """, (cutoff,)).fetchall()

    # Best performing models
    best_models = conn.execute("""
        SELECT model, COUNT(*) as total,
               SUM(CASE WHEN rating > 0 THEN 1 ELSE 0 END) as pos
        FROM feedback
        WHERE timestamp > ? AND model != ''
        GROUP BY model
        HAVING total >= 3
        ORDER BY CAST(pos AS REAL) / total DESC
    """, (cutoff,)).fetchall()

    conn.close()

    return {
        "period_days": days,
        "total_feedback": total,
        "positive": positive,
        "negative": negative,
        "satisfaction_rate": positive / total if total > 0 else 0.0,
        "problem_intents": [{"intent": r[0], "count": r[1]} for r in neg_intents],
        "model_rankings": [
            {"model": r[0], "total": r[1], "positive": r[2],
             "rate": r[2] / r[1] if r[1] > 0 else 0.0}
            for r in best_models
        ],
    }


# ── Route Recommendations ────────────────────────────────────

def get_routing_overrides() -> dict:
    """Generate routing override recommendations based on feedback data.

    Returns a dict of intent -> recommended model/template overrides.
    These can be fed back into the Query Intelligence Engine.
    """
    stats = get_route_stats(min_ratings=5)  # Need at least 5 ratings to recommend

    overrides = {}
    # Group by intent
    by_intent = {}
    for s in stats:
        if s.intent not in by_intent:
            by_intent[s.intent] = []
        by_intent[s.intent].append(s)

    for intent, routes in by_intent.items():
        if len(routes) < 2:
            continue  # Need at least 2 routes to compare

        best = max(routes, key=lambda r: r.success_rate)
        worst = min(routes, key=lambda r: r.success_rate)

        # Only override if there's a meaningful difference
        if best.success_rate - worst.success_rate >= 0.15:
            overrides[intent] = {
                "recommended_model": best.model,
                "recommended_template": best.template,
                "success_rate": best.success_rate,
                "avoid_model": worst.model if worst.success_rate < 0.5 else None,
                "reason": f"Based on {best.total} ratings, {best.model} has {best.success_rate:.0%} success rate for {intent} queries",
            }

    return overrides


# ── Fine-Tuning Data Export ───────────────────────────────────

def export_training_data(min_rating: int = 1, limit: int = 1000) -> list:
    """Export positive feedback as training data for model fine-tuning.

    Returns a list of dicts in the format expected by Unsloth/LoRA:
    [{"instruction": "...", "input": "...", "output": "..."}, ...]
    """
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)

    rows = conn.execute("""
        SELECT question, response, intent, model
        FROM feedback
        WHERE rating >= ?
        ORDER BY quality_score DESC, timestamp DESC
        LIMIT ?
    """, (min_rating, limit)).fetchall()
    conn.close()

    training_data = []
    for question, response, intent, model in rows:
        training_data.append({
            "instruction": f"Answer the following {intent or 'question'} using only the provided context.",
            "input": question,
            "output": response,
        })

    return training_data


def export_training_jsonl(filepath: str = None, min_rating: int = 1) -> str:
    """Export training data as JSONL file for fine-tuning.

    Returns the file path.
    """
    if filepath is None:
        filepath = os.path.join(FEEDBACK_DIR, "training_data.jsonl")

    data = export_training_data(min_rating=min_rating)
    with open(filepath, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

    return filepath


# ── Insight Generation ────────────────────────────────────────

def _maybe_generate_insights():
    """Check if we have enough new data to generate insights.

    Runs automatically after every 10 feedback entries.
    """
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)

    total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    last_insight_count = conn.execute(
        "SELECT COUNT(*) FROM learning_insights"
    ).fetchone()[0]

    # Generate insights every 10 feedback entries
    if total > 0 and total % 10 == 0 and total // 10 > last_insight_count:
        _generate_insights(conn)

    conn.close()


def _generate_insights(conn):
    """Generate learning insights from accumulated feedback."""
    now = datetime.utcnow().isoformat()

    # Insight 1: Model performance shift
    model_stats = conn.execute("""
        SELECT model,
               SUM(CASE WHEN rating > 0 THEN 1 ELSE 0 END) as pos,
               COUNT(*) as total
        FROM feedback
        WHERE model != ''
        GROUP BY model
        HAVING total >= 5
    """).fetchall()

    if model_stats:
        best_model = max(model_stats, key=lambda r: r[1] / r[2] if r[2] > 0 else 0)
        conn.execute("""
            INSERT INTO learning_insights (timestamp, insight_type, description, data)
            VALUES (?, 'model_performance', ?, ?)
        """, (
            now,
            f"Best performing model: {best_model[0]} ({best_model[1]}/{best_model[2]} positive)",
            json.dumps({"model": best_model[0], "positive": best_model[1], "total": best_model[2]}),
        ))

    # Insight 2: Struggling intents
    struggling = conn.execute("""
        SELECT intent,
               SUM(CASE WHEN rating < 0 THEN 1 ELSE 0 END) as neg,
               COUNT(*) as total
        FROM feedback
        WHERE intent != ''
        GROUP BY intent
        HAVING total >= 5 AND CAST(neg AS REAL) / total > 0.4
    """).fetchall()

    for intent, neg, total in struggling:
        conn.execute("""
            INSERT INTO learning_insights (timestamp, insight_type, description, data)
            VALUES (?, 'struggling_intent', ?, ?)
        """, (
            now,
            f"Intent '{intent}' has high failure rate: {neg}/{total} negative",
            json.dumps({"intent": intent, "negative": neg, "total": total}),
        ))

    conn.commit()


def get_insights(limit: int = 20) -> list:
    """Get recent learning insights."""
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT timestamp, insight_type, description, data
        FROM learning_insights
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()

    return [
        {
            "timestamp": r[0],
            "type": r[1],
            "description": r[2],
            "data": json.loads(r[3]),
        }
        for r in rows
    ]
