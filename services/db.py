import sqlite3
import os
import pandas as pd
from datetime import datetime

DB_PATH = "stats.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS query_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            query TEXT,
            models_used TEXT,
            input_tokens INTEGER,
            output_tokens INTEGER,
            estimated_cost REAL
        )
    ''')
    
    # Safely add new columns if they don't exist
    try:
        c.execute("ALTER TABLE query_stats ADD COLUMN model_breakdown TEXT")
    except sqlite3.OperationalError:
        pass # Column already exists
    try:
        c.execute("ALTER TABLE query_stats ADD COLUMN stage_breakdown TEXT")
    except sqlite3.OperationalError:
        pass # Column already exists
        
    conn.commit()
    conn.close()

def log_stats(query: str, models_used: list, input_tokens: int, output_tokens: int, estimated_cost: float, model_breakdown: str = "{}", stage_breakdown: str = "{}"):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    models_str = ", ".join(models_used) if models_used else "unknown"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    c.execute('''
        INSERT INTO query_stats (timestamp, query, models_used, input_tokens, output_tokens, estimated_cost, model_breakdown, stage_breakdown)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (timestamp, query, models_str, input_tokens, output_tokens, estimated_cost, model_breakdown, stage_breakdown))
    
    conn.commit()
    conn.close()

def get_stats_history() -> pd.DataFrame:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM query_stats ORDER BY timestamp DESC", conn)
    conn.close()
    return df

def get_aggregate_stats() -> dict:
    df = get_stats_history()
    if df.empty:
        return {
            "total_queries": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost": 0.0
        }
        
    return {
        "total_queries": len(df),
        "total_input_tokens": int(df["input_tokens"].sum()),
        "total_output_tokens": int(df["output_tokens"].sum()),
        "total_cost": float(df["estimated_cost"].sum())
    }
