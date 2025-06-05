import sqlite3
import datetime

# Default database name
_DATABASE_NAME = "strategies.db"

def set_database_name(name):
    """Sets the database name for connections. Used for testing."""
    global _DATABASE_NAME
    _DATABASE_NAME = name

def get_database_name():
    """Returns the current database name."""
    return _DATABASE_NAME

def get_db_connection():
    """Connects to the SQLite database and enables row factory."""
    conn = sqlite3.connect(_DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def create_tables():
    """Creates the strategies and strategy_versions tables if they don't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # SQL statement for strategies table
    create_strategies_table_sql = """
    CREATE TABLE IF NOT EXISTS strategies (
        strategy_id TEXT PRIMARY KEY,
        strategy_name TEXT NOT NULL,
        description TEXT,
        is_active BOOLEAN DEFAULT TRUE,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        is_deleted BOOLEAN DEFAULT FALSE
    );
    """

    # SQL statement for strategy_versions table
    create_strategy_versions_table_sql = """
    CREATE TABLE IF NOT EXISTS strategy_versions (
        version_id INTEGER PRIMARY KEY AUTOINCREMENT,
        strategy_id TEXT NOT NULL,
        strategy_name TEXT NOT NULL,
        description TEXT,
        conditions_group TEXT NOT NULL, -- Stored as JSON string
        actions TEXT NOT NULL, -- Stored as JSON string
        version_notes TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (strategy_id) REFERENCES strategies (strategy_id)
    );
    """

    cursor.execute(create_strategies_table_sql)
    cursor.execute(create_strategy_versions_table_sql)

    conn.commit()
    conn.close()

if __name__ == '__main__':
    # This ensures tables are created when the module is run directly,
    # but not necessarily when imported.
    # For ensuring table creation on first import, it's better to call
    # create_tables() directly at the module level,
    # or have an explicit setup function called by the application.
    # For this task, placing it here to run on first import as requested.
    create_tables()
    print(f"Database '{_DATABASE_NAME}' and tables created successfully (if they didn't exist).")
