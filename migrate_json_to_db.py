import json
import sqlite3
from datetime import datetime, timezone
import uuid
import os # For checking file existence

# Assuming database.py is in the same directory or accessible in PYTHONPATH
from database import get_db_connection, create_tables

STRATEGIES_JSON_FILE = "custom_strategies.json"
DATABASE_NAME = "strategies.db" # Ensure this matches the one in database.py

def parse_timestamp(timestamp_str, default_time):
    """Helper to parse ISO format timestamp, fallback to default_time if invalid or missing."""
    if not timestamp_str:
        return default_time
    try:
        # Attempt to parse with timezone info if present
        if 'Z' in timestamp_str:
            dt_obj = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
        elif '+' in timestamp_str: # Handles +HH:MM format
            # Python's %z doesn't handle ':' in offset well before 3.7, so manual parsing or dateutil might be better.
            # For simplicity, assuming consistent ISO format or UTC 'Z'.
            # This part might need adjustment based on actual timestamp format in JSON.
            # A common format from JS `toISOString()` is like "2023-10-03T10:30:00.123Z"
            # Another one is "2024-03-11T12:35:17.817629+00:00"
             dt_obj = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

        else: # Assume naive datetime string, treat as UTC
            dt_obj = datetime.fromisoformat(timestamp_str).replace(tzinfo=timezone.utc)

        return dt_obj.isoformat()
    except ValueError as e:
        print(f"Warning: Could not parse timestamp '{timestamp_str}': {e}. Using default.")
        return default_time
    except Exception as e: # Catch any other parsing errors
        print(f"Warning: Unexpected error parsing timestamp '{timestamp_str}': {e}. Using default.")
        return default_time


def migrate_data():
    """Main function to migrate strategies from JSON file to SQLite database."""
    print("Starting migration process...")

    # 1. Ensure database and tables exist
    print(f"Ensuring database '{DATABASE_NAME}' and tables are created...")
    create_tables()
    print("Database tables checked/created.")

    # 2. Load strategies from JSON
    if not os.path.exists(STRATEGIES_JSON_FILE):
        print(f"Error: JSON file '{STRATEGIES_JSON_FILE}' not found. Migration cannot proceed.")
        return

    print(f"Loading strategies from '{STRATEGIES_JSON_FILE}'...")
    try:
        with open(STRATEGIES_JSON_FILE, 'r') as f:
            strategies_from_json = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{STRATEGIES_JSON_FILE}'. File might be corrupted or empty.")
        return
    except Exception as e:
        print(f"Error opening or reading JSON file '{STRATEGIES_JSON_FILE}': {e}")
        return

    if not strategies_from_json:
        print("JSON file is empty. No data to migrate.")
        return

    # The JSON stores strategies as a dictionary {strategy_id: strategy_object}
    # We need to iterate through its values if that's the case, or items if it's a list.
    # Based on previous run.py, it's a dictionary.
    if not isinstance(strategies_from_json, dict):
        print("Error: Expected JSON root to be a dictionary of strategies. Migration aborted.")
        return

    strategies_to_migrate = list(strategies_from_json.values()) # Get list of strategy objects
    print(f"Loaded {len(strategies_to_migrate)} strategies from JSON file.")

    # 3. Iterate and migrate
    conn = None
    migrated_count = 0
    skipped_count = 0
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        print("Database connection established.")

        for strategy_json_obj in strategies_to_migrate:
            if not isinstance(strategy_json_obj, dict):
                print(f"Warning: Skipping an item that is not a strategy object (dict): {strategy_json_obj}")
                skipped_count +=1
                continue

            strategy_id = strategy_json_obj.get('strategy_id')
            if not strategy_id:
                strategy_id = uuid.uuid4().hex
                print(f"Warning: Strategy missing 'strategy_id'. Generated new ID: {strategy_id} for '{strategy_json_obj.get('strategy_name', 'Unknown Name')}'")

            strategy_name = strategy_json_obj.get('strategy_name')
            if not strategy_name:
                print(f"Warning: Strategy ID {strategy_id} missing 'strategy_name'. Skipping this strategy.")
                skipped_count += 1
                continue

            # Check if strategy already exists in DB
            cursor.execute("SELECT strategy_id FROM strategies WHERE strategy_id = ?", (strategy_id,))
            if cursor.fetchone():
                print(f"Strategy '{strategy_name}' (ID: {strategy_id}) already exists in the database. Skipping.")
                skipped_count += 1
                continue

            # Proceed with migration for this strategy
            print(f"Migrating strategy '{strategy_name}' (ID: {strategy_id})...")

            current_time_iso = datetime.now(timezone.utc).isoformat()

            # Timestamps: use from JSON if available and valid, else current_time_iso
            # The old format might not have timezone, treat as UTC if naive.
            created_at_str = strategy_json_obj.get('created_at')
            updated_at_str = strategy_json_obj.get('updated_at')

            created_at_iso = parse_timestamp(created_at_str, current_time_iso)
            updated_at_iso = parse_timestamp(updated_at_str, created_at_iso) # If updated_at is missing/invalid, use created_at

            description = strategy_json_obj.get('description', '')
            is_active = strategy_json_obj.get('is_active', True) # Default to True if missing

            conditions_group = strategy_json_obj.get('conditions_group', {}) # Should exist
            actions = strategy_json_obj.get('actions', [])

            # Insert into strategies table
            cursor.execute("""
                INSERT INTO strategies (strategy_id, strategy_name, description, is_active, created_at, updated_at, is_deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (strategy_id, strategy_name, description, bool(is_active), created_at_iso, updated_at_iso, False))

            # Insert into strategy_versions table
            # For migration, the first version's created_at should match the strategy's created_at
            version_created_at_iso = created_at_iso
            version_notes = "Migrated from JSON file on " + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

            cursor.execute("""
                INSERT INTO strategy_versions (strategy_id, strategy_name, description, conditions_group, actions, version_notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                strategy_id,
                strategy_name, # Denormalized name for this version
                description,   # Denormalized description for this version
                json.dumps(conditions_group),
                json.dumps(actions),
                version_notes,
                version_created_at_iso
            ))
            migrated_count += 1
            print(f"Successfully migrated strategy '{strategy_name}' (ID: {strategy_id}).")

        conn.commit()
        print("Database changes committed.")

    except sqlite3.Error as e:
        if conn:
            conn.rollback()
        print(f"Database error during migration: {e}")
        print("Migration might be incomplete. Please check logs and database.")
        return
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"An unexpected error occurred during migration: {e}")
        print("Migration might be incomplete. Please check logs and database.")
        return
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

    print("\n--- Migration Summary ---")
    print(f"Successfully migrated: {migrated_count} strategies.")
    print(f"Skipped (e.g., already exist or missing data): {skipped_count} strategies.")
    print("Migration process complete.")

if __name__ == "__main__":
    migrate_data()
