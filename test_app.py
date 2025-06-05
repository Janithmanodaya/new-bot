import unittest
import json
import os
import sys
import copy
from datetime import datetime, timezone, timedelta
import uuid
import sqlite3

# Add the parent directory to sys.path to allow imports from run.py and database.py
# This assumes test_app.py is in a subdirectory or a specific test folder.
# If test_app.py is in the same directory as run.py, this might not be strictly necessary
# but is good practice for discoverability if tests are run from a different working directory.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Now import from your application
from run import app, custom_strategies, strategies_lock, load_strategies_from_db
import database # Import the whole module to access its functions dynamically

# Store the original database name from database.py
ORIGINAL_DATABASE_NAME = database.get_database_name()
TEST_DATABASE_NAME = "test_strategies.db"

class TestStrategyAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # This is run once before all tests in the class
        pass

    @classmethod
    def tearDownClass(cls):
        # This is run once after all tests in the class
        pass

    def setUp(self):
        """Set up for each test method."""
        # Configure database for testing
        database.set_database_name(TEST_DATABASE_NAME)

        # Ensure a clean state for the test database file
        if os.path.exists(TEST_DATABASE_NAME):
            os.remove(TEST_DATABASE_NAME)

        # Configure Flask app for testing
        app.config['TESTING'] = True
        self.client = app.test_client()

        # Create tables in the test database
        database.create_tables()

        # Clear in-memory cache before each test
        with strategies_lock:
            custom_strategies.clear()
        # Load strategies from the (empty) test DB to ensure cache is in a known state related to DB
        load_strategies_from_db()


    def tearDown(self):
        """Tear down after each test method."""
        # Clean up the test database file
        if os.path.exists(TEST_DATABASE_NAME):
            os.remove(TEST_DATABASE_NAME)

        # Restore the original database name
        database.set_database_name(ORIGINAL_DATABASE_NAME)

        # Clear in-memory cache
        with strategies_lock:
            custom_strategies.clear()

    # --- Helper Functions ---
    def _create_sample_strategy_direct_db(self, strategy_id, name, description, conditions, actions,
                                          is_active=True, is_deleted=False,
                                          created_at=None, updated_at=None, version_notes="Initial version."):
        conn = database.get_db_connection() # Should connect to TEST_DATABASE_NAME
        cursor = conn.cursor()

        now_iso = datetime.now(timezone.utc).isoformat()
        created_at_iso = created_at or now_iso
        updated_at_iso = updated_at or created_at_iso

        try:
            cursor.execute("""
                INSERT INTO strategies (strategy_id, strategy_name, description, is_active, created_at, updated_at, is_deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (strategy_id, name, description, is_active, created_at_iso, updated_at_iso, is_deleted))

            cursor.execute("""
                INSERT INTO strategy_versions (strategy_id, strategy_name, description, conditions_group, actions, version_notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (strategy_id, name, description, json.dumps(conditions), json.dumps(actions), version_notes, created_at_iso))
            conn.commit()
        except Exception as e:
            conn.rollback()
            self.fail(f"Helper _create_sample_strategy_direct_db failed: {e}")
        finally:
            conn.close()

        # Return the created data for convenience in tests
        return {
            "strategy_id": strategy_id, "strategy_name": name, "description": description,
            "conditions_group": conditions, "actions": actions, "is_active": is_active,
            "created_at": created_at_iso, "updated_at": updated_at_iso, "version_notes": version_notes
        }

    # --- Test Cases ---

    def test_01_create_strategy(self):
        """Test creating a new strategy via POST /api/strategies."""
        strategy_payload = {
            "strategy_name": "Test Strategy 1",
            "description": "A test strategy for creation.",
            "conditions_group": {"type": "ALL", "conditions": [{"field": "price", "operator": ">", "value": 100}]},
            "actions": [{"type": "BUY", "amount": 10}]
        }
        response = self.client.post('/api/strategies', data=json.dumps(strategy_payload), content_type='application/json')
        self.assertEqual(response.status_code, 201, f"Response: {response.data.decode()}")

        created_strategy_data = json.loads(response.data.decode())

        self.assertIn("strategy_id", created_strategy_data)
        self.assertIsNotNone(created_strategy_data["strategy_id"])
        strategy_id = created_strategy_data["strategy_id"]

        self.assertEqual(created_strategy_data["strategy_name"], strategy_payload["strategy_name"])
        self.assertEqual(created_strategy_data["description"], strategy_payload["description"])
        self.assertEqual(created_strategy_data["conditions_group"], strategy_payload["conditions_group"])
        self.assertEqual(created_strategy_data["actions"], strategy_payload["actions"])
        self.assertTrue(created_strategy_data["is_active"])
        self.assertIn("created_at", created_strategy_data)
        self.assertIn("updated_at", created_strategy_data)

        # Verify in-memory cache
        with strategies_lock:
            self.assertIn(strategy_id, custom_strategies)
            self.assertEqual(custom_strategies[strategy_id]["strategy_name"], strategy_payload["strategy_name"])

        # Verify in database
        conn = database.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM strategies WHERE strategy_id = ?", (strategy_id,))
        db_strategy = cursor.fetchone()
        self.assertIsNotNone(db_strategy)
        self.assertEqual(db_strategy["strategy_name"], strategy_payload["strategy_name"])
        self.assertTrue(db_strategy["is_active"])
        self.assertFalse(db_strategy["is_deleted"])

        cursor.execute("SELECT * FROM strategy_versions WHERE strategy_id = ? ORDER BY version_id DESC LIMIT 1", (strategy_id,))
        db_version = cursor.fetchone()
        self.assertIsNotNone(db_version)
        self.assertEqual(json.loads(db_version["conditions_group"]), strategy_payload["conditions_group"])
        conn.close()

    def test_02_get_all_strategies(self):
        """Test GET /api/strategies to retrieve all strategies."""
        # Create some strategies directly in DB for this test
        s1_id = uuid.uuid4().hex
        self._create_sample_strategy_direct_db(s1_id, "Strategy Alpha", "Desc Alpha", {"cond": "A"}, [{"act": "X"}])
        s2_id = uuid.uuid4().hex
        self._create_sample_strategy_direct_db(s2_id, "Strategy Beta", "Desc Beta", {"cond": "B"}, [{"act": "Y"}])

        # Manually call load_strategies_from_db to populate the cache, as it normally runs on app startup
        load_strategies_from_db()

        response = self.client.get('/api/strategies')
        self.assertEqual(response.status_code, 200)
        strategies_list = json.loads(response.data.decode())

        self.assertEqual(len(strategies_list), 2)

        # Check if strategy IDs are present (order might vary)
        returned_ids = {s["strategy_id"] for s in strategies_list}
        self.assertIn(s1_id, returned_ids)
        self.assertIn(s2_id, returned_ids)

    def test_03_get_strategy_by_id(self):
        """Test GET /api/strategies/<strategy_id>."""
        s_id = uuid.uuid4().hex
        sample_strategy = self._create_sample_strategy_direct_db(s_id, "Strategy Gamma", "Desc Gamma", {"cond": "C"}, [{"act": "Z"}])
        load_strategies_from_db() # Load into cache

        # Test get existing strategy
        response = self.client.get(f'/api/strategies/{s_id}')
        self.assertEqual(response.status_code, 200)
        strategy_data = json.loads(response.data.decode())
        self.assertEqual(strategy_data["strategy_id"], s_id)
        self.assertEqual(strategy_data["strategy_name"], sample_strategy["strategy_name"])

        # Test get non-existent strategy
        non_existent_id = uuid.uuid4().hex
        response = self.client.get(f'/api/strategies/{non_existent_id}')
        self.assertEqual(response.status_code, 404)

    def test_04_update_strategy(self):
        """Test PUT /api/strategies/<strategy_id> to update a strategy."""
        s_id = uuid.uuid4().hex
        self._create_sample_strategy_direct_db(s_id, "Original Name", "Original Desc", {"orig_cond": True}, [{"orig_act": True}])
        load_strategies_from_db() # Load into cache

        update_payload = {
            "strategy_name": "Updated Name",
            "description": "Updated Description",
            "conditions_group": {"updated_cond": True},
            "actions": [{"updated_act": True}],
            "is_active": False,
            "version_notes": "Major update for test."
        }
        response = self.client.put(f'/api/strategies/{s_id}', data=json.dumps(update_payload), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        updated_strategy_data = json.loads(response.data.decode())

        self.assertEqual(updated_strategy_data["strategy_name"], update_payload["strategy_name"])
        self.assertEqual(updated_strategy_data["description"], update_payload["description"])
        self.assertEqual(updated_strategy_data["conditions_group"], update_payload["conditions_group"])
        self.assertEqual(updated_strategy_data["actions"], update_payload["actions"])
        self.assertEqual(updated_strategy_data["is_active"], update_payload["is_active"])

        # Verify in DB
        conn = database.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT strategy_name, description, is_active, updated_at FROM strategies WHERE strategy_id = ?", (s_id,))
        db_strategy = cursor.fetchone()
        self.assertEqual(db_strategy["strategy_name"], update_payload["strategy_name"])
        self.assertEqual(db_strategy["description"], update_payload["description"])
        self.assertEqual(bool(db_strategy["is_active"]), update_payload["is_active"])

        # Check updated_at timestamp (it should be newer than original)
        # This requires storing original updated_at or checking it's different.
        # For simplicity, just ensure it's a valid timestamp string.
        self.assertTrue(isinstance(db_strategy["updated_at"], str) and len(db_strategy["updated_at"]) > 0)

        # Verify new version was created
        cursor.execute("SELECT version_id, conditions_group, actions, version_notes FROM strategy_versions WHERE strategy_id = ? ORDER BY version_id DESC", (s_id,))
        versions = cursor.fetchall()
        self.assertEqual(len(versions), 2) # Original + updated version
        latest_version = versions[0]
        self.assertEqual(json.loads(latest_version["conditions_group"]), update_payload["conditions_group"])
        self.assertEqual(json.loads(latest_version["actions"]), update_payload["actions"])
        self.assertEqual(latest_version["version_notes"], update_payload["version_notes"])
        conn.close()

    def test_05_delete_strategy(self):
        """Test DELETE /api/strategies/<strategy_id> for soft delete."""
        s_id = uuid.uuid4().hex
        self._create_sample_strategy_direct_db(s_id, "To Be Deleted", "Delete test", {}, [])
        load_strategies_from_db()

        response = self.client.delete(f'/api/strategies/{s_id}')
        self.assertEqual(response.status_code, 200)

        # Verify in DB (soft deleted)
        conn = database.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT is_deleted, is_active FROM strategies WHERE strategy_id = ?", (s_id,))
        db_strategy = cursor.fetchone()
        self.assertIsNotNone(db_strategy)
        self.assertTrue(db_strategy["is_deleted"])
        self.assertFalse(db_strategy["is_active"]) # Should also be deactivated
        conn.close()

        # Verify removed from cache
        with strategies_lock:
            self.assertNotIn(s_id, custom_strategies)

        # Try to GET it - should be 404
        response_get = self.client.get(f'/api/strategies/{s_id}')
        self.assertEqual(response_get.status_code, 404)

        # Try to GET its history - should also be 404 as base strategy is considered deleted
        response_history = self.client.get(f'/api/strategies/{s_id}/history')
        self.assertEqual(response_history.status_code, 404)


    def test_06_enable_disable_strategy(self):
        """Test POST enabling and disabling strategies."""
        s_id = uuid.uuid4().hex
        self._create_sample_strategy_direct_db(s_id, "Activatable", "Test enable/disable", {}, [], is_active=True)
        load_strategies_from_db()

        # Disable
        response_disable = self.client.post(f'/api/strategies/{s_id}/disable')
        self.assertEqual(response_disable.status_code, 200)
        data_disable = json.loads(response_disable.data.decode())
        self.assertFalse(data_disable["is_active"])
        with strategies_lock: # Check cache
            self.assertFalse(custom_strategies[s_id]["is_active"])
        conn = database.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT is_active FROM strategies WHERE strategy_id = ?", (s_id,))
        self.assertFalse(bool(cursor.fetchone()["is_active"])) # Check DB

        # Enable
        response_enable = self.client.post(f'/api/strategies/{s_id}/enable')
        self.assertEqual(response_enable.status_code, 200)
        data_enable = json.loads(response_enable.data.decode())
        self.assertTrue(data_enable["is_active"])
        with strategies_lock: # Check cache
            self.assertTrue(custom_strategies[s_id]["is_active"])
        cursor.execute("SELECT is_active FROM strategies WHERE strategy_id = ?", (s_id,))
        self.assertTrue(bool(cursor.fetchone()["is_active"])) # Check DB
        conn.close()

    def test_07_get_strategy_history(self):
        """Test GET /api/strategies/<strategy_id>/history."""
        s_id = uuid.uuid4().hex
        # Version 1 (creation)
        self._create_sample_strategy_direct_db(s_id, "History Test", "V1", {"cond": "v1"}, [{"act": "v1"}], version_notes="Version 1")
        load_strategies_from_db()

        # Version 2 (update)
        update_1_payload = {"strategy_name": "History Test Updated", "description": "V2", "conditions_group": {"cond": "v2"}, "actions": [{"act": "v2"}], "version_notes": "Version 2"}
        self.client.put(f'/api/strategies/{s_id}', data=json.dumps(update_1_payload), content_type='application/json')

        # Version 3 (another update)
        update_2_payload = {"strategy_name": "History Test Final", "description": "V3", "conditions_group": {"cond": "v3"}, "actions": [{"act": "v3"}], "version_notes": "Version 3"}
        self.client.put(f'/api/strategies/{s_id}', data=json.dumps(update_2_payload), content_type='application/json')

        response = self.client.get(f'/api/strategies/{s_id}/history')
        self.assertEqual(response.status_code, 200)
        history_data = json.loads(response.data.decode())

        self.assertEqual(len(history_data), 3) # Should have 3 versions

        # Versions are ordered version_id DESC (latest first)
        self.assertEqual(history_data[0]["version_notes"], "Version 3")
        self.assertEqual(history_data[0]["strategy_name"], "History Test Final")
        self.assertEqual(history_data[0]["conditions_group"], {"cond": "v3"})

        self.assertEqual(history_data[1]["version_notes"], "Version 2")
        self.assertEqual(history_data[1]["strategy_name"], "History Test Updated")

        self.assertEqual(history_data[2]["version_notes"], "Version 1")
        self.assertEqual(history_data[2]["strategy_name"], "History Test") # Original name for that version

        # Test history for non-existent strategy
        response_404 = self.client.get(f'/api/strategies/{uuid.uuid4().hex}/history')
        self.assertEqual(response_404.status_code, 404)


    def test_08_rollback_strategy(self):
        """Test POST /api/strategies/<strategy_id>/rollback/<version_id>."""
        s_id = uuid.uuid4().hex
        # Version 1 (creation)
        self._create_sample_strategy_direct_db(s_id, "Rollback Test", "V1 Desc", {"cond": "v1"}, [{"act": "v1"}], version_notes="V1 Notes")
        load_strategies_from_db()

        conn_temp = database.get_db_connection()
        cursor_temp = conn_temp.cursor()
        cursor_temp.execute("SELECT version_id FROM strategy_versions WHERE strategy_id = ? AND version_notes = ?", (s_id, "V1 Notes"))
        v1_id = cursor_temp.fetchone()['version_id']
        conn_temp.close()

        # Version 2 (update)
        update_payload = {"strategy_name": "Rollback Test V2", "description": "V2 Desc", "conditions_group": {"cond": "v2"}, "actions": [{"act": "v2"}], "version_notes": "V2 Notes"}
        self.client.put(f'/api/strategies/{s_id}', data=json.dumps(update_payload), content_type='application/json')

        # Rollback to Version 1
        response_rollback = self.client.post(f'/api/strategies/{s_id}/rollback/{v1_id}')
        self.assertEqual(response_rollback.status_code, 200)
        rollback_data = json.loads(response_rollback.data.decode())

        self.assertIn("strategy_details", rollback_data)
        rolled_back_strategy = rollback_data["strategy_details"]

        self.assertEqual(rolled_back_strategy["strategy_name"], "Rollback Test") # Name from V1
        self.assertEqual(rolled_back_strategy["description"], "V1 Desc")       # Desc from V1
        self.assertEqual(rolled_back_strategy["conditions_group"], {"cond": "v1"}) # Cond from V1
        self.assertEqual(rolled_back_strategy["actions"], [{"act": "v1"}])     # Actions from V1

        # Verify DB state
        conn = database.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT strategy_name, description FROM strategies WHERE strategy_id = ?", (s_id,))
        db_strategy = cursor.fetchone()
        self.assertEqual(db_strategy["strategy_name"], "Rollback Test")
        self.assertEqual(db_strategy["description"], "V1 Desc")

        # Verify a new version (V3) was created due to rollback
        cursor.execute("SELECT version_id, strategy_name, description, conditions_group, actions, version_notes FROM strategy_versions WHERE strategy_id = ? ORDER BY version_id DESC", (s_id,))
        versions = cursor.fetchall()
        self.assertEqual(len(versions), 3) # V1, V2, V3 (rollback version)

        latest_version = versions[0] # This is V3
        self.assertEqual(latest_version["strategy_name"], "Rollback Test")
        self.assertEqual(json.loads(latest_version["conditions_group"]), {"cond": "v1"})
        self.assertTrue(f"Rolled back to version {v1_id}" in latest_version["version_notes"])
        conn.close()

        # Test rollback to a non-existent version
        response_bad_version = self.client.post(f'/api/strategies/{s_id}/rollback/99999')
        self.assertEqual(response_bad_version.status_code, 404)

        # Test rollback for a non-existent strategy_id
        response_bad_strategy = self.client.post(f'/api/strategies/{uuid.uuid4().hex}/rollback/{v1_id}')
        self.assertEqual(response_bad_strategy.status_code, 404)

if __name__ == '__main__':
    # Ensure the script is runnable from the command line
    # This setup allows running tests like: python -m unittest test_app.py
    # Or, if test_app.py is in a 'tests' subdirectory: python -m unittest tests.test_app
    unittest.main()
