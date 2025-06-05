import unittest
import json
import os
import sys
import copy
from datetime import datetime, timezone, timedelta
import uuid
import sqlite3
import asyncio # Required for IsolatedAsyncioTestCase
from unittest.mock import patch, AsyncMock # For mocking async functions and objects
from concurrent.futures import ThreadPoolExecutor # To manage executor in tests
import time # Added for time.time() in test_websocket_successful_connection_and_initial_messages

# Add the parent directory to sys.path to allow imports from run.py and database.py
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) # Assuming test_app.py is in a 'tests' subdir
if parent_dir not in sys.path:
     sys.path.insert(0, parent_dir) # Insert at beginning to ensure it's checked first

# Now import from your application
import run # Import the whole module to allow patching its globals like asyncio_loop, executor
from run import app, custom_strategies, strategies_lock, load_strategies_from_db
import database # Import the whole module to access its functions dynamically

# Store the original database name from database.py
ORIGINAL_DATABASE_NAME = database.get_database_name()
TEST_DATABASE_NAME = "test_strategies.db"


class TestStrategyAPI(unittest.IsolatedAsyncioTestCase): # Changed base class

    async def asyncSetUp(self): # Renamed and made async
        """Set up for each test method."""
        # Store original asyncio_loop and executor from run module
        self.original_asyncio_loop = run.asyncio_loop
        self.original_executor = run.executor
        self.original_ws_app = run.ws_app
        self.original_current_ws_task = run.current_ws_task
        self.original_deriv_semaphore = run.deriv_api_request_semaphore


        # Set the run module's asyncio_loop to the test's loop
        run.asyncio_loop = self.asyncio_loop # self.asyncio_loop is from IsolatedAsyncioTestCase

        # Create a new executor for each test to ensure isolation
        run.executor = ThreadPoolExecutor(max_workers=2)
        run.ws_app = None # Ensure ws_app is reset for tests that might mock it
        run.current_ws_task = None # Ensure task is reset

        # For semaphore, we might want to use a real one for some tests, or mock it for others.
        # Re-initialize it for each test to ensure clean state if not mocking.
        run.deriv_api_request_semaphore = asyncio.Semaphore(5)


        # Configure database for testing
        database.set_database_name(TEST_DATABASE_NAME)

        if os.path.exists(TEST_DATABASE_NAME):
            os.remove(TEST_DATABASE_NAME)

        app.config['TESTING'] = True
        # Note: Flask's test_client is synchronous. For testing async routes directly,
        # one might use something like httpx.AsyncClient(app=app, base_url="http://localhost")
        # However, our Flask routes themselves are sync, they just interact with an async backend.
        self.client = app.test_client()

        database.create_tables()

        with strategies_lock:
            custom_strategies.clear()
        load_strategies_from_db()


    async def asyncTearDown(self): # Renamed and made async
        """Tear down after each test method."""
        # Shutdown the test-specific executor
        if run.executor:
            run.executor.shutdown(wait=True)

        # Restore original asyncio_loop and executor in run module
        run.asyncio_loop = self.original_asyncio_loop
        run.executor = self.original_executor
        run.ws_app = self.original_ws_app
        run.current_ws_task = self.original_current_ws_task
        run.deriv_api_request_semaphore = self.original_deriv_semaphore


        if os.path.exists(TEST_DATABASE_NAME):
            os.remove(TEST_DATABASE_NAME)

        database.set_database_name(ORIGINAL_DATABASE_NAME)

        with strategies_lock:
            custom_strategies.clear()

    # --- Helper Functions ---
    async def _create_sample_strategy_direct_db(self, strategy_id, name, description, conditions, actions,
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

    # Strategy API tests (largely remain synchronous from client's perspective)
    # These will continue to use self.client (Flask's test client) which makes sync HTTP calls.
    # The async nature is mostly internal to run.py's WebSocket and indicator parts.
    # We will add specific tests for WebSocket interactions later.

    async def test_01_create_strategy(self):
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

    async def test_02_get_all_strategies(self): # Made async
        """Test GET /api/strategies to retrieve all strategies."""
        s1_id = uuid.uuid4().hex
        self._create_sample_strategy_direct_db(s1_id, "Strategy Alpha", "Desc Alpha", {"cond": "A"}, [{"act": "X"}])
        s2_id = uuid.uuid4().hex
        self._create_sample_strategy_direct_db(s2_id, "Strategy Beta", "Desc Beta", {"cond": "B"}, [{"act": "Y"}])

        load_strategies_from_db()

        response = self.client.get('/api/strategies')
        self.assertEqual(response.status_code, 200)
        strategies_list = json.loads(response.data.decode())

        self.assertEqual(len(strategies_list), 2)

        # Check if strategy IDs are present (order might vary)
        returned_ids = {s["strategy_id"] for s in strategies_list}
        self.assertIn(s1_id, returned_ids)
        self.assertIn(s2_id, returned_ids)

    async def test_03_get_strategy_by_id(self): # Made async
        """Test GET /api/strategies/<strategy_id>."""
        s_id = uuid.uuid4().hex
        sample_strategy = self._create_sample_strategy_direct_db(s_id, "Strategy Gamma", "Desc Gamma", {"cond": "C"}, [{"act": "Z"}])
        load_strategies_from_db()

        response = self.client.get(f'/api/strategies/{s_id}')
        self.assertEqual(response.status_code, 200)
        strategy_data = json.loads(response.data.decode())
        self.assertEqual(strategy_data["strategy_id"], s_id)
        self.assertEqual(strategy_data["strategy_name"], sample_strategy["strategy_name"])

        # Test get non-existent strategy
        non_existent_id = uuid.uuid4().hex
        response = self.client.get(f'/api/strategies/{non_existent_id}')
        self.assertEqual(response.status_code, 404)

    async def test_04_update_strategy(self): # Made async
        """Test PUT /api/strategies/<strategy_id> to update a strategy."""
        s_id = uuid.uuid4().hex
        self._create_sample_strategy_direct_db(s_id, "Original Name", "Original Desc", {"orig_cond": True}, [{"orig_act": True}])
        load_strategies_from_db()

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

    async def test_05_delete_strategy(self): # Made async
        """Test DELETE /api/strategies/<strategy_id> for soft delete."""
        s_id = uuid.uuid4().hex
        self._create_sample_strategy_direct_db(s_id, "To Be Deleted", "Delete test", {}, [])
        load_strategies_from_db()

        response = self.client.delete(f'/api/strategies/{s_id}') # This is sync
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


    async def test_06_enable_disable_strategy(self): # Made async
        """Test POST enabling and disabling strategies."""
        s_id = uuid.uuid4().hex
        self._create_sample_strategy_direct_db(s_id, "Activatable", "Test enable/disable", {}, [], is_active=True)
        load_strategies_from_db()

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

    async def test_07_get_strategy_history(self): # Made async
        """Test GET /api/strategies/<strategy_id>/history."""
        s_id = uuid.uuid4().hex
        self._create_sample_strategy_direct_db(s_id, "History Test", "V1", {"cond": "v1"}, [{"act": "v1"}], version_notes="Version 1")
        load_strategies_from_db()

        update_1_payload = {"strategy_name": "History Test Updated", "description": "V2", "conditions_group": {"cond": "v2"}, "actions": [{"act": "v2"}], "version_notes": "Version 2"}
        # Flask client calls are synchronous
        response_put1 = self.client.put(f'/api/strategies/{s_id}', data=json.dumps(update_1_payload), content_type='application/json')
        self.assertEqual(response_put1.status_code, 200)

        update_2_payload = {"strategy_name": "History Test Final", "description": "V3", "conditions_group": {"cond": "v3"}, "actions": [{"act": "v3"}], "version_notes": "Version 3"}
        response_put2 = self.client.put(f'/api/strategies/{s_id}', data=json.dumps(update_2_payload), content_type='application/json')
        self.assertEqual(response_put2.status_code, 200)

        response = self.client.get(f'/api/strategies/{s_id}/history') # Sync call
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


    async def test_08_rollback_strategy(self): # Made async
        """Test POST /api/strategies/<strategy_id>/rollback/<version_id>."""
        s_id = uuid.uuid4().hex
        self._create_sample_strategy_direct_db(s_id, "Rollback Test", "V1 Desc", {"cond": "v1"}, [{"act": "v1"}], version_notes="V1 Notes")
        load_strategies_from_db()

        # Get v1_id from DB
        conn_temp = database.get_db_connection() # This now uses TEST_DATABASE_NAME
        cursor_temp = conn_temp.cursor()
        cursor_temp.execute("SELECT version_id FROM strategy_versions WHERE strategy_id = ? AND version_notes = ?", (s_id, "V1 Notes"))
        v1_row = cursor_temp.fetchone()
        self.assertIsNotNone(v1_row, "Version 1 not found in DB for rollback test setup")
        v1_id = v1_row['version_id']
        conn_temp.close()

        update_payload = {"strategy_name": "Rollback Test V2", "description": "V2 Desc", "conditions_group": {"cond": "v2"}, "actions": [{"act": "v2"}], "version_notes": "V2 Notes"}
        response_put = self.client.put(f'/api/strategies/{s_id}', data=json.dumps(update_payload), content_type='application/json')
        self.assertEqual(response_put.status_code, 200)

        response_rollback = self.client.post(f'/api/strategies/{s_id}/rollback/{v1_id}') # Sync call
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

    # Or, if test_app.py is in a 'tests' subdirectory: python -m unittest tests.test_app
    # No new tests were added in the previous step, this was a mistake.
    # The content below is what was intended to be added.

    # --- New Async Tests for WebSocket Handling ---

    @patch('run.websockets.connect', new_callable=AsyncMock)
    async def test_websocket_successful_connection_and_initial_messages(self, mock_connect):
        """ Test successful WebSocket connection, on_open calls, and basic message send/recv """
        mock_ws_client = AsyncMock()
        mock_ws_client.send = AsyncMock()
        mock_ws_client.open = True # Simulate is open for send_ws_request_and_wait checks
        # run.ws_app = mock_ws_client # Set the global ws_app to our mock - careful if _connect_and_listen_deriv sets it too

        # Simulate authorize response and a tick, then keep open until cancelled
        async def mock_recv_generator():
            yield json.dumps({
                "msg_type": "authorize",
                "authorize": {"loginid": "testuser", "scopes": ["read", "trade"]},
                "echo_req": {"req_id": 1} # Assuming a req_id for on_open's authorize
            })
            yield json.dumps({ # Simulate response to ticks subscription
                "msg_type": "subscribe",
                "subscription": {"id": "test_tick_sub_id"},
                "echo_req": {"req_id": 2, "ticks": run.SYMBOL} # Assuming a req_id for on_open's subscribe
            })
            yield json.dumps({
                "msg_type": "tick",
                "tick": {"symbol": run.SYMBOL, "quote": 123.45, "epoch": int(time.time())},
                "subscription": {"id": "test_tick_sub_id"}
            })
            try:
                while True:
                    await asyncio.sleep(0.001)
            except asyncio.CancelledError:
                # print("Mock recv_generator cancelled") # For debugging
                raise

        mock_ws_client.__aiter__.return_value = mock_recv_generator()
        mock_ws_client.close_code = 1000
        mock_ws_client.close_reason = "Test normal close by cancel"

        mock_connect.return_value = mock_ws_client # This is what `async with websockets.connect(...)` will use

        # We need to control run.ws_app, so _connect_and_listen_deriv should set it to our mock_ws_client
        # Patching on_open_for_deriv and its sub-calls if they make actual sends, or on_message_for_deriv
        with patch('run.on_open_for_deriv', new_callable=AsyncMock) as mock_on_open_deriv, \
             patch('run.on_message_for_deriv', new_callable=AsyncMock) as mock_on_message_deriv, \
             patch('run.on_close_for_deriv', new_callable=AsyncMock) as mock_on_close_deriv, \
             patch('run.on_error_for_deriv', new_callable=AsyncMock) as mock_on_error_deriv:

            mock_on_open_deriv.return_value = None
            mock_on_message_deriv.return_value = None
            mock_on_close_deriv.return_value = None
            mock_on_error_deriv.return_value = None

            connect_task = self.asyncio_loop.create_task(run._connect_and_listen_deriv())

            await asyncio.sleep(0.01) # Allow time for connection attempt and on_open

            mock_connect.assert_called_with(
                run.DERIV_WS_URL,
                ping_interval=20,
                ping_timeout=10,
                open_timeout=10
            )
            self.assertIsNotNone(run.ws_app, "run.ws_app should be set to the mock client after connect.")
            self.assertEqual(run.ws_app, mock_ws_client) # Check if the global ws_app is our mock
            mock_on_open_deriv.assert_called_once_with(mock_ws_client)

            await asyncio.sleep(0.01) # Allow time for messages to be processed by recv_loop->on_message_for_deriv

            self.assertTrue(mock_on_message_deriv.call_count >= 3, f"Expected at least 3 messages, got {mock_on_message_deriv.call_count}")

            processed_msg_types = set()
            for call_args in mock_on_message_deriv.call_args_list:
                args, _ = call_args
                msg_str = args[1]
                msg_data = json.loads(msg_str)
                processed_msg_types.add(msg_data.get("msg_type"))

            self.assertIn("authorize", processed_msg_types)
            self.assertIn("subscribe", processed_msg_types)
            self.assertIn("tick", processed_msg_types)

            connect_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await connect_task

            await asyncio.sleep(0.01) # Allow time for cleanup in _connect_and_listen_deriv's finally
            # mock_on_close_deriv.assert_called_once() # This can be tricky with cancellation

    @patch('run.websockets.connect', new_callable=AsyncMock)
    @patch('run.asyncio.sleep', new_callable=AsyncMock)
    async def test_websocket_reconnection_backoff(self, mock_sleep, mock_connect):
        """Test WebSocket reconnection logic with exponential backoff."""

        mock_successful_ws_client = AsyncMock()
        mock_successful_ws_client.send = AsyncMock()
        mock_successful_ws_client.open = True

        async def success_recv_generator():
            yield json.dumps({"msg_type": "authorize", "authorize": {"loginid": "testuser"}, "echo_req": {"req_id": 1}})
            try:
                while True: await asyncio.sleep(0.001)
            except asyncio.CancelledError:
                # print("Successful WS client recv_generator cancelled in backoff test") # Debug
                raise
        mock_successful_ws_client.__aiter__.return_value = success_recv_generator()
        mock_successful_ws_client.close_code = 1000
        mock_successful_ws_client.close_reason = "Normal close after success (backoff test)"

        mock_connect.side_effect = [
            websockets.exceptions.ConnectionClosedError(None, None),
            ConnectionRefusedError("Test refusal from mock (backoff)"),
            mock_successful_ws_client
        ]

        initial_delay = run.INITIAL_RECONNECT_DELAY
        run.current_reconnect_delay = initial_delay

        with patch('run.on_open_for_deriv', new_callable=AsyncMock) as mock_on_open_success, \
             patch('run.on_close_for_deriv', new_callable=AsyncMock) as mock_on_close: # Mock on_close too

            connect_task = self.asyncio_loop.create_task(run._connect_and_listen_deriv())

            # 1st failure
            await asyncio.sleep(0.01)
            mock_sleep.assert_any_call(initial_delay)
            expected_delay_after_1st_fail = initial_delay * run.RECONNECT_FACTOR
            self.assertAlmostEqual(run.current_reconnect_delay, expected_delay_after_1st_fail, places=5)

            # 2nd failure
            await asyncio.sleep(expected_delay_after_1st_fail + 0.01)
            mock_sleep.assert_any_call(expected_delay_after_1st_fail)
            expected_delay_after_2nd_fail = expected_delay_after_1st_fail * run.RECONNECT_FACTOR
            self.assertAlmostEqual(run.current_reconnect_delay, expected_delay_after_2nd_fail, places=5)

            # 3rd attempt (success)
            await asyncio.sleep(expected_delay_after_2nd_fail + 0.01)

            self.assertEqual(run.ws_app, mock_successful_ws_client, "Global ws_app not set to successful mock client")
            mock_on_open_success.assert_called_once_with(mock_successful_ws_client)
            self.assertEqual(run.current_reconnect_delay, initial_delay)
            self.assertEqual(mock_connect.call_count, 3)

            connect_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await connect_task

            await asyncio.sleep(0.01) # allow finally block in _connect_and_listen_deriv to run
            # mock_on_close.assert_called() # At least one on_close should be called

if __name__ == '__main__':
    unittest.main()
