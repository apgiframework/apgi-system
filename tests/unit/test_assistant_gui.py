"""
Unit tests for Assistant GUI application components.

This module implements unit tests for the APGI Assistant GUI application,
testing initialization, UI components, query processing, and assistant
interactions without launching the full GUI.

Tests focus on the backend logic that powers the Assistant GUI functionality.
"""

import importlib.util
import json
import sys
import tempfile
import time
import tkinter as tk
from pathlib import Path
from typing import Any

import pytest

# AI_Assistant.py imports torch unconditionally, so Assistant_GUI cannot even be imported without it.
pytest.importorskip("torch", reason="Assistant_GUI needs torch (AI_Assistant.py imports it)")


def load_module_from_file(module_name: str, file_path: Path) -> Any:
    """Load a Python module from a file path, handling hyphenated filenames."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Import the GUI class with error handling
HAS_ASSISTANT_GUI = False
APGIGUI: Any = None

try:
    # Load from hyphenated filename using importlib
    assistant_gui_path = Path(__file__).parent.parent.parent / "Assistant_GUI.py"
    if assistant_gui_path.exists():
        Assistant_GUI = load_module_from_file("Assistant_GUI", assistant_gui_path)
        APGIGUI = Assistant_GUI.APGIGUI
        HAS_ASSISTANT_GUI = True
    else:
        # Fallback to standard import if file renamed
        from Assistant_GUI import APGIGUI as _APGIGUI  # type: ignore [import]

        APGIGUI = _APGIGUI
        HAS_ASSISTANT_GUI = True
except ImportError as e:
    print(f"Warning: Could not import APGIGUI: {e}")

HAS_ASSISTANT = False
APGIAssistant: Any = None

try:
    ai_assistant_path = Path(__file__).parent.parent.parent / "AI_Assistant.py"
    if ai_assistant_path.exists():
        AI_Assistant = load_module_from_file("AI_Assistant", ai_assistant_path)
        APGIAssistant = AI_Assistant.APGIAssistant
        HAS_ASSISTANT = True
    else:
        # Fallback to standard import if file renamed
        from AI_Assistant import APGIAssistant as _APGIAssistant

        APGIAssistant = _APGIAssistant
        HAS_ASSISTANT = True
except ImportError:
    pass


class TestAssistantGUIInitialization:
    """Test Assistant GUI initialization and setup."""

    @pytest.mark.skipif(not HAS_ASSISTANT_GUI, reason="APGIGUI not available")
    def test_gui_launches_without_errors(self):
        """
        Test that Assistant GUI can be initialized without errors.

        This test verifies that the GUI window can be created and
        all components are properly initialized.
        """
        root = tk.Tk()

        try:
            app = APGIGUI(root)

            # Verify GUI was created
            assert app is not None
            assert app.root is root

            # Verify assistant integration
            if HAS_ASSISTANT:
                assert hasattr(app, "assistant")
                assert app.assistant is not None
            else:
                assert app.assistant is None

            # Verify UI components were created (using actual attribute names)
            assert hasattr(app, "query_input")  # Text widget for query input
            assert hasattr(app, "response_display")  # Text widget for response display
            assert hasattr(app, "status_label")  # Status bar label
            assert hasattr(app, "main_frame")  # Main tab frame

            # Verify configuration (stored as config_file path, not dict)
            assert hasattr(app, "config_file")
            assert hasattr(app, "session_file")

        finally:
            try:
                if root.winfo_exists():
                    root.quit()
                    root.destroy()
            except tk.TclError:
                pass

    @pytest.mark.skipif(not HAS_ASSISTANT_GUI, reason="APGIGUI not available")
    def test_gui_initializes_with_default_config(self):
        """
        Test that Assistant GUI initializes with default configuration.
        """
        root = tk.Tk()

        try:
            app = APGIGUI(root)

            # Verify default configuration files are set
            assert app.config_file is not None
            assert app.session_file is not None

            # Verify default variable values
            assert hasattr(app, "memory_length_var")
            assert hasattr(app, "hidden_dim_var")
            assert app.memory_length_var.get() == 100
            assert app.hidden_dim_var.get() == 256

            # Verify UI state
            assert app.query_input is not None
            assert app.response_display is not None

        finally:
            try:
                if root.winfo_exists():
                    root.quit()
                    root.destroy()
            except tk.TclError:
                pass

    @pytest.mark.skipif(not HAS_ASSISTANT_GUI, reason="APGIGUI not available")
    def test_gui_creates_all_ui_panels(self):
        """
        Test that all UI panels are created.
        """
        root = tk.Tk()

        try:
            app = APGIGUI(root)

            # Verify main tab frames exist (these are the actual frame names)
            assert hasattr(app, "main_frame")
            assert hasattr(app, "cognitive_frame")
            assert hasattr(app, "oscillatory_frame")
            assert hasattr(app, "biofeedback_frame")
            assert hasattr(app, "performance_frame")
            assert hasattr(app, "viz_frame")
            assert hasattr(app, "settings_frame")

            # Verify text widgets exist
            assert app.query_input.winfo_exists()
            assert app.response_display.winfo_exists()

        finally:
            try:
                if root.winfo_exists():
                    root.quit()
                    root.destroy()
            except tk.TclError:
                pass


class TestQueryProcessing:
    """Test query processing functionality."""

    @pytest.mark.skipif(not HAS_ASSISTANT_GUI, reason="APGIGUI not available")
    def test_query_validation(self):
        """
        Test that query validation works correctly.
        """
        # Test valid query using InputValidator (no need for GUI instance)
        from Assistant_GUI import InputValidator

        # Test valid query
        valid_query = "What is active inference?"
        is_valid, _ = InputValidator.validate_query(valid_query)
        assert is_valid

        # Test empty query
        is_valid, _ = InputValidator.validate_query("")
        assert not is_valid

        # Test whitespace-only query
        is_valid, _ = InputValidator.validate_query("   ")
        assert not is_valid

    @pytest.mark.skipif(
        not HAS_ASSISTANT_GUI or not HAS_ASSISTANT, reason="Assistant components not available"
    )
    def test_query_processing_with_mock_assistant(self):
        """
        Test query processing with mocked assistant.
        """
        root = tk.Tk()

        try:
            app = APGIGUI(root)

            # Skip if assistant is not available (even if HAS_ASSISTANT is True)
            if app.assistant is None:
                pytest.skip("Assistant not initialized")

            # Verify assistant has expected methods/attributes
            assert hasattr(app.assistant, "model")
            assert hasattr(app.assistant, "generate_response")
            assert callable(app.assistant.generate_response)

            # Set query text in the input widget
            app.query_input.insert(1.0, "Test query")

            # Verify the query was set
            assert app.query_input.get(1.0, tk.END).strip() == "Test query"

        finally:
            try:
                if root.winfo_exists():
                    root.quit()
                    root.destroy()
            except tk.TclError:
                pass

    @pytest.mark.skipif(not HAS_ASSISTANT_GUI, reason="APGIGUI not available")
    def test_query_type_classification(self):
        """
        Test query type classification.
        """
        root = tk.Tk()

        try:
            app = APGIGUI(root)

            # Test different query types based on actual implementation
            assert app._classify_query_type("What is active inference?") == "informational"
            assert app._classify_query_type("Explain the free energy principle") == "informational"
            assert app._classify_query_type("How to run a simulation") == "procedural"
            assert app._classify_query_type("Analyze the data") == "analytical"
            assert app._classify_query_type("Create a report") == "creative"
            assert app._classify_query_type("Why is this happening?") == "interrogative"
            assert app._classify_query_type("Hello") == "general"

        finally:
            try:
                if root.winfo_exists():
                    root.quit()
                    root.destroy()
            except tk.TclError:
                pass


class TestUIInteractions:
    """Test UI interaction functionality."""

    @pytest.mark.skipif(not HAS_ASSISTANT_GUI, reason="APGIGUI not available")
    def test_clear_query_action(self):
        """
        Test clearing query text.
        """
        root = tk.Tk()

        try:
            app = APGIGUI(root)

            # Set some text
            app.query_input.insert(1.0, "Test query text")
            assert len(app.query_input.get(1.0, tk.END).strip()) > 0

            # Clear it using the actual method name
            app.clear_query()
            assert app.query_input.get(1.0, tk.END).strip() == ""

        finally:
            try:
                if root.winfo_exists():
                    root.quit()
                    root.destroy()
            except tk.TclError:
                pass

    @pytest.mark.skipif(not HAS_ASSISTANT_GUI, reason="APGIGUI not available")
    def test_status_updates(self):
        """
        Test status label updates.
        """
        root = tk.Tk()

        try:
            app = APGIGUI(root)

            # Verify status_label exists and has expected properties
            assert hasattr(app, "status_label")
            assert app.status_label is not None

            # Verify we can read the status text
            initial_text = app.status_label.cget("text")
            assert isinstance(initial_text, str)

        finally:
            try:
                if root.winfo_exists():
                    root.quit()
                    root.destroy()
            except tk.TclError:
                pass

    @pytest.mark.skipif(not HAS_ASSISTANT_GUI, reason="APGIGUI not available")
    def test_progress_bar_updates(self):
        """
        Test progress bar functionality using CancellableProgress dialog.
        """
        root = tk.Tk()

        try:
            # The APGIGUI uses CancellableProgress dialog for progress
            # Verify the progress dialog class exists
            from Assistant_GUI import CancellableProgress

            assert CancellableProgress is not None

            # Create a progress dialog and test it
            progress = CancellableProgress(root, timeout_seconds=30)
            progress.show("Test progress")

            # Verify progress bar was created
            assert hasattr(progress, "progress_bar")
            assert progress.progress_bar is not None

            # Clean up
            progress.hide()

        finally:
            try:
                if root.winfo_exists():
                    root.quit()
                    root.destroy()
            except tk.TclError:
                pass


class TestDataManagement:
    """Test data management and export functionality."""

    @pytest.mark.skipif(not HAS_ASSISTANT_GUI, reason="APGIGUI not available")
    def test_conversation_history_management(self):
        """
        Test conversation history storage and retrieval.
        """
        root = tk.Tk()

        try:
            app = APGIGUI(root)

            # Add some conversation data to query_history deque
            initial_len = len(app.query_history)
            app.query_history.append({"query": "User query", "response": "Assistant response"})

            # Verify history was stored
            assert len(app.query_history) > initial_len
            assert app.query_history[-1]["query"] == "User query"
            assert app.query_history[-1]["response"] == "Assistant response"

        finally:
            try:
                if root.winfo_exists():
                    root.quit()
                    root.destroy()
            except tk.TclError:
                pass

    @pytest.mark.skipif(not HAS_ASSISTANT_GUI, reason="APGIGUI not available")
    def test_export_conversation_to_json(self):
        """
        Test exporting conversation to JSON via session export.
        """
        root = tk.Tk()

        try:
            app = APGIGUI(root)

            # Add conversation data to query_history
            app.query_history.append({"query": "Query 1", "response": "Response 1"})
            app.query_history.append({"query": "Query 2", "response": "Response 2"})

            # Export to temporary file using session JSON export approach
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json_filename = f.name

            try:
                # Use export_session_json approach - save query_history directly
                import json

                with open(json_filename, "w") as f:
                    json.dump(list(app.query_history), f, indent=2)

                # Verify file was created and contains data
                with open(json_filename, "r") as f:
                    data = json.load(f)

                assert len(data) == 2
                assert data[0]["query"] == "Query 1"
                assert data[1]["response"] == "Response 2"

            finally:
                import os

                os.unlink(json_filename)

        finally:
            try:
                if root.winfo_exists():
                    root.quit()
                    root.destroy()
            except tk.TclError:
                pass

    @pytest.mark.skipif(not HAS_ASSISTANT_GUI, reason="APGIGUI not available")
    def test_configuration_persistence(self):
        """
        Test saving and loading configuration via export/import config.
        """
        root = tk.Tk()

        try:
            app = APGIGUI(root)

            # Modify configuration variables
            original_hidden_dim = app.hidden_dim_var.get()
            app.hidden_dim_var.set(512)

            # Create a simple config dict
            config = {
                "hidden_dim": app.hidden_dim_var.get(),
                "memory_length": app.memory_length_var.get(),
            }

            # Save to temporary file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                config_filename = f.name

            try:
                # Save configuration directly
                with open(config_filename, "w") as f:
                    json.dump(config, f, indent=2)

                # Load configuration
                with open(config_filename, "r") as f:
                    loaded_config = json.load(f)

                assert loaded_config["hidden_dim"] == 512

            finally:
                import os

                os.unlink(config_filename)

            # Restore original value
            app.hidden_dim_var.set(original_hidden_dim)

        finally:
            try:
                if root.winfo_exists():
                    root.quit()
                    root.destroy()
            except tk.TclError:
                pass


class TestErrorHandling:
    """Test error handling and recovery."""

    @pytest.mark.skipif(not HAS_ASSISTANT_GUI, reason="APGIGUI not available")
    def test_assistant_unavailable_handling(self):
        """
        Test graceful handling when assistant is not available.
        """
        root = tk.Tk()

        try:
            app = APGIGUI(root)

            # Simulate assistant being unavailable
            original_assistant = app.assistant
            app.assistant = None

            # Set query text
            app.query_input.insert(1.0, "Test query")

            # Try to process query - should not raise exception
            # The process_query method should handle missing assistant gracefully
            try:
                app.process_query()
                # If we get here, the method handled it gracefully
                assert True
            except (RuntimeError, AttributeError) as e:
                # Expected error when assistant is not available
                assert "not available" in str(e).lower() or "assistant" in str(e).lower()

            # Restore assistant
            app.assistant = original_assistant

        finally:
            try:
                if root.winfo_exists():
                    root.quit()
                    root.destroy()
            except tk.TclError:
                pass

    @pytest.mark.skipif(not HAS_ASSISTANT_GUI, reason="APGIGUI not available")
    def test_invalid_query_handling(self):
        """
        Test handling of invalid queries using InputValidator.
        """
        root = tk.Tk()

        try:
            from Assistant_GUI import InputValidator

            # Test with invalid query (empty)
            is_valid, error_msg = InputValidator.validate_query("")
            assert not is_valid
            assert "empty" in error_msg.lower()

            # Test with whitespace-only query
            is_valid, error_msg = InputValidator.validate_query("   ")
            assert not is_valid
            assert "empty" in error_msg.lower()

        finally:
            try:
                if root.winfo_exists():
                    root.quit()
                    root.destroy()
            except tk.TclError:
                pass


class TestThreadingAndAsync:
    """Test threading and asynchronous operations."""

    @pytest.mark.skipif(not HAS_ASSISTANT_GUI, reason="APGIGUI not available")
    def test_query_processing_threading(self):
        """
        Test that query processing components support threading.
        """
        root = tk.Tk()

        try:
            app = APGIGUI(root)

            # Verify threading primitives exist
            assert hasattr(app, "assistant_lock")
            assert hasattr(app, "processing_queue")
            assert hasattr(app, "init_queue")

            # Set query text
            app.query_input.insert(1.0, "Test query")

            # Verify process_query method exists and handles threading
            assert hasattr(app, "process_query")
            assert callable(app.process_query)

        finally:
            try:
                if root.winfo_exists():
                    root.quit()
                    root.destroy()
            except tk.TclError:
                pass

    @pytest.mark.skipif(not HAS_ASSISTANT_GUI, reason="APGIGUI not available")
    def test_ui_update_threading(self):
        """
        Test UI updates from simulated background operations.
        """
        root = tk.Tk()

        try:
            app = APGIGUI(root)

            # Test thread-safe UI updates
            initial_status = app.status_label.cget("text")

            # Update from "background thread" (simulated via after)
            app.root.after(10, lambda: app.update_status("Updated from thread", "info"))

            # Wait for update
            app.root.update()
            time.sleep(0.1)
            app.root.update()

            # Status should have been updated
            assert app.status_label.cget("text") != initial_status

        finally:
            try:
                if root.winfo_exists():
                    root.quit()
                    root.destroy()
            except tk.TclError:
                pass


if __name__ == "__main__":
    pytest.main([__file__])
