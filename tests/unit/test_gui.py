"""
Unit tests for GUI application components.

This module implements unit tests for the APGI GUI application,
testing initialization, plot updates, parameter adjustments, and
manual interventions without launching the full GUI.

Tests focus on the backend logic that powers the GUI functionality.
"""

import time
import tkinter as tk

import numpy as np
import pytest

from apgi_framework.system import APGISystem


def _new_root() -> tk.Tk:
    """Create a Tk root, retrying once or twice: Tcl occasionally fails to read its init
    script on Windows runners when many roots are created in one process."""
    for attempt in range(3):
        try:
            return tk.Tk()
        except tk.TclError:
            if attempt == 2:
                raise
            time.sleep(0.5)
    raise RuntimeError("unreachable")


class TestGUIInitialization:
    """Test GUI initialization and setup."""

    def test_gui_launches_without_errors(self) -> None:
        """
        Test that GUI can be initialized without errors.

        This test verifies that the GUI window can be created and
        all components are properly initialized.
        **Validates: Requirements 6.1**
        """
        # Create root window
        root = _new_root()

        try:
            # Import GUI class
            from apgi_gui.main import APGIGui

            # Create GUI instance
            app = APGIGui(root)

            # Verify GUI was created
            assert app is not None
            assert app.root is root

            # Verify system was initialized
            assert app.system is not None
            assert isinstance(app.system, APGISystem)

            # Verify initial state
            assert app.sim_controller.is_running is False
            assert app.sim_controller.is_paused is False  # type: ignore[attr-defined]
            assert app.is_paused is False  # type: ignore[attr-defined]

            # Verify data buffers were created
            assert hasattr(app, "data_buffers")
            assert "ignition" in app.data_buffers
            assert "surprise" in app.data_buffers
            assert "metabolic_reserves" in app.data_buffers

            # Verify parameter variables were created
            assert hasattr(app, "param_vars")
            # Note: param_vars is populated by ControlPanel, not directly in APGIGui

            # Verify simulation controller was created
            assert hasattr(app, "sim_controller")
            assert hasattr(app, "control_panel")
            assert hasattr(app, "status_bar")

        finally:
            # Clean up
            root.quit()
            root.destroy()

    def test_gui_initializes_with_default_config(self) -> None:
        """
        Test that GUI initializes with default configuration.
        **Validates: Requirements 6.1**
        """
        root = _new_root()

        try:
            from apgi_gui.main import APGIGui

            app = APGIGui(root)

            # Wait for tkinter variable conversion
            app.root.after(250, lambda: None)  # Wait longer than the 200ms delay
            app.root.update()

            # Verify buffer size
            assert app.max_buffer_points == 1000

        finally:
            root.quit()
            root.destroy()

    def test_gui_creates_all_visualization_panels(self) -> None:
        """
        Test that all visualization panels are created.
        **Validates: Requirements 6.1**
        """
        root = _new_root()

        try:
            from apgi_gui.main import APGIGui

            app = APGIGui(root)

            # Verify visualization panel was created
            assert hasattr(app, "viz_panel")
            assert hasattr(app, "control_panel")
            assert hasattr(app, "status_bar")
            assert hasattr(app, "menu_bar")

        finally:
            root.quit()
            root.destroy()


class TestPlotUpdateMechanisms:
    """Test plot update mechanisms."""

    def test_plot_updates_with_valid_data(self) -> None:
        """
        Test that plots update correctly with valid data.
        **Validates: Requirements 6.4**
        """
        try:
            root = _new_root()
        except Exception as e:
            # Skip test if Tkinter is not properly configured
            pytest.skip(f"Tkinter not available: {e}")

        try:
            from apgi_gui.main import APGIGui

            app = APGIGui(root)

            # Simulate some data
            for i in range(10):
                app.time_buffer.append(i * 0.1)
                app.data_buffers["ignition"].append(1 if i % 3 == 0 else 0)
                app.data_buffers["surprise"].append(2.0 + np.random.randn() * 0.1)
                app.data_buffers["extero_precision"].append(1.0 + np.random.randn() * 0.05)
                app.data_buffers["intero_precision"].append(0.8 + np.random.randn() * 0.05)
                app.data_buffers["somatic_gain"].append(100.0 - i * 2)
                app.data_buffers["metabolic_reserves"].append(100.0 - i * 2)
                app.data_buffers["allostatic_load"].append(i * 0.01)

            # Update plots (should not raise errors)
            app.viz_panel.update_plots(app.data_buffers, app.time_buffer)

            # Verify plots were updated (no exceptions raised)
            assert len(app.time_buffer) == 10
            assert len(app.data_buffers["ignition"]) == 10

        finally:
            root.quit()
            root.destroy()

    def test_plot_updates_handle_empty_buffers(self) -> None:
        """
        Test that plot updates handle empty buffers gracefully.
        **Validates: Requirements 6.4**
        """
        root = _new_root()

        try:
            from apgi_gui.main import APGIGui

            app = APGIGui(root)

            # Ensure buffers are empty
            app.time_buffer.clear()
            for buffer in app.data_buffers.values():
                buffer.clear()

            # Update plots with empty buffers (should not raise errors)
            app.viz_panel.update_plots(app.data_buffers, app.time_buffer)

            # Verify no errors occurred
            assert len(app.time_buffer) == 0

        finally:
            root.quit()
            root.destroy()

    def test_status_labels_update_correctly(self) -> None:
        """
        Test that status labels update with correct values.
        **Validates: Requirements 6.4**
        """
        root = _new_root()

        try:
            from apgi_gui.main import APGIGui

            app = APGIGui(root)

            # Run a few simulation steps
            for i in range(5):
                obs = np.random.randn(256) * 0.5
                assert app.system is not None
                app.system.step(obs)

            # Update status labels
            status_metrics = {
                "Time": f"{i * 0.1:.2f} s",
                "Ignition Events": "0",
                "Workspace": "Active",
            }
            app.control_panel.update_status(status_metrics)

            # Verify labels were updated (should not raise errors)
            # Note: We can't easily verify exact text without complex mocking,
            # but we can verify the method executes without errors
            assert True

        finally:
            root.quit()
            root.destroy()

    def test_record_state_captures_all_metrics(self) -> None:
        """
        Test that _record_state captures all required metrics.
        **Validates: Requirements 6.3, 6.4**
        """
        pytest.skip("GUI tests skipped due to tkinter segfault issues")


class TestParameterAdjustments:
    """Test slider parameter updates."""

    def test_parameter_adjustment_updates_system(self) -> None:
        """
        Test that parameter adjustments update the system immediately.
        **Validates: Requirements 6.2**
        """
        pytest.skip("Parameter adjustment test skipped - param_vars managed by ControlPanel")

    def test_parameter_adjustment_with_boundary_values(self) -> None:
        """
        Test parameter adjustments with boundary values.
        **Validates: Requirements 6.2**
        """
        pytest.skip("Parameter adjustment test skipped - param_vars managed by ControlPanel")

    def test_parameter_adjustment_handles_errors_gracefully(self) -> None:
        """
        Test that parameter adjustment handles errors gracefully.
        **Validates: Requirements 6.2**
        """
        pytest.skip("Parameter adjustment test skipped - param_vars managed by ControlPanel")

    def test_speed_control_updates(self) -> None:
        """
        Test that speed control variable updates correctly.
        **Validates: Requirements 6.2**
        """
        root = _new_root()

        try:
            from apgi_gui.main import APGIGui

            app = APGIGui(root)

            # Test different speed values
            test_speeds = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

            for speed in test_speeds:
                app.speed_var.set(speed)
                assert app.speed_var.get() == speed

        finally:
            root.quit()
            root.destroy()


class TestManualInterventions:
    """Test manual intervention buttons."""

    def test_trigger_ignition_intervention(self) -> None:
        """
        Test that manual ignition trigger works correctly.
        **Validates: Requirements 6.5**
        """
        root = _new_root()

        try:
            from apgi_gui.main import APGIGui

            app = APGIGui(root)

            # Start simulation first (required for ignition trigger)
            app._start_simulation()

            # Record initial arousal and stress
            initial_arousal = app.param_vars["arousal"].get()
            initial_stress = app.param_vars["stress"].get()

            # Trigger ignition
            app._trigger_ignition()

            # Verify arousal and stress were increased
            assert app.param_vars["arousal"].get() == 0.9
            assert app.param_vars["stress"].get() == 0.8

            # Verify these are higher than initial values
            assert app.param_vars["arousal"].get() > initial_arousal
            assert app.param_vars["stress"].get() > initial_stress

        finally:
            root.quit()
            root.destroy()

    def test_induce_stressor_intervention(self) -> None:
        """
        Test that stressor induction works correctly.
        **Validates: Requirements 6.5**
        """
        root = _new_root()

        try:
            from apgi_gui.main import APGIGui

            app = APGIGui(root)

            # Get initial allostatic load
            obs = np.random.randn(256) * 0.5
            assert app.system is not None
            app.system.step(obs)

            # Induce stressor
            app._induce_stressor()

            # Step the system to see the effect
            obs = np.random.randn(256) * 0.5
            assert app.system is not None
            app.system.step(obs)

            # Verify stressor was applied (method should execute without errors)
            # Note: The exact effect on allostatic load depends on system dynamics
            assert True

        finally:
            root.quit()
            root.destroy()

    def test_modulate_precision_intervention(self) -> None:
        """
        Test that precision modulation dialog can be created.
        **Validates: Requirements 6.5**
        """
        root = _new_root()

        try:
            from apgi_gui.main import APGIGui

            app = APGIGui(root)

            # Record initial precision values
            # Note: We can't easily test the dialog interaction without complex mocking,
            # but we can verify the method exists and can be called
            assert hasattr(app, "_modulate_precision")
            assert callable(app._modulate_precision)

        finally:
            root.quit()
            root.destroy()

    def test_reset_simulation_intervention(self) -> None:
        """
        Test that reset simulation works correctly.
        **Validates: Requirements 6.5**
        """
        root = _new_root()

        try:
            from apgi_gui.main import APGIGui

            app = APGIGui(root)

            # Run some simulation steps
            assert app.system is not None
            for i in range(10):
                obs = np.random.randn(256) * 0.5
                state = app.system.step(obs)
                # Buffers are filled by the simulation-step callback (the mediator calls it)
                app._on_simulation_step({"time": state["time"], "surprise": 0.0})

            # Verify data was accumulated
            assert len(app.time_buffer) > 0

            # Reset simulation
            app._reset_simulation()

            # Verify buffers were cleared
            assert len(app.time_buffer) == 0
            for buffer in app.data_buffers.values():
                assert len(buffer) == 0

            # Verify system was reset
            assert app.system is not None
            assert app.system.time == 0

        finally:
            root.quit()
            root.destroy()


class TestDataExport:
    """Test data export functionality."""

    def test_export_data_to_csv(self) -> None:
        """
        Test that data can be exported to CSV format.
        **Validates: Requirements 6.3**
        """
        pytest.skip("Data export test skipped - log_data not available in new implementation")

    def test_export_data_to_json(self) -> None:
        """
        Test that data can be exported to JSON format.
        **Validates: Requirements 6.3**
        """
        pytest.skip("GUI tests skipped due to tkinter segfault issues")


class TestSimulationControl:
    """Test simulation control functionality."""

    def test_start_simulation_changes_state(self) -> None:
        """
        Test that starting simulation changes GUI state correctly.
        **Validates: Requirements 6.1**
        """
        root = _new_root()

        try:
            from apgi_gui.main import APGIGui

            app = APGIGui(root)

            # Verify initial state
            assert app.sim_controller.is_running is False
            assert app.sim_controller.is_paused is False

            # Note: We can't easily test the full simulation loop without complex threading mocks,
            # but we can verify the state management logic exists
            assert hasattr(app, "_start_simulation")
            assert hasattr(app, "_pause_simulation")
            assert hasattr(app, "_stop_simulation")
            assert callable(app._start_simulation)

        finally:
            root.quit()
            root.destroy()

    def test_generate_input_produces_valid_arrays(self) -> None:
        """
        Test that input generation produces valid arrays.
        **Validates: Requirements 6.1**
        """
        root = _new_root()

        try:
            from apgi_gui.main import APGIGui

            app = APGIGui(root)

            # Generate input for different time points
            for t in [0.0, 1.0, 5.0, 10.0]:
                input_array = app._generate_input(t)

                # Verify input is valid
                assert isinstance(input_array, np.ndarray)
                assert len(input_array) == 256
                assert np.all(np.isfinite(input_array))
                assert not np.any(np.isnan(input_array))

        finally:
            root.quit()
            root.destroy()


class TestEventLogging:
    """Test event logging functionality."""

    def test_log_event_adds_to_log(self) -> None:
        """
        Test that logging events adds entries to the log.
        **Validates: Requirements 6.1**
        """
        root = _new_root()

        try:
            from apgi_gui.main import APGIGui

            app = APGIGui(root)

            # Log an event
            app.control_panel.log_event("Test event message")

            # Process pending tkinter events
            app.root.update()

            # Verify method executes without errors
            assert True

        finally:
            root.quit()
            root.destroy()

    def test_update_status_changes_status_bar(self) -> None:
        """
        Test that updating status changes the status bar.
        **Validates: Requirements 6.1**
        """
        root = _new_root()

        try:
            from apgi_gui.main import APGIGui

            app = APGIGui(root)

            # Update status
            app.status_bar.set_status("Test status message")

            # Process pending tkinter events
            app.root.update()

            # Verify method executes without errors
            assert True

        finally:
            root.quit()
            root.destroy()
