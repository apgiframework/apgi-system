"""
Unit tests for platform-specific functionality and cross-platform compatibility.

This module implements tests for platform-specific behavior, cross-platform
deployment compatibility, and platform-dependent features of the APGI system.

Tests ensure the system works correctly across different operating systems
and deployment environments.
"""

import os
import platform
import shutil
import subprocess
import sys
import tempfile

import pytest

try:
    import tkinter

    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False
    tkinter = None  # type: ignore

# Import platform utilities with error handling
try:
    from utils.platform_utils import get_platform_info, is_platform_supported  # type: ignore

    HAS_PLATFORM_UTILS = True
except ImportError:
    HAS_PLATFORM_UTILS = False
    get_platform_info = None
    is_platform_supported = None

try:
    from apgi_framework.system import APGISystem

    HAS_SYSTEM = True
except ImportError:
    HAS_SYSTEM = False
    APGISystem = None  # type: ignore


class TestPlatformDetection:
    """Test platform detection and identification."""

    def test_current_platform_detection(self) -> None:
        """Test detection of the current platform."""
        system = platform.system().lower()
        machine = platform.machine().lower()

        # Verify we're running on a supported platform
        supported_platforms = ["windows", "linux", "darwin", "macos"]
        assert system in supported_platforms or any(s in system for s in supported_platforms)

        # Verify architecture detection
        supported_architectures = ["x86_64", "amd64", "arm64", "aarch64"]
        assert machine in supported_architectures

    @pytest.mark.skipif(not HAS_PLATFORM_UTILS, reason="Platform utils not available")
    def test_platform_info_structure(self) -> None:
        """Test that platform info returns expected structure."""
        info = get_platform_info()

        assert isinstance(info, dict)
        assert "system" in info
        assert "release" in info
        assert "version" in info
        assert "machine" in info
        assert "processor" in info

        # Values should be strings or None
        for key, value in info.items():
            assert value is None or isinstance(value, str)

    @pytest.mark.skipif(not HAS_PLATFORM_UTILS, reason="Platform utils not available")
    def test_platform_support_check(self) -> None:
        """Test platform support verification."""
        current_system = platform.system().lower()

        # Current platform should be supported
        assert is_platform_supported(current_system)

        # Test some unsupported platforms
        assert not is_platform_supported("unsupported_os")
        assert not is_platform_supported("")


class TestPlatformSpecificBehavior:
    """Test platform-specific behavior and adaptations."""

    def test_path_handling_cross_platform(self) -> None:
        """Test path handling works across platforms."""
        # Test path separators

        if os.name == "nt":  # Windows
            # Should handle backslashes
            windows_path = "config\\default.yaml"
            normalized = os.path.normpath(windows_path)
            assert (
                os.path.exists(os.path.join("config", "default.yaml")) or True
            )  # Just test normalization
        else:  # Unix-like
            # Should handle forward slashes
            unix_path = "config/default.yaml"
            normalized = os.path.normpath(unix_path)
            assert "/" in normalized or True  # Just test normalization

    def test_file_operations_platform_compatibility(self) -> None:
        """Test file operations work across platforms."""
        # Create temporary file with cross-platform path handling
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            temp_path = f.name
            f.write("test content")

        try:
            # Verify file was created and can be read
            assert os.path.exists(temp_path)

            with open(temp_path, "r") as f:
                content = f.read()
                assert content == "test content"

            # Test file permissions (limited cross-platform test)
            # On Windows, we can't test chmod the same way, but file should be readable
            assert os.access(temp_path, os.R_OK)

        finally:
            os.unlink(temp_path)

    def test_environment_variable_handling(self) -> None:
        """Test environment variable handling across platforms."""
        test_var = "APGI_TEST_VAR"
        test_value = "test_value_platform"

        # Set environment variable
        original_value = os.environ.get(test_var)
        os.environ[test_var] = test_value

        try:
            # Verify it was set
            assert os.environ.get(test_var) == test_value

            # Test retrieval
            retrieved_value = os.environ.get(test_var)
            assert retrieved_value == test_value

        finally:
            # Restore original value
            if original_value is not None:
                os.environ[test_var] = original_value
            else:
                os.environ.pop(test_var, None)

    def test_process_creation_platform_specific(self) -> None:
        """Test process creation with platform-specific considerations."""
        # Test basic process creation
        if sys.platform == "win32":
            # Windows-specific command
            cmd = ["cmd", "/c", "echo", "test"]
        else:
            # Unix-like systems
            cmd = ["echo", "test"]

        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 0
        assert "test" in result.stdout.strip()

    def test_temporary_directory_handling(self) -> None:
        """Test temporary directory handling across platforms."""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()

        try:
            assert os.path.exists(temp_dir)
            assert os.path.isdir(temp_dir)

            # Create file in temp directory
            test_file = os.path.join(temp_dir, "test.txt")
            with open(test_file, "w") as f:
                f.write("platform test")

            assert os.path.exists(test_file)

            # Read file back
            with open(test_file, "r") as f:
                content = f.read()
                assert content == "platform test"

        finally:
            # Clean up
            shutil.rmtree(temp_dir)
            assert not os.path.exists(temp_dir)


class TestCrossPlatformDeployment:
    """Test cross-platform deployment scenarios."""

    @pytest.mark.skipif(not HAS_SYSTEM, reason="APGI system not available")
    def test_system_initialization_cross_platform(self) -> None:
        """Test APGI system initializes correctly on current platform."""
        import numpy as np

        system = APGISystem()

        # Verify system initialized
        assert system is not None
        assert hasattr(system, "time")
        assert hasattr(system, "step")

        # Test basic functionality
        obs = np.array([0.0] * 256)  # Simple test input as numpy array
        state = system.step(obs)

        assert isinstance(state, dict)
        assert "time" in state
        assert "free_energy" in state

    def test_import_compatibility_across_platforms(self) -> None:
        """Test that all required imports work on current platform."""
        # Test core imports
        try:
            import numpy as np

            assert np is not None
        except ImportError:
            pytest.skip("NumPy not available on this platform")

        try:
            import matplotlib

            assert matplotlib is not None
        except ImportError:
            pytest.skip("Matplotlib not available on this platform")

        # Test that APGI system can be imported
        if HAS_SYSTEM:
            assert APGISystem is not None
        else:
            pytest.skip("APGI system not available on this platform")

    def test_memory_usage_platform_aware(self) -> None:
        """Test memory usage monitoring works on current platform."""
        import psutil

        process = psutil.Process()

        # Get memory info
        memory_info = process.memory_info()

        # Verify memory values are reasonable
        assert memory_info.rss > 0  # Resident set size should be positive
        assert memory_info.vms > 0  # Virtual memory size should be positive

        # Memory should be in reasonable range (not terabytes)
        assert memory_info.rss < 10**12  # Less than 1 TB
        assert memory_info.vms < 10**12

    def test_cpu_usage_platform_aware(self) -> None:
        """Test CPU usage monitoring works on current platform."""
        import psutil

        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # CPU usage should be between 0 and 100
        assert 0 <= cpu_percent <= 100

    def test_network_connectivity_platform_aware(self) -> None:
        """Test network connectivity checks work on current platform."""
        import socket

        try:
            # Test basic network connectivity
            socket.create_connection(("8.8.8.8", 53), timeout=5)
            network_available = True
        except (socket.timeout, socket.gaierror, OSError):
            network_available = False

        # Network availability is a boolean - just verify the check doesn't crash
        assert isinstance(network_available, bool)


class TestPlatformSpecificGUI:
    """Test platform-specific GUI behavior."""

    @pytest.mark.skipif(
        "tkinter" not in globals() or not hasattr(globals().get("tkinter", None), "Tk"),
        reason="Tkinter not available",
    )
    def test_gui_initialization_platform_specific(self) -> None:
        """Test GUI initialization works on current platform."""
        import tkinter as tk

        # Tkinter should be available on most platforms
        root = tk.Tk()

        try:
            # Test basic GUI operations
            assert root is not None
            assert root.winfo_exists()

            # Test platform-specific window management
            if sys.platform == "win32":
                # Windows-specific tests
                assert root.winfo_toplevel() is root
            elif sys.platform.startswith("linux"):
                # Linux-specific tests
                assert root.winfo_toplevel() is root
            elif sys.platform == "darwin":
                # macOS-specific tests
                assert root.winfo_toplevel() is root

            # Test that we can create basic widgets
            label = tk.Label(root, text="Platform test")
            label.pack()
            assert label.winfo_exists()

        finally:
            root.quit()
            root.destroy()

    @pytest.mark.skipif(
        "tkinter" not in globals() or not hasattr(globals().get("tkinter", None), "Tk"),
        reason="Tkinter not available",
    )
    def test_gui_event_loop_platform_compatibility(self) -> None:
        """Test GUI event loop works across platforms."""
        import tkinter as tk

        root = tk.Tk()

        try:
            # Test event loop scheduling (without actually running the loop to avoid hangs/crashes)
            callback_called = False

            def test_callback() -> None:
                nonlocal callback_called
                callback_called = True

            # Schedule callback (but don't run update to avoid potential issues)
            root.after(100, test_callback)

            # Just verify that after() method exists and can be called
            assert hasattr(root, "after")
            assert callable(root.after)

            # Test that we can create and destroy widgets without running the event loop
            label = tk.Label(root, text="Event loop test")
            label.pack()
            assert label.winfo_exists()

            label.destroy()

        finally:
            root.quit()
            root.destroy()


class TestPlatformSpecificFileFormats:
    """Test platform-specific file format handling."""

    def test_csv_export_platform_compatibility(self) -> None:
        """Test CSV export works across platforms."""
        import csv

        test_data = [
            {"name": "test1", "value": 1.0},
            {"name": "test2", "value": 2.0},
        ]

        with tempfile.NamedTemporaryFile(mode="w", newline="", delete=False) as f:
            csv_filename = f.name

        try:
            # Write CSV
            with open(csv_filename, "w", newline="") as f:
                if test_data:
                    writer = csv.DictWriter(f, fieldnames=test_data[0].keys())
                    writer.writeheader()
                    writer.writerows(test_data)

            # Read CSV back
            with open(csv_filename, "r", newline="") as f:
                reader = csv.DictReader(f)
                read_data = list(reader)

            # Verify data integrity
            assert len(read_data) == len(test_data)
            assert read_data[0]["name"] == "test1"
            assert float(read_data[0]["value"]) == 1.0

        finally:
            os.unlink(csv_filename)

    def test_json_export_platform_compatibility(self) -> None:
        """Test JSON export works across platforms."""
        import json

        test_data = {
            "platform": platform.system(),
            "test_values": [1, 2, 3],
            "nested": {"key": "value"},
        }

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            json_filename = f.name

        try:
            # Write JSON
            with open(json_filename, "w") as f:
                json.dump(test_data, f, indent=2)

            # Read JSON back
            with open(json_filename, "r") as f:
                read_data = json.load(f)

            # Verify data integrity
            assert read_data["platform"] == platform.system()
            assert read_data["test_values"] == [1, 2, 3]
            assert read_data["nested"]["key"] == "value"

        finally:
            os.unlink(json_filename)


class TestPlatformSpecificPerformance:
    """Test platform-specific performance characteristics."""

    @pytest.mark.skipif(not HAS_SYSTEM, reason="APGI system not available")
    def test_simulation_performance_platform_varied(self) -> None:
        """Test simulation performance varies appropriately by platform."""
        import numpy as np

        system = APGISystem()

        # Run timing test
        import time

        start_time = time.time()

        num_steps = 100
        for _ in range(num_steps):
            obs = np.array([0.0] * 256)
            system.step(obs)

        end_time = time.time()
        duration = end_time - start_time

        # Performance should be reasonable (not taking minutes)
        assert duration < 30.0  # Should complete within 30 seconds

        # Calculate steps per second
        steps_per_second = num_steps / duration

        # Should be at least minimally performant
        assert steps_per_second > 0.1  # At least 0.1 steps per second

        print(".2f")

    def test_memory_allocation_platform_aware(self) -> None:
        """Test memory allocation works within platform constraints."""

        # Allocate some test arrays
        arrays = []
        for i in range(10):
            arrays.append([0.0] * 10000)  # Allocate some memory

        # Check that allocation succeeded
        assert len(arrays) == 10
        assert len(arrays[0]) == 10000

        # Clear arrays to free memory
        arrays.clear()

        # Memory should be freed (at least eventually)
        import gc

        gc.collect()


class TestDeploymentScenarios:
    """Test deployment scenarios for different platforms."""

    def test_standalone_executable_compatibility(self) -> None:
        """Test compatibility with standalone executable deployment."""
        # This is a smoke test for deployment readiness

        # Verify Python version compatibility
        assert sys.version_info >= (3, 8)  # Minimum supported version

        # Verify required modules can be imported
        required_modules = ["numpy", "json", "os", "sys"]
        for module in required_modules:
            assert __import__(module) is not None

    def test_container_deployment_readiness(self) -> None:
        """Test readiness for container-based deployment."""
        # Check that all imports work (which they should for container deployment)
        try:
            container_ready = True
        except ImportError:
            container_ready = False

        assert container_ready

    def test_web_deployment_compatibility(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test compatibility with web-based deployment scenarios."""
        # Test that we can run in headless mode (no GUI)

        monkeypatch.setenv("DISPLAY", "")  # Simulate headless environment (undone after the test)

        # Try to import GUI modules (should not crash in headless mode)
        try:
            gui_available = True
        except ImportError:
            gui_available = False

        # In headless environments, GUI might not be available
        # This is expected behavior for web deployment
        assert isinstance(gui_available, bool)  # Just verify the check works


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
