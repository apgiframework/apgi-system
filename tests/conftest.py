"""
Shared test fixtures and configuration for APGI system tests.

This module provides common fixtures used across unit, property-based,
and integration tests.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Generator

import matplotlib
import numpy as np
import pytest
import yaml
from hypothesis import HealthCheck, settings
from numpy.typing import NDArray

from apgi_framework.interoception.body_model import BodyModel
from apgi_framework.system import APGISystem

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure xvfb for GUI tests
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

# Set up headless environment BEFORE any GUI imports
# Force headless mode for GUI tests
if not os.environ.get("DISPLAY") or os.environ.get("CI"):
    os.environ.setdefault("DISPLAY", ":99")

# Set matplotlib to non-interactive backend BEFORE importing pyplot or tkagg
matplotlib.use("Agg", force=True)

# Set environment variables for testing
os.environ["ENVIRONMENT"] = "testing"
os.environ["JWT_SECRET_KEY"] = (
    "testing-secret-key-at-least-64-characters-long-for-production-compliance-verification"
)

# Configure Hypothesis profiles
settings.register_profile(
    "ci",
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
    print_blob=True,
)

settings.register_profile(
    "dev",
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
    print_blob=True,
)

settings.register_profile(
    "thorough",
    max_examples=1000,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
    print_blob=True,
)

# Load the appropriate profile from environment or default to 'dev'
profile = os.getenv("HYPOTHESIS_PROFILE", "dev")
settings.load_profile(profile)


# GUI test configuration
def pytest_configure(config):
    """Configure pytest for GUI tests."""
    config.addinivalue_line("markers", "gui: mark test as requiring GUI/display")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle GUI tests."""
    # Add xvfb marker to GUI tests automatically
    for item in items:
        if "gui" in item.keywords or "GUI" in str(item.fspath):
            item.add_marker(pytest.mark.xvfb)
            item.add_marker(pytest.mark.gui)


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment."""
    # Set environment variables for headless testing if needed
    os.environ.setdefault("DISPLAY", ":99")

    # Set matplotlib to non-interactive backend for headless testing
    try:
        matplotlib.use("Agg")
    except ImportError:
        pass


@pytest.fixture(autouse=True)
def no_blocking_dialogs(monkeypatch):
    """Replace tkinter message boxes with non-blocking stubs.

    A real modal dialog never returns on a headless CI runner, so a test that
    reaches ``messagebox.showwarning`` would hang until the job is cancelled.
    Tests that assert on a dialog patch it themselves, which overrides this.
    """
    try:
        from tkinter import messagebox
    except ImportError:
        yield
        return

    for name in ("showinfo", "showwarning", "showerror"):
        monkeypatch.setattr(messagebox, name, lambda *a, **k: "ok")
    for name in ("askyesno", "askokcancel", "askretrycancel"):
        monkeypatch.setattr(messagebox, name, lambda *a, **k: True)
    monkeypatch.setattr(messagebox, "askquestion", lambda *a, **k: "yes")
    monkeypatch.setattr(messagebox, "askyesnocancel", lambda *a, **k: True)
    yield


@pytest.fixture(autouse=True)
def cleanup_gui_state():
    """Clean up GUI state after each test to prevent segfaults."""
    yield
    # Cleanup after test
    try:
        import matplotlib.pyplot as plt

        plt.close("all")
        plt.clf()
    except Exception:
        pass

    # Force Tkinter cleanup
    try:
        import tkinter as tk

        # Destroy any lingering root windows
        for widget in tk._default_root.children.values() if tk._default_root else []:
            try:
                widget.destroy()
            except Exception:
                pass
        if tk._default_root:
            try:
                tk._default_root.destroy()
            except Exception:
                pass
            tk._default_root = None
    except Exception:
        pass

    # Force garbage collection to clean up resources
    import gc

    gc.collect()


@pytest.fixture(scope="session", autouse=True)
def cleanup_session_gui():
    """Final cleanup after all tests complete."""
    yield
    # Final matplotlib cleanup
    try:
        import matplotlib.pyplot as plt

        plt.close("all")
    except Exception:
        pass

    # Final Tkinter cleanup
    try:
        import tkinter as tk

        if tk._default_root:
            try:
                tk._default_root.quit()
                tk._default_root.destroy()
            except Exception:
                pass
    except Exception:
        pass


@pytest.fixture
def config() -> Dict[str, Any]:
    """
    Load default configuration from apgi_framework/resources/config/default.yaml.

    Returns
    -------
    dict
        Configuration dictionary with all system parameters.
    """
    config_path = (
        Path(__file__).parent.parent / "apgi_framework" / "resources" / "config" / "default.yaml"
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        return config if isinstance(config, dict) else {}


@pytest.fixture
def apgi_framework() -> Generator[APGISystem, None, None]:
    """
    Provide a fresh APGI system instance.

    Yields
    ------
    APGISystem
        A newly initialized APGI system instance.

    Notes
    -----
    The system is reset after the test completes to ensure clean state.
    """
    from apgi_framework.system import APGISystem

    config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    system = APGISystem(str(config_path))
    yield system
    system.reset()


@pytest.fixture
def body_model(config: Dict[str, Any]) -> BodyModel:
    """
    Provide a body model instance.

    Parameters
    ----------
    config : dict
        Configuration dictionary (from config fixture).

    Returns
    -------
    BodyModel
        A newly initialized body model instance.
    """
    from apgi_framework.interoception.body_model import BodyModel

    return BodyModel(config)


@pytest.fixture
def random_observation() -> NDArray[np.float64]:
    """
    Generate a random observation vector.

    Returns
    -------
    np.ndarray
        Random observation vector of shape (256,) with values ~ N(0, 0.5^2).

    Notes
    -----
    The observation is scaled to have moderate variance (0.5) to represent
    typical sensory input magnitudes. Dimension matches the sensory level
    in the hierarchy configuration (256 nodes).
    """
    return np.random.randn(256) * 0.5


@pytest.fixture
def random_body_state() -> Dict[str, float]:
    """
    Generate a random but physiologically valid body state.

    Returns
    -------
    dict
        Dictionary containing physiological variables within valid ranges:
        - heart_rate: 60-100 bpm (normal resting range)
        - cortisol: 8-15 μg/dL (normal range)
        - temperature: 36.5-37.5°C (normal range)
        - glucose: 4.0-6.0 mmol/L (normal fasting range)
        - respiration: 12-18 breaths/min (normal range)
        - systolic_bp: 110-130 mmHg (normal range)

    Notes
    -----
    All values are uniformly sampled within physiologically plausible ranges
    to ensure valid test inputs.
    """
    return {
        "heart_rate": np.random.uniform(60, 100),
        "cortisol": np.random.uniform(8, 15),
        "temperature": np.random.uniform(36.5, 37.5),
        "glucose": np.random.uniform(4.0, 6.0),
        "respiration": np.random.uniform(12, 18),
        "systolic_bp": np.random.uniform(110, 130),
    }


# Database fixtures for API tests
@pytest.fixture
def db() -> Generator[Any, None, None]:
    """
    Provide a test database session.

    Creates an in-memory SQLite database for testing with thread-safety enabled.

    Yields
    ------
    Session
        SQLAlchemy database session for testing.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from api.database.models import Base

    # Create in-memory SQLite database for testing with thread-safety
    # check_same_thread=False allows the connection to be used across threads
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)

    # Create session
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    yield session

    # Cleanup
    session.close()
    Base.metadata.drop_all(engine)


# Additional fixtures for comprehensive testing


@pytest.fixture
def valid_belief_state() -> NDArray[np.float64]:
    """
    Provide a valid normalized belief state.

    Returns
    -------
    np.ndarray
        Probability distribution that sums to 1.0, shape (10,).
    """
    beliefs = np.random.rand(10)
    return beliefs / beliefs.sum()  # type: ignore[no-any-return]


@pytest.fixture
def valid_precision() -> float:
    """
    Provide a valid precision value.

    Returns
    -------
    float
        Precision value in range [0.1, 10.0].
    """
    return np.random.uniform(0.1, 10.0)


@pytest.fixture
def empty_input_cases():
    """
    Provide various empty input cases for edge case testing.

    Returns
    -------
    dict
        Dictionary of empty input types.
    """
    return {
        "empty_list": [],
        "empty_array": np.array([]),
        "empty_dict": {},
        "empty_string": "",
        "none_value": None,
        "zero_array": np.zeros(0),
    }


@pytest.fixture
def boundary_values() -> Dict[str, Any]:
    """
    Provide boundary value test cases.

    Returns
    -------
    dict
        Dictionary of boundary values for different types.
    """
    return {
        "zero": 0,
        "one": 1,
        "negative_one": -1,
        "small_positive": 1e-10,
        "small_negative": -1e-10,
        "large_positive": 1e10,
        "large_negative": -1e10,
        "max_float": np.finfo(np.float64).max,
        "min_float": np.finfo(np.float64).min,
        "epsilon": np.finfo(np.float64).eps,
    }


@pytest.fixture
def invalid_inputs() -> Dict[str, Any]:
    """
    Provide invalid input cases for error handling tests.

    Returns
    -------
    dict
        Dictionary of invalid input types.
    """
    return {
        "nan": float("nan"),
        "inf": float("inf"),
        "neg_inf": float("-inf"),
        "wrong_type_str": "not_a_number",
        "wrong_type_list": ["a", "b", "c"],
        "negative_precision": -1.0,
        "out_of_range": 1000.0,
    }


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory: Any) -> Path:
    """
    Provide a temporary directory for test data.

    Parameters
    ----------
    tmp_path_factory : pytest fixture
        Factory for creating temporary directories.

    Returns
    -------
    Path
        Path to temporary test data directory.
    """
    return tmp_path_factory.mktemp("test_data")  # type: ignore[no-any-return]


@pytest.fixture
def mock_config_minimal() -> Dict[str, Any]:
    """
    Provide a minimal valid configuration for testing.

    Returns
    -------
    dict
        Minimal configuration with only required fields.
    """
    return {
        "system": {
            "timestep_ms": 1.0,
            "random_seed": 42,
        },
        "active_inference": {
            "learning_rate": 0.01,
            "precision_init": 1.0,
        },
    }


@pytest.fixture
def sample_time_series() -> NDArray[np.float64]:
    """
    Generate a sample time series for testing temporal dynamics.

    Returns
    -------
    np.ndarray
        Time series data of shape (100, 10).
    """
    t = np.linspace(0, 10, 100)
    # Create multi-dimensional time series with different frequencies
    data = np.column_stack(
        [np.sin(2 * np.pi * f * t) for f in [0.5, 1.0, 2.0, 3.0, 5.0]]
        + [np.cos(2 * np.pi * f * t) for f in [0.5, 1.0, 2.0, 3.0, 5.0]]
    )
    return data


@pytest.fixture
def performance_timer() -> Any:
    """
    Provide a context manager for timing test execution.

    Yields
    ------
    callable
        Context manager that measures execution time.
    """
    import time
    from contextlib import contextmanager

    @contextmanager
    def timer() -> Any:
        start = time.perf_counter()
        yield lambda: time.perf_counter() - start

    return timer
