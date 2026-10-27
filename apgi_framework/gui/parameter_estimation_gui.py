"""Main parameter estimation experiment control interface.

Provides unified GUI for all three parameter estimation tasks (detection,
heartbeat detection, and dual-modality oddball).
"""

import logging
import threading
import tkinter as tk
from datetime import datetime, timedelta
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk
from typing import Any, List, Optional

# Setup logger
logger = logging.getLogger(__name__)

# Import available modules
from apgi_framework.gui.progress_monitoring import RealTimeProgressMonitor

# Import core modules with corrected paths
try:
    from apgi_framework.research.core_mechanisms.behavioral_tasks import (
        DetectionTask,
        DualModalityOddballTask,
        HeartbeatDetectionTask,
    )

    logger.info("Successfully imported behavioral task classes")
except ImportError as e:
    logger.warning(f"Failed to import behavioral tasks: {e}")

    # Fallback placeholder classes only if not already imported
    if "DetectionTask" not in globals():

        class DetectionTask:  # type: ignore[no-redef]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                logger.warning("Using placeholder DetectionTask class - functionality limited")
                self.task_id = kwargs.get("task_id", "placeholder")
                self.n_trials = kwargs.get("n_trials", 0)

            def run(self) -> bool:
                logger.warning("Placeholder DetectionTask.run() called - no actual execution")
                return False

            def stop(self) -> None:
                logger.warning("Placeholder DetectionTask.stop() called - no actual stopping")

    if "HeartbeatDetectionTask" not in globals():

        class HeartbeatDetectionTask:  # type: ignore[no-redef]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                logger.warning(
                    "Using placeholder HeartbeatDetectionTask class - functionality limited"
                )
                self.task_id = kwargs.get("task_id", "placeholder")
                self.n_trials = kwargs.get("n_trials", 0)

            def run(self) -> bool:
                logger.warning(
                    "Placeholder HeartbeatDetectionTask.run() called - no actual execution"
                )
                return False

            def stop(self) -> None:
                logger.warning(
                    "Placeholder HeartbeatDetectionTask.stop() called - no actual stopping"
                )

    if "DualModalityOddballTask" not in globals():

        class DualModalityOddballTask:  # type: ignore[no-redef]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                logger.warning(
                    "Using placeholder DualModalityOddballTask class - functionality limited"
                )
                self.task_id = kwargs.get("task_id", "placeholder")
                self.n_trials = kwargs.get("n_trials", 0)

            def run(self) -> bool:
                logger.warning(
                    "Placeholder DualModalityOddballTask.run() called - no actual execution"
                )
                return False

            def stop(self) -> None:
                logger.warning(
                    "Placeholder DualModalityOddballTask.stop() called - no actual stopping"
                )

else:
    # Import successful, don't define fallback classes
    pass


try:
    from apgi_framework.data.parameter_estimation_dao import ParameterEstimationDAO
except ImportError:
    if "ParameterEstimationDAO" not in globals():

        class ParameterEstimationDAO:  # type: ignore[no-redef]
            def __init__(self, db_path: str) -> None:
                self.db_path = db_path

            def list_sessions(self, limit: Optional[int] = None) -> List[Any]:
                """Return empty list for fallback implementation."""
                return []

            def get_session(self, session_id: str) -> Optional[Any]:
                """Return None for fallback implementation."""
                return None

            def update_session(self, session_data: Any) -> None:
                """No-op for fallback implementation."""

            def save_session(self, session_data: Any) -> None:
                """No-op for fallback implementation."""

else:
    # Import successful, don't define fallback classes
    pass


try:
    from apgi_framework.data.parameter_estimation_models import SessionData, TaskType
except ImportError:
    if "SessionData" not in globals():

        class SessionData:  # type: ignore[no-redef]
            def __init__(self) -> None:
                self.session_id = ""
                self.participant_id = ""
                self.start_time = datetime.now()
                self.researcher_name = ""
                self.task_type = ""
                self.researcher = ""  # Add researcher attribute

    if "TaskType" not in globals():

        class TaskType:  # type: ignore[no-redef]
            DETECTION = "detection"
            HEARTBEAT = "heartbeat"

else:
    # Import successful, don't define fallback classes
    pass


try:
    logger.info("Importing session_management...")
    from .session_management import ParticipantManager, SessionSetupManager

    logger.info("All GUI component imports successful")
except ImportError:
    if "SessionSetupManager" not in globals():

        class SessionSetupManager:  # type: ignore[no-redef]
            def __init__(self, dao: Any) -> None:
                self.dao = dao

    if "ParticipantManager" not in globals():

        class ParticipantManager:  # type: ignore[no-redef]
            def __init__(self, dao: Any) -> None:
                self.dao = dao

else:
    # Import successful, don't define fallback function
    pass


try:
    from .task_configuration import TaskParameterConfigurator
except ImportError:
    if "TaskParameterConfigurator" not in globals():

        class TaskParameterConfigurator:  # type: ignore[no-redef]
            def __init__(self) -> None:
                pass

else:
    # Import successful, don't define fallback function
    pass


class ParameterEstimationGUI:
    """Main GUI for parameter estimation experiments.

    Provides unified interface for running all three tasks with real-time
    monitoring, session management, and task configuration.
    """

    def __init__(self, db_path: Path, title: str = "APGI Parameter Estimation System"):
        """Initialize parameter estimation GUI.

        Args:
            db_path: Path to database for data storage
            title: Window title
        """
        self.db_path = db_path
        self.dao = ParameterEstimationDAO(db_path)  # Using fallback DAO implementation

        # Create main window
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("1400x900")

        # Session management with fallback implementations
        self.session_manager = SessionSetupManager(self.dao)
        self.participant_manager = ParticipantManager(self.dao)

        # Task configuration with fallback implementation
        self.task_configurator = TaskParameterConfigurator()  # type: ignore[no-untyped-call]

        # Progress monitoring
        self.progress_monitor = RealTimeProgressMonitor()  # type: ignore[no-untyped-call]

        # Current session and tasks
        self.current_session: Optional[SessionData] = None
        self.detection_task: Optional[DetectionTask] = None
        self.heartbeat_task: Optional[HeartbeatDetectionTask] = None
        self.oddball_task: Optional[DualModalityOddballTask] = None

        # Task execution state
        self.task_running = False
        self.task_thread: Optional[threading.Thread] = None
        self.sequential_execution = False
        self.task_stop_requested = False
        self.current_session_id: Optional[str] = None

        # Status variable for UI updates
        self.status_var = tk.StringVar()

        # Build UI
        self._build_ui()

        logger.info("ParameterEstimationGUI initialized")

    def _build_ui(self) -> None:
        """Build the main user interface."""
        # Create menu bar
        self._create_menu_bar()

        # Create main layout with notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create tabs
        self._create_session_tab()
        self._create_detection_task_tab()
        self._create_heartbeat_task_tab()
        self._create_oddball_task_tab()
        self._create_monitoring_tab()

        # Create status bar
        self._create_status_bar()

    def _create_menu_bar(self) -> None:
        """Create menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Session", command=self._new_session)
        file_menu.add_command(label="Load Session", command=self._load_session)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_closing)

        # Tasks menu
        tasks_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tasks", menu=tasks_menu)
        tasks_menu.add_command(label="Run Detection Task", command=self._run_detection_task)
        tasks_menu.add_command(label="Run Heartbeat Task", command=self._run_heartbeat_task)
        tasks_menu.add_command(label="Run Oddball Task", command=self._run_oddball_task)
        tasks_menu.add_separator()
        tasks_menu.add_command(label="Run All Tasks", command=self._run_all_tasks)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)

    def _create_session_tab(self) -> None:
        """Create session setup and management tab."""
        session_frame = ttk.Frame(self.notebook)
        self.notebook.add(session_frame, text="Session Setup")

        # Participant information
        participant_frame = ttk.LabelFrame(
            session_frame, text="Participant Information", padding=10
        )
        participant_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(participant_frame, text="Participant ID:").grid(
            row=0, column=0, sticky=tk.W, pady=2
        )
        self.participant_id_var = tk.StringVar()
        ttk.Entry(participant_frame, textvariable=self.participant_id_var, width=30).grid(
            row=0, column=1, pady=2
        )

        ttk.Label(participant_frame, text="Researcher:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.researcher_var = tk.StringVar()
        ttk.Entry(participant_frame, textvariable=self.researcher_var, width=30).grid(
            row=1, column=1, pady=2
        )

        # Session configuration
        config_frame = ttk.LabelFrame(session_frame, text="Session Configuration", padding=10)
        config_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(config_frame, text="Protocol Version:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.protocol_version_var = tk.StringVar(value="1.0.0")
        ttk.Entry(config_frame, textvariable=self.protocol_version_var, width=20).grid(
            row=0, column=1, pady=2
        )

        # Session controls
        control_frame = ttk.Frame(session_frame, padding=10)
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(control_frame, text="Start New Session", command=self._start_new_session).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(control_frame, text="End Session", command=self._end_session).pack(
            side=tk.LEFT, padx=5
        )

        # Session status
        status_frame = ttk.LabelFrame(session_frame, text="Session Status", padding=10)
        status_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.session_status_text = scrolledtext.ScrolledText(
            status_frame, height=15, state=tk.DISABLED
        )
        self.session_status_text.pack(fill=tk.BOTH, expand=True)

    def _create_detection_task_tab(self) -> None:
        """Create detection task configuration tab."""
        task_frame = ttk.Frame(self.notebook)
        self.notebook.add(task_frame, text="Detection Task")

        # Task parameters
        params_frame = ttk.LabelFrame(task_frame, text="Task Parameters", padding=10)
        params_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(params_frame, text="Modality:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.detection_modality_var = tk.StringVar(value="visual")
        ttk.Combobox(
            params_frame,
            textvariable=self.detection_modality_var,
            values=["visual", "auditory"],
            state="readonly",
            width=20,
        ).grid(row=0, column=1, pady=2)

        ttk.Label(params_frame, text="Number of Trials:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.detection_trials_var = tk.IntVar(value=200)
        ttk.Spinbox(
            params_frame,
            from_=50,
            to=500,
            textvariable=self.detection_trials_var,
            width=20,
        ).grid(row=1, column=1, pady=2)

        # Task controls
        control_frame = ttk.Frame(task_frame, padding=10)
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(control_frame, text="Run Task", command=self._run_detection_task).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(control_frame, text="Stop Task", command=self._stop_current_task).pack(
            side=tk.LEFT, padx=5
        )

    def _create_heartbeat_task_tab(self) -> None:
        """Create heartbeat detection task configuration tab."""
        task_frame = ttk.Frame(self.notebook)
        self.notebook.add(task_frame, text="Heartbeat Task")

        # Task parameters
        params_frame = ttk.LabelFrame(task_frame, text="Task Parameters", padding=10)
        params_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(params_frame, text="Number of Trials:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.heartbeat_trials_var = tk.IntVar(value=60)
        ttk.Spinbox(
            params_frame,
            from_=30,
            to=120,
            textvariable=self.heartbeat_trials_var,
            width=20,
        ).grid(row=0, column=1, pady=2)

        # Task controls
        control_frame = ttk.Frame(task_frame, padding=10)
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(control_frame, text="Run Task", command=self._run_heartbeat_task).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(control_frame, text="Stop Task", command=self._stop_current_task).pack(
            side=tk.LEFT, padx=5
        )

    def _create_oddball_task_tab(self) -> None:
        """Create oddball task configuration tab."""
        task_frame = ttk.Frame(self.notebook)
        self.notebook.add(task_frame, text="Oddball Task")

        # Task parameters
        params_frame = ttk.LabelFrame(task_frame, text="Task Parameters", padding=10)
        params_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(params_frame, text="Number of Trials:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.oddball_trials_var = tk.IntVar(value=120)
        ttk.Spinbox(
            params_frame,
            from_=60,
            to=240,
            textvariable=self.oddball_trials_var,
            width=20,
        ).grid(row=0, column=1, pady=2)

        ttk.Label(params_frame, text="Deviant Probability:").grid(
            row=1, column=0, sticky=tk.W, pady=2
        )
        self.oddball_deviant_prob_var = tk.DoubleVar(value=0.2)
        ttk.Spinbox(
            params_frame,
            from_=0.1,
            to=0.5,
            increment=0.05,
            textvariable=self.oddball_deviant_prob_var,
            width=20,
        ).grid(row=1, column=1, pady=2)

        # Task controls
        control_frame = ttk.Frame(task_frame, padding=10)
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(control_frame, text="Run Task", command=self._run_oddball_task).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(control_frame, text="Stop Task", command=self._stop_current_task).pack(
            side=tk.LEFT, padx=5
        )

    def _create_monitoring_tab(self) -> None:
        """Create real-time monitoring tab."""
        monitor_frame = ttk.Frame(self.notebook)
        self.notebook.add(monitor_frame, text="Monitoring")

        # Progress display
        progress_frame = ttk.LabelFrame(monitor_frame, text="Task Progress", padding=10)
        progress_frame.pack(fill=tk.X, padx=10, pady=5)

        self.progress_bar = ttk.Progressbar(progress_frame, mode="determinate", length=400)
        self.progress_bar.pack(fill=tk.X, pady=5)

        self.progress_label = ttk.Label(progress_frame, text="No task running")
        self.progress_label.pack(pady=5)

        # Data quality display
        quality_frame = ttk.LabelFrame(monitor_frame, text="Data Quality", padding=10)
        quality_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.quality_text = scrolledtext.ScrolledText(quality_frame, height=20, state=tk.DISABLED)
        self.quality_text.pack(fill=tk.BOTH, expand=True)

    def _create_status_bar(self) -> None:
        """Create status bar at bottom of window."""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_label = ttk.Label(status_frame, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.time_label = ttk.Label(status_frame, text="", relief=tk.SUNKEN)
        self.time_label.pack(side=tk.RIGHT)

        self._update_time()

    def _update_time(self) -> None:
        """Update time display in status bar with drift prevention."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.config(text=current_time)

        # Schedule next update to run at the start of the next second
        now = datetime.now()
        next_second = (now.replace(microsecond=0) + timedelta(seconds=1)).timestamp()
        current_timestamp = now.timestamp()
        delay_ms = int((next_second - current_timestamp) * 1000)

        self.root.after(delay_ms, self._update_time)

    def _new_session(self) -> None:
        """Create new session."""
        self._start_new_session()

    def _load_session(self) -> None:
        """Load existing session."""
        try:
            # Get list of available sessions
            sessions = self.dao.list_sessions(limit=50)

            if not sessions:
                messagebox.showinfo("Load Session", "No saved sessions found")
                return

            # Create session selection dialog
            dialog = tk.Toplevel(self.root)
            dialog.title("Load Session")
            dialog.geometry("600x400")
            dialog.transient(self.root)
            dialog.grab_set()

            # Session list
            ttk.Label(dialog, text="Select session to load:").pack(pady=5)

            list_frame = ttk.Frame(dialog)
            list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

            scrollbar = ttk.Scrollbar(list_frame)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            session_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set)
            session_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.config(command=session_listbox.yview)

            # Populate session list with details
            for session_id in sessions:
                try:
                    session_data = self.dao.get_session(session_id)
                    if session_data:
                        display_text = f"{session_id[:8]}... - {session_data.participant_id} - {session_data.start_time.strftime('%Y-%m-%d %H:%M')}"  # type: ignore
                        session_listbox.insert(tk.END, display_text)
                        session_listbox.insert(tk.END, session_id)  # Store full ID as hidden item
                except Exception:
                    session_listbox.insert(tk.END, f"{session_id[:8]}... - (Error loading details)")

            def load_selected() -> None:
                selection = session_listbox.curselection()  # type: ignore[no-untyped-call]
                if not selection:
                    messagebox.showwarning("No Selection", "Please select a session to load")
                    return

                # Get the full session ID (it's the item after the display text)
                selected_index = selection[0]
                if selected_index % 2 == 0:  # Display text
                    session_id = sessions[selected_index // 2]
                else:  # Should not happen, but just in case
                    session_id = sessions[selected_index // 2]

                try:
                    session_data = self.dao.get_session(session_id)
                    if session_data:
                        self._restore_session_state(session_data)
                        dialog.destroy()
                        messagebox.showinfo(
                            "Success",
                            f"Session {session_id[:8]}... loaded successfully",
                        )
                    else:
                        messagebox.showerror("Error", "Failed to load session data")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load session: {str(e)}")

            # Buttons
            button_frame = ttk.Frame(dialog)
            button_frame.pack(pady=10)

            ttk.Button(button_frame, text="Load", command=load_selected).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(
                side=tk.LEFT, padx=5
            )

        except Exception as e:
            logger.error(f"Session loading failed: {e}")
            messagebox.showerror("Error", f"Failed to load session: {str(e)}")

    def _restore_session_state(self, session_data: SessionData) -> None:
        """Restore GUI state from loaded session data."""
        try:
            # Restore basic session info
            self.participant_id_var.set(session_data.participant_id)
            self.researcher_var.set(session_data.researcher)

            # Restore session metadata
            self.current_session_id = session_data.session_id

            # Update status
            self.status_var.set(f"Session loaded: {session_data.session_id[:8]}...")

            # Enable task controls
            self._enable_task_controls()

            # Load task-specific data if available
            if session_data.task_type:  # type: ignore
                self._restore_task_data(session_data)

            logger.info(f"Session state restored for {session_data.session_id}")

        except Exception as e:
            logger.error(f"Failed to restore session state: {e}")
            raise

    def _start_new_session(self) -> None:
        """Start a new parameter estimation session."""
        participant_id = self.participant_id_var.get().strip()
        researcher = self.researcher_var.get().strip()

        if not participant_id:
            messagebox.showerror("Error", "Please enter a participant ID")
            return

        try:
            # Create new session
            self.current_session = self.session_manager.create_session(
                participant_id=participant_id,
                researcher=researcher,
                protocol_version=self.protocol_version_var.get(),
            )

            self._update_session_status(f"Session started: {self.current_session.session_id}")
            self._update_status(f"Session active for participant {participant_id}")

            logger.info(f"Started new session: {self.current_session.session_id}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start session: {str(e)}")
            logger.error(f"Failed to start session: {e}")

    def _end_session(self) -> None:
        """End current session."""
        if not self.current_session:
            messagebox.showwarning("Warning", "No active session")
            return

        if self.task_running:
            messagebox.showwarning("Warning", "Cannot end session while task is running")
            return

        try:
            # Update session status
            self.current_session.completion_status = "completed"
            self.current_session.total_duration_minutes = (
                datetime.now() - self.current_session.session_date
            ).total_seconds() / 60

            # Save session
            self.dao.update_session(self.current_session)

            self._update_session_status(f"Session ended: {self.current_session.session_id}")
            self._update_status("Session ended")

            self.current_session = None

            logger.info("Session ended")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to end session: {str(e)}")
            logger.error(f"Failed to end session: {e}")

    def _run_detection_task(self) -> None:
        """Run detection task."""
        if not self.current_session:
            messagebox.showerror("Error", "Please start a session first")
            return

        if self.task_running:
            messagebox.showwarning("Warning", "A task is already running")
            return

        try:
            # Create detection task
            self.detection_task = DetectionTask(
                task_id=f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                modality=self.detection_modality_var.get(),
                n_trials=self.detection_trials_var.get(),
            )

            # Run task in separate thread
            self.task_running = True
            self.task_thread = threading.Thread(target=self._execute_detection_task)
            self.task_thread.start()

            self._update_status("Running detection task...")
            self.notebook.select(4)  # type: ignore[no-untyped-call]  # Switch to monitoring tab

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start detection task: {str(e)}")
            logger.error(f"Failed to start detection task: {e}")

    def _execute_detection_task(self) -> None:
        """Execute detection task (runs in separate thread)."""
        try:
            if self.detection_task is None:
                raise ValueError("Detection task not initialized")

            success = self.detection_task.run()

            if success:
                self.root.after(
                    0,
                    lambda: self._task_completed("Detection task completed successfully"),
                )
            else:
                self.root.after(0, lambda: self._task_failed("Detection task failed"))

        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda: self._task_failed(f"Detection task error: {error_msg}"))
            logger.error(f"Detection task error: {e}")
        finally:
            self.task_running = False

    def _run_heartbeat_task(self) -> None:
        """Run heartbeat detection task."""
        if not self.current_session:
            messagebox.showerror("Error", "Please start a session first")
            return

        if self.task_running:
            messagebox.showwarning("Warning", "A task is already running")
            return

        try:
            # Create heartbeat task
            self.heartbeat_task = HeartbeatDetectionTask(
                task_id=f"heartbeat_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                n_trials=self.heartbeat_trials_var.get(),
            )

            # Run task in separate thread
            self.task_running = True
            self.task_thread = threading.Thread(target=self._execute_heartbeat_task)
            self.task_thread.start()

            self._update_status("Running heartbeat detection task...")
            self.notebook.select(4)  # type: ignore[no-untyped-call]  # Switch to monitoring tab

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start heartbeat task: {str(e)}")
            logger.error(f"Failed to start heartbeat task: {e}")

    def _execute_heartbeat_task(self) -> None:
        """Execute heartbeat task (runs in separate thread)."""
        try:
            if self.heartbeat_task is None:
                raise ValueError("Heartbeat task not initialized")

            success = self.heartbeat_task.run()

            if success:
                self.root.after(
                    0,
                    lambda: self._task_completed("Heartbeat task completed successfully"),
                )
            else:
                self.root.after(0, lambda: self._task_failed("Heartbeat task failed"))

        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda: self._task_failed(f"Heartbeat task error: {error_msg}"))
            logger.error(f"Heartbeat task error: {e}")
        finally:
            self.task_running = False

    def _run_oddball_task(self) -> None:
        """Run oddball task."""
        if not self.current_session:
            messagebox.showerror("Error", "Please start a session first")
            return

        if self.task_running:
            messagebox.showwarning("Warning", "A task is already running")
            return

        try:
            # Create oddball task
            self.oddball_task = DualModalityOddballTask(
                task_id=f"oddball_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                n_trials=self.oddball_trials_var.get(),
                oddball_probability=self.oddball_deviant_prob_var.get(),
            )

            # Run task in separate thread
            self.task_running = True
            self.task_thread = threading.Thread(target=self._execute_oddball_task)
            self.task_thread.start()

            self._update_status("Running oddball task...")
            self.notebook.select(4)  # type: ignore[no-untyped-call]  # Switch to monitoring tab

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start oddball task: {str(e)}")
            logger.error(f"Failed to start oddball task: {e}")

    def _execute_oddball_task(self) -> None:
        """Execute oddball task (runs in separate thread)."""
        try:
            if self.oddball_task is None:
                raise ValueError("Oddball task not initialized")

            success = self.oddball_task.run()

            if success:
                self.root.after(
                    0,
                    lambda: self._task_completed("Oddball task completed successfully"),
                )
            else:
                self.root.after(0, lambda: self._task_failed("Oddball task failed"))

        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda: self._task_failed(f"Oddball task error: {error_msg}"))
            logger.error(f"Oddball task error: {e}")
        finally:
            self.task_running = False

    def _run_all_tasks(self) -> None:
        """Run all three tasks sequentially."""
        if not self.current_session:
            messagebox.showerror("Error", "Please start a session first")
            return

        if self.task_running:
            messagebox.showwarning("Warning", "A task is already running")
            return

        # Confirm sequential execution
        result = messagebox.askyesno(
            "Run All Tasks",
            "This will run all three tasks sequentially:\n\n"
            "1. Detection Task\n"
            "2. Heartbeat Detection Task\n"
            "3. Dual-Modality Oddball Task\n\n"
            "This may take 30-45 minutes. Continue?",
        )

        if not result:
            return

        # Start sequential execution in background thread
        self.task_running = True
        self.sequential_execution = True
        self.task_thread = threading.Thread(target=self._execute_sequential_tasks)
        self.task_thread.start()

    def _stop_current_task(self) -> None:
        """Stop currently running task."""
        if not self.task_running:
            messagebox.showinfo("Info", "No task is currently running")
            return

        # Confirm stopping
        result = messagebox.askyesno(
            "Stop Task",
            "Are you sure you want to stop the current task?\n\n" "Any unsaved data will be lost.",
        )

        if not result:
            return

        try:
            # Set stop flag for tasks to check
            self.task_stop_requested = True

            # Update status
            self.status_var.set("Stopping task...")

            # Give tasks a moment to clean up
            if hasattr(self, "detection_task") and self.detection_task:
                self.detection_task.stop()
            if hasattr(self, "heartbeat_task") and self.heartbeat_task:
                self.heartbeat_task.stop()
            if hasattr(self, "oddball_task") and self.oddball_task:
                self.oddball_task.stop()

            # Wait for thread to finish (with timeout)
            if self.task_thread and self.task_thread.is_alive():
                self.task_thread.join(timeout=5.0)

            # Reset state
            self.task_running = False
            self.sequential_execution = False
            self.task_stop_requested = False

            self.status_var.set("Task stopped")
            messagebox.showinfo("Success", "Task stopped successfully")

        except Exception as e:
            logger.error(f"Error stopping task: {e}")
            messagebox.showerror("Error", f"Failed to stop task: {str(e)}")

    def _execute_sequential_tasks(self) -> None:
        """Execute all three tasks sequentially."""
        tasks = [
            ("Detection Task", self._run_detection_task),
            ("Heartbeat Detection Task", self._run_heartbeat_task),
            ("Dual-Modality Oddball Task", self._run_oddball_task),
        ]

        for task_name, task_method in tasks:
            if self.task_stop_requested:
                logger.info(f"Sequential execution stopped before {task_name}")
                break

            try:
                logger.info(f"Starting {task_name}")
                self.root.after(0, lambda: self.status_var.set(f"Running {task_name}..."))

                # Run the task
                task_method()

                logger.info(f"Completed {task_name}")

                def set_status(tn: str = task_name) -> None:
                    self.status_var.set(f"Completed {tn}")

                self.root.after(0, set_status)

                # Brief pause between tasks
                import time

                time.sleep(2)

            except Exception as e:
                logger.error(f"Error in {task_name}: {e}")

                def show_error(tn: str = task_name, err: str = str(e)) -> None:
                    messagebox.showerror(f"{tn} Error", f"Error in {tn}: {err}")

                self.root.after(0, show_error)
                break

        # Reset sequential execution state
        self.sequential_execution = False
        self.task_running = False
        self.task_stop_requested = False

        self.root.after(0, lambda: self.status_var.set("All tasks completed"))
        self.root.after(
            0,
            lambda: messagebox.showinfo("Complete", "Sequential task execution finished"),
        )

    def _task_completed(self, message: str) -> None:
        """Handle task completion."""
        self._update_status(message)
        self._update_quality_display(message)
        messagebox.showinfo("Task Complete", message)

    def _task_failed(self, message: str) -> None:
        """Handle task failure."""
        self._update_status(message)
        self._update_quality_display(message)
        messagebox.showerror("Task Failed", message)

    def _update_session_status(self, message: str) -> None:
        """Update session status display."""
        self.session_status_text.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.session_status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.session_status_text.see(tk.END)
        self.session_status_text.config(state=tk.DISABLED)

    def _update_status(self, message: str) -> None:
        """Update status bar."""
        self.status_label.config(text=message)
        logger.info(message)

    def _update_quality_display(self, message: str) -> None:
        """Update data quality display."""
        self.quality_text.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.quality_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.quality_text.see(tk.END)
        self.quality_text.config(state=tk.DISABLED)

    def _show_about(self) -> None:
        """Show about dialog."""
        about_text = """APGI Parameter Estimation System
        
Version 1.0.0

A comprehensive system for estimating APGI framework parameters
through behavioral tasks and neural measurements.

© 2024 APGI Research Team"""

        messagebox.showinfo("About", about_text)

    def _on_closing(self) -> None:
        """Handle window closing."""
        if self.task_running:
            if not messagebox.askokcancel(
                "Quit", "A task is running. Are you sure you want to quit?"
            ):
                return

        if self.current_session and self.current_session.completion_status == "in_progress":
            if messagebox.askyesno("End Session", "End current session before closing?"):
                self._end_session()

        self.root.destroy()

    def run(self) -> None:
        """Run the GUI application."""
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.root.mainloop()

    def _enable_task_controls(self) -> None:
        """Enable task control buttons when session is active."""
        # This would enable task buttons in the UI
        # Implementation depends on the specific UI structure

    def _restore_task_data(self, session_data: SessionData) -> None:
        """Restore task-specific data from loaded session."""
        try:
            # Restore task-specific state based on session data
            if session_data.detection_trials:
                logger.info(f"Restored {len(session_data.detection_trials)} detection trials")

            if session_data.heartbeat_trials:
                logger.info(f"Restored {len(session_data.heartbeat_trials)} heartbeat trials")

            if session_data.oddball_trials:
                logger.info(f"Restored {len(session_data.oddball_trials)} oddball trials")

            # Update UI to reflect loaded data
            # This would update progress displays, charts, etc.

        except Exception as e:
            logger.error(f"Failed to restore task data: {e}")
            raise


def launch_gui(db_path: Path) -> None:
    """Launch the parameter estimation GUI.

    Args:
        db_path: Path to database file
    """
    gui = ParameterEstimationGUI(db_path)
    gui.run()


if __name__ == "__main__":
    from pathlib import Path

    db_path = Path("data/parameter_estimation.db")
    db_path.parent.mkdir(exist_ok=True)
    launch_gui(db_path)
