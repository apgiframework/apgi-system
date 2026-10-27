"""Session management GUI for APGI Framework experiments.

Provides session setup, participant management, and session tracking with standardized
UI components and error handling.
"""

import json
import logging
import tkinter as tk
import uuid
from datetime import datetime
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional, Type

# Import standardized GUI utilities
try:
    from .utils.standard_gui import (
        ErrorHandler,
        GUIStyleManager,
        StandardMenuBar,
        StandardWindow,
        ask_yes_no_dialog,
        create_standard_button,
        create_standard_button_frame,
        create_standard_notebook,
        show_error_dialog,
        show_info_dialog,
        show_warning_dialog,
    )
except ImportError:
    # Fallback for direct execution
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from apgi_framework.gui.utils.standard_gui import (
        ErrorHandler,
        GUIStyleManager,
        StandardMenuBar,
        StandardWindow,
        ask_yes_no_dialog,
        create_standard_button,
        create_standard_button_frame,
        create_standard_notebook,
        show_error_dialog,
        show_info_dialog,
        show_warning_dialog,
    )

try:
    from ..data.parameter_estimation_dao import ParameterEstimationDAO
    from ..data.parameter_estimation_models import SessionData
except ImportError:
    # Handle relative import when run directly
    if "SessionData" not in globals():
        SessionData: Optional[Type] = None  # type: ignore[no-redef]
    if "ParameterEstimationDAO" not in globals():
        ParameterEstimationDAO: Optional[Type] = None  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


class SessionSetupManager:
    """Manages session setup and configuration.

    Handles creation, loading, and configuration of experimental sessions.
    """

    def __init__(self, dao: ParameterEstimationDAO):
        """Initialize session setup manager.

        Args:
            dao: Data access object for persistence
        """
        self.dao = dao
        self.current_session: Optional[SessionData] = None

        logger.info("SessionSetupManager initialized")

    def create_session(
        self,
        participant_id: str,
        researcher: str = "",
        protocol_version: str = "1.0.0",
        notes: str = "",
    ) -> SessionData:
        """Create a new experimental session.

        Args:
            participant_id: Participant identifier
            researcher: Researcher name
            protocol_version: Protocol version string
            notes: Session notes

        Returns:
            SessionData: Created session
        """
        session = SessionData(
            session_id=str(uuid.uuid4()),
            participant_id=participant_id,
            session_date=datetime.now(),
            protocol_version=protocol_version,
            completion_status="in_progress",
            researcher=researcher,
            notes=notes,
        )

        # Save to database
        self.dao.create_session(session)
        self.current_session = session

        logger.info(f"Created session {session.session_id} for participant {participant_id}")
        return session

    def load_session(self, session_id: str) -> Optional[SessionData]:
        """Load an existing session.

        Args:
            session_id: Session identifier

        Returns:
            SessionData or None if not found
        """
        session = self.dao.get_session(session_id)

        if session:
            self.current_session = session
            logger.info(f"Loaded session {session_id}")
        else:
            logger.warning(f"Session {session_id} not found")

        return session

    def update_session(self, session: SessionData) -> None:
        """Update session information.

        Args:
            session: Updated session data
        """
        self.dao.update_session(session)
        logger.info(f"Updated session {session.session_id}")

    def end_session(self, session: SessionData) -> None:
        """End a session and mark as completed.

        Args:
            session: Session to end
        """
        session.completion_status = "completed"
        session.total_duration_minutes = (
            datetime.now() - session.session_date
        ).total_seconds() / 60

        self.dao.update_session(session)
        logger.info(f"Ended session {session.session_id}")

    def get_session_summary(self, session: SessionData) -> Dict[str, Any]:
        """Get summary information for a session.

        Args:
            session: Session to summarize

        Returns:
            Dictionary with session summary
        """
        trial_counts = session.get_trial_count_by_task()

        return {
            "session_id": session.session_id,
            "participant_id": session.participant_id,
            "session_date": session.session_date.isoformat(),
            "completion_status": session.completion_status,
            "duration_minutes": session.total_duration_minutes,
            "trial_counts": trial_counts,
            "quality_score": session.session_quality_score,
            "researcher": session.researcher,
        }


class ParticipantManager:
    """Manages participant information and tracking.

    Handles participant registration, session history, and metadata.
    """

    def __init__(self, dao: ParameterEstimationDAO):
        """Initialize participant manager.

        Args:
            dao: Data access object for persistence
        """
        self.dao = dao
        self.participants: Dict[str, Dict[str, Any]] = {}

        logger.info("ParticipantManager initialized")

    def register_participant(
        self, participant_id: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a new participant.

        Args:
            participant_id: Participant identifier
            metadata: Optional participant metadata
        """
        if participant_id in self.participants:
            logger.warning(f"Participant {participant_id} already registered")
            return

        self.participants[participant_id] = {
            "participant_id": participant_id,
            "registration_date": datetime.now(),
            "metadata": metadata or {},
            "sessions": [],
        }

        logger.info(f"Registered participant {participant_id}")

    def get_participant_sessions(self, participant_id: str) -> List[str]:
        """Get all sessions for a participant.

        Args:
            participant_id: Participant identifier

        Returns:
            List of session IDs
        """
        return self.dao.list_sessions(participant_id=participant_id)

    def get_participant_info(self, participant_id: str) -> Optional[Dict[str, Any]]:
        """Get participant information.

        Args:
            participant_id: Participant identifier

        Returns:
            Participant information dictionary or None
        """
        return self.participants.get(participant_id)

    def update_participant_metadata(self, participant_id: str, metadata: Dict[str, Any]) -> None:
        """Update participant metadata.

        Args:
            participant_id: Participant identifier
            metadata: Updated metadata
        """
        if participant_id not in self.participants:
            logger.warning(f"Participant {participant_id} not found")
            return

        self.participants[participant_id]["metadata"].update(metadata)
        logger.info(f"Updated metadata for participant {participant_id}")

    def get_participant_summary(self, participant_id: str) -> Dict[str, Any]:
        """Get summary information for a participant.

        Args:
            participant_id: Participant identifier

        Returns:
            Dictionary with participant summary
        """
        sessions = self.get_participant_sessions(participant_id)
        participant_info = self.get_participant_info(participant_id)

        return {
            "participant_id": participant_id,
            "n_sessions": len(sessions),
            "session_ids": sessions,
            "metadata": (participant_info.get("metadata", {}) if participant_info else {}),
        }


if __name__ == "__main__":
    """Standardized GUI for session management."""

    class SessionManagementGUI(StandardWindow):
        def __init__(self) -> None:
            """Initialize the session management GUI."""
            super().__init__("APGI Session Management", GUIStyleManager.DEFAULT_WINDOW_SIZE)

            # Initialize managers
            self.session_manager = (
                SessionSetupManager(None)  # type: ignore[arg-type]
                if ParameterEstimationDAO is not None
                else None
            )
            self.participant_manager = (
                ParticipantManager(None)  # type: ignore[arg-type]
                if ParameterEstimationDAO is not None
                else None
            )

            # Create menu bar
            self.menu_bar = StandardMenuBar(self.root, self)

            # Create main interface
            self.create_main_interface()

            logger.info("SessionManagementGUI initialized")

        def create_main_interface(self) -> None:
            """Create the main interface components."""
            # Create notebook for tabs
            self.notebook = create_standard_notebook(self.root)

            # Session setup tab
            self.create_session_tab()

            # Participant management tab
            self.create_participant_tab()

            # Session list tab
            self.create_session_list_tab()

            # Button frame
            self.create_button_frame()

        def create_session_tab(self) -> None:
            """Create session setup tab."""
            session_frame = ttk.Frame(self.notebook)
            self.notebook.add(session_frame, text="Session Setup")

            # Session info frame
            info_frame = ttk.LabelFrame(session_frame, text="Session Information", padding=10)
            info_frame.pack(fill="x", padx=10, pady=10)

            # Session ID
            ttk.Label(info_frame, text="Session ID:").grid(row=0, column=0, sticky="w", pady=5)
            self.session_id_var = tk.StringVar()
            ttk.Entry(info_frame, textvariable=self.session_id_var, width=30).grid(
                row=0, column=1, sticky="w", pady=5, padx=5
            )
            create_standard_button(info_frame, "Generate", self.generate_session_id).grid(
                row=0, column=2, pady=5
            )

            # Participant ID
            ttk.Label(info_frame, text="Participant ID:").grid(row=1, column=0, sticky="w", pady=5)
            self.participant_id_var = tk.StringVar()
            ttk.Entry(info_frame, textvariable=self.participant_id_var, width=30).grid(
                row=1, column=1, sticky="w", pady=5, padx=5
            )

            # Session type
            ttk.Label(info_frame, text="Session Type:").grid(row=2, column=0, sticky="w", pady=5)
            self.session_type_var = tk.StringVar(value="parameter_estimation")
            session_type_combo = ttk.Combobox(
                info_frame, textvariable=self.session_type_var, width=28
            )
            session_type_combo["values"] = (
                "parameter_estimation",
                "detection_task",
                "heartbeat_task",
                "falsification_test",
            )
            session_type_combo.grid(row=2, column=1, sticky="w", pady=5, padx=5)

            # Notes
            ttk.Label(info_frame, text="Notes:").grid(row=3, column=0, sticky="w", pady=5)
            self.session_notes_text = tk.Text(info_frame, width=40, height=3)
            self.session_notes_text.grid(row=3, column=1, columnspan=2, sticky="w", pady=5, padx=5)

        def create_participant_tab(self) -> None:
            """Create participant management tab."""
            participant_frame = ttk.Frame(self.notebook)
            self.notebook.add(participant_frame, text="Participant Management")

            # Participant info frame
            info_frame = ttk.LabelFrame(
                participant_frame, text="Participant Information", padding=10
            )
            info_frame.pack(fill="x", padx=10, pady=10)

            # Participant ID
            ttk.Label(info_frame, text="Participant ID:").grid(row=0, column=0, sticky="w", pady=5)
            self.new_participant_id_var = tk.StringVar()
            ttk.Entry(info_frame, textvariable=self.new_participant_id_var, width=30).grid(
                row=0, column=1, sticky="w", pady=5, padx=5
            )

            # Age
            ttk.Label(info_frame, text="Age:").grid(row=1, column=0, sticky="w", pady=5)
            self.age_var = tk.IntVar(value=25)
            ttk.Spinbox(info_frame, from_=18, to=100, textvariable=self.age_var, width=28).grid(
                row=1, column=1, sticky="w", pady=5, padx=5
            )

            # Gender
            ttk.Label(info_frame, text="Gender:").grid(row=2, column=0, sticky="w", pady=5)
            self.gender_var = tk.StringVar(value="other")
            gender_frame = ttk.Frame(info_frame)
            gender_frame.grid(row=2, column=1, sticky="w", pady=5, padx=5)
            ttk.Radiobutton(gender_frame, text="Male", variable=self.gender_var, value="male").pack(
                side="left", padx=5
            )
            ttk.Radiobutton(
                gender_frame, text="Female", variable=self.gender_var, value="female"
            ).pack(side="left", padx=5)
            ttk.Radiobutton(
                gender_frame, text="Other", variable=self.gender_var, value="other"
            ).pack(side="left", padx=5)

            # Button frame for participant
            participant_button_frame = ttk.Frame(info_frame)
            participant_button_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=10)
            create_standard_button(
                participant_button_frame, "Add Participant", self.add_participant
            ).pack(side="left", padx=5)
            create_standard_button(
                participant_button_frame, "View Participants", self.view_participants
            ).pack(side="left", padx=5)

        def create_session_list_tab(self) -> None:
            """Create session list tab."""
            list_frame = ttk.Frame(self.notebook)
            self.notebook.add(list_frame, text="Session List")

            # Create treeview for sessions
            columns = ("Session ID", "Participant", "Type", "Date", "Status")
            self.session_tree = ttk.Treeview(
                list_frame, columns=columns, show="headings", height=15
            )

            for col in columns:
                self.session_tree.heading(col, text=col)
                self.session_tree.column(col, width=150)

            # Scrollbars
            v_scrollbar = ttk.Scrollbar(
                list_frame, orient="vertical", command=self.session_tree.yview
            )
            h_scrollbar = ttk.Scrollbar(
                list_frame, orient="horizontal", command=self.session_tree.xview
            )
            self.session_tree.configure(
                yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set
            )

            # Pack widgets
            self.session_tree.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
            v_scrollbar.grid(row=0, column=1, sticky="ns", pady=10)
            h_scrollbar.grid(row=1, column=0, sticky="ew", padx=10)

            list_frame.grid_rowconfigure(0, weight=1)
            list_frame.grid_columnconfigure(0, weight=1)

            # Add sample data
            self.add_sample_sessions()

        def create_button_frame(self) -> None:
            """Create the button frame."""
            button_frame = create_standard_button_frame(self.root)

            create_standard_button(button_frame, "Create Session", self.create_session).pack(
                side="left", padx=5
            )
            create_standard_button(button_frame, "Save Session", self.save_session).pack(
                side="left", padx=5
            )
            create_standard_button(button_frame, "Load Session", self.load_session).pack(
                side="left", padx=5
            )
            create_standard_button(button_frame, "Export Data", self.export_data).pack(
                side="left", padx=5
            )
            create_standard_button(button_frame, "Refresh", self.refresh_data).pack(
                side="left", padx=5
            )

        def generate_session_id(self) -> None:
            """Generate a unique session ID."""
            session_id = (
                f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            )
            self.session_id_var.set(session_id)

        def add_participant(self) -> None:
            """Add a new participant."""
            participant_id = self.new_participant_id_var.get().strip()
            if not participant_id:
                show_error_dialog("Error", "Please enter a participant ID")
                return

            # Check if participant already exists
            if self.participant_manager and participant_id in self.participant_manager.participants:
                show_warning_dialog("Warning", "Participant already exists")
                return

            # Add participant
            if self.participant_manager:
                try:
                    self.participant_manager.register_participant(
                        participant_id,
                        metadata={
                            "age": self.age_var.get(),
                            "gender": self.gender_var.get(),
                        },
                    )
                    show_info_dialog("Success", f"Participant {participant_id} added successfully")
                    self.new_participant_id_var.set("")
                except Exception as e:
                    show_error_dialog("Error", f"Failed to add participant: {e}")
            else:
                show_warning_dialog("Info", "Participant manager not available")

        def view_participants(self) -> None:
            """View all participants."""
            if not self.participant_manager:
                show_warning_dialog("Info", "Participant manager not available")
                return

            participants = list(self.participant_manager.participants.keys())
            if not participants:
                show_info_dialog("Participants", "No participants found")
                return

            participant_list = "\n".join(participants)
            show_info_dialog("Participants", f"Current participants:\n\n{participant_list}")

        def create_session(self) -> None:
            """Create a new session."""
            session_id = self.session_id_var.get().strip()
            participant_id = self.participant_id_var.get().strip()

            if not session_id:
                show_error_dialog("Error", "Please generate or enter a session ID")
                return

            if not participant_id:
                show_error_dialog("Error", "Please enter a participant ID")
                return

            # Create session
            if self.session_manager:
                try:
                    self.session_manager.create_session(session_id)
                    show_info_dialog("Success", f"Session {session_id} created successfully")
                    self.add_session_to_tree(
                        session_id, participant_id, self.session_type_var.get()
                    )
                except Exception as e:
                    show_error_dialog("Error", f"Failed to create session: {e}")
            else:
                show_warning_dialog("Info", "Session manager not available")

        def save_session(self) -> None:
            """Save session data."""
            filename = filedialog.asksaveasfilename(
                title="Save Session Data",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            )

            if filename:
                try:
                    session_data = {
                        "session_id": self.session_id_var.get(),
                        "participant_id": self.participant_id_var.get(),
                        "session_type": self.session_type_var.get(),
                        "notes": self.session_notes_text.get("1.0", tk.END).strip(),
                        "created_at": datetime.now().isoformat(),
                    }

                    with open(filename, "w") as f:
                        json.dump(session_data, f, indent=2)

                    show_info_dialog("Success", f"Session data saved to {filename}")
                except Exception:
                    ErrorHandler.handle_file_error(filename, "save")

        def load_session(self) -> None:
            """Load session data."""
            filename = filedialog.askopenfilename(
                title="Load Session Data",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            )

            if filename:
                try:
                    with open(filename, "r") as f:
                        session_data = json.load(f)

                    self.session_id_var.set(session_data.get("session_id", ""))
                    self.participant_id_var.set(session_data.get("participant_id", ""))
                    self.session_type_var.set(session_data.get("session_type", ""))
                    self.session_notes_text.delete("1.0", tk.END)
                    self.session_notes_text.insert("1.0", session_data.get("notes", ""))

                    show_info_dialog("Success", f"Session data loaded from {filename}")
                except Exception:
                    ErrorHandler.handle_file_error(filename, "load")

        def export_data(self) -> None:
            """Export session data."""
            show_info_dialog(
                "Export",
                "Export functionality would save all session data to CSV/Excel format",
            )

        def refresh_data(self) -> None:
            """Refresh the session list."""
            # Clear existing items
            for item in self.session_tree.get_children():
                self.session_tree.delete(item)

            # Add sample data again
            self.add_sample_sessions()

        def add_session_to_tree(
            self, session_id: str, participant_id: str, session_type: str
        ) -> None:
            """Add a session to the tree view."""
            status = "Active"
            date = datetime.now().strftime("%Y-%m-%d %H:%M")
            self.session_tree.insert(
                "",
                "end",
                values=(session_id, participant_id, session_type, date, status),
            )

        def add_sample_sessions(self) -> None:
            """Add sample session data."""
            sample_sessions = [
                (
                    "session_20260110_120000_abc123",
                    "P001",
                    "parameter_estimation",
                    "2026-01-10 12:00",
                    "Completed",
                ),
                (
                    "session_20260110_130000_def456",
                    "P002",
                    "detection_task",
                    "2026-01-10 13:00",
                    "Active",
                ),
                (
                    "session_20260110_140000_ghi789",
                    "P003",
                    "heartbeat_task",
                    "2026-01-10 14:00",
                    "Paused",
                ),
            ]

            for session in sample_sessions:
                self.session_tree.insert("", "end", values=session)

        def _confirm_quit(self) -> bool:
            """Confirm before quitting with unsaved changes."""
            return bool(
                ask_yes_no_dialog(
                    "Quit",
                    "Are you sure you want to quit? Any unsaved session data will be lost.",
                )
            )


if __name__ == "__main__":
    # Launch GUI
    try:
        app = SessionManagementGUI()  # type: ignore[no-untyped-call]
        app.run()
    except Exception as e:
        logger.error(f"Error launching Session Management GUI: {e}")
        messagebox.showerror("Error", f"Failed to launch Session Management GUI: {e}")
