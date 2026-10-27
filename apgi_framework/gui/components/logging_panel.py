"""Logging Panel for APGI Framework GUI.

Extracted from the monolithic GUI to provide a focused component
for system logging and message display.
"""

import json
import logging
import queue
import sys
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any, Callable, Dict, List, Optional

import customtkinter as ctk

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    pass  # Removed unused import
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("logging_panel")
    logger.warning(f"Could not import standardized logging: {e}")

logger = logging.getLogger("logging_panel")


class LogHandler(logging.Handler):
    """Custom logging handler that sends logs to a queue for GUI display."""

    def __init__(self, log_queue: queue.Queue[Any]) -> None:
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: Any) -> None:
        """Emit a log record to the queue."""
        try:
            log_entry = self.format(record)
            self.log_queue.put(log_entry)
        except Exception:
            # Don't let logging errors crash the application
            pass


class LoggingPanel(ctk.CTkFrame):
    """Panel for displaying system logs and messages.

    Provides comprehensive logging capabilities including real-time display,
    log level filtering, search functionality, and export options.
    """

    def __init__(self, parent: Any, max_log_lines: int = 1000) -> None:
        """Initialize the logging panel.

        Args:
            parent: Parent widget
            max_log_lines: Maximum number of log lines to display
        """
        super().__init__(parent)

        self.max_log_lines = max_log_lines
        self.log_queue: queue.Queue[Any] = queue.Queue()
        self.log_handler: Optional[LogHandler] = None
        self.is_polling = True

        # Log filtering
        self.current_log_level = logging.INFO
        self.log_filter_text = ""

        # Log statistics
        self.log_counts = {
            "DEBUG": 0,
            "INFO": 0,
            "WARNING": 0,
            "ERROR": 0,
            "CRITICAL": 0,
        }

        # Callbacks for external components
        self.on_log_added: Optional[Callable] = None
        self.on_log_cleared: Optional[Callable] = None

        # Color scheme for log levels
        self.log_colors = {
            "DEBUG": "#808080",  # Gray
            "INFO": "#2E8B57",  # Sea green
            "WARNING": "#FF8C00",  # Dark orange
            "ERROR": "#DC143C",  # Crimson
            "CRITICAL": "#8B0000",  # Dark red
        }

        self._create_widgets()
        self._setup_logging()
        self._start_log_polling()

        logger.info("LoggingPanel initialized")

    def _create_widgets(self) -> None:
        """Create logging panel widgets."""
        # Create scrollable frame
        self.scrollable_frame = ctk.CTkScrollableFrame(self)
        self.scrollable_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Control section
        self._create_control_section()

        # Filter section
        self._create_filter_section()

        # Statistics section
        self._create_statistics_section()

        # Log display section
        self._create_log_display_section()

    def _create_control_section(self) -> None:
        """Create control buttons section."""
        control_frame = ctk.CTkFrame(self.scrollable_frame)
        control_frame.pack(fill="x", padx=5, pady=5)

        # Control buttons
        self.clear_btn = ctk.CTkButton(
            control_frame,
            text="Clear Logs",
            command=self._clear_logs,
            fg_color=self.log_colors["WARNING"],
        )
        self.clear_btn.pack(side="left", padx=2, pady=2)

        self.save_btn = ctk.CTkButton(
            control_frame,
            text="Save Logs",
            command=self._save_logs,
            fg_color=self.log_colors["INFO"],
        )
        self.save_btn.pack(side="left", padx=2, pady=2)

        self.export_btn = ctk.CTkButton(
            control_frame,
            text="Export JSON",
            command=self._export_json,
            fg_color=self.log_colors["DEBUG"],
        )
        self.export_btn.pack(side="left", padx=2, pady=2)

        # Pause/Resume button
        self.pause_btn = ctk.CTkButton(
            control_frame,
            text="Pause",
            command=self._toggle_polling,
            fg_color=self.log_colors["WARNING"],
        )
        self.pause_btn.pack(side="left", padx=2, pady=2)

        # Log level selection
        ctk.CTkLabel(control_frame, text="Level:").pack(side="right", padx=(10, 2))

        self.log_level_var = tk.StringVar(value="INFO")
        self.log_level_combo = ctk.CTkOptionMenu(
            control_frame,
            variable=self.log_level_var,
            values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            width=10,
        )
        self.log_level_combo.pack(side="right", padx=2)
        self.log_level_combo.configure(command=lambda choice: self._change_log_level(choice))

    def _create_filter_section(self) -> None:
        """Create log filtering section."""
        filter_frame = ctk.CTkFrame(self.scrollable_frame)
        filter_frame.pack(fill="x", padx=5, pady=5)

        # Search/filter entry
        ctk.CTkLabel(filter_frame, text="Filter:").pack(side="left", padx=5)

        self.filter_var = tk.StringVar()
        self.filter_entry = ctk.CTkEntry(filter_frame, textvariable=self.filter_var, width=30)
        self.filter_entry.pack(side="left", padx=5, fill="x", expand=True)

        # Filter button
        self.filter_btn = ctk.CTkButton(
            filter_frame, text="Apply", command=self._apply_filter, width=10
        )
        self.filter_btn.pack(side="left", padx=5)

        # Clear filter button
        self.clear_filter_btn = ctk.CTkButton(
            filter_frame, text="Clear", command=self._clear_filter, width=10
        )
        self.clear_filter_btn.pack(side="left", padx=2)

        # Bind Enter key to filter
        self.filter_entry.bind("<Return>", lambda e: self._apply_filter())

    def _create_statistics_section(self) -> None:
        """Create log statistics section."""
        stats_frame = ctk.CTkFrame(self.scrollable_frame)
        stats_frame.pack(fill="x", padx=5, pady=5)

        stats_title = ctk.CTkLabel(
            stats_frame,
            text="Log Statistics",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        stats_title.pack(padx=10, pady=(5, 2))

        # Statistics display
        self.stats_text = tk.Text(stats_frame, height=3, wrap="word", font=("Courier", 9))
        stats_scrollbar = ctk.CTkScrollbar(stats_frame, command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=stats_scrollbar.set)

        self.stats_text.pack(side="left", fill="x", expand=True, padx=(5, 0), pady=2)
        stats_scrollbar.pack(side="right", fill="y", padx=(0, 5), pady=2)

        # Configure text tags
        for level, color in self.log_colors.items():
            self.stats_text.tag_configure(level.lower(), foreground=color)

        self._update_statistics_display()

    def _create_log_display_section(self) -> None:
        """Create log display section."""
        log_frame = ctk.CTkFrame(self.scrollable_frame)
        log_frame.pack(fill="both", expand=True, padx=5, pady=5)

        log_title = ctk.CTkLabel(
            log_frame,
            text="System Logs",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        log_title.pack(padx=10, pady=(5, 2))

        # Log text widget with scrollbar
        log_container = ctk.CTkFrame(log_frame)
        log_container.pack(fill="both", expand=True, padx=5, pady=5)

        self.log_text = tk.Text(
            log_container,
            wrap="word",
            height=15,
            font=("Courier", 9),
            bg="#1a1a1a",
            fg="#ffffff",
        )
        log_scrollbar = ctk.CTkScrollbar(log_container, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)

        self.log_text.pack(side="left", fill="both", expand=True, padx=(5, 0), pady=5)
        log_scrollbar.pack(side="right", fill="y", padx=(0, 5), pady=5)

        # Configure text tags for different log levels
        for level, color in self.log_colors.items():
            self.log_text.tag_configure(level.lower(), foreground=color)
        self.log_text.tag_configure("timestamp", foreground="#888888")
        self.log_text.tag_configure("logger_name", foreground="#aaaaaa")

        # Add context menu
        self._create_context_menu()

    def _create_context_menu(self) -> None:
        """Create context menu for log text widget."""
        self.context_menu = tk.Menu(self.log_text, tearoff=0)
        self.context_menu.add_command(label="Copy Selected", command=self._copy_selected)
        self.context_menu.add_command(label="Copy All", command=self._copy_all)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Find...", command=self._find_text)
        self.context_menu.add_command(label="Clear Selection", command=self._clear_selection)

        # Bind right-click
        self.log_text.bind("<Button-3>", self._show_context_menu)

    def _show_context_menu(self, event: Any) -> None:
        """Show context menu."""
        try:
            self.context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.context_menu.grab_release()

    def _setup_logging(self) -> None:
        """Setup logging handler."""
        from typing import cast

        self.log_handler = LogHandler(self.log_queue)
        log_handler = cast(LogHandler, self.log_handler)
        log_handler.setLevel(self.current_log_level)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        log_handler.setFormatter(formatter)

        # Add handler to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(log_handler)
        root_logger.setLevel(self.current_log_level)

        logger.info("Logging handler configured")

    def _start_log_polling(self) -> None:
        """Start polling for log messages."""
        self._poll_logs()

    def _poll_logs(self) -> None:
        """Poll for new log messages."""
        if not self.is_polling:
            # Schedule next poll even when paused to check for resume
            self.after(1000, self._poll_logs)
            return

        try:
            # Process all available log messages
            while True:
                log_message = self.log_queue.get_nowait()
                self._process_log_message(log_message)
        except queue.Empty:
            pass

        # Schedule next poll
        self.after(100, self._poll_logs)

    def _process_log_message(self, message: str) -> None:
        """Process a log message and display it if it meets criteria."""
        try:
            # Parse log message
            parts = message.split(" - ")
            if len(parts) >= 4:
                timestamp, logger_name, level, log_content = (
                    parts[0],
                    parts[1],
                    parts[2],
                    " - ".join(parts[3:]),
                )

                # Check log level
                level_numeric = getattr(logging, level, logging.INFO)
                if level_numeric >= self.current_log_level:
                    # Check filter
                    if not self.log_filter_text or self.log_filter_text.lower() in message.lower():
                        self._append_formatted_log(timestamp, logger_name, level, log_content)

                        # Update statistics
                        self.log_counts[level] = self.log_counts.get(level, 0) + 1
                        self._update_statistics_display()

                        # Notify log added
                        if self.on_log_added:
                            self.on_log_added(message)
        except Exception as e:
            # Don't let logging errors crash the application
            logger.error(f"Error processing log message: {e}")

    def _append_formatted_log(
        self, timestamp: str, logger_name: str, level: str, content: str
    ) -> None:
        """Append a formatted log message to the display."""
        # Format the log message with colors
        self.log_text.insert("end", timestamp, "timestamp")
        self.log_text.insert("end", " - ")
        self.log_text.insert("end", logger_name, "logger_name")
        self.log_text.insert("end", " - ")
        self.log_text.insert("end", level, level.lower())
        self.log_text.insert("end", " - ")
        self.log_text.insert("end", content + "\n")

        # Auto-scroll to bottom
        self.log_text.see("end")

        # Limit log size
        self._limit_log_size()

    def _limit_log_size(self) -> None:
        """Limit the number of lines in the log display."""
        try:
            lines = int(self.log_text.index("end-1c").split(".")[0])
            if lines > self.max_log_lines:
                # Remove oldest lines
                lines_to_remove = lines - self.max_log_lines
                self.log_text.delete("1.0", f"{lines_to_remove + 1}.0")
        except Exception:
            pass  # Don't let errors in log management crash the app

    def _clear_logs(self) -> None:
        """Clear all log messages."""
        if messagebox.askyesno("Clear Logs", "Clear all log messages? This cannot be undone."):
            self.log_text.delete("1.0", "end")

            # Reset statistics
            for level in self.log_counts:
                self.log_counts[level] = 0
            self._update_statistics_display()

            # Notify log cleared
            if self.on_log_cleared:
                self.on_log_cleared()

            logger.info("Logs cleared by user")

    def _save_logs(self) -> None:
        """Save logs to file."""
        try:
            filename = filedialog.asksaveasfilename(
                title="Save Logs",
                defaultextension=".log",
                filetypes=[
                    ("Log files", "*.log"),
                    ("Text files", "*.txt"),
                    ("All files", "*.*"),
                ],
            )

            if filename:
                with open(filename, "w") as f:
                    f.write(self.log_text.get("1.0", "end"))
                messagebox.showinfo("Success", f"Logs saved to {filename}")
                logger.info(f"Logs saved to {filename}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save logs: {e}")
            logger.error(f"Failed to save logs: {e}")

    def _export_json(self) -> None:
        """Export logs as structured JSON."""
        try:
            filename = filedialog.asksaveasfilename(
                title="Export Logs as JSON",
                defaultextension=".json",
                filetypes=[
                    ("JSON files", "*.json"),
                    ("All files", "*.*"),
                ],
            )

            if filename:
                # Parse log entries and export as JSON
                log_entries = self._parse_log_entries()

                export_data = {
                    "export_timestamp": datetime.now().isoformat(),
                    "log_level_filter": self.log_level_var.get(),
                    "text_filter": self.log_filter_text,
                    "statistics": self.log_counts,
                    "total_entries": len(log_entries),
                    "logs": log_entries,
                }

                with open(filename, "w") as f:
                    json.dump(export_data, f, indent=2)

                messagebox.showinfo("Success", f"Logs exported to {filename}")
                logger.info(f"Logs exported to {filename}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to export logs: {e}")
            logger.error(f"Failed to export logs: {e}")

    def _parse_log_entries(self) -> List[Dict[str, str]]:
        """Parse log entries from the text widget."""
        entries = []
        log_content = self.log_text.get("1.0", "end").strip()

        for line in log_content.split("\n"):
            if line.strip():
                parts = line.split(" - ")
                if len(parts) >= 4:
                    entries.append(
                        {
                            "timestamp": parts[0],
                            "logger": parts[1],
                            "level": parts[2],
                            "message": " - ".join(parts[3:]),
                        }
                    )

        return entries

    def _change_log_level(self, level: str) -> None:
        """Change logging level."""
        try:
            self.current_log_level = getattr(logging, level)
            from typing import cast

            log_handler = cast(logging.Handler, self.log_handler)
            log_handler.setLevel(self.current_log_level)

            # Update root logger level
            root_logger = logging.getLogger()
            root_logger.setLevel(self.current_log_level)

            logger.info(f"Log level changed to {level}")

        except Exception as e:
            logger.error(f"Failed to change log level: {e}")

    def _toggle_polling(self) -> None:
        """Toggle log polling pause/resume."""
        self.is_polling = not self.is_polling

        if self.is_polling:
            self.pause_btn.configure(text="Pause")
            logger.info("Log polling resumed")
        else:
            self.pause_btn.configure(text="Resume")
            logger.info("Log polling paused")

    def _apply_filter(self) -> None:
        """Apply text filter to logs."""
        self.log_filter_text = self.filter_var.get().strip()

        # Clear and re-display logs with new filter
        self.log_text.delete("1.0", "end")

        # Reset statistics
        for level in self.log_counts:
            self.log_counts[level] = 0

        # Re-process logs from queue (this is a simplified approach)
        # In a full implementation, we'd store filtered logs
        logger.info(f"Applied filter: '{self.log_filter_text}'")

    def _clear_filter(self) -> None:
        """Clear text filter."""
        self.filter_var.set("")
        self._apply_filter()

    def _update_statistics_display(self) -> None:
        """Update the statistics display."""
        self.stats_text.configure(state="normal")
        self.stats_text.delete("1.0", "end")

        total = sum(self.log_counts.values())

        stats_text = f"Total: {total} | "
        stats_text += f"DEBUG: {self.log_counts['DEBUG']} | "
        stats_text += f"INFO: {self.log_counts['INFO']} | "
        stats_text += f"WARNING: {self.log_counts['WARNING']} | "
        stats_text += f"ERROR: {self.log_counts['ERROR']} | "
        stats_text += f"CRITICAL: {self.log_counts['CRITICAL']}"

        self.stats_text.insert("1.0", stats_text)

        # Apply color coding
        start_pos = 0
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            level_pos = stats_text.find(level)
            if level_pos != -1:
                end_pos = level_pos + len(level)
                self.stats_text.tag_add(level.lower(), f"1.{start_pos}", f"1.{end_pos}")
                start_pos = end_pos + 3  # Skip " | "

        self.stats_text.configure(state="disabled")

    def _copy_selected(self) -> None:
        """Copy selected text to clipboard."""
        try:
            selected_text = self.log_text.get(tk.SEL_FIRST, tk.SEL_LAST)
            if selected_text:
                self.clipboard_clear()
                self.clipboard_append(selected_text)
                self.show_status("Selected text copied to clipboard", "success")
            else:
                self.show_status("No text selected to copy", "warning")
        except tk.TclError:
            self.show_status("No text selected to copy", "warning")
        except Exception as e:
            self.show_status(f"Failed to copy selected text: {e}", "error")
            self.logger.error(f"Copy selected text failed: {e}")

    def _copy_all(self) -> None:
        """Copy all log text to clipboard."""
        try:
            all_text = self.log_text.get("1.0", "end")
            if all_text.strip():
                self.clipboard_clear()
                self.clipboard_append(all_text)
                self.show_status("All log text copied to clipboard", "success")
            else:
                self.show_status("No log text to copy", "warning")
        except Exception as e:
            self.show_status(f"Failed to copy all log text: {e}", "error")
            self.logger.error(f"Copy all log text failed: {e}")

    def _find_text(self) -> None:
        """Find text in log display."""
        # Simple find dialog (could be enhanced)
        find_dialog = ctk.CTkToplevel(self)
        find_dialog.title("Find in Logs")
        find_dialog.geometry("300x100")

        ctk.CTkLabel(find_dialog, text="Find:").pack(pady=10)

        find_var = tk.StringVar()
        find_entry = ctk.CTkEntry(find_dialog, textvariable=find_var, width=30)
        find_entry.pack(pady=5)

        def do_find() -> None:
            search_text = find_var.get()
            if search_text:
                # Simple search (could be enhanced with regex, case sensitivity, etc.)
                self.log_text.tag_remove(tk.SEL, "1.0", "end")
                pos = self.log_text.search(search_text, "1.0", "end")
                if pos:
                    self.log_text.tag_add(tk.SEL, pos, f"{pos}+{len(search_text)}c")
                    self.log_text.see(pos)
                else:
                    messagebox.showinfo("Find", "Text not found")

        ctk.CTkButton(find_dialog, text="Find", command=do_find).pack(pady=10)

    def _clear_selection(self) -> None:
        """Clear text selection."""
        self.log_text.tag_remove(tk.SEL, "1.0", "end")

    def add_message(self, message: str, level: str = "INFO") -> None:
        """Add a message directly to the log display."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if level.upper() in self.log_colors:
            self._append_formatted_log(timestamp, "GUI", level.upper(), message)
            self.log_counts[level.upper()] = self.log_counts.get(level.upper(), 0) + 1
            self._update_statistics_display()
        else:
            self._append_formatted_log(timestamp, "GUI", "INFO", message)
            self.log_counts["INFO"] = self.log_counts.get("INFO", 0) + 1
            self._update_statistics_display()

    def get_log_statistics(self) -> Dict[str, int]:
        """Get current log statistics."""
        return self.log_counts.copy()

    def set_log_added_callback(self, callback: Callable[..., None]) -> None:
        """Set callback for log added events."""
        self.on_log_added = callback

    def set_log_cleared_callback(self, callback: Callable[..., None]) -> None:
        """Set callback for log cleared events."""
        self.on_log_cleared = callback

    def set_max_log_lines(self, max_lines: int) -> None:
        """Set maximum number of log lines to display."""
        self.max_log_lines = max_lines
        self._limit_log_size()

    def export_filtered_logs(self, filename: str) -> None:
        """Export only filtered logs."""
        try:
            with open(filename, "w") as f:
                log_content = self.log_text.get("1.0", "end")
                f.write(log_content)
            logger.info(f"Filtered logs exported to {filename}")
        except Exception as e:
            logger.error(f"Failed to export filtered logs: {e}")

    def get_recent_logs(self, count: int = 50) -> List[str]:
        """Get the most recent log entries."""
        try:
            all_content = self.log_text.get("1.0", "end").strip()
            lines: List[str] = all_content.split("\n")
            return lines[-count:] if len(lines) > count else lines
        except Exception:
            return []

    def search_logs(self, pattern: str, case_sensitive: bool = False) -> List[str]:
        """Search logs for a pattern."""
        try:
            all_content = self.log_text.get("1.0", "end").strip()
            lines = all_content.split("\n")

            if not case_sensitive:
                pattern = pattern.lower()
                return [line for line in lines if pattern in line.lower()]
            else:
                return [line for line in lines if pattern in line]
        except Exception:
            return []


# Factory function for easy instantiation
def create_logging_panel(parent: Any, max_log_lines: int = 1000) -> LoggingPanel:
    """Create a logging panel with default settings.

    Args:
        parent: Parent widget
        max_log_lines: Maximum number of log lines to display

    Returns:
        Configured LoggingPanel instance
    """
    return LoggingPanel(parent, max_log_lines)
