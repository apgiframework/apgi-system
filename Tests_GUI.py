#!/usr/bin/env python3
"""
GUI to run all tests folder scripts
==========================================

A tkinter-based GUI that allows running all test scripts in the tests folder
with output display and error handling.
"""

# Deprecated legacy entry point (README: "Legacy Entry Points (Deprecated)").
try:
    from apgi_framework.utils.deprecation import warn_deprecated_module

    warn_deprecated_module("Tests_GUI", "apgi_gui.main", since="1.1.0", removal="2.0.0")
except ImportError:  # framework not importable: the README and docstring still state the deprecation
    pass

import sys
import tkinter as tk

from utils.script_runner_gui import ScriptRunnerGUI


class TestsRunnerGUI(ScriptRunnerGUI):
    """Simple GUI for running tests scripts.

    Provides a tkinter-based interface to run test scripts from the tests folder
    with real-time output display, error handling, and process management.
    """

    def __init__(self, root: tk.Tk):
        super().__init__(
            root,
            "tests",
            "APGI Tests Scripts Runner",
            "Available Test Scripts",
            "Run Selected",
            "Run the currently selected test script",
            "Run All",
            "Run all test scripts in sequence",
            "Stop",
        )

        # Set tests_dir alias for script_dir
        self.tests_dir = self.script_dir

        # Add summary panel
        self.add_summary_panel()

        # Initialize summary counters
        self.reset_summary()

    def add_summary_panel(self) -> None:
        """Add a summary panel to display test results statistics."""
        # Create summary frame
        self.summary_frame = tk.LabelFrame(
            self.control_frame, text="Test Results Summary", padx=10, pady=5
        )
        self.summary_frame.pack(fill=tk.X, pady=(10, 0))

        # Create summary labels
        self.total_label = tk.Label(
            self.summary_frame, text="Total Tests: 0", font=("Arial", 10, "bold")
        )
        self.total_label.pack(anchor=tk.W)

        self.passed_label = tk.Label(
            self.summary_frame, text="Passed: 0", fg="green", font=("Arial", 10)
        )
        self.passed_label.pack(anchor=tk.W)

        self.failed_label = tk.Label(
            self.summary_frame, text="Failed: 0", fg="red", font=("Arial", 10)
        )
        self.failed_label.pack(anchor=tk.W)

        self.success_rate_label = tk.Label(
            self.summary_frame, text="Success Rate: 0%", font=("Arial", 10, "bold")
        )
        self.success_rate_label.pack(anchor=tk.W, pady=(5, 0))

    def reset_summary(self) -> None:
        """Reset the summary counters."""
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.update_summary_display()

    def update_summary_display(self) -> None:
        """Update the summary panel display."""
        self.total_label.config(text=f"Total Tests: {self.total_tests}")
        self.passed_label.config(text=f"Passed: {self.passed_tests}")
        self.failed_label.config(text=f"Failed: {self.failed_tests}")

        if self.total_tests > 0:
            success_rate = (self.passed_tests / self.total_tests) * 100
            self.success_rate_label.config(text=f"Success Rate: {success_rate:.1f}%")
        else:
            self.success_rate_label.config(text="Success Rate: 0%")

    def run_all_scripts(self) -> None:
        """Run all scripts sequentially."""
        if not self.scripts:
            self.log_output("No test scripts found", self.TAG_ERROR)
            return

        # Reset summary for new run
        self.reset_summary()

        def run_all() -> None:
            for i, script in enumerate(self.scripts):
                self.total_tests += 1
                relative_path = script.relative_to(self.tests_dir)
                self.log_output(
                    f"Running test {i + 1}/{len(self.scripts)}: {relative_path}",
                    self.TAG_INFO,
                )
                self.scripts_listbox.selection_clear(0, tk.END)
                self.scripts_listbox.selection_set(i)
                self.scripts_listbox.see(i)

                success = self.run_script(script, wait=True)
                if success:
                    self.passed_tests += 1
                else:
                    self.failed_tests += 1

                # Update display after each test
                self.root.after(0, self.update_summary_display)

            self.log_output(
                f"All tests completed: {self.passed_tests} passed, {self.failed_tests} failed",
                self.TAG_SUCCESS if self.failed_tests == 0 else self.TAG_ERROR,
            )
            self.update_status("Ready")
            self.progress.stop()

        # Run in separate thread to avoid blocking GUI
        import threading

        thread = threading.Thread(target=run_all, daemon=True)
        thread.start()


def main() -> None:
    """Launch the tests runner GUI."""
    try:
        # Create and run the GUI
        root = tk.Tk()
        TestsRunnerGUI(root)

        # Center window on screen
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f"{width}x{height}+{x}+{y}")

        root.mainloop()

    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("This script requires tkinter, which should come with Python.")
        sys.exit(1)
    except tk.TclError as e:
        print(f"❌ Tkinter Error: {e}")
        print("There was an error initializing the GUI.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
