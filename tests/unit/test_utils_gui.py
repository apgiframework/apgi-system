"""
Unit tests for the script-runner GUI used by the APGI utilities.

``ScriptRunnerGUI`` (utils/script_runner_gui.py) lists ``test_*.py`` scripts from a directory,
runs the selected one in a subprocess and shows its output. These tests drive it against a
temporary script directory. They need a display (Xvfb on Linux CI) and are skipped, with a
reason, when Tk cannot open one.
"""

import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest

from utils.script_runner_gui import ScriptRunnerGUI


@pytest.fixture
def root() -> Generator[tk.Tk, None, None]:
    try:
        window = tk.Tk()
    except tk.TclError as exc:  # no display available
        pytest.skip(f"Tk display unavailable: {exc}")
    window.withdraw()
    yield window
    window.destroy()


@pytest.fixture
def script_dir(tmp_path: Path) -> Path:
    (tmp_path / "test_ok.py").write_text("print('hello from ok')\n")
    (tmp_path / "test_fail.py").write_text("import sys\nprint('about to fail')\nsys.exit(3)\n")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "test_nested.py").write_text("print('nested')\n")
    (tmp_path / "helper.py").write_text("print('not a test script')\n")
    return tmp_path


def make_gui(root: tk.Tk, directory: Path) -> ScriptRunnerGUI:
    return ScriptRunnerGUI(
        root,
        script_dir=str(directory),
        title="APGI Utils Scripts Runner",
        listbox_label="Available Scripts",
        run_button_text="Run Selected Script",
        run_button_tooltip="Run the currently selected script",
        run_all_button_text="Run All Scripts",
        run_all_button_tooltip="Run all scripts in sequence",
        stop_button_text="Stop",
    )


def pump(root: tk.Tk, rounds: int = 20) -> None:
    """Run pending Tk callbacks (script output is delivered with ``root.after`` from a thread)."""
    for _ in range(rounds):
        root.update()
        time.sleep(0.01)


def run_waiting(root: tk.Tk, gui: ScriptRunnerGUI, script: Path) -> bool:
    """Run ``script`` with ``wait=True`` the way the app does: from a worker thread.

    Widget calls and ``root.after`` from a worker thread only work while the main thread is in
    ``mainloop``, so the loop runs until the worker ends.
    """
    result: list[bool] = []
    worker = threading.Thread(target=lambda: result.append(gui.run_script(script, wait=True)))

    def poll() -> None:
        if worker.is_alive():
            root.after(10, poll)
        else:
            root.quit()

    worker.start()
    root.after(10, poll)
    root.mainloop()  # Tk marshals worker-thread widget calls through the running main loop
    worker.join()
    pump(root)
    return result[0]


def output_of(gui: ScriptRunnerGUI) -> str:
    return str(gui.output_text.get("1.0", tk.END))


class TestInitialization:
    def test_gui_launches_without_errors(self, root: tk.Tk, script_dir: Path) -> None:
        gui = make_gui(root, script_dir)

        assert gui.root is root
        assert root.title() == "APGI Utils Scripts Runner"
        for widget in ("main_frame", "control_frame", "output_frame", "output_text", "progress"):
            assert hasattr(gui, widget), widget
        assert gui.status_var.get() == "Ready"

    def test_lists_test_scripts_recursively(self, root: tk.Tk, script_dir: Path) -> None:
        gui = make_gui(root, script_dir)

        names = sorted(p.name for p in gui.scripts)
        assert names == ["test_fail.py", "test_nested.py", "test_ok.py"]  # helper.py is ignored
        shown = list(gui.scripts_listbox.get(0, tk.END))
        assert len(shown) == 3
        assert str(Path("nested") / "test_nested.py") in shown

    def test_buttons_start_in_idle_state(self, root: tk.Tk, script_dir: Path) -> None:
        gui = make_gui(root, script_dir)

        assert str(gui.stop_button["state"]) == tk.DISABLED
        assert str(gui.run_button["state"]) == tk.NORMAL
        assert str(gui.run_all_button["state"]) == tk.NORMAL

    def test_missing_directory_is_reported(self, root: tk.Tk, tmp_path: Path) -> None:
        gui = make_gui(root, tmp_path / "does-not-exist")

        assert "Script directory not found" in output_of(gui)

    def test_empty_directory_is_reported(self, root: tk.Tk, tmp_path: Path) -> None:
        gui = make_gui(root, tmp_path)

        assert gui.scripts == []
        assert "No scripts found" in output_of(gui)


class TestExecution:
    def test_run_selected_without_selection_warns(self, root: tk.Tk, script_dir: Path) -> None:
        gui = make_gui(root, script_dir)
        with patch.object(gui, "run_script") as run_script:
            gui.run_selected_script()

        run_script.assert_not_called()
        assert "No script selected" in output_of(gui)

    def test_run_selected_runs_the_chosen_script(self, root: tk.Tk, script_dir: Path) -> None:
        gui = make_gui(root, script_dir)
        gui.scripts_listbox.selection_set(1)
        with patch.object(gui, "run_script") as run_script:
            gui.run_selected_script()

        run_script.assert_called_once_with(gui.scripts[1])

    def test_successful_script_reports_success(self, root: tk.Tk, script_dir: Path) -> None:
        gui = make_gui(root, script_dir)

        assert run_waiting(root, gui, script_dir / "test_ok.py") is True
        text = output_of(gui)
        assert "hello from ok" in text
        assert "Script completed successfully: test_ok.py" in text
        assert gui.status_var.get() == "Ready"
        assert gui.current_process is None

    def test_failing_script_reports_return_code(self, root: tk.Tk, script_dir: Path) -> None:
        gui = make_gui(root, script_dir)

        assert run_waiting(root, gui, script_dir / "test_fail.py") is False

        assert "Script failed with return code 3" in output_of(gui)
        assert str(gui.run_button["state"]) == tk.NORMAL  # buttons are restored

    def test_popen_failure_is_handled(self, root: tk.Tk, script_dir: Path) -> None:
        gui = make_gui(root, script_dir)
        with patch("utils.script_runner_gui.subprocess.Popen", side_effect=OSError("no exec")):
            assert run_waiting(root, gui, script_dir / "test_ok.py") is False

        assert "Error running script: no exec" in output_of(gui)
        assert gui.status_var.get() == "Ready"

    def test_running_script_uses_current_interpreter(self, root: tk.Tk, script_dir: Path) -> None:
        gui = make_gui(root, script_dir)
        process = MagicMock()
        process.stdout.readline.return_value = ""
        process.returncode = 0
        with patch("utils.script_runner_gui.subprocess.Popen", return_value=process) as popen:
            run_waiting(root, gui, script_dir / "test_ok.py")

        command = popen.call_args.args[0]
        assert command == [sys.executable, str(script_dir / "test_ok.py")]

    def test_run_all_is_a_documented_noop_in_base_class(
        self, root: tk.Tk, script_dir: Path
    ) -> None:
        gui = make_gui(root, script_dir)
        gui.run_all_scripts()

        assert "not implemented in base class" in output_of(gui)

    def test_stop_script_terminates_the_process(self, root: tk.Tk, script_dir: Path) -> None:
        gui = make_gui(root, script_dir)
        process = MagicMock()
        gui.current_process = process
        gui.stop_script()

        process.terminate.assert_called_once()
        assert gui.stop_requested is True
        assert gui.current_process is None
        assert "Script stopped" in output_of(gui)

    def test_stop_script_without_process_does_nothing(self, root: tk.Tk, script_dir: Path) -> None:
        gui = make_gui(root, script_dir)
        gui.stop_script()

        assert "Stopping script" not in output_of(gui)


class TestOutputAndStatus:
    def test_log_output_appends_lines_with_tags(self, root: tk.Tk, script_dir: Path) -> None:
        gui = make_gui(root, script_dir)
        gui.log_output("first")
        gui.log_output("second", ScriptRunnerGUI.TAG_ERROR)

        text = output_of(gui)
        assert text.index("first") < text.index("second")
        ranges = gui.output_text.tag_ranges(ScriptRunnerGUI.TAG_ERROR)
        assert len(ranges) >= 2  # at least one tagged span (start, end)

    def test_output_can_be_cleared(self, root: tk.Tk, script_dir: Path) -> None:
        gui = make_gui(root, script_dir)
        gui.log_output("something")
        gui.output_text.delete("1.0", tk.END)

        assert output_of(gui).strip() == ""

    def test_update_status_sets_status_bar(self, root: tk.Tk, script_dir: Path) -> None:
        gui = make_gui(root, script_dir)
        gui.update_status("Working")

        assert gui.status_var.get() == "Working"

    def test_output_text_supports_export_to_file(
        self, root: tk.Tk, script_dir: Path, tmp_path: Path
    ) -> None:
        gui = make_gui(root, script_dir)
        gui.log_output("line to export")
        target = tmp_path / "out.txt"
        target.write_text(output_of(gui))

        assert "line to export" in target.read_text()


class TestThreading:
    def test_script_output_is_delivered_via_the_ui_thread(
        self, root: tk.Tk, script_dir: Path
    ) -> None:
        """Reader threads must hand lines to Tk through ``root.after`` (not touch widgets)."""
        gui = make_gui(root, script_dir)
        calls: list[Any] = []
        original_after = root.after

        def spy(delay: int, func: Any = None, *args: Any) -> Any:
            calls.append((func, args))
            return original_after(delay, func, *args)

        with patch.object(root, "after", side_effect=spy):
            run_waiting(root, gui, script_dir / "test_ok.py")

        assert any(func == gui.log_output and args == ("hello from ok",) for func, args in calls)
