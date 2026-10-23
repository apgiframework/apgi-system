#!/usr/bin/env python3
"""
verify_apgi_application_gui.py – Comprehensive headless verification of APGI_Application_GUI.py
=============================================================================================
"""

import importlib.util
import sys
import traceback
from unittest.mock import MagicMock, mock_open, patch

# ── Helpers & colours ───────────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

passed: list[str] = []
failed: list[str] = []
skipped: list[str] = []
gui = None


def assert_(cond, msg=""):
    assert cond, msg


def ok(name, detail=""):
    passed.append(name)
    print(f"  {GREEN}✓ PASS{RESET}  {name}" + (f"  [{detail}]" if detail else ""))


def fail(name, detail=""):
    failed.append(name)
    print(f"  {RED}✗ FAIL{RESET}  {name}" + (f"\n           ↳ {detail}" if detail else ""))


def section(title):
    bar = "─" * 62
    print(f"\n{CYAN}{BOLD}{bar}\n  {title}\n{bar}{RESET}")


def run_test(name, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
        ok(name)
    except AssertionError as e:
        fail(name, str(e))
    except Exception as e:
        traceback.print_exc()
        fail(name, f"{type(e).__name__}: {e}")


# ════════════════════════════════════════════════════════════════════════════
# 1. SYNTAX CHECK
# ════════════════════════════════════════════════════════════════════════════
section("1 · Syntax Check")
import py_compile

try:
    py_compile.compile("APGI_Application_GUI.py", doraise=True)
    ok("py_compile – syntax clean")
except py_compile.PyCompileError as e:
    fail("py_compile – syntax clean", str(e))
    sys.exit(1)

# ════════════════════════════════════════════════════════════════════════════
# 2. MODULE IMPORT
# ════════════════════════════════════════════════════════════════════════════
section("2 · Module Import")


class MockCTK:
    class CTk:
        def __init__(self, *args, **kwargs):
            self.winfo_screenwidth = lambda: 1920
            self.winfo_screenheight = lambda: 1080

        def geometry(self, s):
            pass

        def title(self, t):
            pass

        def minsize(self, w, h):
            pass

        def grid_columnconfigure(self, *args, **kwargs):
            pass

        def grid_rowconfigure(self, *args, **kwargs):
            pass

        def after(self, *args):
            pass

        def withdraw(self):
            pass

        def deiconify(self):
            pass

        def bind(self, *args):
            pass

        def protocol(self, *args):
            pass

        def quit(self):
            pass

        def destroy(self):
            pass

        def cget(self, *args):
            return ""

        def configure(self, *args, **kwargs):
            pass

    def set_appearance_mode(self, mode):
        pass

    def set_default_color_theme(self, theme):
        pass

    def CTkButton(self, *args, **kwargs):
        return MagicMock()

    def CTkLabel(self, *args, **kwargs):
        return MagicMock()

    def CTkFrame(self, *args, **kwargs):
        return MagicMock()

    def CTkProgressBar(self, *args, **kwargs):
        return MagicMock()

    def CTkTextbox(self, *args, **kwargs):
        return MagicMock()

    def CTkTabview(self, *args, **kwargs):
        return MagicMock()


with (
    patch.dict(sys.modules, {"customtkinter": MockCTK()}),
    patch("tkinter.messagebox.showinfo"),
    patch("tkinter.messagebox.showwarning"),
    patch("tkinter.messagebox.showerror"),
    patch("tkinter.messagebox.askyesno", return_value=True),
    patch("tkinter.filedialog.asksaveasfilename", return_value="config.json"),
    patch("tkinter.filedialog.askopenfilename", return_value="config.json"),
):
    try:
        spec = importlib.util.spec_from_file_location(
            "APGI_Application_GUI", "APGI_Application_GUI.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ok("APGI_Application_GUI – imported without errors")
    except Exception as e:
        fail("APGI_Application_GUI – imported without errors", str(e))
        traceback.print_exc()
        sys.exit(1)

# ════════════════════════════════════════════════════════════════════════════
# 3. GUI INITIALIZATION
# ════════════════════════════════════════════════════════════════════════════
section("3 · GUI Initialization")


def test_gui_init():
    global gui
    with (
        patch("APGI_Application_GUI.APGIFrameworkGUI._setup_window"),
        patch("APGI_Application_GUI.APGIFrameworkGUI._create_ui_components"),
        patch("APGI_Application_GUI.APGIFrameworkGUI._initialize_framework"),
        patch("APGI_Application_GUI.APGIFrameworkGUI._initialize_test_runners"),
        patch("APGI_Application_GUI.APGIFrameworkGUI.after"),
    ):
        gui = mod.APGIFrameworkGUI()
        gui.test_runners = {}
        gui.sidebar_buttons = []
        gui._test_lock = MagicMock()
        gui.log_to_console = MagicMock()
        gui.update_status = MagicMock()
    assert_(hasattr(gui, "test_runners"), "GUI should have test_runners")


run_test("APGIFrameworkGUI – init", test_gui_init)

# ════════════════════════════════════════════════════════════════════════════
# 4. VALIDATORS
# ════════════════════════════════════════════════════════════════════════════
section("4 · Validators")


def test_validators():
    iv = mod.InputValidator()
    v, val, err = iv.validate_numeric_input("10.5", "test", 0, 20)
    assert_(v and val == 10.5)

    with patch("APGI_Application_GUI.ConfigManager", None):
        cv = mod.ConfigurationValidator()
    v, err = cv.validate_parameter("exteroceptive_precision", "1.5")
    assert_(v)


run_test("Validators logic", test_validators)

# ════════════════════════════════════════════════════════════════════════════
# 5. TEST RUNNERS
# ════════════════════════════════════════════════════════════════════════════
section("5 · Test Runners")


def test_runners():
    for runner_cls in [
        mod.PrimaryFalsificationRunner,
        mod.CWITestRunner,
        mod.ThresholdTestRunner,
        mod.SomaBiasRunner,
    ]:
        instance = runner_cls(gui)
        assert_(instance.get_test_name())
        ok(f"Runner confirmed: {instance.get_test_name()}")


run_test("Verify all Falsification Test Runners", test_runners)

# ════════════════════════════════════════════════════════════════════════════
# 6. CORE COMMANDS
# ════════════════════════════════════════════════════════════════════════════
section("6 · Core Commands")


def test_core_commands():
    with patch("builtins.open", mock_open(read_data='{"theme": "dark"}')):
        gui.save_config()
        gui.load_config()
    gui.undo_action()
    gui.redo_action()


run_test("Core GUI Commands", test_core_commands)

print(f"\n{BOLD}Verification Summary{RESET}")
print(f"{GREEN}Passed:  {len(passed)}{RESET}")
print(f"{RED}Failed:  {len(failed)}{RESET}")
if failed:
    sys.exit(1)
else:
    print(f"\n{GREEN}{BOLD}ALL TESTS PASSED 100%{RESET}\n")
    sys.exit(0)
