"""Standardized GUI utilities for APGI Framework applications.

Provides consistent styling, window management, error handling, and cross-platform
compatibility across all GUI applications.
"""

import logging
import os
import sys
import tkinter as tk
import traceback
from abc import ABC
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any, Callable, Optional

# Try to import CustomTkinter, fallback to tkinter if not available
try:
    import customtkinter as ctk

    CUSTOMTKINTER_AVAILABLE = True
except ImportError:
    CUSTOMTKINTER_AVAILABLE = False
    ctk = None

logger = logging.getLogger(__name__)


class GUIStyleManager:
    """Manages consistent styling across GUI applications."""

    # Standard window dimensions
    DEFAULT_WINDOW_SIZE = "1200x800"
    COMPACT_WINDOW_SIZE = "800x600"
    LARGE_WINDOW_SIZE = "1400x900"

    # Standard color scheme
    COLORS = {
        "primary": "#2E86AB",
        "secondary": "#A23B72",
        "success": "#5C946E",
        "warning": "#F18F01",
        "error": "#C73E1D",
        "background": "#F8F9FA",
        "text": "#212529",
        "text_secondary": "#6C757D",
    }

    # Standard font configurations
    FONTS = {
        "title": ("Arial", 16, "bold"),
        "heading": ("Arial", 12, "bold"),
        "normal": ("Arial", 10),
        "small": ("Arial", 9),
        "code": ("Courier New", 9),
    }

    @classmethod
    def configure_style(cls, root: tk.Tk, use_customtkinter: bool = False) -> None:
        """Apply consistent styling to a tkinter root window."""
        if use_customtkinter and CUSTOMTKINTER_AVAILABLE:
            cls._configure_customtkinter_style(root)
        else:
            cls._configure_tkinter_style(root)

    @classmethod
    def _configure_tkinter_style(cls, root: tk.Tk) -> None:
        """Configure standard tkinter styling."""
        style = ttk.Style()
        style.theme_use("clam")

        # Configure colors
        style.configure(
            "TLabel", background=cls.COLORS["background"], foreground=cls.COLORS["text"]
        )
        style.configure("TButton", background=cls.COLORS["primary"], foreground="white")
        style.configure("TFrame", background=cls.COLORS["background"])
        style.configure(
            "TLabelframe",
            background=cls.COLORS["background"],
            foreground=cls.COLORS["text"],
        )

        root.configure(bg=cls.COLORS["background"])

    @classmethod
    def _configure_customtkinter_style(cls, root: ctk.CTk) -> None:
        """Configure CustomTkinter styling."""
        if CUSTOMTKINTER_AVAILABLE:
            ctk.set_appearance_mode("light")
            ctk.set_default_color_theme("blue")


class StandardWindow(ABC):
    """Base class for standardized GUI windows."""

    def __init__(self, title: str, size: Optional[str] = None, use_customtkinter: bool = False):
        """Initialize a standardized window.

        Args:
            title: Window title
            size: Window size string (e.g., "1200x800")
            use_customtkinter: Whether to use CustomTkinter if available
        """
        self.title = title
        self.size = size or GUIStyleManager.DEFAULT_WINDOW_SIZE
        self.use_customtkinter = use_customtkinter and CUSTOMTKINTER_AVAILABLE

        # Create root window
        if self.use_customtkinter:
            self.root = ctk.CTk()
        else:
            self.root = tk.Tk()

        # Configure window
        self.root.title(title)
        self.root.geometry(self.size)

        # Apply styling
        GUIStyleManager.configure_style(self.root, self.use_customtkinter)

        # Center window on screen
        self._center_window()

        # Add keyboard shortcuts
        self._setup_keyboard_shortcuts()

        # Setup error handling
        self._setup_error_handling()

    def _center_window(self) -> None:
        """Center the window on the screen."""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def _setup_keyboard_shortcuts(self) -> None:
        """Setup standard keyboard shortcuts."""
        # Ctrl+Q: Quit
        self.root.bind("<Control-q>", lambda e: self.quit())
        self.root.bind("<Control-Q>", lambda e: self.quit())

        # Escape: Close window
        self.root.bind("<Escape>", lambda e: self.quit())

        # F1: Help
        self.root.bind("<F1>", lambda e: self.show_help())

    def _setup_error_handling(self) -> None:
        """Setup global error handling for the window."""
        try:
            from .enhanced_error_handling import setup_global_error_handling

            setup_global_error_handling(self.root)
        except ImportError:
            # Fallback to basic error handling
            def handle_tk_error(exc: type[BaseException], val: BaseException, tb: Any) -> None:
                error_msg = f"An error occurred:\n\n{type(val).__name__}: {val}\n\n"
                error_msg += "Please check the logs for more details."
                messagebox.showerror("Error", error_msg)
                logger.error(f"GUI Error: {type(val).__name__}: {val}\n{traceback.format_exc()}")

            # Install error handler
            if hasattr(sys, "excepthook"):
                sys.excepthook = handle_tk_error

    def show_help(self) -> None:
        """Show help dialog."""
        help_text = f"{self.title}\n\n"
        help_text += "Keyboard Shortcuts:\n"
        help_text += "Ctrl+Q: Quit application\n"
        help_text += "Escape: Close window\n"
        help_text += "F1: Show this help\n\n"
        help_text += "For more help, check the documentation."

        messagebox.showinfo("Help", help_text)

    def quit(self) -> None:
        """Quit the application."""
        if self._confirm_quit():
            self.root.quit()
            self.root.destroy()

    def _confirm_quit(self) -> bool:
        """Confirm before quitting (can be overridden)."""
        return True

    def run(self) -> None:
        """Start the main loop."""
        self.root.mainloop()


class StandardMenuBar:
    """Standard menu bar for GUI applications."""

    def __init__(self, root: tk.Tk, window_instance: StandardWindow) -> None:
        """Create a standard menu bar.

        Args:
            root: Root window
            window_instance: Window instance for callbacks
        """
        self.root = root
        self.window = window_instance
        self.menubar = tk.Menu(root)
        self._create_menus()
        root.config(menu=self.menubar)

    def _create_menus(self) -> None:
        """Create standard menus."""
        # File menu
        file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New", command=self.new_file, accelerator="Ctrl+N")
        file_menu.add_command(label="Open", command=self.open_file, accelerator="Ctrl+O")
        file_menu.add_command(label="Save", command=self.save_file, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.window.quit, accelerator="Ctrl+Q")

        # Edit menu
        edit_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Undo", command=self.undo, accelerator="Ctrl+Z")
        edit_menu.add_command(label="Redo", command=self.redo, accelerator="Ctrl+Y")
        edit_menu.add_separator()
        edit_menu.add_command(label="Preferences", command=self.preferences)

        # View menu
        view_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Zoom In", command=self.zoom_in, accelerator="Ctrl++")
        view_menu.add_command(label="Zoom Out", command=self.zoom_out, accelerator="Ctrl+-")
        view_menu.add_command(label="Reset View", command=self.reset_view)

        # Help menu
        help_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Help", command=self.window.show_help, accelerator="F1")
        help_menu.add_command(label="About", command=self.about)

        # Bind keyboard shortcuts
        self.root.bind("<Control-n>", lambda e: self.new_file())
        self.root.bind("<Control-o>", lambda e: self.open_file())
        self.root.bind("<Control-s>", lambda e: self.save_file())
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<Control-y>", lambda e: self.redo())
        self.root.bind("<Control-plus>", lambda e: self.zoom_in())
        self.root.bind("<Control-minus>", lambda e: self.zoom_out())

    def new_file(self) -> None:
        """Create new file (to be implemented by subclasses)."""

    def open_file(self) -> None:
        """Open file (to be implemented by subclasses)."""

    def save_file(self) -> None:
        """Save file (to be implemented by subclasses)."""

    def undo(self) -> None:
        """Undo action (to be implemented by subclasses)."""

    def redo(self) -> None:
        """Redo action (to be implemented by subclasses)."""

    def preferences(self) -> None:
        """Show preferences dialog (to be overridden)."""
        messagebox.showinfo("Preferences", "Preferences functionality not implemented")

    def zoom_in(self) -> None:
        """Zoom in (to be overridden)."""
        messagebox.showinfo("Zoom In", "Zoom in functionality not implemented")

    def zoom_out(self) -> None:
        """Zoom out (to be overridden)."""
        messagebox.showinfo("Zoom Out", "Zoom out functionality not implemented")

    def reset_view(self) -> None:
        """Reset view (to be overridden)."""
        messagebox.showinfo("Reset View", "Reset view functionality not implemented")

    def about(self) -> None:
        """Show about dialog."""
        about_text = "APGI Framework\n\n"
        about_text += "Active Perception and General Intelligence Framework\n"
        about_text += "Version 1.0\n\n"
        about_text += "A comprehensive framework for consciousness research\n"
        about_text += "and active inference modeling."

        messagebox.showinfo("About", about_text)


class ErrorHandler:
    """Enhanced error handling for GUI applications."""

    @staticmethod
    def handle_import_error(module_name: str, fallback_available: bool = True) -> None:
        """Handle missing optional dependencies gracefully."""
        if fallback_available:
            messagebox.showwarning(
                "Optional Module Missing",
                f"The module '{module_name}' is not available.\n\n"
                f"Some features may be limited, but the application will continue to work.\n\n"
                f"To enable all features, install the missing dependencies.",
            )
        else:
            messagebox.showerror(
                "Required Module Missing",
                f"The required module '{module_name}' is not available.\n\n"
                f"Please install the missing dependencies and restart the application.",
            )

    @staticmethod
    def handle_file_error(file_path: str, operation: str) -> None:
        """Handle file operation errors."""
        messagebox.showerror(
            "File Error",
            f"Could not {operation.lower()} file:\n{file_path}\n\n"
            f"Please check if the file exists and you have the necessary permissions.",
        )

    @staticmethod
    def handle_data_error(error_message: str) -> None:
        """Handle data processing errors."""
        messagebox.showerror(
            "Data Error",
            f"A data processing error occurred:\n\n{error_message}\n\n"
            f"Please check your input data and try again.",
        )

    @staticmethod
    def handle_network_error(url: str) -> None:
        """Handle network connection errors."""
        messagebox.showerror(
            "Network Error",
            f"Could not connect to:\n{url}\n\n"
            f"Please check your internet connection and try again.",
        )


class PathManager:
    """Cross-platform path management with robust error handling."""

    @staticmethod
    def get_project_root() -> Path:
        """Get the project root directory with fallback."""
        try:
            # Try to get from file location first
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent

            # Verify this looks like the project root
            if (project_root / "apgi_framework").exists():
                return project_root

            # Fallback to current working directory
            cwd = Path.cwd()
            if (cwd / "apgi_framework").exists():
                return cwd

            # Final fallback - assume we're in the project
            return Path.cwd()

        except Exception:
            # Ultimate fallback
            return Path.cwd()

    @staticmethod
    def get_data_dir() -> Path:
        """Get the data directory with creation."""
        data_dir = PathManager.get_project_root() / "data"
        return PathManager.ensure_dir_exists(data_dir)

    @staticmethod
    def get_outputs_dir() -> Path:
        """Get the outputs directory with creation."""
        outputs_dir = PathManager.get_project_root() / "apgi_outputs"
        return PathManager.ensure_dir_exists(outputs_dir)

    @staticmethod
    def get_config_dir() -> Path:
        """Get the config directory with creation."""
        config_dir = PathManager.get_project_root() / "apgi_framework" / "config"
        return PathManager.ensure_dir_exists(config_dir)

    @staticmethod
    def ensure_dir_exists(path: Path) -> Path:
        """Ensure a directory exists, create if necessary."""
        try:
            path.mkdir(parents=True, exist_ok=True)
            return path
        except Exception as e:
            logger.warning(f"Could not create directory {path}: {e}")
            # Fallback to temp directory
            import tempfile

            fallback_dir = Path(tempfile.gettempdir()) / "apgi_framework" / path.name
            fallback_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using fallback directory: {fallback_dir}")
            return fallback_dir

    @staticmethod
    def get_user_config_dir() -> Path:
        """Get user-specific config directory with cross-platform support."""
        try:
            if sys.platform == "win32":
                base_dir = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
            elif sys.platform == "darwin":
                base_dir = Path.home() / "Library" / "Application Support"
            else:  # Linux and others
                base_dir = Path.home() / ".config"

            config_dir = base_dir / "apgi-framework"
            return PathManager.ensure_dir_exists(config_dir)

        except Exception as e:
            logger.warning(f"Could not create user config dir: {e}")
            # Fallback to project config
            return PathManager.get_config_dir()

    @staticmethod
    def get_user_data_dir() -> Path:
        """Get user-specific data directory."""
        try:
            if sys.platform == "win32":
                base_dir = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
            elif sys.platform == "darwin":
                base_dir = Path.home() / "Library" / "Application Support"
            else:  # Linux and others
                base_dir = Path.home() / ".local" / "share"

            data_dir = base_dir / "apgi-framework" / "data"
            return PathManager.ensure_dir_exists(data_dir)

        except Exception as e:
            logger.warning(f"Could not create user data dir: {e}")
            # Fallback to project data
            return PathManager.get_data_dir()

    @staticmethod
    def normalize_path(path: Path) -> Path:
        """Normalize path for cross-platform compatibility."""
        try:
            return path.resolve()
        except Exception:
            # Fallback to absolute path
            return path.absolute()

    @staticmethod
    def safe_file_operation(operation: Callable, file_path: Path, *args: Any, **kwargs: Any) -> Any:
        """Safely perform file operations with error handling."""
        try:
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Perform operation
            return operation(file_path, *args, **kwargs)

        except PermissionError:
            # Try user directory as fallback
            user_data_dir = PathManager.get_user_data_dir()
            fallback_path = user_data_dir / file_path.name
            logger.warning(f"Permission denied for {file_path}, trying {fallback_path}")

            try:
                return operation(fallback_path, *args, **kwargs)
            except Exception as e:
                logger.error(f"Failed file operation even with fallback: {e}")
                raise
        except Exception as e:
            logger.error(f"File operation failed: {e}")
            raise


def create_standard_notebook(parent: tk.Widget) -> ttk.Notebook:
    """Create a standard styled notebook."""
    notebook = ttk.Notebook(parent)
    notebook.pack(fill="both", expand=True, padx=10, pady=10)
    return notebook


def create_standard_button_frame(parent: tk.Widget) -> ttk.Frame:
    """Create a standard button frame."""
    frame = ttk.Frame(parent)
    frame.pack(fill="x", padx=10, pady=5)
    return frame


def create_standard_button(
    parent: tk.Widget, text: str, command: Callable, **kwargs: Any
) -> ttk.Button:
    """Create a standard styled button."""
    return ttk.Button(parent, text=text, command=command, **kwargs)


def show_info_dialog(title: str, message: str) -> None:
    """Show a standard info dialog."""
    messagebox.showinfo(title, message)


def show_warning_dialog(title: str, message: str) -> None:
    """Show a standard warning dialog."""
    messagebox.showwarning(title, message)


def show_error_dialog(title: str, message: str) -> None:
    """Show a standard error dialog."""
    messagebox.showerror(title, message)


def ask_yes_no_dialog(title: str, message: str) -> bool:
    """Show a standard yes/no dialog."""
    return messagebox.askyesno(title, message)
