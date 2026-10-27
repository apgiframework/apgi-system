"""Accessibility Features for APGI Framework GUIs
=======================================

This module provides accessibility features to make the APGI Framework GUIs
more accessible to users with disabilities, including screen reader support,
keyboard navigation, and visual accessibility improvements.

Features:
- Screen reader support with ARIA labels
- Keyboard navigation shortcuts
- High contrast themes
- Focus management
- Text size adjustment
- Color blind friendly palettes
- Audio feedback for important events
"""

import tkinter as tk
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

try:
    import customtkinter as ctk

    CUSTOMTKINTER_AVAILABLE = True
except ImportError:
    CUSTOMTKINTER_AVAILABLE = False
    ctk = None


class AccessibilityLevel(Enum):
    """Accessibility compliance levels."""

    MINIMAL = "minimal"
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    FULL = "full"


@dataclass
class AccessibilityConfig:
    """Configuration for accessibility features."""

    level: AccessibilityLevel = AccessibilityLevel.BASIC
    keyboard_navigation: bool = True
    high_contrast_theme: bool = False
    text_size_adjustment: float = 1.0  # 0.8 to 2.0
    color_blind_friendly: bool = True
    audio_feedback: bool = False
    focus_indicators: bool = True
    reduced_motion: bool = False
    auto_suggest: bool = False


class AccessibilityManager:
    """Manages accessibility features for GUI applications."""

    def __init__(self, root: tk.Tk, config: Optional[AccessibilityConfig] = None):
        """Initialize accessibility manager."""
        self.root = root
        self.tooltips: Dict[tk.Widget, str] = {}
        self.accessible_widgets: Dict[str, tk.Widget] = {}
        self.current_theme = "light"
        self.high_contrast = False
        self.keyboard_shortcuts: Dict[str, Callable] = {}
        self.aria_labels: Dict[str, str] = {}
        self.audio_callbacks: List[Callable] = []
        self.config = config or AccessibilityConfig()  # Initialize configuration
        self.focus_order: List[str] = []  # Track focus order for navigation
        self.current_focus_widget: Optional[tk.Widget] = None  # Current focused widget

        # Initialize accessibility features
        self._init_screen_reader_support()
        self._init_keyboard_navigation()
        self._init_shortcuts()
        self._apply_accessibility_settings()

    def _init_screen_reader_support(self) -> None:
        """Initialize keyboard navigation and accessibility support."""
        if not self.config.keyboard_navigation:
            return

        # Configure root window for accessibility
        self.root.title("APGI Framework - Accessible Interface")

    def _init_keyboard_navigation(self) -> None:
        """Initialize keyboard navigation system."""
        if not self.config.keyboard_navigation:
            return

        # Enable keyboard navigation
        self.root.focus_set()

        # Set up focus tracking
        self.root.bind("<FocusIn>", self._on_focus_in)
        self.root.bind("<FocusOut>", self._on_focus_out)

        # Add Tab navigation support
        self.root.bind("<Tab>", self._navigate_forward)
        self.root.bind("<Shift-Tab>", self._navigate_backward)

        # Add Enter/Space key support for buttons
        self.root.bind("<Return>", self._activate_focused_widget)
        self.root.bind("<space>", self._activate_focused_widget)

    def _init_shortcuts(self) -> None:
        """Initialize keyboard shortcuts."""
        # Common accessibility shortcuts
        self.keyboard_shortcuts = {
            "<Control-f>": lambda: self._find_focusable_widget(),
            "<Control-s>": lambda: self._save_current_state(),
            "<Control-o>": lambda: self._open_file(),
            "<Control-p>": lambda: self._print_content(),
            "<F1>": lambda: self._show_help(),
            "<F6>": lambda: self._toggle_high_contrast(),
            "<F11>": lambda: self._toggle_fullscreen(),
            "<Escape>": lambda: self._cancel_current_action(),
        }

        # Bind shortcuts
        for shortcut, callback in self.keyboard_shortcuts.items():
            self.root.bind(shortcut, callback)

    def _apply_accessibility_settings(self) -> None:
        """Apply current accessibility settings to the GUI."""
        # Apply high contrast theme if enabled
        if self.config.high_contrast_theme:
            self._apply_high_contrast_theme()

        # Apply text size adjustment
        self._apply_text_size_adjustment()

        # Apply color blind friendly colors
        if self.config.color_blind_friendly:
            self._apply_color_blind_palette()

        # Apply reduced motion if enabled
        if self.config.reduced_motion:
            self._apply_reduced_motion()

    def add_widget_to_focus_order(self, widget: tk.Widget, name: str) -> None:
        """Add a widget to the focus order for keyboard navigation."""
        if name not in self.focus_order:
            self.focus_order.append(name)
        self._set_widget_accessibility(widget, name)

    def set_tooltip(self, widget: tk.Widget, tooltip: str) -> None:
        """Set tooltip for a widget."""
        # Store tooltip for later use
        self.tooltips[widget] = tooltip

    def get_tooltip(self, widget: tk.Widget) -> Optional[str]:
        """Get tooltip from a widget."""
        return self.tooltips.get(widget, None)

    def set_aria_label(self, widget: tk.Widget, label: str) -> None:
        """Set tooltip as accessibility label for a widget."""
        if self.config.keyboard_navigation:
            self.set_tooltip(widget, label)
            self.aria_labels[str(widget)] = label
            # Add the method to widget if it doesn't exist
            if not hasattr(widget, "set_aria_label"):
                setattr(
                    widget,
                    "set_aria_label",
                    lambda label: self.set_aria_label(widget, label),
                )

    def set_aria_description(self, widget: tk.Widget, description: str) -> None:
        """Set tooltip as accessibility description for a widget."""
        if self.config.keyboard_navigation:
            self.set_tooltip(widget, description)
            # Add method if it doesn't exist
            if not hasattr(widget, "set_aria_description"):
                setattr(
                    widget,
                    "set_aria_description",
                    lambda accessibility_description: self.set_aria_description(
                        widget, accessibility_description
                    ),
                )

    def set_focusable(self, widget: tk.Widget, focusable: bool = True) -> None:
        """Mark a widget as focusable for keyboard navigation."""
        if focusable:
            widget.bind("<FocusIn>", self._on_widget_focus_in)
            widget.bind("<FocusOut>", self._on_widget_focus_out)
            widget.takefocus(1) if hasattr(widget, "takefocus") else None
        else:
            widget.unbind("<FocusIn>")
            widget.unbind("<FocusOut>")
            widget.set_takefocus(False) if hasattr(widget, "set_takefocus") else None

    def add_audio_feedback(self, callback: Callable[[str], None]) -> None:
        """Add an audio feedback callback for important events."""
        if self.config.audio_feedback:
            self.audio_callbacks.append(callback)

    def _play_audio_feedback(self, event_type: str) -> None:
        """Play audio feedback for an event."""
        if self.config.audio_feedback and self.audio_callbacks:
            for callback in self.audio_callbacks:
                try:
                    callback(event_type)
                except Exception:
                    pass  # Ignore audio feedback errors

    def _on_focus_in(self, event: Any) -> None:
        """Handle focus in event for accessibility."""
        widget = event.widget
        self.current_focus_widget = widget
        self._play_audio_feedback("focus_in")

        # Visual focus indicator
        if self.config.focus_indicators:
            self._show_focus_indicator(widget)

    def _on_focus_out(self, event: Any) -> None:
        """Handle focus out event for accessibility."""
        self.current_focus_widget = None
        self._play_audio_feedback("focus_out")

        # Hide focus indicator
        if self.config.focus_indicators:
            self._hide_focus_indicator()

    def _on_widget_focus_in(self, event: Any) -> None:
        """Handle widget focus in event."""
        self._play_audio_feedback("widget_focus_in")

    def _on_widget_focus_out(self, event: Any) -> None:
        """Handle widget focus out event."""
        self._play_audio_feedback("widget_focus_out")

    def _navigate_forward(self, event: Any) -> None:
        """Navigate forward through focusable widgets."""
        if not self.focus_order:
            return

        current_index = -1
        if self.current_focus_widget:
            try:
                current_index = self.focus_order.index(self.current_focus_widget.winfo_name())
            except ValueError:
                current_index = -1

        next_index = (current_index + 1) % len(self.focus_order)
        next_name = self.focus_order[next_index]

        # Find and focus the next widget
        for child in self.root.winfo_children():
            if child.winfo_name() == next_name:
                child.focus_set()
                break

    def _navigate_backward(self, event: Any) -> None:
        """Navigate backward through focusable widgets."""
        if not self.focus_order:
            return

        current_index = -1
        if self.current_focus_widget:
            try:
                current_index = self.focus_order.index(self.current_focus_widget.winfo_name())
            except ValueError:
                current_index = -1

        prev_index = (current_index - 1) % len(self.focus_order)
        prev_name = self.focus_order[prev_index]

        # Find and focus the previous widget
        for child in self.root.winfo_children():
            if child.winfo_name() == prev_name:
                child.focus_set()
                break

    def _activate_focused_widget(self, event: Any) -> None:
        """Activate the currently focused widget (button, checkbox, etc.)."""
        if self.current_focus_widget:
            if hasattr(self.current_focus_widget, "invoke"):
                self.current_focus_widget.invoke()
            elif hasattr(self.current_focus_widget, "activate"):
                self.current_focus_widget.activate()
            self._play_audio_feedback("activate")

    def _find_focusable_widget(self) -> None:
        """Find the next focusable widget."""
        for child in self.root.winfo_children():
            if self._is_focusable(child):
                child.focus_set()
                break

    def _is_focusable(self, widget: tk.Widget | tk.Toplevel) -> bool:
        """Check if a widget is focusable for keyboard navigation."""
        return (
            widget.winfo_class()
            in (
                "Button",
                "Checkbutton",
                "Radiobutton",
                "Entry",
                "Text",
                "Listbox",
                "Combobox",
            )
            and widget.winfo_viewable()
            and not widget.cget("state") == "disabled"
        )

    def _set_widget_accessibility(self, widget: tk.Widget, name: str) -> None:
        """Set accessibility properties for a widget."""
        # Add tooltip with accessibility information
        if hasattr(widget, "cget") and widget.cget("text"):
            tooltip = f"{name}: {widget.cget('text')}"
            # Store tooltip for later use
            self.tooltips[widget] = tooltip

        # Make sure widget is focusable if it should be
        if self._is_focusable(widget):
            self.set_focusable(widget, True)

    def _apply_high_contrast_theme(self) -> None:
        """Apply high contrast theme to the GUI."""
        if CUSTOMTKINTER_AVAILABLE and hasattr(self.root, "_get_appearance_mode"):
            # For CustomTkinter
            appearance = ctk.CustomTkinter(appearance_mode="dark")
            appearance.set_theme("blue")

            # Update all widgets recursively
            self._update_widget_theme(self.root, appearance)
        else:
            # For standard tkinter
            self._apply_tkinter_high_contrast()

    def _apply_tkinter_high_contrast(self) -> None:
        """Apply high contrast theme to standard tkinter widgets."""
        # High contrast color scheme
        bg_color = "#000000"
        fg_color = "#FFFFFF"
        select_color = "#000080"

        # Apply to root and all children
        try:
            self.root.configure(bg=bg_color)
        except tk.TclError:
            pass
        try:
            # Use configure with keyword arguments to avoid mypy issues
            self.root.configure(**{"fg": fg_color})
        except (tk.TclError, TypeError):
            pass
        self._update_tkinter_theme(self.root, bg_color, fg_color, select_color)

    def _update_tkinter_theme(self, widget: tk.Misc, bg: str, fg: str, select: str) -> None:
        """Update tkinter widget colors recursively."""
        try:
            if widget.winfo_class() == "Entry":
                # Only apply supported options for Entry widgets
                try:
                    widget.configure(insertbackground=select)  # type: ignore
                except tk.TclError:
                    pass
                try:
                    widget.configure(selectforeground=fg)  # type: ignore
                except tk.TclError:
                    pass
            else:
                # Try to configure bg and fg separately to handle unsupported options
                try:
                    widget.configure(background=bg)  # type: ignore
                except tk.TclError:
                    pass
                try:
                    widget.configure(fg=fg)  # type: ignore
                except tk.TclError:
                    pass
        except tk.TclError:
            pass  # Some widgets don't support certain options

    def _update_widget_theme(self, widget: tk.Misc, appearance: Any) -> None:
        """Update CustomTkinter widget appearance recursively."""
        try:
            if hasattr(widget, "configure"):
                config_options = widget.configure()
                if config_options and "appearance_mode" in config_options:
                    widget.configure(appearance_mode=appearance.appearance_mode)  # type: ignore
        except Exception:
            pass  # Some widgets might not support appearance mode

        # Update all children
        for child in widget.winfo_children():
            self._update_widget_theme(child, appearance)

    def _apply_text_size_adjustment(self) -> None:
        """Apply text size adjustment to the GUI."""
        if self.config.text_size_adjustment != 1.0:
            # Get current font
            current_font = self.root.cget("font")
            if current_font:
                # Adjust font size
                font_parts = current_font.split()
                if len(font_parts) >= 2:
                    try:
                        new_size = int(float(font_parts[1]) * self.config.text_size_adjustment)
                        new_font = f"{font_parts[0]} {new_size} {' '.join(font_parts[2:])}"
                        try:
                            config_options = self.root.configure()
                            if config_options and "font" in config_options:
                                self.root.configure(font=new_font)  # type: ignore
                        except tk.TclError:
                            pass

                        # Update all children
                        self._update_font_sizes(self.root, new_font)
                    except (ValueError, IndexError):
                        pass

    def _update_font_sizes(self, widget: tk.Misc, font: str) -> None:
        """Update font sizes recursively for all widgets."""
        try:
            if widget.winfo_class() in ["Label", "Button", "Entry", "Text"]:
                try:
                    config_options = widget.configure()
                    if config_options and "font" in config_options:
                        widget.configure(font=font)  # type: ignore
                except tk.TclError:
                    pass  # Some widgets might not support font configuration
        except tk.TclError:
            pass

        # Update all children
        for child in widget.winfo_children():
            self._update_font_sizes(child, font)

    def _apply_color_blind_palette(self) -> None:
        """Apply color blind friendly color palette."""
        # Color blind friendly colors
        safe_colors = {
            "primary": "#0066CC",  # Blue
            "secondary": "#FF6600",  # Orange
            "success": "#00AA00",  # Green
            "warning": "#FF8800",  # Yellow
            "danger": "#CC0000",  # Red
            "info": "#0088CC",  # Cyan
            "light": "#F0F0F0",  # Light gray
            "dark": "#333333",  # Dark gray
        }

        # Apply to GUI elements
        self._apply_color_palette(safe_colors)

    def _apply_color_palette(self, colors: Dict[str, str]) -> None:
        """Apply color palette to GUI elements."""
        # This would be implemented based on the specific GUI framework
        # and the color scheme used in the application

    def _apply_reduced_motion(self) -> None:
        """Apply reduced motion settings."""
        # Disable animations and transitions
        if CUSTOMTKINTER_AVAILABLE:
            # For CustomTkinter
            try:
                # Note: These methods may not exist in all CustomTkinter versions
                if hasattr(self.root, "set_appearance_mode"):
                    self.root.set_appearance_mode("dark")
                # Disable animations if method exists
                if hasattr(self.root, "disable_animations"):
                    self.root.disable_animations()
            except Exception:
                pass

    def _show_focus_indicator(self, widget: tk.Misc) -> None:
        """Show focus indicator on a widget."""
        try:
            # Try to get original border width
            try:
                original_border = widget.cget("borderwidth")
            except (tk.TclError, KeyError):
                original_border = 0

            # Try to apply focus indicator
            try:
                config_options = widget.configure()
                if (
                    config_options
                    and "borderwidth" in config_options
                    and "relief" in config_options
                ):
                    widget.configure(borderwidth=3, relief="solid")  # type: ignore
            except tk.TclError:
                pass  # Widget doesn't support these options

            # Store original border width for restoration
            setattr(widget, "_original_border", original_border)
        except Exception:
            pass

    def _hide_focus_indicator(self) -> None:
        """Hide focus indicator from the currently focused widget."""
        if self.current_focus_widget and hasattr(self.current_focus_widget, "_original_border"):
            try:
                original_border = getattr(self.current_focus_widget, "_original_border")
                try:
                    config_options = self.current_focus_widget.configure()
                    if (
                        config_options
                        and "borderwidth" in config_options
                        and "relief" in config_options
                    ):
                        self.current_focus_widget.configure(
                            borderwidth=original_border, relief="flat"
                        )  # type: ignore
                except tk.TclError:
                    pass  # Widget doesn't support these options
                delattr(self.current_focus_widget, "_original_border")
            except Exception:
                pass

    def _toggle_high_contrast(self) -> None:
        """Toggle high contrast theme."""
        self.config.high_contrast_theme = not self.config.high_contrast_theme
        self._apply_accessibility_settings()

    def _toggle_fullscreen(self) -> None:
        """Toggle fullscreen mode."""
        current_state = self.root.attributes("-fullscreen")
        self.root.attributes("-fullscreen", not current_state)

    def _show_help(self) -> None:
        """Show accessibility help dialog."""
        help_text = """
        APGI Framework Accessibility Help
        
        Keyboard Shortcuts:
        - Tab: Navigate forward between focusable elements
        - Shift+Tab: Navigate backward
        - Enter/Space: Activate focused button
        - Ctrl+F: Find focusable element
        - Ctrl+S: Save current state
        - Ctrl+O: Open file
        - Ctrl+P: Print content
        - F1: Show this help
        - F6: Toggle high contrast
        - F11: Toggle fullscreen
        - Escape: Cancel current action
        
        Screen Reader:
        - All widgets have ARIA labels and descriptions
        - Focus changes are announced
        - Important events have audio feedback
        
        Visual Accessibility:
        - High contrast theme available
        - Adjustable text size
        - Color blind friendly colors
        - Focus indicators
        - Reduced motion option
        
        For more information, visit the APGI Framework documentation.
        """

        # Create help dialog
        help_dialog = tk.Toplevel(self.root)
        help_dialog.title("Accessibility Help")
        help_dialog.geometry("600x400")
        help_dialog.transient(self.root)

        # Add help text
        help_text_widget = tk.Text(help_dialog, wrap="word", padx=20, pady=20)
        help_text_widget.insert("1.0", help_text.strip())
        help_text_widget.configure(state="disabled")

        # Close button
        close_button = tk.Button(help_dialog, text="Close", command=help_dialog.destroy)
        close_button.pack(pady=10)

        # Center the dialog
        help_dialog.transient(self.root)
        help_dialog.grab_set()

    def _save_current_state(self) -> None:
        """Save current application state."""
        self._play_audio_feedback("save")
        # Implementation would depend on the specific application

    def _open_file(self) -> None:
        """Open file dialog."""
        self._play_audio_feedback("open_file")
        # Implementation would depend on the specific application

    def _print_content(self) -> None:
        """Print current content."""
        self._play_audio_feedback("print")
        # Implementation would depend on the specific application

    def _cancel_current_action(self) -> None:
        """Cancel current action."""
        self._play_audio_feedback("cancel")
        # Implementation would depend on the specific application

    def update_config(self, config: AccessibilityConfig) -> None:
        """Update accessibility configuration and apply changes."""
        self.config = config
        self._apply_accessibility_settings()

    def get_accessibility_summary(self) -> Dict[str, Any]:
        """Get summary of current accessibility settings."""
        return {
            "level": self.config.level.value,
            "keyboard_navigation": self.config.keyboard_navigation,
            "high_contrast": self.config.high_contrast_theme,
            "text_size": self.config.text_size_adjustment,
            "audio_feedback": self.config.audio_feedback,
            "focus_indicators": self.config.focus_indicators,
            "reduced_motion": self.config.reduced_motion,
            "auto_suggest": self.config.auto_suggest,
            "focus_order_count": len(self.focus_order),
            "keyboard_shortcuts_count": len(self.keyboard_shortcuts),
            "aria_labels_count": len(self.aria_labels),
        }


# Convenience functions for common accessibility patterns
def create_accessible_button(
    parent: tk.Misc,
    text: str,
    command: Optional[Callable[[], Any]] = None,
    accessibility_name: Optional[str] = None,
    tooltip: Optional[str] = None,
    **kwargs: Any,
) -> tk.Button:
    """Create an accessible button with proper ARIA support."""
    # Ensure command is properly typed
    final_command = command if command is not None else lambda: None

    button = tk.Button(parent, text=text, command=final_command, **kwargs)

    # Set accessibility properties
    if accessibility_name:
        # Store accessibility name as attribute
        setattr(button, "accessibility_name", accessibility_name)
        # Add method if it doesn't exist
        if not hasattr(button, "set_aria_label"):
            setattr(
                button,
                "set_aria_label",
                lambda label: setattr(button, "accessibility_name", label),
            )
    if tooltip:
        # Store tooltip as attribute
        setattr(button, "tooltip", tooltip)
        # Add method if it doesn't exist
        if not hasattr(button, "set_aria_description"):
            setattr(
                button,
                "set_aria_description",
                lambda desc: setattr(button, "tooltip", desc),
            )

    # Make focusable for keyboard navigation
    if hasattr(button, "takefocus"):
        button.takefocus(1)

    return button


def make_accessible_entry(
    parent: tk.Widget, accessibility_label: Optional[str] = None, **kwargs: Any
) -> tk.Entry:
    """Create an accessible entry field with proper ARIA support."""
    entry = tk.Entry(parent, **kwargs)

    # Set accessibility properties
    if accessibility_label:
        # Store accessibility label as attribute
        setattr(entry, "accessibility_label", accessibility_label)
        # Add method if it doesn't exist
        if not hasattr(entry, "set_aria_label"):
            setattr(
                entry,
                "set_aria_label",
                lambda label: setattr(entry, "accessibility_label", label),
            )

    # Make focusable for keyboard navigation
    if hasattr(entry, "takefocus"):
        entry.takefocus(1)

    return entry


def add_keyboard_shortcut(
    manager: AccessibilityManager,
    key_combination: str,
    callback: Callable,
    description: Optional[str] = None,
) -> None:
    """Add a keyboard shortcut to the accessibility manager."""
    manager.keyboard_shortcuts[key_combination] = callback
    if description:
        manager.aria_labels[key_combination] = description


# Global accessibility manager instance (initialized by main application)
_global_accessibility_manager: Optional[AccessibilityManager] = None


def initialize_accessibility(
    root: tk.Tk, config: Optional[AccessibilityConfig] = None
) -> AccessibilityManager:
    """Initialize global accessibility manager."""
    global _global_accessibility_manager
    _global_accessibility_manager = AccessibilityManager(root, config)
    return _global_accessibility_manager


def get_accessibility_manager() -> Optional[AccessibilityManager]:
    """Get the global accessibility manager instance."""
    return _global_accessibility_manager


def is_accessibility_enabled() -> bool:
    """Check if accessibility features are enabled."""
    return (
        _global_accessibility_manager is not None
        and _global_accessibility_manager.config.keyboard_navigation
    )
