"""Font management system for APGI Framework.
Provides cross-platform font compatibility with fallback chains.
"""

import logging
import platform
import tkinter as tk
from tkinter import font as tkFont
from typing import Dict, List, Optional, Tuple, Union, cast

logger = logging.getLogger(__name__)


class FontManager:
    """Manages font selection with cross-platform fallback chains.

    Ensures consistent typography across Windows, macOS, and Linux systems
    by providing appropriate font families for each platform.
    """

    # Platform-specific font families
    PLATFORM_FONTS = {
        "Windows": {
            "sans-serif": ["Segoe UI", "Arial", "Helvetica", "sans-serif"],
            "serif": ["Cambria", "Times New Roman", "Georgia", "serif"],
            "monospace": ["Consolas", "Courier New", "monospace"],
            "ui": ["Segoe UI", "Tahoma", "Arial", "Helvetica"],
            "code": ["Consolas", "Courier New", "monospace"],
            "heading": ["Segoe UI Light", "Arial", "Helvetica"],
        },
        "Darwin": {  # macOS
            "sans-serif": ["SF Pro Display", "Helvetica Neue", "Arial", "sans-serif"],
            "serif": ["SF Pro Text", "Georgia", "Times New Roman", "serif"],
            "monospace": ["SF Mono", "Monaco", "Menlo", "monospace"],
            "ui": ["SF Pro Display", "Helvetica Neue", "Arial"],
            "code": ["SF Mono", "Monaco", "Menlo", "monospace"],
            "heading": ["SF Pro Display Light", "Helvetica Neue Light", "Arial"],
        },
        "Linux": {
            "sans-serif": ["DejaVu Sans", "Liberation Sans", "Arial", "sans-serif"],
            "serif": ["DejaVu Serif", "Liberation Serif", "Times New Roman", "serif"],
            "monospace": [
                "DejaVu Sans Mono",
                "Liberation Mono",
                "Courier New",
                "monospace",
            ],
            "ui": ["DejaVu Sans", "Ubuntu", "Arial", "Helvetica"],
            "code": ["DejaVu Sans Mono", "Ubuntu Mono", "monospace"],
            "heading": ["DejaVu Sans Light", "Ubuntu Light", "Arial"],
        },
    }

    # Generic fallback fonts (should work on any system)
    GENERIC_FALLBACKS = {
        "sans-serif": ["sans-serif"],
        "serif": ["serif"],
        "monospace": ["monospace"],
        "ui": ["sans-serif"],
        "code": ["monospace"],
        "heading": ["sans-serif"],
    }

    def __init__(self) -> None:
        """Initialize font manager and detect available fonts."""
        self.platform = platform.system()
        self.available_fonts = self._detect_available_fonts()
        self.font_cache: Dict[str, str] = {}

        logger.info(f"Font manager initialized for {self.platform}")
        logger.info(f"Found {len(self.available_fonts)} available fonts")

    def _detect_available_fonts(self) -> List[str]:
        """Detect available fonts on the current system.

        Returns:
            List of available font family names
        """
        try:
            # Create temporary root to access font families
            temp_root = tk.Tk()
            temp_root.withdraw()  # Hide the window

            # Get all available font families
            font_families = sorted(temp_root.tk.call("font", "families"))

            # Clean up
            temp_root.destroy()

            # Filter out system fonts that aren't suitable for UI
            filtered_fonts = []
            for font_name in font_families:
                # Skip decorative/symbol fonts
                if any(
                    skip in font_name.lower()
                    for skip in [
                        "symbol",
                        "dingbat",
                        "webding",
                        "wingding",
                        "braille",
                        "cursor",
                        "marlett",
                    ]
                ):
                    continue

                # Skip fonts with very long names (likely system fonts)
                if len(font_name) > 50:
                    continue

                filtered_fonts.append(font_name)

            return filtered_fonts

        except Exception as e:
            logger.warning(f"Failed to detect system fonts: {e}")
            return []

    def get_font_family(self, font_type: str = "ui") -> str:
        """Get the best available font family for the specified type.

        Args:
            font_type: Type of font ('ui', 'code', 'heading', 'sans-serif', etc.)

        Returns:
            Best available font family name
        """
        if font_type in self.font_cache:
            return cast(str, self.font_cache[font_type])

        # Get platform-specific font list
        platform_key = "Darwin" if self.platform == "Darwin" else self.platform
        font_list = self.PLATFORM_FONTS.get(platform_key, {}).get(font_type, [])

        # Add generic fallbacks
        font_list.extend(self.GENERIC_FALLBACKS.get(font_type, ["sans-serif"]))

        # Find first available font
        for font_name in font_list:
            if font_name in self.available_fonts:
                self.font_cache[font_type] = font_name
                logger.debug(f"Selected font for {font_type}: {font_name}")
                return font_name

        # Ultimate fallback
        fallback_font = "sans-serif"
        self.font_cache[font_type] = fallback_font
        logger.warning(f"No suitable font found for {font_type}, using fallback: {fallback_font}")
        return fallback_font

    def get_font(
        self,
        font_type: str = "ui",
        size: int = 12,
        weight: str = "normal",
        slant: str = "roman",
    ) -> tkFont.Font:
        """Get a configured font object.

        Args:
            font_type: Type of font ('ui', 'code', 'heading', etc.)
            size: Font size in points
            weight: Font weight ('normal', 'bold')
            slant: Font slant ('roman', 'italic')

        Returns:
            Configured tkinter Font object
        """
        family = self.get_font_family(font_type)

        try:
            font_obj = tkFont.Font(
                family=family,
                size=size,
                weight="normal" if weight == "normal" else "bold",
                slant="roman" if slant == "roman" else "italic",
            )
            return font_obj
        except Exception as e:
            logger.warning(f"Failed to create font {family}: {e}")
            # Try with generic font
            try:
                return tkFont.Font(
                    family="sans-serif",
                    size=size,
                    weight="normal" if weight == "normal" else "bold",
                    slant="roman" if slant == "roman" else "italic",
                )
            except Exception as e2:
                logger.error(f"Failed to create fallback font: {e2}")
                # Return system default
                return tkFont.nametofont("TkDefaultFont")

    def get_ui_font(self, size: int = 12) -> tkFont.Font:
        """Get UI-appropriate font."""
        return self.get_font("ui", size)

    def get_code_font(self, size: int = 11) -> tkFont.Font:
        """Get code/monospace font."""
        return self.get_font("code", size)

    def get_heading_font(self, size: int = 16, weight: str = "bold") -> tkFont.Font:
        """Get heading font."""
        return self.get_font("heading", size, weight=weight)

    def get_body_font(self, size: int = 12) -> tkFont.Font:
        """Get body text font."""
        return self.get_font("sans-serif", size)

    def get_font_info(self) -> dict:
        """Get information about available fonts and selections.

        Returns:
            Dictionary with font information
        """
        return {
            "platform": self.platform,
            "total_fonts": len(self.available_fonts),
            "selected_fonts": self.font_cache.copy(),
            "available_sample": self.available_fonts[:10],  # First 10 fonts
        }

    def test_font(self, font_name: str) -> bool:
        """Test if a font is available and usable.

        Args:
            font_name: Font family name to test

        Returns:
            True if font is available and usable
        """
        try:
            font_obj = tkFont.Font(family=font_name, size=12)
            # Try to measure text
            width = font_obj.measure("Test")
            return width > 0
        except Exception:
            return False

    def get_best_font_for_text(
        self, text: str, font_type: str = "ui", size: int = 12
    ) -> Tuple[str, tkFont.Font]:
        """Get the best font for rendering specific text.

        Args:
            text: Text to be rendered
            font_type: Preferred font type
            size: Font size

        Returns:
            Tuple of (font_name, font_object)
        """
        # Get preferred font list
        platform_key = "Darwin" if self.platform == "Darwin" else self.platform
        font_list = self.PLATFORM_FONTS.get(platform_key, {}).get(font_type, [])
        font_list.extend(self.GENERIC_FALLBACKS.get(font_type, ["sans-serif"]))

        # Test each font with the text
        for font_name in font_list:
            if font_name in self.available_fonts:
                try:
                    font_obj = tkFont.Font(family=font_name, size=size)
                    # Test if font can render the text
                    width = font_obj.measure(text)
                    if width > 0:
                        return font_name, font_obj
                except Exception:
                    continue

        # Fallback to generic font
        fallback_font = "sans-serif"
        try:
            font_obj = tkFont.Font(family=fallback_font, size=size)
            return fallback_font, font_obj
        except Exception:
            # System default
            default_font = tkFont.nametofont("TkDefaultFont")
            return "system", default_font


# Global font manager instance
_font_manager = None


def get_font_manager() -> FontManager:
    """Get global font manager with lazy initialization."""
    global _font_manager
    if _font_manager is None:
        _font_manager = FontManager()
    return _font_manager


def get_ui_font(size: int = 12) -> tkFont.Font:
    """Convenience function to get UI font."""
    return get_font_manager().get_ui_font(size)


def get_code_font(size: int = 11) -> tkFont.Font:
    """Convenience function to get code font."""
    return get_font_manager().get_code_font(size)


def get_heading_font(size: int = 16, weight: str = "bold") -> tkFont.Font:
    """Convenience function to get heading font."""
    return get_font_manager().get_heading_font(size, weight)


def get_body_font(size: int = 12) -> tkFont.Font:
    """Convenience function to get body font."""
    return get_font_manager().get_body_font(size)


def configure_widget_fonts(
    widget: Union[tk.Widget, tk.Toplevel],
    font_type: str = "ui",
    size: Optional[int] = None,
) -> None:
    """Configure font for a widget and its children.

    Args:
        widget: Root widget to configure
        font_type: Type of font to use
        size: Font size (uses default if None)
    """
    if size is None:
        # Get default size for widget type
        if isinstance(widget, (tk.Label, tk.Button)):
            size = 12
        elif isinstance(widget, tk.Entry):
            size = 11
        elif isinstance(widget, tk.Text):
            size = 11
        else:
            size = 12

    font_obj = get_font_manager().get_font(font_type, size)

    try:
        widget.configure(font=font_obj)  # type: ignore
    except Exception as e:
        logger.warning(f"Failed to configure font for widget: {e}")

    # Recursively configure children
    for child in widget.winfo_children():
        configure_widget_fonts(child, font_type, size)


# Font presets for common use cases
UI_FONT_PRESETS = {
    "small": ("ui", 10),
    "normal": ("ui", 12),
    "large": ("ui", 14),
    "xlarge": ("ui", 16),
    "heading_small": ("heading", 14, "bold"),
    "heading_normal": ("heading", 16, "bold"),
    "heading_large": ("heading", 18, "bold"),
    "code_small": ("code", 10),
    "code_normal": ("code", 11),
    "code_large": ("code", 12),
}


def get_preset_font(preset_name: str) -> tkFont.Font:
    """Get a font by preset name.

    Args:
        preset_name: Name of the preset

    Returns:
        Font object for the preset
    """
    if preset_name not in UI_FONT_PRESETS:
        logger.warning(f"Unknown font preset: {preset_name}")
        preset_name = "normal"

    preset = UI_FONT_PRESETS[preset_name]
    if len(preset) == 2:
        font_type, size = preset
        return get_font_manager().get_font(str(font_type), int(size), weight="normal")
    else:
        font_type, size, weight = preset
        return get_font_manager().get_font(str(font_type), int(size), weight=str(weight))
