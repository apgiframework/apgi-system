# APGI System Accessibility Guide

This document provides comprehensive information about accessibility features, WCAG compliance, and inclusive design practices in the APGI System GUI applications.

## Keyboard Navigation

### Enhanced Global Shortcuts

| Shortcut | Function | Context | Available In |
| ---------- | ---------- | --------- | ------------- |
| `Ctrl/Cmd + Q` | Quit application | Global | All GUIs |
| `Ctrl/Cmd + N` | New session | Global | apgi_gui.py, Assistant_GUI.py |
| `Ctrl/Cmd + O` | Open configuration | Global | All GUIs |
| `Ctrl/Cmd + S` | Save configuration | Global | All GUIs |
| `Ctrl/Cmd + E` | Export data | Global | All GUIs |
| `Ctrl/Cmd + R` | Reset simulation | Global | apgi_gui.py |
| `F5` | Start simulation | Global | apgi_gui.py |
| `F6` | Pause/Resume simulation | Global | apgi_gui.py |
| `F7` | Stop simulation | Global | apgi_gui.py |
| `F8` | Reset simulation | Global | apgi_gui.py |
| `F1` | Show help | Global | All GUIs |
| `F11` | Toggle fullscreen | Global | All GUIs |
| `Ctrl/Cmd + +` | Zoom in | Global | All GUIs |
| `Ctrl/Cmd + -` | Zoom out | Global | All GUIs |
| `Ctrl/Cmd + 0` | Reset zoom | Global | All GUIs |
| `Ctrl/Cmd + F` | Find/Search | Global | Assistant_GUI.py, Utils-GUI.py |
| `Ctrl/Cmd + G` | Find next | Global | Assistant_GUI.py, Utils-GUI.py |
| `Ctrl/Cmd + Shift + G` | Find previous | Global | Assistant_GUI.py, Utils-GUI.py |
| `Ctrl/Cmd + A` | Select all | Global | Text areas and inputs |
| `Ctrl/Cmd + C` | Copy | Global | Text areas and inputs |
| `Ctrl/Cmd + V` | Paste | Global | Text areas and inputs |
| `Ctrl/Cmd + X` | Cut | Global | Text areas and inputs |
| `Ctrl/Cmd + Z` | Undo | Global | Text areas and inputs |
| `Ctrl/Cmd + Y` | Redo | Global | Text areas and inputs |

### Advanced Navigation

#### Skip Links

- **Ctrl+Home**: Jump to main content area
- **Ctrl+End**: Jump to status/control area
- **Tab**: Skip to next major section
- **Shift+Tab**: Skip to previous major section

#### Focus Management

- **Arrow Keys**: Navigate within complex controls (tables, lists)
- **Page Up/Down**: Navigate through long content
- **Home/End**: Jump to beginning/end of lists or text
- **Ctrl+Arrow Keys**: Extended navigation in text areas

### Keyboard Customization

Users can customize keyboard shortcuts through the configuration file:

```yaml
accessibility:
  keyboard_shortcuts:
    quit: "Ctrl+Q"  # Default: Ctrl+Q
    new_session: "Ctrl+N"  # Default: Ctrl+N
    save: "Ctrl+S"  # Default: Ctrl+S
    zoom_in: "Ctrl+Plus"  # Default: Ctrl++
    zoom_out: "Ctrl+Minus"  # Default: Ctrl+-
    fullscreen: "F11"  # Default: F11
    help: "F1"  # Default: F1
  sticky_keys: false  # Enable sticky keys support
  key_repeat:
    delay: 500  # ms before repeat starts
    interval: 50  # ms between repeats
```

## High Contrast and Theme Support

### Built-in High Contrast Themes

#### High Contrast Dark Theme

```python
HIGH_CONTRAST_DARK = {
    'bg_color': '#000000',
    'fg_color': '#FFFFFF',
    'accent_color': '#00FF00',
    'error_color': '#FF0000',
    'warning_color': '#FFFF00',
    'success_color': '#00FF00',
    'button_bg': '#404040',
    'button_fg': '#FFFFFF',
    'input_bg': '#202020',
    'input_fg': '#FFFFFF',
    'border_color': '#FFFFFF',
    'font_family': 'Arial',
    'font_size': 12,
    'border_width': 2
}
```

#### High Contrast Light Theme

```python
HIGH_CONTRAST_LIGHT = {
    'bg_color': '#FFFFFF',
    'fg_color': '#000000',
    'accent_color': '#0000FF',
    'error_color': '#FF0000',
    'warning_color': '#FFA500',
    'success_color': '#008000',
    'button_bg': '#C0C0C0',
    'button_fg': '#000000',
    'input_bg': '#FFFFFF',
    'input_fg': '#000000',
    'border_color': '#000000',
    'font_family': 'Arial',
    'font_size': 12,
    'border_width': 2
}
```

### Color Blind Friendly Themes

#### Deuteranopia (Green-Weak) Theme

```python
DEUTERANOPIA_THEME = {
    'bg_color': '#FFFFFF',
    'fg_color': '#000000',
    'accent_color': '#FF6B35',  # Orange instead of green
    'error_color': '#FF0000',
    'warning_color': '#FFA500',
    'success_color': '#4169E1',  # Blue instead of green
    'info_color': '#9370DB'     # Purple for information
}
```

#### Protanopia (Red-Weak) Theme

```python
PROTANOPIA_THEME = {
    'bg_color': '#FFFFFF',
    'fg_color': '#000000',
    'accent_color': '#FFD700',  # Gold instead of red
    'error_color': '#8B0000',   # Dark red
    'warning_color': '#FFA500',
    'success_color': '#228B22', # Forest green
    'info_color': '#4169E1'     # Royal blue
}
```

#### Tritanopia (Blue-Weak) Theme

```python
TRITANOPIA_THEME = {
    'bg_color': '#FFFFFF',
    'fg_color': '#000000',
    'accent_color': '#FF6347',  # Tomato red
    'error_color': '#DC143C',   # Crimson
    'warning_color': '#FF8C00', # Dark orange
    'success_color': '#32CD32', # Lime green
    'info_color': '#9932CC'     # Dark orchid
}
```

## Font Scaling and Typography

### Responsive Font Scaling

```python
class ResponsiveTypography:
    def __init__(self):
        self.base_font_size = 10
        self.scale_factors = {
            'small': 0.875,
            'normal': 1.0,
            'large': 1.125,
            'extra_large': 1.25,
            'xx_large': 1.5
        }

    def get_font_size(self, scale_name: str = 'normal') -> int:
        """Get font size for given scale."""
        factor = self.scale_factors.get(scale_name, 1.0)
        return int(self.base_font_size * factor)

    def scale_font(self, base_size: int, scale_factor: float) -> int:
        """Scale font size by factor."""
        return max(8, int(base_size * scale_factor))  # Minimum 8pt

# Usage in GUI applications
typography = ResponsiveTypography()

# For different UI elements
heading_font = ("Arial", typography.get_font_size('large'), "bold")
body_font = ("Arial", typography.get_font_size('normal'))
caption_font = ("Arial", typography.get_font_size('small'))
```

### Font Accessibility Settings

```yaml
accessibility:
  typography:
    font_family: "Arial"  # Default system font
    base_size: 10  # Base font size in points
    scaling: "normal"  # small, normal, large, extra_large
    line_height: 1.5  # Line height multiplier
    letter_spacing: 0.5  # Additional letter spacing in pixels
    word_spacing: 1.0  # Word spacing multiplier
  readability:
    hyphenation: true  # Enable automatic hyphenation
    justification: false  # Avoid full justification
    uppercase_limit: 0.3  # Maximum uppercase text ratio
```

## GUI-Specific Accessibility Features

### APGI Main GUI (apgi_gui.py)

#### Keyboard Navigation Features

- **Tab Order**: Logical progression through parameter controls, buttons, and plots
- **Plot Navigation**: Arrow keys to navigate plot focus areas
- **Parameter Adjustment**: Up/Down arrows for slider controls
- **Simulation Control**: Dedicated function keys for simulation lifecycle

#### Dynamic Content Announcements

- **Live Regions**: Status updates announced automatically
- **Chart Descriptions**: Alt-text equivalents for plots and visualizations
- **Progress Announcements**: Simulation progress announced during long operations

#### Focus Indicators

```python
class AccessibleFocusFrame(ttk.Frame):
    """Frame with high-contrast focus indicators."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(style='Accessible.TFrame')

        # Bind focus events
        self.bind('<FocusIn>', self._on_focus_in)
        self.bind('<FocusOut>', self._on_focus_out)

    def _on_focus_in(self, event):
        """Handle focus gained."""
        self.configure(style='AccessibleFocus.TFrame')
        # Announce to screen readers
        self._announce_focus()

    def _on_focus_out(self, event):
        """Handle focus lost."""
        self.configure(style='Accessible.TFrame')

    def _announce_focus(self):
        """Announce focus change to screen readers."""
        if hasattr(self, 'accessible_name'):
            announce_to_screen_reader(f"Focused on {self.accessible_name}")
```

### Assistant GUI (Assistant_GUI.py)

#### Query Input Accessibility

- **Auto-complete**: Announced completions for partial queries
- **Input Validation**: Real-time feedback on query format
- **Response Navigation**: Structured navigation through response sections

#### Conversation History

- **Message Navigation**: Jump between user queries and assistant responses
- **Timestamp Information**: Accessible time information for each message
- **Response Categories**: Semantic markup for different response types

### Psychological States GUI (Psychological-States-GUI.py)

#### Visualization Accessibility

- **Alternative Representations**: Text descriptions of visual states
- **Parameter Descriptions**: Detailed explanations of psychological parameters
- **State Change Announcements**: Audio feedback for state transitions

#### Interactive Controls

- **Slider Accessibility**: Value announcements during adjustment
- **Color Indicators**: Text descriptions of color-coded elements
- **Graph Navigation**: Keyboard navigation through data points

### Utils GUI (Utils-GUI.py)

#### Utility Execution Feedback

- **Progress Announcements**: Step-by-step execution feedback
- **Result Navigation**: Structured access to utility outputs
- **Error Descriptions**: Detailed error information with suggested fixes

## Testing Procedures

### Automated Accessibility Testing

```python
class AccessibilityTestSuite:
    def test_keyboard_navigation(self):
        """Test keyboard navigation coverage."""
        # Verify all interactive elements are keyboard accessible
        accessible_elements = self._get_accessible_elements()
        keyboard_reachable = self._test_keyboard_reachability()

        assert len(keyboard_reachable) >= len(accessible_elements) * 0.95

    def test_screen_reader_compatibility(self):
        """Test screen reader compatibility."""
        # Simulate screen reader interactions
        reader_announcements = self._simulate_screen_reader()

        # Verify essential information is announced
        required_announcements = [
            "Application title",
            "Main content area",
            "Navigation controls",
            "Status information"
        ]

        for announcement in required_announcements:
            assert announcement in reader_announcements

    def test_color_contrast(self):
        """Test color contrast ratios."""
        color_pairs = self._get_ui_color_pairs()

        for fg_color, bg_color in color_pairs:
            contrast_ratio = self._calculate_contrast_ratio(fg_color, bg_color)
            assert contrast_ratio >= 4.5, f"Insufficient contrast: {contrast_ratio}"

    def test_focus_indicators(self):
        """Test focus indicator visibility."""
        focusable_elements = self._get_focusable_elements()

        for element in focusable_elements:
            focus_visible = self._test_focus_visibility(element)
            assert focus_visible, f"Focus indicator not visible for {element}"

    def test_alt_text_completeness(self):
        """Test alt-text completeness for visual elements."""
        visual_elements = self._get_visual_elements()
        elements_with_alt = self._get_elements_with_alt_text()

        coverage = len(elements_with_alt) / len(visual_elements)
        assert coverage >= 0.9, f"Alt-text coverage: {coverage:.1%}"

    def test_responsive_design(self):
        """Test responsive design at different scales."""
        test_scales = [100, 125, 150, 200]  # Percentage scaling

        for scale in test_scales:
            self._set_display_scale(scale)
            layout_valid = self._test_layout_integrity()
            assert layout_valid, f"Layout broken at {scale}% scale"
```

### Manual Testing Checklist

#### Keyboard Navigation Testing

- [ ] All interactive elements reachable via keyboard
- [ ] Logical tab order maintained
- [ ] Skip links functional
- [ ] No keyboard traps present
- [ ] Custom shortcuts documented and functional

#### Screen Reader Testing

- [ ] All UI elements properly labeled
- [ ] Dynamic content announced
- [ ] Complex widgets properly described
- [ ] Error messages announced
- [ ] Progress information conveyed

#### Visual Accessibility Testing

- [ ] Color contrast meets WCAG standards
- [ ] Focus indicators clearly visible
- [ ] High contrast themes functional
- [ ] Text scaling works correctly
- [ ] Content doesn't break at high zoom levels

#### Motor Accessibility Testing

- [ ] Large click targets (44x44px minimum)
- [ ] Adequate spacing between controls
- [ ] Touch-friendly interface elements
- [ ] Sticky keys support
- [ ] Key repeat settings configurable

## Future Accessibility Roadmap

### Phase 1: Core Improvements (Next Release)

- [ ] Complete WCAG 2.1 AA compliance audit
- [ ] Implement screen reader announcements for all dynamic content
- [ ] Add high contrast theme toggle
- [ ] Enhance keyboard navigation for complex visualizations
- [ ] Add customizable keyboard shortcuts

### Phase 2: Advanced Features (Q2 2024)

- [ ] Voice control integration
- [ ] Braille display support
- [ ] Advanced gesture navigation
- [ ] AI-powered accessibility assistance
- [ ] Multi-modal interaction support

### Phase 3: Industry Leadership (Q4 2024)

- [ ] WCAG 2.2 compliance
- [ ] Advanced accessibility analytics
- [ ] Third-party accessibility tool integration
- [ ] Community accessibility contribution guidelines
- [ ] Accessibility certification targets

## Configuration

### Accessibility Configuration File

```yaml
# accessibility.yaml
accessibility:
  enabled: true
  theme: "high_contrast_dark"  # default, high_contrast_light, deuteranopia, etc.
  screen_reader:
    announcements: true
    live_regions: true
    detailed_descriptions: true
  keyboard:
    navigation: true
    shortcuts: true
    sticky_keys: false
    key_repeat:
      delay: 500
      interval: 50
  typography:
    scaling: "normal"
    font_family: "Arial"
    dyslexia_friendly: false
  motor:
    large_targets: true
    touch_friendly: true
    gesture_support: false
  cognitive:
    simplified_interface: false
    progress_indicators: true
    error_prevention: true
## GUI-Specific Accessibility Features
  apgi_gui:
    theme: "deuteranopia"
    keyboard_shortcuts:
      start_simulation: "Ctrl+Enter"
  assistant_gui:
    screen_reader:
      detailed_descriptions: true
  psychological_states_gui:
    motor:
      gesture_support: true
```

## Support and Resources

### Getting Help

- **Documentation**: This accessibility guide and inline help (F1)
- **Community**: GitHub discussions for accessibility topics
- **Professional Support**: Contact <accessibility@apgi-system.com>
- **Standards References**:
  - [WCAG 2.1 Guidelines](https://www.w3.org/TR/WCAG21/)
  - [Section 508 Standards](https://www.section508.gov/)
  - [EN 301 549](https://www.etsi.org/deliver/etsi_en/301500_301599/301549/03.02.01_60/en_301549v030201p.pdf)

### Contributing to Accessibility

We welcome contributions that improve accessibility:

1. **Report Issues**: Use the accessibility label on GitHub issues
2. **Test Changes**: Include accessibility testing in PRs
3. **Design Reviews**: Accessibility review required for UI changes
4. **Documentation**: Update this guide when adding accessibility features

### Accessibility Metrics

We track accessibility compliance through automated metrics:

- **Keyboard Navigation Coverage**: Target >95%
- **Screen Reader Compatibility**: Target >90%
- **Color Contrast Compliance**: Target 100%
- **Focus Indicator Visibility**: Target 100%
- **Alt-text Coverage**: Target >90%
- **WCAG Success Criteria**: Target 100% AA compliance
