# APGI Design System

## Overview

This document defines the unified design system for the APGI (Allostatic Precision-Gated Ignition) system. The design system ensures consistency, accessibility, and user experience quality across all GUI applications and interfaces.

## Design Principles

### Core Principles

1. **Consistency**: Uniform appearance and behavior across all applications
2. **Accessibility**: WCAG 2.1 AA compliance with inclusive design practices
3. **Simplicity**: Clean, uncluttered interfaces focused on functionality
4. **Scalability**: Design that works across different screen sizes and contexts
5. **Feedback**: Clear visual and auditory feedback for all user interactions

### Scientific Accuracy

The design system incorporates principles from cognitive psychology and neuroscience:

- **Information Hierarchy**: Based on Gestalt principles and visual perception research
- **Color Psychology**: Evidence-based color choices for different information types
- **Cognitive Load**: Minimized through progressive disclosure and chunking
- **Error Prevention**: Design patterns that reduce user errors and provide clear guidance

## Color Palette

### Primary Colors

```python
PRIMARY_COLORS = {
    'primary': '#2563eb',      # Blue - main brand color
    'primary_dark': '#1d4ed8', # Darker blue for hover states
    'primary_light': '#3b82f6', # Lighter blue for accents
    'secondary': '#64748b',    # Slate - secondary actions
    'accent': '#f59e0b',       # Amber - highlights and warnings
}
```

### Semantic Colors

```python
SEMANTIC_COLORS = {
    'success': '#10b981',      # Green - positive actions and states
    'warning': '#f59e0b',      # Amber - warnings and cautions
    'error': '#ef4444',        # Red - errors and critical issues
    'info': '#3b82f6',         # Blue - informational content
    'neutral': '#6b7280',      # Gray - neutral states and text
}
```

### Data Visualization Colors

```python
DATA_COLORS = {
    'delta': '#8b5cf6',        # Purple - delta waves
    'theta': '#06b6d4',        # Cyan - theta waves
    'alpha': '#10b981',        # Green - alpha waves
    'beta': '#f59e0b',         # Amber - beta waves
    'gamma': '#ef4444',        # Red - gamma waves
    'background': '#f8fafc',   # Light gray background
    'grid': '#e2e8f0',         # Light blue grid lines
}
```

### Accessibility Color Variants

```python
ACCESSIBILITY_COLORS = {
    'high_contrast': {
        'bg_primary': '#ffffff',
        'bg_secondary': '#f8fafc',
        'text_primary': '#000000',
        'text_secondary': '#374151',
        'border': '#000000',
        'focus': '#2563eb'
    },
    'deuteranopia': {  # Green-weak color blindness
        'success': '#2563eb',   # Blue instead of green
        'error': '#dc2626',    # Red stays red
        'warning': '#ea580c',  # Orange for warnings
        'info': '#7c3aed'      # Purple for info
    },
    'protanopia': {    # Red-weak color blindness
        'success': '#059669',  # Green stays green
        'error': '#7c2d12',    # Brown instead of red
        'warning': '#d97706',  # Amber stays amber
        'info': '#2563eb'      # Blue stays blue
    }
}
```

## Typography

### Font Families

```python
TYPOGRAPHY = {
    'primary': 'Inter',        # Main UI font - clean, modern
    'monospace': 'JetBrains Mono',  # Code and data display
    'fallback': 'system-ui, -apple-system, sans-serif'
}
```

### Font Scales

```python
FONT_SCALES = {
    'xs': 10,     # Small captions
    'sm': 12,     # Body text
    'base': 14,   # Default body
    'lg': 16,     # Large body
    'xl': 18,     # Headings
    '2xl': 20,    # Large headings
    '3xl': 24,    # Page titles
    '4xl': 32     # Hero text
}
```

### Font Weights

```python
FONT_WEIGHTS = {
    'light': 300,
    'normal': 400,
    'medium': 500,
    'semibold': 600,
    'bold': 700,
    'extrabold': 800
}
```

### Typography Usage

```python
TEXT_STYLES = {
    'h1': {
        'font_family': 'Inter',
        'font_size': 32,
        'font_weight': 700,
        'line_height': 1.2,
        'color': '#111827'
    },
    'h2': {
        'font_family': 'Inter',
        'font_size': 24,
        'font_weight': 600,
        'line_height': 1.3,
        'color': '#111827'
    },
    'body': {
        'font_family': 'Inter',
        'font_size': 14,
        'font_weight': 400,
        'line_height': 1.5,
        'color': '#374151'
    },
    'caption': {
        'font_family': 'Inter',
        'font_size': 12,
        'font_weight': 400,
        'line_height': 1.4,
        'color': '#6b7280'
    },
    'code': {
        'font_family': 'JetBrains Mono',
        'font_size': 12,
        'font_weight': 400,
        'line_height': 1.4,
        'color': '#111827',
        'background': '#f8fafc'
    }
}
```

## Spacing Scale

### Spacing Tokens

```python
SPACING = {
    'px': 1,
    '0.5': 2,     # 2px
    '1': 4,       # 4px
    '1.5': 6,     # 6px
    '2': 8,       # 8px
    '2.5': 10,    # 10px
    '3': 12,      # 12px
    '3.5': 14,    # 14px
    '4': 16,      # 16px
    '5': 20,      # 20px
    '6': 24,      # 24px
    '7': 28,      # 28px
    '8': 32,      # 32px
    '9': 36,      # 36px
    '10': 40,     # 40px
    '11': 44,     # 44px
    '12': 48,     # 48px
    '16': 64,     # 64px
    '20': 80,     # 80px
    '24': 96,     # 96px
    '32': 128     # 128px
}
```

### Layout Spacing

```python
LAYOUT_SPACING = {
    'page_padding': SPACING['8'],      # 32px
    'section_gap': SPACING['6'],       # 24px
    'element_gap': SPACING['4'],       # 16px
    'control_gap': SPACING['3'],       # 12px
    'border_radius': SPACING['1'],     # 4px
    'border_width': 1,                 # 1px
    'shadow_blur': SPACING['2'],       # 8px
}
```

## Component Library

### Buttons

#### Primary Button

```python
PRIMARY_BUTTON = {
    'background': PRIMARY_COLORS['primary'],
    'color': '#ffffff',
    'border': f"1px solid {PRIMARY_COLORS['primary']}",
    'border_radius': LAYOUT_SPACING['border_radius'],
    'padding': f"{SPACING['2']}px {SPACING['4']}px",
    'font_family': TYPOGRAPHY['primary'],
    'font_size': FONT_SCALES['base'],
    'font_weight': FONT_WEIGHTS['medium'],
    'cursor': 'pointer',
    'transition': 'all 0.2s ease',
    'hover': {
        'background': PRIMARY_COLORS['primary_dark'],
        'border': f"1px solid {PRIMARY_COLORS['primary_dark']}",
        'transform': 'translateY(-1px)',
        'box_shadow': f"0 {SPACING['1']}px {SPACING['2']}px rgba(0, 0, 0, 0.1)"
    },
    'active': {
        'transform': 'translateY(0)',
        'box_shadow': 'none'
    },
    'disabled': {
        'background': '#e5e7eb',
        'color': '#9ca3af',
        'cursor': 'not-allowed',
        'transform': 'none'
    }
}
```

#### Secondary Button

```python
SECONDARY_BUTTON = {
    'background': 'transparent',
    'color': PRIMARY_COLORS['primary'],
    'border': f"1px solid {PRIMARY_COLORS['primary']}",
    'border_radius': LAYOUT_SPACING['border_radius'],
    'padding': f"{SPACING['2']}px {SPACING['4']}px",
    'font_family': TYPOGRAPHY['primary'],
    'font_size': FONT_SCALES['base'],
    'font_weight': FONT_WEIGHTS['medium'],
    'cursor': 'pointer',
    'transition': 'all 0.2s ease',
    'hover': {
        'background': PRIMARY_COLORS['primary'],
        'color': '#ffffff'
    }
}
```

#### Icon Button

```python
ICON_BUTTON = {
    'width': SPACING['8'],      # 32px
    'height': SPACING['8'],     # 32px
    'border_radius': LAYOUT_SPACING['border_radius'],
    'background': 'transparent',
    'border': 'none',
    'color': '#6b7280',
    'cursor': 'pointer',
    'display': 'flex',
    'align_items': 'center',
    'justify_content': 'center',
    'transition': 'all 0.2s ease',
    'hover': {
        'background': '#f3f4f6',
        'color': '#374151'
    },
    'focus': {
        'outline': f"2px solid {PRIMARY_COLORS['primary']}",
        'outline_offset': 2
    }
}
```

### Form Controls

#### Text Input

```python
TEXT_INPUT = {
    'width': '100%',
    'padding': f"{SPACING['2']}px {SPACING['3']}px",
    'border': f"1px solid {SEMANTIC_COLORS['neutral']}",
    'border_radius': LAYOUT_SPACING['border_radius'],
    'font_family': TYPOGRAPHY['primary'],
    'font_size': FONT_SCALES['base'],
    'color': '#111827',
    'background': '#ffffff',
    'transition': 'border-color 0.2s ease, box-shadow 0.2s ease',
    'focus': {
        'border': f"1px solid {PRIMARY_COLORS['primary']}",
        'outline': 'none',
        'box_shadow': f"0 0 0 3px rgba(37, 99, 235, 0.1)"
    },
    'error': {
        'border': f"1px solid {SEMANTIC_COLORS['error']}",
        'box_shadow': f"0 0 0 3px rgba(239, 68, 68, 0.1)"
    },
    'disabled': {
        'background': '#f9fafb',
        'color': '#9ca3af',
        'cursor': 'not-allowed'
    }
}
```

#### Slider Control

```python
SLIDER_CONTROL = {
    'track': {
        'height': 4,
        'background': '#e5e7eb',
        'border_radius': 2
    },
    'track_filled': {
        'background': PRIMARY_COLORS['primary'],
        'border_radius': 2
    },
    'thumb': {
        'width': SPACING['4'],      # 16px
        'height': SPACING['4'],     # 16px
        'background': '#ffffff',
        'border': f"2px solid {PRIMARY_COLORS['primary']}",
        'border_radius': '50%',
        'cursor': 'pointer',
        'box_shadow': '0 1px 3px rgba(0, 0, 0, 0.1)',
        'hover': {
            'transform': 'scale(1.1)'
        },
        'focus': {
            'outline': f"2px solid {PRIMARY_COLORS['primary']}",
            'outline_offset': 2
        }
    },
    'value_display': {
        'font_family': TYPOGRAPHY['monospace'],
        'font_size': FONT_SCALES['sm'],
        'color': '#6b7280',
        'margin_left': SPACING['2']
    }
}
```

### Data Visualization

#### Chart Container

```python
CHART_CONTAINER = {
    'background': '#ffffff',
    'border': f"1px solid {SEMANTIC_COLORS['neutral']}",
    'border_radius': LAYOUT_SPACING['border_radius'] * 2,
    'padding': SPACING['4'],
    'box_shadow': '0 1px 3px rgba(0, 0, 0, 0.1)',
    'title': {
        'font_family': TYPOGRAPHY['primary'],
        'font_size': FONT_SCALES['lg'],
        'font_weight': FONT_WEIGHTS['semibold'],
        'color': '#111827',
        'margin_bottom': SPACING['3']
    }
}
```

#### Plot Styles

```python
PLOT_STYLES = {
    'line': {
        'width': 2,
        'colors': DATA_COLORS,
        'marker': {
            'size': 4,
            'colors': DATA_COLORS
        }
    },
    'grid': {
        'color': DATA_COLORS['grid'],
        'width': 1,
        'style': '--'
    },
    'axes': {
        'label_font': {
            'family': TYPOGRAPHY['primary'],
            'size': FONT_SCALES['sm'],
            'color': '#374151'
        },
        'tick_font': {
            'family': TYPOGRAPHY['primary'],
            'size': FONT_SCALES['xs'],
            'color': '#6b7280'
        }
    },
    'legend': {
        'background': '#ffffff',
        'border': f"1px solid {DATA_COLORS['grid']}",
        'border_radius': LAYOUT_SPACING['border_radius'],
        'font_family': TYPOGRAPHY['primary'],
        'font_size': FONT_SCALES['sm'],
        'padding': SPACING['2']
    }
}
```

### Status Indicators

#### Status Badge

```python
STATUS_BADGE = {
    'base': {
        'display': 'inline-flex',
        'align_items': 'center',
        'padding': f"{SPACING['0.5']}px {SPACING['2']}px",
        'border_radius': LAYOUT_SPACING['border_radius'] * 2,
        'font_family': TYPOGRAPHY['primary'],
        'font_size': FONT_SCALES['xs'],
        'font_weight': FONT_WEIGHTS['medium'],
        'text_transform': 'uppercase',
        'letter_spacing': '0.05em'
    },
    'success': {
        'background': f"rgba(16, 185, 129, 0.1)",
        'color': SEMANTIC_COLORS['success'],
        'border': f"1px solid rgba(16, 185, 129, 0.2)"
    },
    'warning': {
        'background': f"rgba(245, 158, 11, 0.1)",
        'color': SEMANTIC_COLORS['warning'],
        'border': f"1px solid rgba(245, 158, 11, 0.2)"
    },
    'error': {
        'background': f"rgba(239, 68, 68, 0.1)",
        'color': SEMANTIC_COLORS['error'],
        'border': f"1px solid rgba(239, 68, 68, 0.2)"
    },
    'info': {
        'background': f"rgba(59, 130, 246, 0.1)",
        'color': SEMANTIC_COLORS['info'],
        'border': f"1px solid rgba(59, 130, 246, 0.2)"
    }
}
```

#### Progress Indicator

```python
PROGRESS_INDICATOR = {
    'container': {
        'width': '100%',
        'height': 8,
        'background': '#e5e7eb',
        'border_radius': 4,
        'overflow': 'hidden'
    },
    'bar': {
        'height': '100%',
        'background': PRIMARY_COLORS['primary'],
        'border_radius': 4,
        'transition': 'width 0.3s ease'
    },
    'animated': {
        'background': 'linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent)',
        'background_size': '200% 100%',
        'animation': 'loading 1.5s infinite'
    },
    'label': {
        'font_family': TYPOGRAPHY['primary'],
        'font_size': FONT_SCALES['sm'],
        'color': '#6b7280',
        'margin_top': SPACING['1'],
        'text_align': 'center'
    }
}
```

## Layout Patterns

### Application Layout

#### Main Application Structure

```python
MAIN_LAYOUT = {
    'container': {
        'width': '100vw',
        'height': '100vh',
        'background': '#f8fafc',
        'font_family': TYPOGRAPHY['primary']
    },
    'header': {
        'height': 64,
        'background': '#ffffff',
        'border_bottom': f"1px solid {SEMANTIC_COLORS['neutral']}",
        'padding': f"0 {LAYOUT_SPACING['page_padding']}px",
        'display': 'flex',
        'align_items': 'center',
        'justify_content': 'space-between'
    },
    'sidebar': {
        'width': 280,
        'background': '#ffffff',
        'border_right': f"1px solid {SEMANTIC_COLORS['neutral']}",
        'padding': LAYOUT_SPACING['page_padding']
    },
    'main_content': {
        'flex': 1,
        'padding': LAYOUT_SPACING['page_padding'],
        'overflow': 'auto'
    },
    'footer': {
        'height': 48,
        'background': '#ffffff',
        'border_top': f"1px solid {SEMANTIC_COLORS['neutral']}",
        'padding': f"0 {LAYOUT_SPACING['page_padding']}px",
        'display': 'flex',
        'align_items': 'center',
        'justify_content': 'space-between'
    }
}
```

### Card Layout

```python
CARD_LAYOUT = {
    'container': {
        'background': '#ffffff',
        'border': f"1px solid {SEMANTIC_COLORS['neutral']}",
        'border_radius': LAYOUT_SPACING['border_radius'] * 2,
        'box_shadow': '0 1px 3px rgba(0, 0, 0, 0.1)',
        'padding': SPACING['6']
    },
    'header': {
        'margin_bottom': SPACING['4'],
        'padding_bottom': SPACING['3'],
        'border_bottom': f"1px solid {SEMANTIC_COLORS['neutral']}"
    },
    'title': {
        'font_family': TYPOGRAPHY['primary'],
        'font_size': FONT_SCALES['xl'],
        'font_weight': FONT_WEIGHTS['semibold'],
        'color': '#111827',
        'margin': 0
    },
    'subtitle': {
        'font_family': TYPOGRAPHY['primary'],
        'font_size': FONT_SCALES['base'],
        'color': '#6b7280',
        'margin': f"{SPACING['1']}px 0 0 0"
    },
    'content': {
        'margin_bottom': SPACING['4']
    },
    'footer': {
        'padding_top': SPACING['3'],
        'border_top': f"1px solid {SEMANTIC_COLORS['neutral']}",
        'display': 'flex',
        'justify_content': 'flex-end',
        'gap': SPACING['3']
    }
}
```

### Form Layout

```python
FORM_LAYOUT = {
    'group': {
        'margin_bottom': SPACING['6']
    },
    'label': {
        'display': 'block',
        'font_family': TYPOGRAPHY['primary'],
        'font_size': FONT_SCALES['sm'],
        'font_weight': FONT_WEIGHTS['medium'],
        'color': '#374151',
        'margin_bottom': SPACING['1.5']
    },
    'control': {
        'margin_bottom': SPACING['3']
    },
    'help_text': {
        'font_family': TYPOGRAPHY['primary'],
        'font_size': FONT_SCALES['xs'],
        'color': '#6b7280',
        'margin_top': SPACING['1']
    },
    'error_message': {
        'font_family': TYPOGRAPHY['primary'],
        'font_size': FONT_SCALES['xs'],
        'color': SEMANTIC_COLORS['error'],
        'margin_top': SPACING['1']
    },
    'actions': {
        'padding_top': SPACING['4'],
        'border_top': f"1px solid {SEMANTIC_COLORS['neutral']}",
        'display': 'flex',
        'justify_content': 'flex-end',
        'gap': SPACING['3']
    }
}
```

## Animation and Transitions

### Micro-interactions

```python
ANIMATIONS = {
    'button_press': {
        'duration': '0.1s',
        'timing': 'ease-out',
        'transform': 'scale(0.98)'
    },
    'hover_lift': {
        'duration': '0.2s',
        'timing': 'ease-out',
        'transform': 'translateY(-2px)',
        'box_shadow': '0 4px 12px rgba(0, 0, 0, 0.15)'
    },
    'fade_in': {
        'duration': '0.3s',
        'timing': 'ease-out',
        'opacity': '0 to 1'
    },
    'slide_in': {
        'duration': '0.3s',
        'timing': 'ease-out',
        'transform': 'translateX(-100%) to translateX(0)'
    },
    'progress_fill': {
        'duration': '0.5s',
        'timing': 'ease-in-out',
        'width': '0% to 100%'
    }
}
```

### Loading States

```python
LOADING_STATES = {
    'spinner': {
        'width': SPACING['4'],
        'height': SPACING['4'],
        'border': f"2px solid {SEMANTIC_COLORS['neutral']}",
        'border_top': f"2px solid {PRIMARY_COLORS['primary']}",
        'border_radius': '50%',
        'animation': 'spin 1s linear infinite'
    },
    'pulse': {
        'background': PRIMARY_COLORS['primary'],
        'animation': 'pulse 2s ease-in-out infinite',
        'opacity': '0.5 to 1'
    },
    'skeleton': {
        'background': '#f3f4f6',
        'animation': 'skeleton 1.5s ease-in-out infinite',
        'gradient': f"linear-gradient(90deg, #f3f4f6 25%, #e5e7eb 50%, #f3f4f6 75%)"
    }
}
```

## Responsive Design

### Breakpoints

```python
BREAKPOINTS = {
    'mobile': 640,    # < 640px
    'tablet': 768,    # >= 640px
    'desktop': 1024,  # >= 768px
    'wide': 1280      # >= 1024px
}
```

### Responsive Typography

```python
RESPONSIVE_TYPOGRAPHY = {
    'mobile': {
        'h1': 24,
        'h2': 20,
        'body': 14,
        'caption': 12
    },
    'tablet': {
        'h1': 28,
        'h2': 22,
        'body': 15,
        'caption': 13
    },
    'desktop': {
        'h1': 32,
        'h2': 24,
        'body': 16,
        'caption': 14
    },
    'wide': {
        'h1': 36,
        'h2': 28,
        'body': 16,
        'caption': 14
    }
}
```

### Responsive Spacing

```python
RESPONSIVE_SPACING = {
    'mobile': {
        'page_padding': SPACING['4'],   # 16px
        'section_gap': SPACING['4'],    # 16px
        'element_gap': SPACING['3']     # 12px
    },
    'tablet': {
        'page_padding': SPACING['6'],   # 24px
        'section_gap': SPACING['5'],    # 20px
        'element_gap': SPACING['4']     # 16px
    },
    'desktop': {
        'page_padding': SPACING['8'],   # 32px
        'section_gap': SPACING['6'],    # 24px
        'element_gap': SPACING['4']     # 16px
    },
    'wide': {
        'page_padding': SPACING['12'],  # 48px
        'section_gap': SPACING['8'],    # 32px
        'element_gap': SPACING['5']     # 20px
    }
}
```

## Implementation Guidelines

### CSS Custom Properties

```css
:root {
  /* Colors */
  --color-primary: #2563eb;
  --color-primary-dark: #1d4ed8;
  --color-primary-light: #3b82f6;
  --color-secondary: #64748b;
  --color-accent: #f59e0b;

  /* Semantic Colors */
  --color-success: #10b981;
  --color-warning: #f59e0b;
  --color-error: #ef4444;
  --color-info: #3b82f6;
  --color-neutral: #6b7280;

  /* Typography */
  --font-primary: 'Inter', system-ui, -apple-system, sans-serif;
  --font-monospace: 'JetBrains Mono', 'Fira Code', monospace;

  /* Font Sizes */
  --text-xs: 10px;
  --text-sm: 12px;
  --text-base: 14px;
  --text-lg: 16px;
  --text-xl: 18px;
  --text-2xl: 20px;
  --text-3xl: 24px;

  /* Spacing */
  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --space-5: 20px;
  --space-6: 24px;
  --space-8: 32px;

  /* Layout */
  --border-radius: 4px;
  --border-width: 1px;
  --shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

/* Dark mode overrides */
@media (prefers-color-scheme: dark) {
  :root {
    --color-primary: #3b82f6;
    --color-primary-dark: #2563eb;
    --color-primary-light: #60a5fa;
  }
}
```

### Python Implementation

```python
class DesignSystem:
    """Central design system implementation."""

    def __init__(self):
        self.colors = PRIMARY_COLORS
        self.typography = TYPOGRAPHY
        self.spacing = SPACING
        self.components = self._load_component_definitions()

    def get_component_style(self, component_name, variant='default'):
        """Get style definition for a component."""
        return self.components.get(component_name, {}).get(variant, {})

    def get_responsive_value(self, value, breakpoint='desktop'):
        """Get responsive value for current breakpoint."""
        if isinstance(value, dict):
            return value.get(breakpoint, value.get('desktop', value))
        return value

    def apply_theme(self, theme_name):
        """Apply a theme to the application."""
        theme = self.themes.get(theme_name, self.themes['default'])
        self._apply_theme_colors(theme)
        self._apply_theme_typography(theme)

    def _load_component_definitions(self):
        """Load component style definitions."""
        return {
            'button': {
                'primary': PRIMARY_BUTTON,
                'secondary': SECONDARY_BUTTON,
                'icon': ICON_BUTTON
            },
            'input': {
                'text': TEXT_INPUT
            },
            'card': CARD_LAYOUT,
            'form': FORM_LAYOUT
        }
```

## Usage Examples

### Applying the Design System in GUI Code

```python
# APGI Main GUI (apgi_gui.py)
class APGIGUI:
    def __init__(self, root):
        self.design_system = DesignSystem()
        self.setup_styling()

    def setup_styling(self):
        """Apply design system styling to the GUI."""
        style = ttk.Style()

        # Apply primary button style
        primary_style = self.design_system.get_component_style('button', 'primary')
        style.configure('Primary.TButton', **primary_style)

        # Apply card styling
        card_style = self.design_system.get_component_style('card')
        self.apply_card_styling(card_style)

    def create_button(self, text, command, style='primary'):
        """Create a styled button."""
        button_style = f"{style.title()}.TButton"
        return ttk.Button(self.root, text=text, command=command, style=button_style)
```

### Theme Application

```python
# Theme management
class ThemeManager:
    def __init__(self, design_system):
        self.design_system = design_system
        self.current_theme = 'default'

    def set_theme(self, theme_name):
        """Apply a theme to the application."""
        self.current_theme = theme_name
        self.design_system.apply_theme(theme_name)

        # Update all GUI components
        self._update_gui_theme()

    def toggle_high_contrast(self):
        """Toggle high contrast mode."""
        if 'high_contrast' in self.current_theme:
            self.set_theme('default')
        else:
            self.set_theme('high_contrast_dark')

    def _update_gui_theme(self):
        """Update all GUI components with new theme."""
        # This would update all open windows and components
        for window in self.open_windows:
            window.update_theme(self.design_system)
```

## Testing the Design System

### Visual Regression Testing

```python
class DesignSystemTests:
    def test_component_visual_consistency(self):
        """Test that components render consistently."""
        # Create component instances
        button1 = self.create_styled_button("Test 1")
        button2 = self.create_styled_button("Test 2")

        # Capture screenshots
        screenshot1 = self.capture_component_screenshot(button1)
        screenshot2 = self.capture_component_screenshot(button2)

        # Compare visual appearance
        assert self.compare_screenshots(screenshot1, screenshot2), \
            "Buttons should look identical"

    def test_responsive_behavior(self):
        """Test responsive design behavior."""
        for breakpoint in ['mobile', 'tablet', 'desktop', 'wide']:
            self.set_viewport_size(BREAKPOINTS[breakpoint])

            # Test component sizing
            component = self.create_test_component()
            assert self.verify_responsive_sizing(component, breakpoint)

    def test_accessibility_compliance(self):
        """Test accessibility compliance."""
        component = self.create_test_component()

        # Test color contrast
        assert self.test_color_contrast(component)

        # Test keyboard navigation
        assert self.test_keyboard_navigation(component)

        # Test screen reader support
        assert self.test_screen_reader_support(component)
```

## Maintenance and Updates

### Version Control

The design system follows semantic versioning:

- **Major**: Breaking changes to components or color palette
- **Minor**: New components or style additions
- **Patch**: Bug fixes and small adjustments

### Deprecation Policy

1. **Announcement**: New versions announce deprecated styles 2 versions in advance
2. **Migration Guide**: Provide migration guides for deprecated styles
3. **Support Period**: Deprecated styles supported for 6 months
4. **Removal**: Clean removal after support period

### Governance

- **Design Review**: All UI changes require design system approval
- **Component Ownership**: Each component has a designated maintainer
- **Usage Analytics**: Track component usage for optimization decisions
- **User Feedback**: Regular user testing and feedback integration

---

### This design system is continuously evolving

Last updated: February 2024
