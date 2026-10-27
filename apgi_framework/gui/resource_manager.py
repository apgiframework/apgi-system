"""GUI Resource Manager for memory management and cleanup.

Provides context managers and utilities for proper resource cleanup
in GUI applications to prevent memory leaks.
"""

import gc
import logging
import weakref
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class GUIResourceManager:
    """Manages GUI resources and ensures proper cleanup.

    Tracks widgets, figures, and other resources for cleanup.
    """

    def __init__(self) -> None:
        """Initialize the resource manager."""
        self._widgets: List[weakref.ref] = []
        self._figures: List[Any] = []
        self._threads: List[Any] = []
        self._cleanup_callbacks: List[Callable] = []

    def register_widget(self, widget: Any) -> None:
        """Register a GUI widget for tracking.

        Args:
            widget: Widget to track
        """
        # Use weak reference to avoid keeping widgets alive
        self._widgets.append(weakref.ref(widget))
        logger.debug("Widget registered: %s", type(widget).__name__)

    def register_figure(self, figure: Any) -> None:
        """Register a matplotlib figure for cleanup.

        Args:
            figure: Matplotlib figure to track
        """
        self._figures.append(figure)
        logger.debug("Figure registered")

    def register_thread(self, thread: Any) -> None:
        """Register a thread for cleanup.

        Args:
            thread: Thread to track
        """
        self._threads.append(thread)
        logger.debug("Thread registered")

    def register_cleanup_callback(self, callback: Callable) -> None:
        """Register a cleanup callback.

        Args:
            callback: Function to call during cleanup
        """
        self._cleanup_callbacks.append(callback)
        logger.debug("Cleanup callback registered")

    def cleanup_widgets(self) -> None:
        """Clean up registered widgets."""
        cleaned = 0
        for widget_ref in self._widgets:
            widget = widget_ref()
            if widget is not None:
                try:
                    if hasattr(widget, "destroy"):
                        widget.destroy()
                    cleaned += 1
                except Exception as e:
                    logger.warning("Failed to cleanup widget: %s", e)

        self._widgets.clear()
        logger.info("Cleaned up %d widgets", cleaned)

    def cleanup_figures(self) -> None:
        """Clean up matplotlib figures."""
        for figure in self._figures:
            try:
                import matplotlib.pyplot as plt

                plt.close(figure)
            except Exception as e:
                logger.warning("Failed to cleanup figure: %s", e)

        self._figures.clear()
        logger.info("Cleaned up %d figures", len(self._figures))

    def cleanup_threads(self) -> None:
        """Clean up threads."""
        for thread in self._threads:
            try:
                if hasattr(thread, "join"):
                    thread.join(timeout=1.0)
            except Exception as e:
                logger.warning("Failed to cleanup thread: %s", e)

        self._threads.clear()
        logger.info("Cleaned up %d threads", len(self._threads))

    def run_cleanup_callbacks(self) -> None:
        """Run all registered cleanup callbacks."""
        for callback in self._cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logger.warning("Cleanup callback failed: %s", e)

        self._cleanup_callbacks.clear()
        logger.info("Ran %d cleanup callbacks", len(self._cleanup_callbacks))

    def cleanup_all(self) -> None:
        """Perform complete cleanup of all resources."""
        logger.info("Starting complete resource cleanup")

        self.cleanup_widgets()
        self.cleanup_figures()
        self.cleanup_threads()
        self.run_cleanup_callbacks()

        # Force garbage collection
        gc.collect()

        logger.info("Resource cleanup complete")

    def get_memory_usage(self) -> dict:
        """Get current memory usage statistics.

        Returns:
            Dictionary with memory usage info
        """
        import os

        import psutil

        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # Resident set size
            "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual memory size
            "percent": process.memory_percent(),
            "widgets_tracked": len([w for w in self._widgets if w() is not None]),
            "figures_tracked": len(self._figures),
            "threads_tracked": len(self._threads),
        }


@contextmanager
def gui_resource_context(resource_manager: GUIResourceManager):
    """Context manager for GUI resource cleanup.

    Usage:
        with gui_resource_context(manager):
            # Create and use GUI resources
            widget = create_widget()
            manager.register_widget(widget)

    Args:
        resource_manager: GUIResourceManager instance

    Yields:
        Resource manager instance
    """
    try:
        yield resource_manager
    finally:
        resource_manager.cleanup_all()


@contextmanager
def widget_context(widget: Any, resource_manager: Optional[GUIResourceManager] = None):
    """Context manager for individual widget cleanup.

    Usage:
        with widget_context(my_widget, manager):
            # Use widget
            pass

    Args:
        widget: Widget to manage
        resource_manager: Optional resource manager to register with

    Yields:
        Widget instance
    """
    if resource_manager:
        resource_manager.register_widget(widget)

    try:
        yield widget
    finally:
        try:
            if hasattr(widget, "destroy"):
                widget.destroy()
        except Exception as e:
            logger.warning("Failed to cleanup widget: %s", e)


@contextmanager
def figure_context(figure: Any, resource_manager: Optional[GUIResourceManager] = None):
    """Context manager for matplotlib figure cleanup.

    Usage:
        with figure_context(fig, manager):
            # Use figure
            pass

    Args:
        figure: Matplotlib figure to manage
        resource_manager: Optional resource manager to register with

    Yields:
        Figure instance
    """
    if resource_manager:
        resource_manager.register_figure(figure)

    try:
        yield figure
    finally:
        try:
            import matplotlib.pyplot as plt

            plt.close(figure)
        except Exception as e:
            logger.warning("Failed to cleanup figure: %s", e)


class LazyGUILoader:
    """Lazy loader for GUI components to reduce startup time.

    Loads GUI components on-demand instead of at startup.
    """

    def __init__(self) -> None:
        """Initialize the lazy loader."""
        self._components: Dict[str, Callable] = {}
        self._loaded: Dict[str, Any] = {}

    def register_component(self, name: str, loader: Callable) -> None:
        """Register a component loader.

        Args:
            name: Component name
            loader: Callable that creates the component
        """
        self._components[name] = loader
        logger.debug("Component registered for lazy loading: %s", name)

    def get_component(self, name: str) -> Any:
        """Get a component, loading it if necessary.

        Args:
            name: Component name

        Returns:
            Component instance

        Raises:
            KeyError: If component not registered
        """
        if name not in self._components:
            raise KeyError(f"Component '{name}' not registered")

        # Return cached component if already loaded
        if name in self._loaded:
            return self._loaded[name]

        # Load component
        logger.debug("Loading component: %s", name)
        loader = self._components[name]
        component = loader()
        self._loaded[name] = component

        return component

    def unload_component(self, name: str) -> None:
        """Unload a component to free memory.

        Args:
            name: Component name
        """
        if name in self._loaded:
            component = self._loaded[name]
            if hasattr(component, "destroy"):
                component.destroy()
            del self._loaded[name]
            logger.debug("Component unloaded: %s", name)

    def unload_all(self) -> None:
        """Unload all loaded components."""
        for name in list(self._loaded.keys()):
            self.unload_component(name)
        logger.info("All components unloaded")
