"""Results Viewer.

This module implements comprehensive result display with expandable test result views,
failure highlighting, quick navigation to failure details, and stack trace display.
"""

import enum
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


class TestStatus(enum.Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class FailureCategory(enum.Enum):
    ASSERTION_ERROR = "assertion_error"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass
class TestFailure:
    test_name: str
    failure_category: FailureCategory
    error_message: str
    file_path: Optional[str] = None
    stack_trace: Optional[str] = None
    failure_context: Optional[Dict[str, Any]] = None


@dataclass
class TestResults:
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    error_tests: int = 0
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    failures: List[TestFailure] = field(default_factory=list)
    test_results: List[Any] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return self.passed_tests

    @property
    def failed(self) -> int:
        return self.failed_tests

    @property
    def skipped(self) -> int:
        return self.skipped_tests

    @property
    def duration(self) -> float:
        return self.execution_time


from apgi_framework.logging.standardized_logging import get_logger

logger = get_logger(__name__)


try:
    from PySide6.QtCore import Qt, Signal  # type: ignore[import-not-found]
    from PySide6.QtGui import QFont  # type: ignore[import-not-found]
    from PySide6.QtGui import QColor, QSyntaxHighlighter, QTextCharFormat
    from PySide6.QtWidgets import (  # type: ignore[import-not-found]
        QApplication,
        QCheckBox,
        QComboBox,
        QFrame,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QTextEdit,
        QTreeWidget,
        QTreeWidgetItem,
        QVBoxLayout,
        QWidget,
    )

    PYSIDE6_AVAILABLE = True
except ImportError:
    # Fallback for environments without PySide6
    PYSIDE6_AVAILABLE = False

    class FallbackQWidget:
        pass

    class FallbackQVBoxLayout:
        pass

    class FallbackQHBoxLayout:
        pass

    class FallbackQTreeWidget:
        pass

    class FallbackQTreeWidgetItem:
        pass

    class FallbackQTextEdit:
        pass

    class FallbackQSplitter:
        pass

    class FallbackQTabWidget:
        pass

    class FallbackQLabel:
        pass

    class FallbackQPushButton:
        pass

    class FallbackQGroupBox:
        pass

    class FallbackQScrollArea:
        pass

    class FallbackQFrame:
        pass

    class FallbackQLineEdit:
        pass

    class FallbackQComboBox:
        pass

    class FallbackQCheckBox:
        pass

    class FallbackQTableWidget:
        pass

    class FallbackQTableWidgetItem:
        pass

    class FallbackQHeaderView:
        Stretch = 1
        ResizeToContents = 2

    class FallbackQProgressBar:
        pass

    class FallbackQApplication:
        @staticmethod
        def clipboard():  # type: ignore[no-untyped-def]
            return None

    class FallbackSignal:
        def __init__(self, *args: Any) -> None:
            pass

    class FallbackQSyntaxHighlighter:
        pass

    class FallbackQPaintEvent:
        pass

    class FallbackQFont:
        Bold = 1

    class FallbackQColor:
        def __init__(self, *args: Any) -> None:
            pass

        def name(self) -> str:
            return "#000000"

    class FallbackQt:
        Horizontal = 1
        UserRole = 256

    class FallbackQTextCharFormat:
        pass

    # Assign fallback classes to expected names
    QWidget = FallbackQWidget
    QVBoxLayout = FallbackQVBoxLayout
    QHBoxLayout = FallbackQHBoxLayout
    QTreeWidget = FallbackQTreeWidget
    QTreeWidgetItem = FallbackQTreeWidgetItem
    QTextEdit = FallbackQTextEdit
    QSplitter = FallbackQSplitter
    QTabWidget = FallbackQTabWidget
    QLabel = FallbackQLabel
    QPushButton = FallbackQPushButton
    QGroupBox = FallbackQGroupBox
    QScrollArea = FallbackQScrollArea
    QFrame = FallbackQFrame
    QLineEdit = FallbackQLineEdit
    QComboBox = FallbackQComboBox
    QCheckBox = FallbackQCheckBox
    QTableWidget = FallbackQTableWidget
    QTableWidgetItem = FallbackQTableWidgetItem
    QHeaderView = FallbackQHeaderView
    QProgressBar = FallbackQProgressBar
    QApplication = FallbackQApplication
    Signal = FallbackSignal
    QSyntaxHighlighter = FallbackQSyntaxHighlighter
    QPaintEvent = FallbackQPaintEvent
    QFont = FallbackQFont
    QColor = FallbackQColor
    Qt = FallbackQt
    QTextCharFormat = FallbackQTextCharFormat


class StackTraceHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for stack traces to improve readability."""

    def __init__(self, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self._setup_formats()

    def _setup_formats(self) -> None:  # type: ignore[no-untyped-def]
        """Set up text formats for different parts of stack traces."""
        self.file_format = QTextCharFormat()
        self.file_format.setForeground(QColor(0, 100, 200))  # Blue
        self.file_format.setFontWeight(QFont.Bold)

        self.line_format = QTextCharFormat()
        self.line_format.setForeground(QColor(200, 100, 0))  # Orange

        self.error_format = QTextCharFormat()
        self.error_format.setForeground(QColor(200, 0, 0))  # Red
        self.error_format.setFontWeight(QFont.Bold)

        self.function_format = QTextCharFormat()
        self.function_format.setForeground(QColor(100, 0, 200))  # Purple

    def highlightBlock(self, text: str) -> None:  # type: ignore[no-untyped-def]
        """Highlight a block of text in the stack trace."""
        # Highlight file paths
        if 'File "' in text:
            start = text.find('File "')
            end = text.find('"', start + 6)
            if end > start:
                self.setFormat(start, end - start + 1, self.file_format)

        # Highlight line numbers
        if ", line " in text:
            start = text.find(", line ")
            end = text.find(",", start + 7)
            if end > start:
                self.setFormat(start, end - start, self.line_format)

        # Highlight function names
        if ", in " in text:
            start = text.find(", in ")
            self.setFormat(start + 5, len(text) - start - 5, self.function_format)

        # Highlight error messages (lines starting with common error types)
        error_types = [
            "AssertionError",
            "ValueError",
            "TypeError",
            "AttributeError",
            "ImportError",
        ]
        for error_type in error_types:
            if text.strip().startswith(error_type):
                self.setFormat(0, len(text), self.error_format)
                break


class TestResultItem(QTreeWidgetItem):
    """Custom tree widget item for test results with additional data."""

    def __init__(
        self,
        parent: Any,
        test_name: str,
        status: TestStatus,
        execution_time: float,
        failure: Optional[TestFailure] = None,
    ) -> None:
        super().__init__(parent)
        self.test_name = test_name
        self.status = status
        self.execution_time = execution_time
        self.failure = failure

        self._setup_display()

    def _setup_display(self) -> None:  # type: ignore[no-untyped-def]
        """Set up the display text and formatting."""
        self.setText(0, self.test_name)
        self.setText(1, self.status.value.upper())
        self.setText(2, f"{self.execution_time:.3f}s")

        # Set status-based coloring
        if self.status == TestStatus.PASSED:
            self.setForeground(1, QColor(0, 150, 0))  # Green
        elif self.status == TestStatus.FAILED:
            self.setForeground(1, QColor(200, 0, 0))  # Red
        elif self.status == TestStatus.SKIPPED:
            self.setForeground(1, QColor(200, 150, 0))  # Orange
        elif self.status == TestStatus.ERROR:
            self.setForeground(1, QColor(150, 0, 150))  # Purple

        # Add failure summary if available
        if self.failure:
            self.setText(3, self.failure.failure_category.value.replace("_", " ").title())
            self.setToolTip(
                0,
                (
                    self.failure.error_message[:200] + "..."
                    if len(self.failure.error_message) > 200
                    else self.failure.error_message
                ),
            )


class TestResultsTree(QTreeWidget):
    """Tree widget for displaying test results with filtering and navigation."""

    result_selected = Signal(object)  # TestFailure or None

    def __init__(self, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self._setup_tree()
        self._test_items: Dict[str, TestResultItem] = {}

    def _setup_tree(self) -> None:  # type: ignore[no-untyped-def]
        """Set up the tree widget."""
        self.setHeaderLabels(["Test Name", "Status", "Time", "Failure Type"])
        self.setAlternatingRowColors(True)
        self.setRootIsDecorated(False)
        self.setSortingEnabled(True)

        # Set column widths
        header = self.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)

        # Connect selection changes
        self.itemSelectionChanged.connect(self._on_selection_changed)

    def populate_results(self, results: TestResults) -> None:  # type: ignore[no-untyped-def]
        """Populate the tree with test results."""
        self.clear()
        self._test_items.clear()

        # Create failure lookup
        failures_by_test = {f.test_name: f for f in results.failures}

        # Add results (we'll need to reconstruct test cases from results)
        # In a real implementation, you'd have access to the original test cases
        for i in range(results.total_tests):
            # This is a simplified approach - in reality, you'd have the actual test data
            test_name = f"test_{i}"  # Placeholder

            # Determine status based on results
            if i < results.passed_tests:
                status = TestStatus.PASSED
                execution_time = 0.1 + (i * 0.05)  # Mock execution time
            elif i < results.passed_tests + results.failed_tests:
                status = TestStatus.FAILED
                execution_time = 0.2 + (i * 0.03)
            elif i < results.passed_tests + results.failed_tests + results.skipped_tests:
                status = TestStatus.SKIPPED
                execution_time = 0.0
            else:
                status = TestStatus.ERROR
                execution_time = 0.1

            failure = failures_by_test.get(test_name)
            item = TestResultItem(self, test_name, status, execution_time, failure)
            self._test_items[test_name] = item

        # Add actual failures if they exist
        for failure in results.failures:
            if failure.test_name not in self._test_items:
                item = TestResultItem(self, failure.test_name, TestStatus.FAILED, 0.0, failure)
                self._test_items[failure.test_name] = item

        self.expandAll()

    def filter_results(
        self, status_filter: Optional[TestStatus] = None, search_text: str = ""
    ) -> None:  # type: ignore[no-untyped-def]
        """Filter results by status and search text."""
        for test_name, item in self._test_items.items():
            show_item = True

            # Status filter
            if status_filter and item.status != status_filter:
                show_item = False

            # Search filter
            if search_text and search_text.lower() not in test_name.lower():
                show_item = False

            item.setHidden(not show_item)

    def navigate_to_failure(self, test_name: str) -> None:  # type: ignore[no-untyped-def]
        """Navigate to a specific test failure."""
        if test_name in self._test_items:
            item = self._test_items[test_name]
            self.setCurrentItem(item)
            self.scrollToItem(item)

    def _on_selection_changed(self) -> None:  # type: ignore[no-untyped-def]
        """Handle selection changes."""
        current_item = self.currentItem()
        if isinstance(current_item, TestResultItem):
            self.result_selected.emit(current_item.failure)
        else:
            self.result_selected.emit(None)


class FailureDetailWidget(QWidget):
    """Widget for displaying detailed failure information."""

    def __init__(self, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self._setup_ui()
        self._current_failure: Optional[TestFailure] = None

    def _setup_ui(self) -> None:  # type: ignore[no-untyped-def]
        """Set up the failure detail UI."""
        layout = QVBoxLayout(self)

        # Failure summary
        summary_group = QGroupBox("Failure Summary")
        summary_layout = QVBoxLayout(summary_group)

        self.test_name_label = QLabel("No failure selected")
        self.test_name_label.setFont(QFont("", 10, QFont.Bold))
        summary_layout.addWidget(self.test_name_label)

        self.category_label = QLabel("")
        summary_layout.addWidget(self.category_label)

        self.file_path_label = QLabel("")
        summary_layout.addWidget(self.file_path_label)

        layout.addWidget(summary_group)

        # Error message
        message_group = QGroupBox("Error Message")
        message_layout = QVBoxLayout(message_group)

        self.error_message_text = QTextEdit()
        self.error_message_text.setMaximumHeight(100)
        self.error_message_text.setReadOnly(True)
        message_layout.addWidget(self.error_message_text)

        layout.addWidget(message_group)

        # Stack trace
        trace_group = QGroupBox("Stack Trace")
        trace_layout = QVBoxLayout(trace_group)

        self.stack_trace_text = QTextEdit()
        self.stack_trace_text.setReadOnly(True)
        self.stack_trace_text.setFont(QFont("Consolas", 9))

        # Add syntax highlighting
        self.highlighter = StackTraceHighlighter(self.stack_trace_text.document())

        trace_layout.addWidget(self.stack_trace_text)

        layout.addWidget(trace_group)

        # Context information
        context_group = QGroupBox("Context")
        context_layout = QVBoxLayout(context_group)

        self.context_text = QTextEdit()
        self.context_text.setMaximumHeight(80)
        self.context_text.setReadOnly(True)
        context_layout.addWidget(self.context_text)

        layout.addWidget(context_group)

        # Navigation buttons
        nav_layout = QHBoxLayout()

        self.open_file_btn = QPushButton("Open File")
        self.open_file_btn.setEnabled(False)
        nav_layout.addWidget(self.open_file_btn)

        self.copy_trace_btn = QPushButton("Copy Stack Trace")
        self.copy_trace_btn.setEnabled(False)
        nav_layout.addWidget(self.copy_trace_btn)

        nav_layout.addStretch()
        layout.addLayout(nav_layout)

        # Connect buttons
        self.copy_trace_btn.clicked.connect(self._copy_stack_trace)

    def display_failure(self, failure: Optional[TestFailure]) -> None:  # type: ignore[no-untyped-def]
        """Display failure details."""
        self._current_failure = failure

        if not failure:
            self._clear_display()
            return

        # Update summary
        self.test_name_label.setText(f"Test: {failure.test_name}")
        self.category_label.setText(
            f"Category: {failure.failure_category.value.replace('_', ' ').title()}"
        )
        self.file_path_label.setText(f"File: {failure.file_path or 'Unknown'}")

        # Update error message
        self.error_message_text.setPlainText(failure.error_message)

        # Update stack trace
        self.stack_trace_text.setPlainText(failure.stack_trace)

        # Update context
        if failure.failure_context:
            context_text = "\n".join([f"{k}: {v}" for k, v in failure.failure_context.items()])
            self.context_text.setPlainText(context_text)
        else:
            self.context_text.setPlainText("No additional context available")

        # Enable buttons
        self.open_file_btn.setEnabled(bool(failure.file_path))
        self.copy_trace_btn.setEnabled(bool(failure.stack_trace))

    def _clear_display(self) -> None:  # type: ignore[no-untyped-def]
        """Clear the failure display."""
        self.test_name_label.setText("No failure selected")
        self.category_label.setText("")
        self.file_path_label.setText("")
        self.error_message_text.clear()
        self.stack_trace_text.clear()
        self.context_text.clear()

        self.open_file_btn.setEnabled(False)
        self.copy_trace_btn.setEnabled(False)

    def _copy_stack_trace(self) -> None:  # type: ignore[no-untyped-def]
        """Copy stack trace to clipboard."""
        if self._current_failure and self._current_failure.stack_trace:
            clipboard = QApplication.clipboard()
            clipboard.setText(self._current_failure.stack_trace)


class ResultsSummaryWidget(QWidget):
    """Widget for displaying test results summary and statistics."""

    def __init__(self, parent: Optional[Any] = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:  # type: ignore[no-untyped-def]
        """Set up the summary UI."""
        layout = QVBoxLayout(self)

        # Overall statistics
        stats_group = QGroupBox("Test Statistics")
        stats_layout = QVBoxLayout(stats_group)

        # Progress bar for pass rate
        self.pass_rate_bar = QProgressBar()
        self.pass_rate_bar.setFormat("Pass Rate: %p%")
        stats_layout.addWidget(self.pass_rate_bar)

        # Statistics table
        self.stats_table = QTableWidget(5, 2)
        self.stats_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.stats_table.verticalHeader().setVisible(False)
        self.stats_table.setMaximumHeight(150)

        # Set up table items
        metrics = ["Total Tests", "Passed", "Failed", "Skipped", "Errors"]
        for i, metric in enumerate(metrics):
            self.stats_table.setItem(i, 0, QTableWidgetItem(metric))
            self.stats_table.setItem(i, 1, QTableWidgetItem("0"))

        self.stats_table.resizeColumnsToContents()
        stats_layout.addWidget(self.stats_table)

        layout.addWidget(stats_group)

        # Execution information
        exec_group = QGroupBox("Execution Information")
        exec_layout = QVBoxLayout(exec_group)

        self.execution_time_label = QLabel("Execution Time: 0.000s")
        exec_layout.addWidget(self.execution_time_label)

        self.timestamp_label = QLabel("Timestamp: Not available")
        exec_layout.addWidget(self.timestamp_label)

        layout.addWidget(exec_group)

        # Quick navigation
        nav_group = QGroupBox("Quick Navigation")
        nav_layout = QVBoxLayout(nav_group)

        nav_buttons_layout = QHBoxLayout()

        self.show_failures_btn = QPushButton("Show Failures Only")
        self.show_all_btn = QPushButton("Show All Tests")
        self.export_results_btn = QPushButton("Export Results")

        nav_buttons_layout.addWidget(self.show_failures_btn)
        nav_buttons_layout.addWidget(self.show_all_btn)
        nav_buttons_layout.addWidget(self.export_results_btn)

        nav_layout.addLayout(nav_buttons_layout)
        layout.addWidget(nav_group)

    def update_summary(self, results: TestResults) -> None:  # type: ignore[no-untyped-def]
        """Update the summary with test results."""
        # Update statistics table
        values = [
            str(results.total_tests),
            str(results.passed_tests),
            str(results.failed_tests),
            str(results.skipped_tests),
            str(results.error_tests),
        ]

        for i, value in enumerate(values):
            self.stats_table.item(i, 1).setText(value)

        # Update pass rate
        pass_rate = (
            (results.passed_tests / results.total_tests * 100) if results.total_tests > 0 else 0
        )
        self.pass_rate_bar.setValue(int(pass_rate))

        # Update execution info
        self.execution_time_label.setText(f"Execution Time: {results.execution_time:.3f}s")
        self.timestamp_label.setText(
            f"Timestamp: {results.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        )


class ResultsViewer(QWidget):
    """Main results viewer widget with comprehensive result display."""

    # Signals
    failure_selected = Signal(object)  # TestFailure
    navigation_requested = Signal(str, str)  # file_path, line_number

    def __init__(self, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self._setup_ui()
        self._setup_connections()
        self._current_results: Optional[TestResults] = None

    def _setup_ui(self) -> None:  # type: ignore[no-untyped-def]
        """Set up the results viewer UI."""
        layout = QVBoxLayout(self)

        # Filter controls
        filter_group = QGroupBox("Filter Results")
        filter_layout = QHBoxLayout(filter_group)

        filter_layout.addWidget(QLabel("Search:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search test names...")
        filter_layout.addWidget(self.search_input)

        filter_layout.addWidget(QLabel("Status:"))
        self.status_combo = QComboBox()
        self.status_combo.addItems(["All", "Passed", "Failed", "Skipped", "Error"])
        filter_layout.addWidget(self.status_combo)

        self.auto_scroll_checkbox = QCheckBox("Auto-scroll to failures")
        self.auto_scroll_checkbox.setChecked(True)
        filter_layout.addWidget(self.auto_scroll_checkbox)

        filter_layout.addStretch()
        layout.addWidget(filter_group)

        # Main content area
        main_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(main_splitter)

        # Left panel - Results tree and summary
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Results tree
        self.results_tree = TestResultsTree()
        left_layout.addWidget(self.results_tree)

        # Summary widget
        self.summary_widget = ResultsSummaryWidget()
        left_layout.addWidget(self.summary_widget)

        main_splitter.addWidget(left_panel)

        # Right panel - Failure details
        self.failure_detail_widget = FailureDetailWidget()
        main_splitter.addWidget(self.failure_detail_widget)

        # Set splitter proportions
        main_splitter.setSizes([600, 400])

    def _setup_connections(self) -> None:  # type: ignore[no-untyped-def]
        """Set up signal connections."""
        # Filter connections
        self.search_input.textChanged.connect(self._apply_filters)
        self.status_combo.currentTextChanged.connect(self._apply_filters)

        # Results tree connections
        self.results_tree.result_selected.connect(self.failure_detail_widget.display_failure)
        self.results_tree.result_selected.connect(self.failure_selected.emit)

        # Summary widget connections
        self.summary_widget.show_failures_btn.clicked.connect(self._show_failures_only)
        self.summary_widget.show_all_btn.clicked.connect(self._show_all_tests)
        self.summary_widget.export_results_btn.clicked.connect(self._export_results)

    def display_results(self, results: TestResults) -> None:  # type: ignore[no-untyped-def]
        """Display test results in the viewer."""
        self._current_results = results

        # Update results tree
        self.results_tree.populate_results(results)

        # Update summary
        self.summary_widget.update_summary(results)

        # Auto-scroll to first failure if enabled
        if self.auto_scroll_checkbox.isChecked() and results.failures:
            first_failure = results.failures[0]
            self.results_tree.navigate_to_failure(first_failure.test_name)

    def _apply_filters(self) -> None:  # type: ignore[no-untyped-def]
        """Apply current filter settings."""
        search_text = self.search_input.text()
        status_text = self.status_combo.currentText()

        # Convert status text to enum
        status_filter = None
        if status_text != "All":
            status_map = {
                "Passed": TestStatus.PASSED,
                "Failed": TestStatus.FAILED,
                "Skipped": TestStatus.SKIPPED,
                "Error": TestStatus.ERROR,
            }
            status_filter = status_map.get(status_text)

        self.results_tree.filter_results(status_filter, search_text)

    def _show_failures_only(self) -> None:  # type: ignore[no-untyped-def]
        """Show only failed tests."""
        self.status_combo.setCurrentText("Failed")

    def _show_all_tests(self) -> None:  # type: ignore[no-untyped-def]
        """Show all tests."""
        self.status_combo.setCurrentText("All")
        self.search_input.clear()

    def _export_results(self) -> None:  # type: ignore[no-untyped-def]
        """Export results to file."""
        if not self._current_results:
            return

        try:
            if PYSIDE6_AVAILABLE:
                from PySide6.QtWidgets import QFileDialog

                file_path, selected_filter = QFileDialog.getSaveFileName(
                    self,
                    "Export Results",
                    "test_results.json",
                    "JSON Files (*.json);;CSV Files (*.csv);;All Files (*)",
                )

                if file_path:
                    if file_path.endswith(".csv"):
                        self._export_csv(file_path)
                    else:
                        self._export_json(file_path)
            else:
                # Fallback for non-GUI export
                from pathlib import Path

                output_file = Path("test_results_export.json")
                self._export_json(str(output_file))
                logger.info(f"Results exported to {output_file}")

        except Exception as e:
            logger.info(f"Export failed: {e}")

    def _export_json(self, file_path: str) -> None:  # type: ignore[no-untyped-def]
        """Export results to JSON format."""
        import json

        if not self._current_results:
            return

        export_data: Dict[str, Any] = {
            "summary": {
                "total_tests": self._current_results.total_tests,
                "passed": self._current_results.passed,
                "failed": self._current_results.failed,
                "skipped": self._current_results.skipped,
                "duration": self._current_results.duration,
            },
            "test_results": [],
        }

        for test_result in self._current_results.test_results:
            result_data: Dict[str, Any] = {
                "name": test_result.name,
                "status": test_result.status.value,
                "duration": test_result.duration,
                "message": test_result.message,
            }

            if test_result.failure:
                result_data["failure"] = {
                    "category": test_result.failure.category.value,
                    "message": test_result.failure.message,
                    "traceback": test_result.failure.traceback,
                }

            export_data["test_results"].append(result_data)

        with open(file_path, "w") as f:
            json.dump(export_data, f, indent=2)

    def _export_csv(self, file_path: str) -> None:  # type: ignore[no-untyped-def]
        """Export results to CSV format."""
        import csv

        if not self._current_results:
            return

        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow(
                [
                    "Test Name",
                    "Status",
                    "Duration",
                    "Message",
                    "Failure Category",
                    "Failure Message",
                ]
            )

            # Write test results
            for test_result in self._current_results.test_results:
                failure_category = test_result.failure.category.value if test_result.failure else ""
                failure_message = test_result.failure.message if test_result.failure else ""

                writer.writerow(
                    [
                        test_result.name,
                        test_result.status.value,
                        test_result.duration,
                        test_result.message,
                        failure_category,
                        failure_message,
                    ]
                )

    def navigate_to_test(self, test_name: str) -> None:  # type: ignore[no-untyped-def]
        """Navigate to a specific test in the results."""
        self.results_tree.navigate_to_failure(test_name)

    def get_current_results(self) -> Optional[TestResults]:
        """Get the currently displayed results."""
        return self._current_results

    def clear_results(self) -> None:  # type: ignore[no-untyped-def]
        """Clear all displayed results."""
        self._current_results = None
        self.results_tree.clear()
        self.failure_detail_widget.display_failure(None)


def main() -> int:  # type: ignore[no-untyped-def]
    """Main function for testing the results viewer."""
    app = QApplication(sys.argv)

    # Create sample test results
    from datetime import datetime

    sample_failures = [
        TestFailure(
            test_name="test_example_failure",
            failure_category=FailureCategory.ASSERTION_ERROR,
            error_message="AssertionError: Expected 5, got 3",
            stack_trace='File "test_example.py", line 42, in test_example_failure\n    assert result == 5\nAssertionError: Expected 5, got 3',
            failure_context={"expected": 5, "actual": 3},
            file_path="tests/test_example.py",
        )
    ]

    sample_results = TestResults(
        total_tests=10,
        passed_tests=8,
        failed_tests=1,
        skipped_tests=1,
        error_tests=0,
        execution_time=2.5,
        failures=sample_failures,
        timestamp=datetime.now(),
    )

    viewer = ResultsViewer()
    viewer.display_results(sample_results)
    viewer.show()

    result = app.exec()
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    sys.exit(main())
