"""Testing module for APGI Framework.

This module provides comprehensive testing capabilities including:
- Batch test execution
- Test result persistence
- Performance monitoring
- Test utilities and fixtures
"""

from .batch_runner import (
    BatchExecutionSummary,
    BatchTestRunner,
    TestResult,
    run_all_tests,
    run_failed_tests,
    run_integration_tests,
    run_research_tests,
    run_unit_tests,
)
from .ci_integrator import (
    ChangeAnalyzer,
    ChangeImpact,
    CIConfiguration,
    CIIntegrator,
    ExecutionResult,
    PreCommitHookManager,
)
from .notification_manager import (
    HistoryTracker,
    NotificationChannel,
    NotificationManager,
    TestFailureNotification,
    create_email_channel,
    create_file_channel,
    create_slack_channel,
    create_teams_channel,
)
from .persistence import (
    BatchExecutionRecord,
    TestExecutionRecord,
    TestResultPersistence,
    get_persistence_manager,
    get_test_performance_report,
    store_test_results,
)

__all__ = [
    "BatchTestRunner",
    "TestResult",
    "BatchExecutionSummary",
    "run_all_tests",
    "run_unit_tests",
    "run_integration_tests",
    "run_research_tests",
    "run_failed_tests",
    "TestResultPersistence",
    "TestExecutionRecord",
    "BatchExecutionRecord",
    "get_persistence_manager",
    "store_test_results",
    "get_test_performance_report",
    "CIIntegrator",
    "CIConfiguration",
    "ChangeImpact",
    "ChangeAnalyzer",
    "PreCommitHookManager",
    "ExecutionResult",
    "NotificationManager",
    "NotificationChannel",
    "TestFailureNotification",
    "HistoryTracker",
    "create_email_channel",
    "create_slack_channel",
    "create_teams_channel",
    "create_file_channel",
]
