"""Notification Manager for CI/CD Test Failure Reporting.

This module provides comprehensive notification and reporting capabilities for CI/CD
test failures, including actionable debugging information, test result history
tracking, and automated report distribution.

Requirements: 8.4, 8.5
"""

import json
import logging
import smtplib
import sqlite3
import ssl
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from .activity_logger import ActivityLevel, ActivityType, get_activity_logger
from .ci_integrator import ExecutionResult


@dataclass
class NotificationChannel:
    """Configuration for a notification channel."""

    name: str
    type: str  # 'email', 'slack', 'teams', 'webhook', 'file'
    config: Dict[str, Any]
    enabled: bool = True
    failure_threshold: int = 1  # Minimum failures to trigger notification
    success_notification: bool = False  # Whether to notify on success


@dataclass
class TestFailureNotification:
    """Notification data for test failures."""

    execution_id: str
    timestamp: datetime
    total_failures: int
    critical_failures: int
    coverage_below_threshold: bool
    failed_tests: List[Dict[str, Any]]
    debugging_info: Dict[str, Any]
    trend_analysis: Dict[str, Any]
    actionable_recommendations: List[str]


@dataclass
class ResultHistory:
    """Historical test result data."""

    execution_id: str
    timestamp: datetime
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    coverage_percentage: float
    execution_time_seconds: float
    pipeline_context: Dict[str, Any]


class HistoryTracker:
    """Tracks test execution history and provides trend analysis."""

    def __init__(self, db_path: str = ".ci/test_history.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self._init_database()

    def _init_database(self) -> None:
        """Initialize the test history database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS test_executions (
                        execution_id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        total_tests INTEGER NOT NULL,
                        passed_tests INTEGER NOT NULL,
                        failed_tests INTEGER NOT NULL,
                        skipped_tests INTEGER NOT NULL,
                        coverage_percentage REAL NOT NULL,
                        execution_time_seconds REAL NOT NULL,
                        pipeline_context TEXT NOT NULL
                    )
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS test_failures (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        execution_id TEXT NOT NULL,
                        test_name TEXT NOT NULL,
                        error_message TEXT NOT NULL,
                        duration REAL NOT NULL,
                        FOREIGN KEY (execution_id) REFERENCES test_executions (execution_id)
                    )
                """)

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_timestamp ON test_executions (timestamp)
                """)

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_execution_id ON test_failures (execution_id)
                """)

        except Exception as e:
            self.logger.error(f"Failed to initialize test history database: {e}")

    def record_execution(self, result: ExecutionResult) -> None:
        """Record a test execution result."""
        try:
            history = ResultHistory(
                execution_id=result.execution_id,
                timestamp=result.start_time,
                total_tests=result.total_tests,
                passed_tests=result.passed_tests,
                failed_tests=result.failed_tests,
                skipped_tests=result.skipped_tests,
                coverage_percentage=result.coverage_percentage,
                execution_time_seconds=result.execution_time_seconds,
                pipeline_context=result.pipeline_context,
            )

            with sqlite3.connect(self.db_path) as conn:
                # Insert execution record
                conn.execute(
                    """
                    INSERT OR REPLACE INTO test_executions 
                    (execution_id, timestamp, total_tests, passed_tests, failed_tests, 
                     skipped_tests, coverage_percentage, execution_time_seconds, pipeline_context)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        history.execution_id,
                        history.timestamp.isoformat(),
                        history.total_tests,
                        history.passed_tests,
                        history.failed_tests,
                        history.skipped_tests,
                        history.coverage_percentage,
                        history.execution_time_seconds,
                        json.dumps(history.pipeline_context),
                    ),
                )

                # Insert failure details
                for failure in result.failed_test_details:
                    conn.execute(
                        """
                        INSERT INTO test_failures 
                        (execution_id, test_name, error_message, duration)
                        VALUES (?, ?, ?, ?)
                    """,
                        (
                            result.execution_id,
                            failure.get("name", ""),
                            failure.get("error", ""),
                            failure.get("duration", 0),
                        ),
                    )

        except Exception as e:
            self.logger.error(f"Failed to record test execution: {e}")

    def get_trend_analysis(self, days: int = 30) -> Dict[str, Any]:
        """Get trend analysis for the specified number of days."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)

            with sqlite3.connect(self.db_path) as conn:
                # Get execution trends
                cursor = conn.execute(
                    """
                    SELECT timestamp, total_tests, passed_tests, failed_tests, 
                           coverage_percentage, execution_time_seconds
                    FROM test_executions 
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                """,
                    (cutoff_date.isoformat(),),
                )

                executions = cursor.fetchall()

                if not executions:
                    return {"error": "No execution data available"}

                # Calculate trends
                recent_executions = executions[:10]  # Last 10 executions

                avg_coverage = sum(row[4] for row in recent_executions) / len(recent_executions)
                avg_execution_time = sum(row[5] for row in recent_executions) / len(
                    recent_executions
                )

                failure_rate = (
                    sum(row[3] for row in recent_executions)
                    / sum(row[1] for row in recent_executions)
                    if recent_executions
                    else 0
                )

                # Get most frequent failures
                cursor = conn.execute(
                    """
                    SELECT test_name, COUNT(*) as failure_count
                    FROM test_failures tf
                    JOIN test_executions te ON tf.execution_id = te.execution_id
                    WHERE te.timestamp >= ?
                    GROUP BY test_name
                    ORDER BY failure_count DESC
                    LIMIT 10
                """,
                    (cutoff_date.isoformat(),),
                )

                frequent_failures = cursor.fetchall()

                return {
                    "period_days": days,
                    "total_executions": len(executions),
                    "average_coverage": avg_coverage,
                    "average_execution_time": avg_execution_time,
                    "failure_rate": failure_rate,
                    "frequent_failures": [
                        {"test_name": name, "failure_count": count}
                        for name, count in frequent_failures
                    ],
                    "coverage_trend": "stable",  # Could be enhanced with actual trend calculation
                    "performance_trend": "stable",  # Could be enhanced with actual trend calculation
                }

        except Exception as e:
            self.logger.error(f"Failed to get trend analysis: {e}")
            return {"error": str(e)}

    def get_flaky_tests(
        self, min_executions: int = 5, failure_threshold: float = 0.2
    ) -> List[Dict[str, Any]]:
        """Identify flaky tests based on failure patterns."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get test execution patterns
                cursor = conn.execute(
                    """
                    SELECT 
                        tf.test_name,
                        COUNT(DISTINCT te.execution_id) as total_executions,
                        COUNT(tf.test_name) as failures
                    FROM test_executions te
                    LEFT JOIN test_failures tf ON te.execution_id = tf.execution_id
                    GROUP BY tf.test_name
                    HAVING total_executions >= ?
                """,
                    (min_executions,),
                )

                results = cursor.fetchall()

                flaky_tests = []
                for test_name, total_execs, failures in results:
                    if test_name and failures > 0:
                        failure_rate = failures / total_execs
                        if 0 < failure_rate <= failure_threshold:
                            flaky_tests.append(
                                {
                                    "test_name": test_name,
                                    "total_executions": total_execs,
                                    "failures": failures,
                                    "failure_rate": failure_rate,
                                    "flakiness_score": failure_rate
                                    * (1 - failure_rate)
                                    * 4,  # Peak at 50% failure rate
                                }
                            )

                # Sort by flakiness score
                flaky_tests.sort(key=lambda x: x["flakiness_score"], reverse=True)

                return flaky_tests

        except Exception as e:
            self.logger.error(f"Failed to identify flaky tests: {e}")
            return []


class NotificationManager:
    """Manages notifications for CI/CD test failures and results."""

    def __init__(
        self,
        channels: Optional[List[NotificationChannel]] = None,
        db_path: Optional[str] = None,
    ):
        self.channels = channels or []
        self.history_tracker = HistoryTracker(db_path or ".ci/test_history.db")
        self.logger = logging.getLogger(__name__)

    def add_channel(self, channel: NotificationChannel) -> None:
        """Add a notification channel."""
        self.channels.append(channel)

    def remove_channel(self, channel_name: str) -> None:
        """Remove a notification channel by name."""
        self.channels = [ch for ch in self.channels if ch.name != channel_name]

    def notify_test_result(self, result: ExecutionResult) -> None:
        """Send notifications based on test execution result."""
        # Record the execution in history
        self.history_tracker.record_execution(result)

        # Log notification activity
        activity_logger = get_activity_logger()
        activity_logger.set_context(
            execution_id=result.execution_id, component="notification_manager"
        )

        # Determine if notification is needed
        should_notify = self._should_notify(result)

        if not should_notify:
            self.logger.info(f"No notification needed for execution {result.execution_id}")
            activity_logger.log_activity(
                ActivityType.NOTIFICATION_SENT,
                ActivityLevel.DEBUG,
                "No notification required",
                data={"reason": "no_failures_or_success_notifications_disabled"},
            )
            return

        # Generate notification data
        notification = self._generate_notification(result)

        # Send notifications through all enabled channels
        sent_count = 0
        failed_count = 0

        for channel in self.channels:
            if not channel.enabled:
                continue

            if result.failed_tests >= channel.failure_threshold or (
                channel.success_notification and result.failed_tests == 0
            ):
                try:
                    self._send_notification(channel, notification, result)
                    sent_count += 1

                    activity_logger.log_notification_sent(
                        channel=channel.name,
                        notification_type=("failure" if result.failed_tests > 0 else "success"),
                        recipient_count=1,  # Simplified for now
                        success=True,
                    )

                except Exception as e:
                    failed_count += 1
                    self.logger.error(f"Failed to send notification via {channel.name}: {e}")

                    activity_logger.log_notification_sent(
                        channel=channel.name,
                        notification_type=("failure" if result.failed_tests > 0 else "success"),
                        recipient_count=1,
                        success=False,
                    )

        # Log summary
        activity_logger.log_activity(
            ActivityType.NOTIFICATION_SENT,
            ActivityLevel.INFO,
            f"Notification summary: {sent_count} sent, {failed_count} failed",
            data={
                "sent_count": sent_count,
                "failed_count": failed_count,
                "total_channels": len(self.channels),
                "notification_type": ("failure" if result.failed_tests > 0 else "success"),
            },
        )

    def _should_notify(self, result: ExecutionResult) -> bool:
        """Determine if a notification should be sent."""
        # Always notify on failures
        if result.failed_tests > 0:
            return True

        # Check if any channel wants success notifications
        return any(ch.success_notification and ch.enabled for ch in self.channels)

    def _generate_notification(self, result: ExecutionResult) -> TestFailureNotification:
        """Generate notification data from test result."""
        # Get trend analysis
        trend_analysis = self.history_tracker.get_trend_analysis()

        # Generate debugging information
        debugging_info = self._generate_debugging_info(result)

        # Generate actionable recommendations
        recommendations = self._generate_recommendations(result, trend_analysis)

        # Count critical failures (core, security modules)
        critical_failures = sum(
            1
            for failure in result.failed_test_details
            if any(
                critical in failure.get("name", "").lower()
                for critical in ["core", "security", "data"]
            )
        )

        return TestFailureNotification(
            execution_id=result.execution_id,
            timestamp=result.start_time,
            total_failures=result.failed_tests,
            critical_failures=critical_failures,
            coverage_below_threshold=result.coverage_percentage < 80.0,  # Default threshold
            failed_tests=result.failed_test_details,
            debugging_info=debugging_info,
            trend_analysis=trend_analysis,
            actionable_recommendations=recommendations,
        )

    def _generate_debugging_info(self, result: ExecutionResult) -> Dict[str, Any]:
        """Generate debugging information for failures."""
        debugging_info = {
            "execution_context": result.pipeline_context,
            "execution_time": result.execution_time_seconds,
            "coverage_info": {
                "percentage": result.coverage_percentage,
                "below_threshold": result.coverage_percentage < 80.0,
            },
            "failure_patterns": {},
            "environment_info": {},
        }

        # Analyze failure patterns
        error_types: Dict[str, int] = {}
        for failure in result.failed_test_details:
            error = failure.get("error", "")
            if "ImportError" in error:
                error_types["import_errors"] = error_types.get("import_errors", 0) + 1
            elif "AssertionError" in error:
                error_types["assertion_errors"] = error_types.get("assertion_errors", 0) + 1
            elif "TimeoutError" in error or "timeout" in error.lower():
                error_types["timeout_errors"] = error_types.get("timeout_errors", 0) + 1
            else:
                error_types["other_errors"] = error_types.get("other_errors", 0) + 1

        debugging_info["failure_patterns"] = error_types

        # Add environment information from pipeline context
        env_info = {}
        for key, value in result.pipeline_context.items():
            if key.lower() in [
                "python_version",
                "os",
                "platform",
                "branch",
                "commit",
            ]:
                env_info[key] = value

        debugging_info["environment_info"] = env_info

        return debugging_info

    def _generate_recommendations(
        self, result: ExecutionResult, trend_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations based on failures and trends."""
        recommendations = []

        # Coverage recommendations
        if result.coverage_percentage < 80.0:
            recommendations.append(
                f"Coverage is {result.coverage_percentage:.1f}%, below 80% threshold. "
                "Consider adding tests for uncovered code paths."
            )

        # Performance recommendations
        if result.execution_time_seconds > 300:  # 5 minutes
            recommendations.append(
                f"Test execution took {result.execution_time_seconds:.1f}s. "
                "Consider optimizing slow tests or enabling parallel execution."
            )

        # Failure pattern recommendations
        debugging_info = self._generate_debugging_info(result)
        error_patterns = debugging_info.get("failure_patterns", {})

        if error_patterns.get("import_errors", 0) > 0:
            recommendations.append(
                "Import errors detected. Check dependencies and Python path configuration."
            )

        if error_patterns.get("timeout_errors", 0) > 0:
            recommendations.append(
                "Timeout errors detected. Consider increasing test timeouts or optimizing test performance."
            )

        # Trend-based recommendations
        if isinstance(trend_analysis, dict) and "failure_rate" in trend_analysis:
            if trend_analysis["failure_rate"] > 0.1:  # 10% failure rate
                recommendations.append(
                    f"High failure rate ({trend_analysis['failure_rate']:.1%}) detected. "
                    "Review recent changes and consider stabilizing flaky tests."
                )

        # Flaky test recommendations
        flaky_tests = self.history_tracker.get_flaky_tests()
        if flaky_tests:
            top_flaky = flaky_tests[0]
            recommendations.append(
                f"Flaky test detected: {top_flaky['test_name']} "
                f"(failure rate: {top_flaky['failure_rate']:.1%}). "
                "Consider investigating and stabilizing this test."
            )

        return recommendations

    def _send_notification(
        self,
        channel: NotificationChannel,
        notification: TestFailureNotification,
        result: ExecutionResult,
    ) -> None:
        """Send notification through a specific channel."""
        if channel.type == "email":
            self._send_email_notification(channel, notification, result)
        elif channel.type == "slack":
            self._send_slack_notification(channel, notification, result)
        elif channel.type == "teams":
            self._send_teams_notification(channel, notification, result)
        elif channel.type == "webhook":
            self._send_webhook_notification(channel, notification, result)
        elif channel.type == "file":
            self._send_file_notification(channel, notification, result)
        else:
            self.logger.warning(f"Unknown notification channel type: {channel.type}")

    def _send_email_notification(
        self,
        channel: NotificationChannel,
        notification: TestFailureNotification,
        result: ExecutionResult,
    ) -> None:
        """Send email notification."""
        config = channel.config

        # Create message
        msg = MIMEMultipart()
        msg["From"] = config.get("from_email", "noreply@apgi-framework.com")
        msg["To"] = ", ".join(config.get("to_emails", []))
        msg["Subject"] = self._generate_email_subject(notification, result)

        # Create email body
        body = self._generate_email_body(notification, result)
        msg.attach(MIMEText(body, "html"))

        # Send email
        try:
            with smtplib.SMTP(
                config.get("smtp_server", "localhost"),
                config.get("smtp_port", 587),
            ) as server:
                if config.get("use_tls", True):
                    server.starttls(context=ssl.create_default_context())

                if config.get("username") and config["password"]:
                    server.login(config["username"], config["password"])

                server.send_message(msg)

            self.logger.info(f"Email notification sent via {channel.name}")

        except Exception as e:
            self.logger.error(f"Failed to send email notification: {e}")
            raise

    def _send_slack_notification(
        self,
        channel: NotificationChannel,
        notification: TestFailureNotification,
        result: ExecutionResult,
    ) -> None:
        """Send Slack notification."""
        config = channel.config
        webhook_url = config.get("webhook_url")

        if not webhook_url:
            raise ValueError("Slack webhook URL not configured")

        # Create Slack message
        message = self._generate_slack_message(notification, result)

        response = requests.post(webhook_url, json=message, timeout=30)
        response.raise_for_status()

        self.logger.info(f"Slack notification sent via {channel.name}")

    def _send_teams_notification(
        self,
        channel: NotificationChannel,
        notification: TestFailureNotification,
        result: ExecutionResult,
    ) -> None:
        """Send Microsoft Teams notification."""
        config = channel.config
        webhook_url = config.get("webhook_url")

        if not webhook_url:
            raise ValueError("Teams webhook URL not configured")

        # Create Teams message
        message = self._generate_teams_message(notification, result)

        response = requests.post(webhook_url, json=message, timeout=30)
        response.raise_for_status()

        self.logger.info(f"Teams notification sent via {channel.name}")

    def _send_webhook_notification(
        self,
        channel: NotificationChannel,
        notification: TestFailureNotification,
        result: ExecutionResult,
    ) -> None:
        """Send generic webhook notification."""
        config = channel.config
        webhook_url = config.get("url")

        if not webhook_url:
            raise ValueError("Webhook URL not configured")

        # Create payload
        payload = {
            "notification": asdict(notification),
            "result": asdict(result),
            "timestamp": datetime.now().isoformat(),
        }

        # Send webhook
        headers = config.get("headers", {})
        response = requests.post(webhook_url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()

        self.logger.info(f"Webhook notification sent via {channel.name}")

    def _send_file_notification(
        self,
        channel: NotificationChannel,
        notification: TestFailureNotification,
        result: ExecutionResult,
    ) -> None:
        """Write notification to file."""
        config = channel.config
        file_path = Path(config.get("file_path", ".ci/notifications.log"))

        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Create notification entry
        entry = {
            "timestamp": datetime.now().isoformat(),
            "channel": channel.name,
            "notification": asdict(notification),
            "result_summary": {
                "execution_id": result.execution_id,
                "total_tests": result.total_tests,
                "failed_tests": result.failed_tests,
                "coverage_percentage": result.coverage_percentage,
            },
        }

        # Append to file
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

        self.logger.info(f"File notification written via {channel.name}")

    def _generate_email_subject(
        self,
        notification: TestFailureNotification,
        result: ExecutionResult,
    ) -> str:
        """Generate email subject line."""
        if result.failed_tests == 0:
            return f"✅ APGI Framework Tests Passed - {result.execution_id}"
        else:
            status = "🔥 CRITICAL" if notification.critical_failures > 0 else "⚠️ FAILED"
            return f"{status} APGI Framework Tests - {result.failed_tests} failures - {result.execution_id}"

    def _generate_email_body(
        self,
        notification: TestFailureNotification,
        result: ExecutionResult,
    ) -> str:
        """Generate HTML email body."""
        if result.failed_tests == 0:
            return self._generate_success_email_body(result)
        else:
            return self._generate_failure_email_body(notification, result)

    def _generate_success_email_body(self, result: ExecutionResult) -> str:
        """Generate success email body."""
        return f"""
        <html>
        <body>
            <h2 style="color: green;">✅ All Tests Passed!</h2>
            <p><strong>Execution ID:</strong> {result.execution_id}</p>
            <p><strong>Total Tests:</strong> {result.total_tests}</p>
            <p><strong>Coverage:</strong> {result.coverage_percentage:.1f}%</p>
            <p><strong>Execution Time:</strong> {result.execution_time_seconds:.1f}s</p>
            <p><strong>Timestamp:</strong> {result.start_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        </body>
        </html>
        """

    def _generate_failure_email_body(
        self,
        notification: TestFailureNotification,
        result: ExecutionResult,
    ) -> str:
        """Generate failure email body."""
        recommendations_html = "".join(
            f"<li>{rec}</li>" for rec in notification.actionable_recommendations
        )

        failures_html = ""
        for failure in notification.failed_tests[:10]:  # Limit to first 10 failures
            failures_html += f"""
            <div style="border-left: 3px solid red; padding-left: 10px; margin: 10px 0;">
                <strong>{failure.get('name', 'Unknown Test')}</strong><br>
                <code style="background-color: #f5f5f5; padding: 5px;">{failure.get('error', 'No error message')[:200]}...</code>
            </div>
            """

        return f"""
        <html>
        <body>
            <h2 style="color: red;">⚠️ Test Failures Detected</h2>
            
            <h3>Summary</h3>
            <ul>
                <li><strong>Execution ID:</strong> {result.execution_id}</li>
                <li><strong>Failed Tests:</strong> {result.failed_tests} / {result.total_tests}</li>
                <li><strong>Critical Failures:</strong> {notification.critical_failures}</li>
                <li><strong>Coverage:</strong> {result.coverage_percentage:.1f}%</li>
                <li><strong>Execution Time:</strong> {result.execution_time_seconds:.1f}s</li>
            </ul>
            
            <h3>Failed Tests</h3>
            {failures_html}
            
            <h3>Actionable Recommendations</h3>
            <ul>
                {recommendations_html}
            </ul>
            
            <h3>Trend Analysis</h3>
            <p>Failure Rate: {notification.trend_analysis.get('failure_rate', 0):.1%}</p>
            <p>Average Coverage: {notification.trend_analysis.get('average_coverage', 0):.1f}%</p>
            
            <p><em>Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
        </body>
        </html>
        """

    def _generate_slack_message(
        self,
        notification: TestFailureNotification,
        result: ExecutionResult,
    ) -> Dict[str, Any]:
        """Generate Slack message payload."""
        if result.failed_tests == 0:
            return {
                "text": f"✅ All tests passed! ({result.total_tests} tests, {result.coverage_percentage:.1f}% coverage)",
                "attachments": [
                    {
                        "color": "good",
                        "fields": [
                            {
                                "title": "Execution ID",
                                "value": result.execution_id,
                                "short": True,
                            },
                            {
                                "title": "Coverage",
                                "value": f"{result.coverage_percentage:.1f}%",
                                "short": True,
                            },
                            {
                                "title": "Duration",
                                "value": f"{result.execution_time_seconds:.1f}s",
                                "short": True,
                            },
                        ],
                    }
                ],
            }
        else:
            color = "danger" if notification.critical_failures > 0 else "warning"
            return {
                "text": f"⚠️ Test failures detected: {result.failed_tests} failed out of {result.total_tests} tests",
                "attachments": [
                    {
                        "color": color,
                        "fields": [
                            {
                                "title": "Execution ID",
                                "value": result.execution_id,
                                "short": True,
                            },
                            {
                                "title": "Failed Tests",
                                "value": str(result.failed_tests),
                                "short": True,
                            },
                            {
                                "title": "Critical Failures",
                                "value": str(notification.critical_failures),
                                "short": True,
                            },
                            {
                                "title": "Coverage",
                                "value": f"{result.coverage_percentage:.1f}%",
                                "short": True,
                            },
                        ],
                        "text": "\n".join(notification.actionable_recommendations[:3]),
                    }
                ],
            }

    def _generate_teams_message(
        self,
        notification: TestFailureNotification,
        result: ExecutionResult,
    ) -> Dict[str, Any]:
        """Generate Microsoft Teams message payload."""
        if result.failed_tests == 0:
            return {
                "@type": "MessageCard",
                "@context": "http://schema.org/extensions",
                "themeColor": "00FF00",
                "summary": "All tests passed",
                "sections": [
                    {
                        "activityTitle": "✅ All Tests Passed",
                        "activitySubtitle": f"Execution ID: {result.execution_id}",
                        "facts": [
                            {
                                "name": "Total Tests",
                                "value": str(result.total_tests),
                            },
                            {
                                "name": "Coverage",
                                "value": f"{result.coverage_percentage:.1f}%",
                            },
                            {
                                "name": "Duration",
                                "value": f"{result.execution_time_seconds:.1f}s",
                            },
                        ],
                    }
                ],
            }
        else:
            theme_color = "FF0000" if notification.critical_failures > 0 else "FFA500"
            return {
                "@type": "MessageCard",
                "@context": "http://schema.org/extensions",
                "themeColor": theme_color,
                "summary": f"{result.failed_tests} test failures detected",
                "sections": [
                    {
                        "activityTitle": "⚠️ Test Failures Detected",
                        "activitySubtitle": f"Execution ID: {result.execution_id}",
                        "facts": [
                            {
                                "name": "Failed Tests",
                                "value": f"{result.failed_tests} / {result.total_tests}",
                            },
                            {
                                "name": "Critical Failures",
                                "value": str(notification.critical_failures),
                            },
                            {
                                "name": "Coverage",
                                "value": f"{result.coverage_percentage:.1f}%",
                            },
                            {
                                "name": "Duration",
                                "value": f"{result.execution_time_seconds:.1f}s",
                            },
                        ],
                        "text": "\n\n".join(notification.actionable_recommendations[:3]),
                    }
                ],
            }


# Convenience functions for common notification setups
def create_email_channel(
    name: str,
    smtp_server: str,
    from_email: str,
    to_emails: List[str],
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> NotificationChannel:
    """Create an email notification channel."""
    return NotificationChannel(
        name=name,
        type="email",
        config={
            "smtp_server": smtp_server,
            "smtp_port": 587,
            "use_tls": True,
            "from_email": from_email,
            "to_emails": to_emails,
            "username": username,
            "password": password,
        },
    )


def create_slack_channel(name: str, webhook_url: str) -> NotificationChannel:
    """Create a Slack notification channel."""
    return NotificationChannel(name=name, type="slack", config={"webhook_url": webhook_url})


def create_teams_channel(name: str, webhook_url: str) -> NotificationChannel:
    """Create a Microsoft Teams notification channel."""
    return NotificationChannel(name=name, type="teams", config={"webhook_url": webhook_url})


def create_file_channel(name: str, file_path: str) -> NotificationChannel:
    """Create a file-based notification channel."""
    return NotificationChannel(name=name, type="file", config={"file_path": file_path})
