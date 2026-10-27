"""Notification module for APGI Framework.

This module provides notification and alert capabilities.
"""

from typing import Any, Dict, List, Optional


# Mock classes for testing
class NotificationManager:
    """Mock notification manager for testing purposes."""

    def __init__(self) -> None:
        self.notifications: List[Dict[str, Any]] = []
        self.subscribers: Dict[str, List[str]] = {}

    def send_notification(self, notification_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send a notification."""
        notification_id = f"notif_{hash(str(notification_data)) % 10000:04d}"
        notification: Dict[str, Any] = {
            "notification_id": notification_id,
            "data": notification_data,
            "sent_at": "2024-01-01T00:00:00Z",
            "delivered": True,
        }
        self.notifications.append(notification)
        return notification

    def subscribe(self, subscriber_id: str, notification_types: List[str]) -> None:
        """Subscribe to notification types."""
        self.subscribers[subscriber_id] = notification_types

    def get_notifications(self, subscriber_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get notifications, optionally filtered by subscriber."""
        if subscriber_id:
            subscriber_types = self.subscribers.get(subscriber_id, [])
            return [n for n in self.notifications if n["data"].get("type") in subscriber_types]
        return self.notifications

    def mark_as_read(self, notification_id: str) -> None:
        """Mark a notification as read."""
        for notification in self.notifications:
            if notification["notification_id"] == notification_id:
                notification["read"] = True
                notification["read_at"] = "2024-01-01T00:01:00:00Z"
                break


class AlertManager:
    """Mock alert manager for testing purposes."""

    def __init__(self) -> None:
        self.alerts: List[Dict[str, Any]] = []
        self.alert_rules: Dict[str, Any] = {}

    def create_alert(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create an alert."""
        alert_id = f"alert_{hash(str(alert_data)) % 10000:04d}"
        alert: Dict[str, Any] = {
            "alert_id": alert_id,
            "data": alert_data,
            "created_at": "2024-01-01T00:00:00Z",
            "active": True,
        }
        self.alerts.append(alert)
        return alert

    def trigger_alert(self, alert_id: str, trigger_data: Any) -> None:
        """Trigger an alert."""
        for alert in self.alerts:
            if alert["alert_id"] == alert_id:
                alert["triggered"] = True
                alert["triggered_at"] = "2024-01-01T00:00:01Z"
                alert["trigger_data"] = trigger_data
                break

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts."""
        return [alert for alert in self.alerts if alert.get("active", False)]


__all__ = [
    "NotificationManager",
    "AlertManager",
]
