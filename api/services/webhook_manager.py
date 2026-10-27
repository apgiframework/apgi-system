"""
Webhook Manager Service

Manages webhook registration, validation, delivery with retry logic and exponential backoff.
"""

import hashlib
import hmac
import ipaddress
import json
import logging
import socket
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import httpx
from sqlalchemy import or_
from sqlalchemy.orm import Session as DBSession

from api.database.models import WebhookDelivery

# Import circuit breaker utilities
from utils.circuit_breaker_utils import CircuitBreakerException, circuit_breaker

logger = logging.getLogger(__name__)


class WebhookStatus(str, Enum):
    """Webhook delivery status."""

    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"


class WebhookManager:
    """
    Manages webhook registration, validation, and delivery.

    Implements retry logic with exponential backoff for failed deliveries.
    """

    # Retry configuration
    MAX_RETRIES = 5
    INITIAL_RETRY_DELAY_SECONDS = 60  # 1 minute
    MAX_RETRY_DELAY_SECONDS = 3600  # 1 hour
    BACKOFF_MULTIPLIER = 2

    # HTTP timeout configuration
    REQUEST_TIMEOUT_SECONDS = 30

    def __init__(self, webhook_config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize webhook manager.

        Args:
            webhook_config: Webhook configuration containing secret and signature settings
        """
        self.webhook_config = webhook_config or {}
        self.webhook_secret = self.webhook_config.get("secret", "")
        self.signature_algorithm = self.webhook_config.get("signature_algorithm", "sha256")
        self.signature_header = self.webhook_config.get("signature_header", "X-Webhook-Signature")

        self.http_client = httpx.AsyncClient(
            timeout=self.REQUEST_TIMEOUT_SECONDS, follow_redirects=True
        )
        logger.info("WebhookManager initialized")

    def _is_private_ip(self, ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
        """
        Check if an IP address is private or blocked.

        Args:
            ip: IP address to check

        Returns:
            True if the IP is private or blocked, False otherwise
        """
        # IPv6 addresses are generally not private in the same way as IPv4
        # For now, treat all IPv6 as not private (can be refined later)
        if isinstance(ip, ipaddress.IPv6Address):
            return False

        private_ranges = [
            ipaddress.IPv4Network("10.0.0.0/8"),  # Private network
            ipaddress.IPv4Network("172.16.0.0/12"),  # Private network
            ipaddress.IPv4Network("192.168.0.0/16"),  # Private network
            ipaddress.IPv4Network("127.0.0.0/8"),  # Loopback
            ipaddress.IPv4Network("169.254.0.0/16"),  # Link-local
            ipaddress.IPv4Network("0.0.0.0/8"),  # This host on this network
        ]

        # Check if IP is in any private range
        for network in private_ranges:
            if ip in network:
                return True

        # Block specific cloud metadata IPs
        if ip == ipaddress.IPv4Address("169.254.169.254"):  # AWS metadata
            return True

        return False

    def _generate_webhook_signature(self, payload: Dict[str, Any]) -> str:
        """
        Generate HMAC signature for webhook payload.

        Args:
            payload: Webhook payload dictionary

        Returns:
            Hex-encoded signature string
        """
        if not self.webhook_secret:
            logger.warning("Webhook secret not configured - signature verification disabled")
            return ""

        # Serialize payload to JSON with sorted keys for consistent signatures
        payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)

        # Generate HMAC signature
        secret_bytes = self.webhook_secret.encode("utf-8")
        payload_bytes = payload_json.encode("utf-8")

        if self.signature_algorithm == "sha256":
            signature = hmac.new(secret_bytes, payload_bytes, hashlib.sha256).hexdigest()
        elif self.signature_algorithm == "sha512":
            signature = hmac.new(secret_bytes, payload_bytes, hashlib.sha512).hexdigest()
        else:
            # Default to sha256
            signature = hmac.new(secret_bytes, payload_bytes, hashlib.sha256).hexdigest()

        return signature

    async def validate_webhook_url(self, url: str) -> bool:
        """
        Validate webhook URL format and accessibility.

        Args:
            url: Webhook URL to validate

        Returns:
            True if URL is valid and accessible

        Raises:
            ValueError: If URL is invalid
        """
        # Basic URL validation
        if not url:
            raise ValueError("Webhook URL cannot be empty")

        if not url.startswith(("http://", "https://")):
            raise ValueError("Webhook URL must start with http:// or https://")

        # Check URL length
        if len(url) > 500:
            raise ValueError("Webhook URL exceeds maximum length of 500 characters")

        # Use centralized security utility for SSRF protection
        from api.utils.security import is_safe_url

        if not is_safe_url(url):
            raise ValueError(
                "Webhook URL points to an unsafe destination (private, reserved, or non-HTTPS)"
            )

        # Optionally perform a HEAD request to verify accessibility
        # (commented out to avoid blocking during validation)
        # try:
        #     response = await self.http_client.head(url, timeout=5.0)
        #     return response.status_code < 500
        # except Exception as e:
        #     logger.warning(f"Webhook URL validation failed for {url}: {e}")
        #     return False

        return True

    async def create_webhook_delivery(
        self, db: DBSession, task_id: str, webhook_url: str, payload: Dict[str, Any]
    ) -> str:
        """
        Create a webhook delivery record.

        Args:
            db: Database session
            task_id: Associated task ID
            webhook_url: Target webhook URL
            payload: Webhook payload

        Returns:
            Delivery ID
        """
        # Validate URL
        await self.validate_webhook_url(webhook_url)

        # Create delivery record
        delivery = WebhookDelivery(
            task_id=task_id,
            webhook_url=webhook_url,
            payload=payload,
            status=WebhookStatus.PENDING.value,
            attempts=0,
        )

        db.add(delivery)
        db.commit()
        db.refresh(delivery)

        logger.info(f"Created webhook delivery {delivery.delivery_id} for task {task_id}")

        return delivery.delivery_id  # type: ignore[return-value]

    async def deliver_webhook(self, db: DBSession, delivery_id: str) -> bool:
        """
        Attempt to deliver a webhook.

        Args:
            db: Database session
            delivery_id: Webhook delivery ID

        Returns:
            True if delivery succeeded, False otherwise
        """
        # Get delivery record with circuit breaker protection
        delivery = await self._get_delivery_record_safe(db, delivery_id)

        if not delivery:
            logger.error(f"Webhook delivery {delivery_id} not found")
            return False

        # Update attempt count and timestamp with circuit breaker
        await self._update_delivery_attempt_safe(db, delivery)

        # Deliver webhook with circuit breaker protection
        return await self._deliver_webhook_request_safe(db, delivery)

    @circuit_breaker(name="webhook_delivery_get_record", failure_threshold=5, recovery_timeout=30.0)
    async def _get_delivery_record_safe(
        self, db: DBSession, delivery_id: str
    ) -> Optional[WebhookDelivery]:
        """Get delivery record with circuit breaker protection."""
        try:
            delivery = (
                db.query(WebhookDelivery).filter(WebhookDelivery.delivery_id == delivery_id).first()
            )
            return delivery
        except Exception as e:
            logger.error(f"Database error getting delivery record {delivery_id}: {e}")
            raise CircuitBreakerException(f"Database operation failed: {e}")

    @circuit_breaker(name="webhook_delivery_update", failure_threshold=3, recovery_timeout=15.0)
    async def _update_delivery_attempt_safe(self, db: DBSession, delivery: WebhookDelivery) -> None:
        """Update delivery attempt with circuit breaker protection."""
        try:
            # Update attempt count and timestamp
            delivery.attempts += 1  # type: ignore[assignment]
            delivery.last_attempt_at = datetime.utcnow()  # type: ignore[assignment]
            delivery.status = (
                WebhookStatus.RETRYING.value  # type: ignore[assignment]
                if delivery.attempts > 1
                else WebhookStatus.PENDING.value
            )
            db.commit()
        except Exception as e:
            db.rollback()
            logger.error(
                f"Database error updating delivery attempt for {delivery.delivery_id}: {e}"
            )
            raise CircuitBreakerException(f"Database operation failed: {e}")

    @circuit_breaker(name="webhook_http_delivery", failure_threshold=3, recovery_timeout=120.0)
    async def _deliver_webhook_request_safe(self, db: DBSession, delivery: WebhookDelivery) -> bool:
        """Deliver webhook request with circuit breaker protection."""
        try:
            # Send POST request to webhook URL
            logger.info(
                f"Attempting webhook delivery {delivery.delivery_id} "
                f"(attempt {delivery.attempts}/{self.MAX_RETRIES}) to {delivery.webhook_url}"
            )

            # Generate webhook signature
            signature = self._generate_webhook_signature(delivery.payload)  # type: ignore[arg-type]

            # Prepare headers
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "APGI-API-Webhook/1.0",
                "X-Webhook-Delivery-ID": str(delivery.delivery_id),
                "X-Webhook-Attempt": str(delivery.attempts),
            }

            # Add signature header if signature was generated
            if signature:
                headers[self.signature_header] = signature

            # Re-validate webhook URL to prevent DNS rebinding attacks
            parsed = urlparse(str(delivery.webhook_url))
            hostname = parsed.hostname

            if hostname:
                try:
                    resolved_ip = socket.gethostbyname(hostname)
                    ip = ipaddress.ip_address(resolved_ip)
                    if self._is_private_ip(ip):
                        delivery.error_message = "Webhook URL resolves to private or blocked IP address"  # type: ignore[assignment]
                        await self._schedule_retry(db, delivery)
                        return False
                except (socket.gaierror, ValueError) as e:
                    delivery.error_message = f"Cannot resolve or validate hostname '{hostname}': {e}"  # type: ignore[assignment]
                    await self._schedule_retry(db, delivery)
                    return False

            response = await self.http_client.post(
                delivery.webhook_url,  # type: ignore[arg-type]
                json=delivery.payload,
                headers=headers,
            )

            # Store response
            delivery.response_status = response.status_code  # type: ignore[assignment]
            delivery.response_body = response.text[:1000]  # type: ignore[assignment]

            # Check if delivery was successful (2xx status code)
            if 200 <= response.status_code < 300:
                delivery.status = WebhookStatus.DELIVERED.value  # type: ignore[assignment]
                delivery.next_retry_at = None  # type: ignore[assignment]

                logger.info(
                    f"Webhook delivery {delivery.delivery_id} succeeded "
                    f"(status: {response.status_code})"
                )
                return True
            else:
                # Non-2xx response - schedule retry
                logger.warning(
                    f"Webhook delivery {delivery.delivery_id} failed with status {response.status_code}"
                )
                delivery.error_message = f"HTTP {response.status_code}: {response.text}"  # type: ignore[assignment]
                await self._schedule_retry(db, delivery)
                return False

        except httpx.TimeoutException as e:
            logger.warning(f"Webhook delivery {delivery.delivery_id} timed out: {e}")
            delivery.error_message = f"Request timeout: {str(e)}"  # type: ignore[assignment]
            await self._schedule_retry(db, delivery)
            return False

        except httpx.RequestError as e:
            logger.warning(
                f"Webhook delivery {delivery.delivery_id} failed with request error: {e}"
            )
            delivery.error_message = f"Request error: {str(e)}"  # type: ignore[assignment]
            await self._schedule_retry(db, delivery)
            return False

        except Exception as e:
            logger.error(
                f"Webhook delivery {delivery.delivery_id} failed with unexpected error: {e}",
                exc_info=True,
            )
            delivery.error_message = f"Unexpected error: {str(e)}"  # type: ignore[assignment]
            await self._schedule_retry(db, delivery)
            return False

    async def _schedule_retry(self, db: DBSession, delivery: WebhookDelivery) -> None:
        """
        Schedule retry for failed webhook delivery with exponential backoff.

        Args:
            db: Database session
            delivery: Webhook delivery record
        """
        if delivery.attempts >= self.MAX_RETRIES:
            # Max retries reached - mark as failed
            delivery.status = WebhookStatus.FAILED.value  # type: ignore[assignment]
            delivery.next_retry_at = None  # type: ignore[assignment]
            logger.error(
                f"Webhook delivery {delivery.delivery_id} failed permanently "
                f"after {delivery.attempts} attempts"
            )
        else:
            # Calculate next retry time with exponential backoff
            retry_delay = min(
                self.INITIAL_RETRY_DELAY_SECONDS
                * pow(self.BACKOFF_MULTIPLIER, delivery.attempts - 1),
                self.MAX_RETRY_DELAY_SECONDS,
            )

            delivery.next_retry_at = datetime.utcnow() + timedelta(seconds=retry_delay)  # type: ignore[assignment]
            delivery.status = WebhookStatus.RETRYING.value  # type: ignore[assignment]

            logger.info(
                f"Webhook delivery {delivery.delivery_id} scheduled for retry "
                f"in {retry_delay} seconds (attempt {delivery.attempts + 1}/{self.MAX_RETRIES})"
            )

        db.commit()

    async def process_pending_webhooks(self, db: DBSession) -> None:
        """
        Process all pending webhook deliveries that are ready for retry.

        Args:
            db: Database session
        """
        # Query pending/retrying webhooks that are ready for delivery
        now = datetime.utcnow()

        pending_deliveries = (
            db.query(WebhookDelivery)
            .filter(
                WebhookDelivery.status.in_(
                    [WebhookStatus.PENDING.value, WebhookStatus.RETRYING.value]
                ),
                or_(
                    WebhookDelivery.next_retry_at.is_(None),
                    WebhookDelivery.next_retry_at <= now,  # type: ignore[arg-type]
                ),
            )
            .all()
        )

        logger.info(f"Processing {len(pending_deliveries)} pending webhook deliveries")

        # Process each delivery
        for delivery in pending_deliveries:
            try:
                await self.deliver_webhook(db, delivery.delivery_id)  # type: ignore[arg-type]
            except Exception as e:
                logger.error(
                    f"Error processing webhook delivery {delivery.delivery_id}: {e}", exc_info=True
                )

    async def get_delivery_status(
        self, db: DBSession, delivery_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get webhook delivery status.

        Args:
            db: Database session
            delivery_id: Webhook delivery ID

        Returns:
            Dict with delivery status information, or None if not found
        """
        delivery = (
            db.query(WebhookDelivery).filter(WebhookDelivery.delivery_id == delivery_id).first()
        )

        if not delivery:
            return None

        return {
            "delivery_id": delivery.delivery_id,
            "task_id": delivery.task_id,
            "webhook_url": delivery.webhook_url,
            "status": delivery.status,
            "attempts": delivery.attempts,
            "last_attempt_at": (
                delivery.last_attempt_at.isoformat() if delivery.last_attempt_at else None
            ),
            "next_retry_at": delivery.next_retry_at.isoformat() if delivery.next_retry_at else None,
            "response_status": delivery.response_status,
            "error_message": delivery.error_message,
            "created_at": delivery.created_at.isoformat(),
        }

    async def get_task_deliveries(self, db: DBSession, task_id: str) -> list[Dict[str, Any]]:
        """
        Get all webhook deliveries for a task.

        Args:
            db: Database session
            task_id: Task ID

        Returns:
            List of delivery status dicts
        """
        deliveries = db.query(WebhookDelivery).filter(WebhookDelivery.task_id == task_id).all()

        return [
            {
                "delivery_id": d.delivery_id,
                "webhook_url": d.webhook_url,
                "status": d.status,
                "attempts": d.attempts,
                "last_attempt_at": d.last_attempt_at.isoformat() if d.last_attempt_at else None,
                "response_status": d.response_status,
                "error_message": d.error_message,
            }
            for d in deliveries
        ]

    async def close(self) -> None:
        """Close HTTP client."""
        await self.http_client.aclose()
        logger.info("WebhookManager closed")
