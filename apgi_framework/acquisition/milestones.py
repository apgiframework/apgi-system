"""Session milestones: the nine-step offline-first lifecycle as recorded facts.

The :class:`~apgi_framework.acquisition.state_machine.SessionMachine` tracks *where a
session is*; milestones record *what has been achieved*, once each, with a timestamp,
in the encrypted store. Interrupted uploads never remove a milestone and never delete
local data.
"""

from __future__ import annotations

import enum
from typing import List


class SessionMilestone(str, enum.Enum):
    SESSION_CREATED = "session_created"
    CONFIGURATION_VERIFIED = "configuration_verified"
    RECORDING_STARTED = "recording_started"
    EVENTS_AND_CHUNKS_JOURNALED = "events_and_chunks_journaled"
    RECORDING_COMPLETED = "recording_completed"
    LOCAL_MANIFEST_FINALIZED = "local_manifest_finalized"
    UPLOAD_PENDING = "upload_pending"
    SERVER_VERIFIED = "server_verified"
    LOCAL_RETENTION_POLICY_APPLIED = "local_retention_policy_applied"


MILESTONE_ORDER: List[SessionMilestone] = list(SessionMilestone)


def missing_predecessors(
    reached: List[SessionMilestone], milestone: SessionMilestone
) -> List[SessionMilestone]:
    """Milestones that should precede ``milestone`` but have not been reached."""
    index = MILESTONE_ORDER.index(milestone)
    return [m for m in MILESTONE_ORDER[:index] if m not in reached]
