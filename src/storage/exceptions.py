"""Custom exceptions for the Blackboard Storage and Concurrency layer."""

from typing import Optional


class BlackboardStorageError(Exception):
    """Base exception for all blackboard storage operations."""
    pass


class SessionNotFoundError(BlackboardStorageError):
    """Raised when an operation targets a blackboard session that does not exist."""
    def __init__(self, session_id: str):
        super().__init__(f"Blackboard session '{session_id}' not found.")
        self.session_id = session_id


class LockAcquisitionError(BlackboardStorageError):
    """Raised when an agent fails to acquire an exclusive write lock."""
    def __init__(self, session_id: str, agent_id: str, holder_id: Optional[str] = None):
        msg = f"Agent '{agent_id}' failed to acquire lock on session '{session_id}'."
        if holder_id:
            msg += f" Current holder is '{holder_id}'."
        super().__init__(msg)
        self.session_id = session_id
        self.agent_id = agent_id
        self.holder_id = holder_id


class LockNotHeldError(BlackboardStorageError):
    """Raised when an agent attempts a write or release without holding a valid lock lease."""
    def __init__(self, session_id: str, agent_id: str, detail: Optional[str] = None):
        msg = f"Agent '{agent_id}' does not hold a valid lock on session '{session_id}'."
        if detail:
            msg += f" {detail}"
        super().__init__(msg)
        self.session_id = session_id
        self.agent_id = agent_id


class VersionConflictError(BlackboardStorageError):
    """Raised when optimistic concurrency control detects a version mismatch."""
    def __init__(self, session_id: str, expected_version: int, current_version: int):
        super().__init__(
            f"Optimistic concurrency conflict on session '{session_id}': "
            f"expected version {expected_version}, but current version is {current_version}."
        )
        self.session_id = session_id
        self.expected_version = expected_version
        self.current_version = current_version
