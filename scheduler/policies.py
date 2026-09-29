"""
scheduler/policies.py
Turn arbitration strategies for multi-agent reasoning in the Intelligible Blackboard Architecture.
Provides Round-Robin, Static Priority, and dynamic Reactive PXP state machine scheduling.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from contracts.schemas import (
    AgentContribution,
    AgentRole,
    BlackboardSnapshot,
    PXPTag,
    ScheduledTurn,
)


class BaseArbitrationPolicy(ABC):
    """Abstract base class for turn arbitration policies."""

    @abstractmethod
    def next_turns(
        self,
        session_id: str,
        current_snapshot: BlackboardSnapshot,
        registered_agents: Dict[str, Dict[str, Any]],
        last_contribution: Optional[AgentContribution] = None,
    ) -> List[ScheduledTurn]:
        """
        Determines the next agent turns to schedule based on blackboard state.
        
        Args:
            session_id: Active session identifier.
            current_snapshot: Latest snapshot of the blackboard state.
            registered_agents: Mapping of agent_id -> metadata (including 'role', 'priority').
            last_contribution: Most recently committed contribution, if any.
            
        Returns:
            List of ScheduledTurn objects to append to the turn queue.
        """
        pass


class RoundRobinPolicy(BaseArbitrationPolicy):
    """
    Fair cyclic rotation across all registered agents in fixed registration order.
    """

    def __init__(self):
        self._last_index: int = -1

    def next_turns(
        self,
        session_id: str,
        current_snapshot: BlackboardSnapshot,
        registered_agents: Dict[str, Dict[str, Any]],
        last_contribution: Optional[AgentContribution] = None,
    ) -> List[ScheduledTurn]:
        if not registered_agents:
            return []

        # Filter out one-shot counterfactual mediator agents from cyclic rotation
        eligible_agents = [
            aid for aid, meta in registered_agents.items()
            if str(meta.get("role", "")).upper() not in ["COUNTERFACTUAL", AgentRole.COUNTERFACTUAL.value]
        ]
        if not eligible_agents:
            eligible_agents = list(registered_agents.keys())

        self._last_index = (self._last_index + 1) % len(eligible_agents)
        next_agent_id = eligible_agents[self._last_index]

        target_id = last_contribution.contribution_id if last_contribution else None
        return [
            ScheduledTurn(
                session_id=session_id,
                agent_id=next_agent_id,
                priority=1,
                target_contribution_id=target_id,
            )
        ]


class PriorityPolicy(BaseArbitrationPolicy):
    """
    Arbitration based strictly on registered agent priority scores.
    Lower numerical priority = higher scheduling precedence.
    """

    def next_turns(
        self,
        session_id: str,
        current_snapshot: BlackboardSnapshot,
        registered_agents: Dict[str, Dict[str, Any]],
        last_contribution: Optional[AgentContribution] = None,
    ) -> List[ScheduledTurn]:
        if not registered_agents:
            return []

        # Exclude the agent that just spoke to avoid immediate self-repetition
        last_agent_id = last_contribution.agent_id if last_contribution else None
        candidates = [
            (aid, meta)
            for aid, meta in registered_agents.items()
            if aid != last_agent_id or len(registered_agents) == 1
        ]

        if not candidates:
            candidates = list(registered_agents.items())

        # Sort by registered priority ascending
        candidates.sort(key=lambda item: item[1].get("priority", 1))
        chosen_agent_id, meta = candidates[0]

        target_id = last_contribution.contribution_id if last_contribution else None
        return [
            ScheduledTurn(
                session_id=session_id,
                agent_id=chosen_agent_id,
                priority=meta.get("priority", 1),
                target_contribution_id=target_id,
            )
        ]


class ReactivePXPPolicy(BaseArbitrationPolicy):
    """
    Dynamic PXP state machine arbitration policy.
    
    Behavior:
    1. If board is empty: Queues PRIMARY agent (or first agent) to formulate initial proposal.
    2. If last tag is PROPOSE / REVISE: Queues CRITIC and DOMAIN_EXPERT agents to evaluate and challenge.
    3. If last tag is REFUTE / REJECT: Queues the author of the disputed claim (or PRIMARY) to rebut/defend.
    4. If last tag is RATIFY: Queues remaining agents who have not yet ratified to confirm consensus.
    """

    def next_turns(
        self,
        session_id: str,
        current_snapshot: BlackboardSnapshot,
        registered_agents: Dict[str, Dict[str, Any]],
        last_contribution: Optional[AgentContribution] = None,
    ) -> List[ScheduledTurn]:
        if not registered_agents:
            return []

        agent_ids = list(registered_agents.keys())

        # Case 1: Initial turn - Dispatch Primary agent to propose
        if last_contribution is None or not current_snapshot.contributions:
            primary_agents = [
                aid for aid, meta in registered_agents.items()
                if meta.get("role") in [AgentRole.PRIMARY, AgentRole.PRIMARY.value, "primary"]
            ]
            first_agent = primary_agents[0] if primary_agents else agent_ids[0]
            return [
                ScheduledTurn(
                    session_id=session_id,
                    agent_id=first_agent,
                    priority=1,
                    target_contribution_id=None,
                )
            ]

        last_tag = last_contribution.tag
        target_id = last_contribution.contribution_id

        # Case 2: PROPOSE or REVISE - Elicit reviews from Critics and Domain Experts
        if last_tag in [PXPTag.REVISE, "PROPOSE", "REVISE"]:
            reviewers = [
                aid for aid in agent_ids
                if aid != last_contribution.agent_id
            ]
            # Prioritize Critics first, then others
            scheduled = []
            for aid in reviewers:
                role = registered_agents[aid].get("role")
                priority = 1 if role in [AgentRole.CRITIC, AgentRole.CRITIC.value, "critic"] else 2
                scheduled.append(
                    ScheduledTurn(
                        session_id=session_id,
                        agent_id=aid,
                        priority=priority,
                        target_contribution_id=target_id,
                    )
                )
            return scheduled if scheduled else [
                ScheduledTurn(
                    session_id=session_id,
                    agent_id=agent_ids[0],
                    priority=1,
                    target_contribution_id=target_id,
                )
            ]

        # Case 3: REFUTE or REJECT - Challenge raised, queue author of parent or Primary
        if last_tag in [PXPTag.REFUTE, PXPTag.REJECT]:
            # Target the parent contribution author if known
            rebuttal_agent_id = None
            if last_contribution.target_contribution_id:
                for c in current_snapshot.contributions:
                    if c.contribution_id == last_contribution.target_contribution_id:
                        rebuttal_agent_id = c.agent_id
                        break

            # Fallback to Primary or alternate agent if target agent not found
            if not rebuttal_agent_id or rebuttal_agent_id == last_contribution.agent_id:
                alternates = [aid for aid in agent_ids if aid != last_contribution.agent_id]
                rebuttal_agent_id = alternates[0] if alternates else agent_ids[0]

            return [
                ScheduledTurn(
                    session_id=session_id,
                    agent_id=rebuttal_agent_id,
                    priority=1,
                    target_contribution_id=target_id,
                )
            ]

        # Case 4: RATIFY - Agent agrees. Elicit remaining ratifications
        if last_tag == PXPTag.RATIFY:
            # Find agents who haven't ratified the current active claim or latest turn
            active_target = last_contribution.target_contribution_id or target_id
            recent_ratifiers = {
                c.agent_id for c in current_snapshot.contributions
                if c.tag == PXPTag.RATIFY and c.target_contribution_id == active_target
            }
            recent_ratifiers.add(last_contribution.agent_id)

            unratified = [aid for aid in agent_ids if aid not in recent_ratifiers]
            if unratified:
                return [
                    ScheduledTurn(
                        session_id=session_id,
                        agent_id=unratified[0],
                        priority=1,
                        target_contribution_id=target_id,
                    )
                ]

        # Default fallback: Next sequential agent
        alternates = [aid for aid in agent_ids if aid != last_contribution.agent_id]
        chosen = alternates[0] if alternates else agent_ids[0]
        return [
            ScheduledTurn(
                session_id=session_id,
                agent_id=chosen,
                priority=2,
                target_contribution_id=target_id,
            )
        ]
