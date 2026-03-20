"""
Base Blender Agent - Foundation for all autonomous Blender agents.

Provides the core agent architecture with:
- Self-awareness and capability tracking
- Task execution with history
- Autonomous self-improvement hooks
- Learning from past operations
"""

import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent lifecycle states."""
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    EVALUATING = "evaluating"
    IMPROVING = "improving"
    ERROR = "error"
    PAUSED = "paused"


class TaskStatus(Enum):
    """Status of a task execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class AgentCapability(Enum):
    """Capabilities an agent can have."""
    MODELING = "modeling"
    MATERIALS = "materials"
    RENDERING = "rendering"
    ANIMATION = "animation"
    SCENE_SETUP = "scene_setup"
    COMPOSITING = "compositing"
    LIGHTING = "lighting"
    TEXTURING = "texturing"
    UV_MAPPING = "uv_mapping"
    RIGGING = "rigging"
    PHYSICS = "physics"
    PARTICLES = "particles"
    EXPORT = "export"


@dataclass
class TaskResult:
    """Result of a task execution with metrics."""
    task_id: str
    status: TaskStatus
    output: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    quality_score: float = 0.0  # 0-1, assessed by improvement engine
    metadata: Dict[str, Any] = field(default_factory=dict)
    improvements_applied: List[str] = field(default_factory=list)
    retry_count: int = 0

    @property
    def succeeded(self) -> bool:
        return self.status == TaskStatus.COMPLETED


@dataclass
class AgentMemory:
    """Agent's memory of past operations and learnings."""
    task_history: List[TaskResult] = field(default_factory=list)
    learned_patterns: Dict[str, Any] = field(default_factory=dict)
    failure_patterns: Dict[str, int] = field(default_factory=dict)
    success_patterns: Dict[str, int] = field(default_factory=dict)
    performance_scores: List[float] = field(default_factory=list)
    improvement_log: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def average_quality(self) -> float:
        if not self.performance_scores:
            return 0.0
        return sum(self.performance_scores) / len(self.performance_scores)

    @property
    def success_rate(self) -> float:
        if not self.task_history:
            return 0.0
        successes = sum(1 for t in self.task_history if t.succeeded)
        return successes / len(self.task_history)

    def record_task(self, result: TaskResult) -> None:
        """Record a task result in memory."""
        self.task_history.append(result)
        if result.quality_score > 0:
            self.performance_scores.append(result.quality_score)

        # Track patterns
        task_type = result.metadata.get("task_type", "unknown")
        if result.succeeded:
            self.success_patterns[task_type] = self.success_patterns.get(task_type, 0) + 1
        else:
            self.failure_patterns[task_type] = self.failure_patterns.get(task_type, 0) + 1

    def get_best_approach(self, task_type: str) -> Optional[Dict[str, Any]]:
        """Find the best approach for a given task type from history."""
        best_result = None
        best_score = 0.0

        for result in self.task_history:
            if (
                result.metadata.get("task_type") == task_type
                and result.succeeded
                and result.quality_score > best_score
            ):
                best_score = result.quality_score
                best_result = result

        if best_result:
            return {
                "approach": best_result.metadata.get("approach", {}),
                "quality_score": best_result.quality_score,
                "params": best_result.metadata.get("params", {}),
            }
        return None


class BlenderAgent(ABC):
    """
    Base class for all Blender automation agents.

    Provides autonomous capabilities including:
    - Self-task creation and planning
    - Execution with error recovery
    - Quality assessment
    - Continuous improvement through learning
    - Memory of past operations
    """

    def __init__(
        self,
        name: str,
        capabilities: List[AgentCapability],
        bridge: Optional[Any] = None,
        max_retries: int = 3,
        auto_improve: bool = True,
    ):
        """
        Initialize the agent.

        Args:
            name: Agent name/identifier
            capabilities: List of agent capabilities
            bridge: BlenderBridge instance for executing Blender operations
            max_retries: Maximum retry attempts for failed tasks
            auto_improve: Enable autonomous self-improvement
        """
        self.agent_id = str(uuid.uuid4())[:8]
        self.name = name
        self.capabilities = capabilities
        self.bridge = bridge
        self.max_retries = max_retries
        self.auto_improve = auto_improve
        self.state = AgentState.IDLE
        self.memory = AgentMemory()
        self._hooks: Dict[str, List[Callable]] = {
            "pre_execute": [],
            "post_execute": [],
            "on_error": [],
            "on_improve": [],
        }

        logger.info(f"Agent '{self.name}' ({self.agent_id}) initialized with "
                     f"capabilities: {[c.value for c in self.capabilities]}")

    def has_capability(self, capability: AgentCapability) -> bool:
        """Check if agent has a specific capability."""
        return capability in self.capabilities

    def register_hook(self, event: str, callback: Callable) -> None:
        """Register a callback hook for an event."""
        if event in self._hooks:
            self._hooks[event].append(callback)

    def _fire_hooks(self, event: str, **kwargs: Any) -> None:
        """Fire all registered hooks for an event."""
        for callback in self._hooks.get(event, []):
            try:
                callback(agent=self, **kwargs)
            except Exception as e:
                logger.warning(f"Hook error in {event}: {e}")

    @abstractmethod
    def get_available_tasks(self) -> List[Dict[str, Any]]:
        """
        Return a list of tasks this agent can perform.

        Returns:
            List of task definitions with name, description, and parameters
        """
        ...

    @abstractmethod
    def execute_task(self, task_name: str, **params: Any) -> TaskResult:
        """
        Execute a specific task.

        Args:
            task_name: Name of the task to execute
            **params: Task parameters

        Returns:
            TaskResult with execution details
        """
        ...

    @abstractmethod
    def assess_quality(self, result: TaskResult) -> float:
        """
        Assess the quality of a task result.

        Args:
            result: The task result to assess

        Returns:
            Quality score from 0.0 to 1.0
        """
        ...

    def run_task(self, task_name: str, **params: Any) -> TaskResult:
        """
        Run a task with full lifecycle management.

        Handles pre/post hooks, error recovery, quality assessment,
        and improvement tracking.

        Args:
            task_name: Task to execute
            **params: Task parameters

        Returns:
            TaskResult with full lifecycle data
        """
        task_id = f"{self.agent_id}-{str(uuid.uuid4())[:8]}"
        self.state = AgentState.PLANNING

        logger.info(f"[{self.name}] Starting task '{task_name}' (id={task_id})")

        # Pre-execution hooks
        self._fire_hooks("pre_execute", task_name=task_name, params=params)

        # Check if we have learned approaches for this task type
        best_approach = self.memory.get_best_approach(task_name)
        if best_approach and self.auto_improve:
            logger.info(f"[{self.name}] Using learned approach "
                        f"(quality={best_approach['quality_score']:.2f})")
            # Merge learned params with user params
            merged_params = {**best_approach.get("params", {}), **params}
            params = merged_params

        # Execute with retries
        self.state = AgentState.EXECUTING
        result = None
        retry_count = 0

        while retry_count <= self.max_retries:
            start_time = time.time()
            try:
                result = self.execute_task(task_name, **params)
                result.task_id = task_id
                result.execution_time = time.time() - start_time
                result.retry_count = retry_count
                result.metadata["task_type"] = task_name
                result.metadata["params"] = params

                if result.succeeded:
                    break

            except Exception as e:
                logger.error(f"[{self.name}] Task failed: {e}")
                result = TaskResult(
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    error=str(e),
                    execution_time=time.time() - start_time,
                    retry_count=retry_count,
                    metadata={"task_type": task_name, "params": params},
                )
                self._fire_hooks("on_error", error=e, result=result)

            if not result.succeeded and retry_count < self.max_retries:
                retry_count += 1
                logger.info(f"[{self.name}] Retrying task (attempt {retry_count}/{self.max_retries})")
                result.status = TaskStatus.RETRYING
                time.sleep(0.5 * retry_count)  # Backoff
            else:
                break

        # Quality assessment
        self.state = AgentState.EVALUATING
        if result.succeeded:
            result.quality_score = self.assess_quality(result)
            logger.info(f"[{self.name}] Task completed (quality={result.quality_score:.2f})")

        # Record in memory
        self.memory.record_task(result)

        # Post-execution hooks
        self._fire_hooks("post_execute", result=result)

        # Self-improvement
        if self.auto_improve and result.succeeded:
            self.state = AgentState.IMPROVING
            improvements = self._self_improve(result)
            result.improvements_applied = improvements

        self.state = AgentState.IDLE
        return result

    def _self_improve(self, result: TaskResult) -> List[str]:
        """
        Autonomous self-improvement based on task result.

        Analyzes the result and adjusts internal strategies.

        Args:
            result: Completed task result

        Returns:
            List of improvements applied
        """
        improvements = []

        # Learn from quality scores
        if result.quality_score >= 0.9:
            # High quality - remember this approach
            self.memory.learned_patterns[result.metadata.get("task_type", "")] = {
                "params": result.metadata.get("params", {}),
                "quality": result.quality_score,
            }
            improvements.append(f"Stored high-quality approach (score={result.quality_score:.2f})")

        elif result.quality_score < 0.5 and len(self.memory.task_history) > 3:
            # Low quality - try to learn from better results
            better_results = [
                r for r in self.memory.task_history
                if r.metadata.get("task_type") == result.metadata.get("task_type")
                and r.quality_score > result.quality_score
            ]
            if better_results:
                improvements.append(
                    f"Identified {len(better_results)} better approaches for future use"
                )

        # Track improvement over time
        if len(self.memory.performance_scores) >= 5:
            recent_avg = sum(self.memory.performance_scores[-5:]) / 5
            overall_avg = self.memory.average_quality
            if recent_avg > overall_avg:
                improvements.append(
                    f"Performance trending up (recent={recent_avg:.2f} vs overall={overall_avg:.2f})"
                )

        self._fire_hooks("on_improve", improvements=improvements)

        if improvements:
            self.memory.improvement_log.append({
                "task_id": result.task_id,
                "improvements": improvements,
                "timestamp": time.time(),
            })

        return improvements

    def suggest_next_tasks(self) -> List[Dict[str, Any]]:
        """
        Autonomously suggest what tasks should be done next.

        Based on memory, patterns, and current state.

        Returns:
            List of suggested tasks with rationale
        """
        suggestions = []

        # Check for commonly failed tasks that could be retried
        for task_type, fail_count in self.memory.failure_patterns.items():
            if fail_count >= 2:
                suggestions.append({
                    "task": task_type,
                    "rationale": f"Previously failed {fail_count} times - retry with improvements",
                    "priority": "medium",
                    "auto_generated": True,
                })

        # Suggest related tasks based on recent successes
        if self.memory.task_history:
            last_task = self.memory.task_history[-1]
            if last_task.succeeded:
                related = self._get_related_tasks(last_task.metadata.get("task_type", ""))
                for task in related:
                    suggestions.append({
                        "task": task,
                        "rationale": f"Follow-up after successful '{last_task.metadata.get('task_type')}'",
                        "priority": "low",
                        "auto_generated": True,
                    })

        return suggestions

    def _get_related_tasks(self, task_type: str) -> List[str]:
        """Get tasks related to a given task type."""
        # Override in subclasses for domain-specific relationships
        return []

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "state": self.state.value,
            "capabilities": [c.value for c in self.capabilities],
            "tasks_completed": len([t for t in self.memory.task_history if t.succeeded]),
            "tasks_failed": len([t for t in self.memory.task_history if not t.succeeded]),
            "success_rate": f"{self.memory.success_rate:.1%}",
            "average_quality": f"{self.memory.average_quality:.2f}",
            "improvements_count": len(self.memory.improvement_log),
        }
