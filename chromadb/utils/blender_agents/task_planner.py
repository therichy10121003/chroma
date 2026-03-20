"""
Autonomous Task Planner - Self-task creation and scheduling.

Provides the brain of the autonomous agent system, capable of:
- Breaking complex goals into sub-tasks
- Dependency management between tasks
- Priority-based scheduling
- Dynamic task creation based on results
- Goal-driven planning
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Set

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    BACKGROUND = 4


@dataclass
class TaskDependency:
    """Dependency between tasks."""
    task_id: str
    dependency_type: str = "requires"  # requires, optional, conflicts
    satisfied: bool = False


@dataclass
class Task:
    """A planned task with metadata and dependencies."""
    task_id: str
    name: str
    description: str
    agent_type: str  # Which agent type should handle this
    priority: TaskPriority = TaskPriority.MEDIUM
    params: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[TaskDependency] = field(default_factory=list)
    created_by: str = "planner"  # "planner", "user", "agent", "improvement_engine"
    status: str = "pending"
    result: Optional[Any] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    subtasks: List["Task"] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        """Check if all dependencies are satisfied."""
        return all(dep.satisfied for dep in self.dependencies)

    @property
    def duration(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None


class TaskPlanner:
    """
    Autonomous task planner with self-task creation.

    Manages the full lifecycle of tasks:
    - Goal decomposition into actionable tasks
    - Dependency graph management
    - Priority-based scheduling
    - Dynamic task generation based on results
    - Continuous re-planning
    """

    def __init__(
        self,
        enable_auto_planning: bool = True,
        max_concurrent_tasks: int = 3,
    ):
        """
        Initialize the task planner.

        Args:
            enable_auto_planning: Allow autonomous task creation
            max_concurrent_tasks: Maximum parallel tasks
        """
        self.tasks: Dict[str, Task] = {}
        self.completed_tasks: List[Task] = []
        self.enable_auto_planning = enable_auto_planning
        self.max_concurrent_tasks = max_concurrent_tasks
        self._goal_templates: Dict[str, List[Dict[str, Any]]] = {}
        self._task_generators: List[Callable] = []

        # Register built-in goal templates
        self._register_default_templates()

    def _register_default_templates(self) -> None:
        """Register default goal decomposition templates."""
        self._goal_templates["create_3d_scene"] = [
            {"name": "setup_scene", "agent_type": "scene", "priority": TaskPriority.HIGH,
             "description": "Initialize scene with world settings"},
            {"name": "create_geometry", "agent_type": "modeling", "priority": TaskPriority.HIGH,
             "description": "Create 3D meshes and geometry"},
            {"name": "apply_materials", "agent_type": "material", "priority": TaskPriority.MEDIUM,
             "description": "Create and apply materials to objects",
             "depends_on": ["create_geometry"]},
            {"name": "setup_lighting", "agent_type": "scene", "priority": TaskPriority.MEDIUM,
             "description": "Set up scene lighting",
             "depends_on": ["setup_scene"]},
            {"name": "setup_camera", "agent_type": "scene", "priority": TaskPriority.MEDIUM,
             "description": "Position and configure camera",
             "depends_on": ["create_geometry"]},
            {"name": "render_scene", "agent_type": "render", "priority": TaskPriority.LOW,
             "description": "Render the final scene",
             "depends_on": ["apply_materials", "setup_lighting", "setup_camera"]},
        ]

        self._goal_templates["create_animation"] = [
            {"name": "setup_scene", "agent_type": "scene", "priority": TaskPriority.HIGH,
             "description": "Initialize scene"},
            {"name": "create_objects", "agent_type": "modeling", "priority": TaskPriority.HIGH,
             "description": "Create objects to animate"},
            {"name": "apply_materials", "agent_type": "material", "priority": TaskPriority.MEDIUM,
             "description": "Apply materials",
             "depends_on": ["create_objects"]},
            {"name": "create_animation", "agent_type": "animation", "priority": TaskPriority.HIGH,
             "description": "Create keyframe animations",
             "depends_on": ["create_objects"]},
            {"name": "setup_lighting", "agent_type": "scene", "priority": TaskPriority.MEDIUM,
             "description": "Set up lighting"},
            {"name": "render_animation", "agent_type": "render", "priority": TaskPriority.LOW,
             "description": "Render animation frames",
             "depends_on": ["create_animation", "apply_materials", "setup_lighting"]},
        ]

        self._goal_templates["product_visualization"] = [
            {"name": "import_or_create_model", "agent_type": "modeling",
             "priority": TaskPriority.CRITICAL,
             "description": "Import or create product model"},
            {"name": "create_pbr_materials", "agent_type": "material",
             "priority": TaskPriority.HIGH,
             "description": "Create photorealistic PBR materials",
             "depends_on": ["import_or_create_model"]},
            {"name": "studio_lighting", "agent_type": "scene",
             "priority": TaskPriority.HIGH,
             "description": "Set up studio lighting for product shots"},
            {"name": "camera_angles", "agent_type": "scene",
             "priority": TaskPriority.MEDIUM,
             "description": "Configure multiple camera angles",
             "depends_on": ["import_or_create_model"]},
            {"name": "render_turntable", "agent_type": "render",
             "priority": TaskPriority.MEDIUM,
             "description": "Render turntable animation",
             "depends_on": ["create_pbr_materials", "studio_lighting", "camera_angles"]},
            {"name": "render_hero_shots", "agent_type": "render",
             "priority": TaskPriority.LOW,
             "description": "Render hero marketing shots",
             "depends_on": ["create_pbr_materials", "studio_lighting", "camera_angles"]},
        ]

    def create_task(
        self,
        name: str,
        agent_type: str,
        description: str = "",
        priority: TaskPriority = TaskPriority.MEDIUM,
        params: Optional[Dict[str, Any]] = None,
        depends_on: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        created_by: str = "user",
    ) -> Task:
        """
        Create a new task.

        Args:
            name: Task name
            agent_type: Type of agent to handle this task
            description: Task description
            priority: Priority level
            params: Task parameters
            depends_on: List of task IDs this depends on
            tags: Optional tags for categorization
            created_by: Who created the task

        Returns:
            Created Task
        """
        task_id = f"task-{str(uuid.uuid4())[:8]}"

        dependencies = []
        if depends_on:
            for dep_id in depends_on:
                dep = TaskDependency(
                    task_id=dep_id,
                    satisfied=dep_id in [t.task_id for t in self.completed_tasks],
                )
                dependencies.append(dep)

        task = Task(
            task_id=task_id,
            name=name,
            agent_type=agent_type,
            description=description,
            priority=priority,
            params=params or {},
            dependencies=dependencies,
            created_by=created_by,
            tags=tags or [],
        )

        self.tasks[task_id] = task
        logger.info(f"Task created: {task.name} ({task_id}) by {created_by}")

        return task

    def plan_from_goal(
        self,
        goal: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Task]:
        """
        Decompose a high-level goal into concrete tasks.

        Args:
            goal: High-level goal (e.g., "create_3d_scene")
            params: Parameters to pass to all generated tasks

        Returns:
            List of generated tasks
        """
        if goal not in self._goal_templates:
            logger.warning(f"No template for goal: {goal}")
            # Auto-create a single task
            task = self.create_task(
                name=goal,
                agent_type="modeling",
                description=f"Execute goal: {goal}",
                params=params or {},
                created_by="planner",
            )
            return [task]

        template = self._goal_templates[goal]
        created_tasks = []
        task_name_to_id: Dict[str, str] = {}

        for step in template:
            # Resolve dependencies
            depends_on = []
            for dep_name in step.get("depends_on", []):
                if dep_name in task_name_to_id:
                    depends_on.append(task_name_to_id[dep_name])

            task = self.create_task(
                name=step["name"],
                agent_type=step["agent_type"],
                description=step["description"],
                priority=step["priority"],
                params=params or {},
                depends_on=depends_on if depends_on else None,
                created_by="planner",
                tags=[goal],
            )

            task_name_to_id[step["name"]] = task.task_id
            created_tasks.append(task)

        logger.info(f"Goal '{goal}' decomposed into {len(created_tasks)} tasks")
        return created_tasks

    def get_next_tasks(self, count: int = 1) -> List[Task]:
        """
        Get the next tasks to execute based on priority and dependencies.

        Args:
            count: Maximum number of tasks to return

        Returns:
            List of ready tasks sorted by priority
        """
        ready_tasks = []

        for task in self.tasks.values():
            if task.status == "pending" and task.is_ready:
                ready_tasks.append(task)

        # Sort by priority (lower enum value = higher priority)
        ready_tasks.sort(key=lambda t: t.priority.value)

        return ready_tasks[:count]

    def complete_task(self, task_id: str, result: Any = None) -> None:
        """
        Mark a task as completed and update dependencies.

        Args:
            task_id: ID of completed task
            result: Task result
        """
        if task_id not in self.tasks:
            logger.warning(f"Task {task_id} not found")
            return

        task = self.tasks[task_id]
        task.status = "completed"
        task.result = result
        task.completed_at = time.time()

        # Update dependencies in other tasks
        for other_task in self.tasks.values():
            for dep in other_task.dependencies:
                if dep.task_id == task_id:
                    dep.satisfied = True

        # Move to completed list
        self.completed_tasks.append(task)
        del self.tasks[task_id]

        logger.info(f"Task completed: {task.name} ({task_id})")

        # Auto-generate follow-up tasks
        if self.enable_auto_planning:
            self._auto_generate_followups(task)

    def fail_task(self, task_id: str, error: str = "") -> None:
        """Mark a task as failed."""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.status = "failed"
            task.result = {"error": error}
            task.completed_at = time.time()

            self.completed_tasks.append(task)
            del self.tasks[task_id]

            logger.warning(f"Task failed: {task.name} ({task_id}): {error}")

            # Auto-create retry task
            if self.enable_auto_planning:
                self.create_task(
                    name=f"retry_{task.name}",
                    agent_type=task.agent_type,
                    description=f"Retry of failed task: {task.name}. Error: {error}",
                    priority=TaskPriority.HIGH,
                    params=task.params,
                    created_by="planner",
                    tags=task.tags + ["retry"],
                )

    def _auto_generate_followups(self, completed_task: Task) -> None:
        """Automatically generate follow-up tasks based on completed task."""
        followups = self._get_followup_tasks(completed_task)
        for followup in followups:
            self.create_task(
                created_by="planner",
                **followup,
            )

    def _get_followup_tasks(self, task: Task) -> List[Dict[str, Any]]:
        """Determine follow-up tasks based on completed task."""
        followups = []

        # Quality improvement suggestions
        if task.result and hasattr(task.result, "quality_score"):
            if task.result.quality_score < 0.7:
                followups.append({
                    "name": f"improve_{task.name}",
                    "agent_type": task.agent_type,
                    "description": f"Improve quality of {task.name} "
                                   f"(current score: {task.result.quality_score:.2f})",
                    "priority": TaskPriority.MEDIUM,
                    "params": task.params,
                    "tags": ["improvement", "auto_generated"],
                })

        return followups

    def register_goal_template(
        self,
        goal_name: str,
        steps: List[Dict[str, Any]],
    ) -> None:
        """
        Register a custom goal template.

        Args:
            goal_name: Name of the goal
            steps: List of step definitions
        """
        self._goal_templates[goal_name] = steps
        logger.info(f"Registered goal template: {goal_name} ({len(steps)} steps)")

    def register_task_generator(self, generator: Callable) -> None:
        """
        Register a function that generates tasks.

        The generator is called periodically to create new tasks autonomously.

        Args:
            generator: Callable that returns List[Dict] of task definitions
        """
        self._task_generators.append(generator)

    def run_task_generators(self) -> List[Task]:
        """Run all registered task generators and create tasks."""
        generated = []
        for generator in self._task_generators:
            try:
                task_defs = generator(
                    completed=self.completed_tasks,
                    pending=list(self.tasks.values()),
                )
                for task_def in task_defs:
                    task = self.create_task(created_by="generator", **task_def)
                    generated.append(task)
            except Exception as e:
                logger.error(f"Task generator error: {e}")

        return generated

    def get_plan_summary(self) -> Dict[str, Any]:
        """Get summary of the current plan."""
        pending = [t for t in self.tasks.values() if t.status == "pending"]
        return {
            "pending_tasks": len(pending),
            "completed_tasks": len(self.completed_tasks),
            "ready_tasks": len([t for t in pending if t.is_ready]),
            "blocked_tasks": len([t for t in pending if not t.is_ready]),
            "auto_generated": len([t for t in pending if t.created_by in ("planner", "generator")]),
            "tasks_by_priority": {
                p.name: len([t for t in pending if t.priority == p])
                for p in TaskPriority
            },
            "tasks_by_agent": {},
            "available_goals": list(self._goal_templates.keys()),
        }

    def get_all_goals(self) -> List[str]:
        """Get all available goal templates."""
        return list(self._goal_templates.keys())
