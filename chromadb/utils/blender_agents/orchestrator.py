"""
Agent Orchestrator - Multi-agent coordination and pipeline management.

The orchestrator coordinates multiple Blender agents to execute complex
workflows, managing dependencies, parallel execution, and quality gates.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

from chromadb.utils.blender_agents.base_agent import (
    BlenderAgent, AgentCapability, AgentState, TaskResult, TaskStatus,
)
from chromadb.utils.blender_agents.task_planner import (
    TaskPlanner, Task, TaskPriority,
)
from chromadb.utils.blender_agents.improvement_engine import (
    ImprovementEngine, MetricType,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineStage:
    """A stage in a processing pipeline."""
    name: str
    agent_type: str
    task_name: str
    params: Dict[str, Any] = field(default_factory=dict)
    quality_gate: float = 0.0  # Minimum quality to proceed
    result: Optional[TaskResult] = None
    skipped: bool = False


@dataclass
class Pipeline:
    """An ordered sequence of pipeline stages."""
    name: str
    stages: List[PipelineStage] = field(default_factory=list)
    status: str = "pending"  # pending, running, completed, failed
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    def add_stage(
        self,
        name: str,
        agent_type: str,
        task_name: str,
        params: Optional[Dict[str, Any]] = None,
        quality_gate: float = 0.0,
    ) -> PipelineStage:
        stage = PipelineStage(
            name=name,
            agent_type=agent_type,
            task_name=task_name,
            params=params or {},
            quality_gate=quality_gate,
        )
        self.stages.append(stage)
        return stage

    @property
    def progress(self) -> float:
        if not self.stages:
            return 0.0
        completed = sum(1 for s in self.stages if s.result and s.result.succeeded)
        return completed / len(self.stages)


class AgentOrchestrator:
    """
    Multi-agent orchestrator for coordinating complex Blender workflows.

    Features:
    - Agent registry and lifecycle management
    - Pipeline-based workflow execution
    - Goal-driven autonomous planning
    - Quality gates between pipeline stages
    - Continuous improvement across all agents
    - Self-task creation and autonomous operation
    """

    def __init__(
        self,
        bridge: Optional[Any] = None,
        auto_improve: bool = True,
        auto_plan: bool = True,
    ):
        """
        Initialize the orchestrator.

        Args:
            bridge: BlenderBridge for agents to use
            auto_improve: Enable continuous improvement
            auto_plan: Enable autonomous task planning
        """
        self.bridge = bridge
        self.agents: Dict[str, BlenderAgent] = {}
        self.planner = TaskPlanner(enable_auto_planning=auto_plan)
        self.improvement_engine = ImprovementEngine()
        self.pipelines: List[Pipeline] = []
        self.auto_improve = auto_improve
        self._running = False

        # Register default metrics
        self.improvement_engine.register_metric(
            "overall_quality", MetricType.ACCURACY, target=0.8,
        )
        self.improvement_engine.register_metric(
            "execution_speed", MetricType.PERFORMANCE, target=0.7,
        )
        self.improvement_engine.register_metric(
            "success_rate", MetricType.EFFICIENCY, target=0.9,
        )

    def register_agent(self, agent_type: str, agent: BlenderAgent) -> None:
        """Register an agent with the orchestrator."""
        self.agents[agent_type] = agent
        logger.info(f"Registered agent: {agent_type} -> {agent.name}")

    def register_default_agents(self) -> None:
        """Register all default agent types."""
        from chromadb.utils.blender_agents.modeling_agent import ModelingAgent
        from chromadb.utils.blender_agents.material_agent import MaterialAgent
        from chromadb.utils.blender_agents.render_agent import RenderAgent
        from chromadb.utils.blender_agents.animation_agent import AnimationAgent
        from chromadb.utils.blender_agents.scene_agent import SceneAgent

        self.register_agent("modeling", ModelingAgent(bridge=self.bridge))
        self.register_agent("material", MaterialAgent(bridge=self.bridge))
        self.register_agent("render", RenderAgent(bridge=self.bridge))
        self.register_agent("animation", AnimationAgent(bridge=self.bridge))
        self.register_agent("scene", SceneAgent(bridge=self.bridge))

    def get_agent(self, agent_type: str) -> Optional[BlenderAgent]:
        """Get an agent by type."""
        return self.agents.get(agent_type)

    def create_pipeline(self, name: str) -> Pipeline:
        """Create a new empty pipeline."""
        pipeline = Pipeline(name=name)
        self.pipelines.append(pipeline)
        return pipeline

    def execute_pipeline(self, pipeline: Pipeline) -> Dict[str, Any]:
        """
        Execute a pipeline sequentially with quality gates.

        Args:
            pipeline: Pipeline to execute

        Returns:
            Execution summary
        """
        pipeline.status = "running"
        results = []

        logger.info(f"Starting pipeline: {pipeline.name} ({len(pipeline.stages)} stages)")

        for i, stage in enumerate(pipeline.stages):
            agent = self.agents.get(stage.agent_type)
            if not agent:
                logger.error(f"No agent for type: {stage.agent_type}")
                stage.skipped = True
                continue

            logger.info(f"Pipeline stage {i+1}/{len(pipeline.stages)}: {stage.name}")

            # Execute the stage
            result = agent.run_task(stage.task_name, **stage.params)
            stage.result = result
            results.append(result)

            # Record metrics
            if result.succeeded:
                self.improvement_engine.record_metric("overall_quality", result.quality_score)
                speed_score = max(0, 1.0 - (result.execution_time / 30.0))
                self.improvement_engine.record_metric("execution_speed", speed_score)

            # Check quality gate
            if stage.quality_gate > 0 and result.quality_score < stage.quality_gate:
                logger.warning(
                    f"Quality gate failed for {stage.name}: "
                    f"{result.quality_score:.2f} < {stage.quality_gate:.2f}"
                )
                pipeline.status = "failed"

                # Auto-create improvement task
                if self.planner.enable_auto_planning:
                    self.planner.create_task(
                        name=f"improve_{stage.name}",
                        agent_type=stage.agent_type,
                        description=f"Improve {stage.name} quality to meet gate {stage.quality_gate}",
                        priority=TaskPriority.HIGH,
                        params=stage.params,
                        created_by="orchestrator",
                    )
                break

            if not result.succeeded:
                logger.error(f"Stage {stage.name} failed: {result.error}")
                pipeline.status = "failed"
                break

        if pipeline.status != "failed":
            pipeline.status = "completed"
        pipeline.completed_at = time.time()

        # Record success rate
        succeeded = sum(1 for r in results if r.succeeded)
        if results:
            self.improvement_engine.record_metric(
                "success_rate", succeeded / len(results),
            )

        # Generate improvement suggestions
        if self.auto_improve:
            self.improvement_engine.generate_suggestions()

        summary = {
            "pipeline": pipeline.name,
            "status": pipeline.status,
            "stages_total": len(pipeline.stages),
            "stages_completed": sum(1 for s in pipeline.stages if s.result and s.result.succeeded),
            "stages_skipped": sum(1 for s in pipeline.stages if s.skipped),
            "progress": f"{pipeline.progress:.0%}",
            "total_time": sum(r.execution_time for r in results),
            "average_quality": sum(r.quality_score for r in results) / max(len(results), 1),
        }

        logger.info(f"Pipeline '{pipeline.name}' {pipeline.status}: {summary}")
        return summary

    def execute_goal(
        self,
        goal: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a high-level goal by decomposing and running tasks.

        Args:
            goal: Goal name (e.g., "create_3d_scene")
            params: Goal parameters

        Returns:
            Execution summary
        """
        # Plan tasks from goal
        tasks = self.planner.plan_from_goal(goal, params)

        # Create pipeline from tasks
        pipeline = self.create_pipeline(f"goal_{goal}")
        for task in tasks:
            pipeline.add_stage(
                name=task.name,
                agent_type=task.agent_type,
                task_name=task.name,
                params=task.params,
            )

        # Execute pipeline
        return self.execute_pipeline(pipeline)

    def run_autonomous(self, max_iterations: int = 10) -> Dict[str, Any]:
        """
        Run the orchestrator autonomously.

        Continuously plans, executes, evaluates, and improves.

        Args:
            max_iterations: Maximum number of autonomous iterations

        Returns:
            Summary of all autonomous work
        """
        self._running = True
        iteration = 0
        all_results = []

        logger.info(f"Starting autonomous operation (max {max_iterations} iterations)")

        while self._running and iteration < max_iterations:
            iteration += 1
            logger.info(f"\n--- Autonomous iteration {iteration}/{max_iterations} ---")

            # 1. Run task generators for new tasks
            self.planner.run_task_generators()

            # 2. Get agents to suggest tasks
            for agent_type, agent in self.agents.items():
                suggestions = agent.suggest_next_tasks()
                for suggestion in suggestions:
                    self.planner.create_task(
                        name=suggestion["task"],
                        agent_type=agent_type,
                        description=suggestion.get("rationale", ""),
                        priority=TaskPriority[suggestion.get("priority", "medium").upper()],
                        created_by="agent",
                    )

            # 3. Get next tasks to execute
            next_tasks = self.planner.get_next_tasks(count=self.planner.max_concurrent_tasks)

            if not next_tasks:
                logger.info("No more tasks to execute. Stopping autonomous operation.")
                break

            # 4. Execute tasks
            for task in next_tasks:
                agent = self.agents.get(task.agent_type)
                if not agent:
                    self.planner.fail_task(task.task_id, f"No agent for: {task.agent_type}")
                    continue

                task.started_at = time.time()
                result = agent.run_task(task.name, **task.params)
                all_results.append(result)

                if result.succeeded:
                    self.planner.complete_task(task.task_id, result)
                    self.improvement_engine.record_metric(
                        "overall_quality", result.quality_score,
                    )
                else:
                    self.planner.fail_task(task.task_id, result.error or "Unknown error")

            # 5. Auto-optimize
            if self.auto_improve and iteration % 3 == 0:
                optimized = self.improvement_engine.auto_optimize()
                if optimized:
                    logger.info(f"Auto-optimized parameters: {list(optimized.keys())}")

        self._running = False

        return {
            "iterations": iteration,
            "tasks_completed": len([r for r in all_results if r.succeeded]),
            "tasks_failed": len([r for r in all_results if not r.succeeded]),
            "average_quality": (
                sum(r.quality_score for r in all_results) / max(len(all_results), 1)
            ),
            "improvement_report": self.improvement_engine.get_improvement_report(),
            "plan_summary": self.planner.get_plan_summary(),
        }

    def stop(self) -> None:
        """Stop autonomous operation."""
        self._running = False

    def get_status(self) -> Dict[str, Any]:
        """Get full orchestrator status."""
        return {
            "agents": {
                name: agent.get_status()
                for name, agent in self.agents.items()
            },
            "plan": self.planner.get_plan_summary(),
            "improvement": self.improvement_engine.get_improvement_report(),
            "pipelines": [
                {
                    "name": p.name,
                    "status": p.status,
                    "progress": f"{p.progress:.0%}",
                    "stages": len(p.stages),
                }
                for p in self.pipelines
            ],
            "available_goals": self.planner.get_all_goals(),
        }
