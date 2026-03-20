"""
Blender Automation Agents for Chroma.

A comprehensive agent framework for automating Blender 3D tasks with
autonomous task planning, continuous improvement, and multi-agent coordination.

Core Components:
    - BlenderAgent: Base agent class with self-improvement capabilities
    - TaskPlanner: Autonomous task creation and scheduling
    - ImprovementEngine: Continuous learning and quality feedback loops
    - AgentOrchestrator: Multi-agent coordination and pipeline management

Concrete Agents:
    - ModelingAgent: 3D mesh creation, modification, sculpting
    - MaterialAgent: Shader and material creation, PBR workflows
    - RenderAgent: Scene rendering, camera setup, lighting
    - AnimationAgent: Keyframe animation, motion paths
    - SceneAgent: Scene composition, layout, environment
    - CompositorAgent: Post-processing, node-based compositing
"""

from chromadb.utils.blender_agents.base_agent import (
    BlenderAgent,
    AgentCapability,
    AgentState,
    TaskResult,
    TaskStatus,
)
from chromadb.utils.blender_agents.task_planner import (
    TaskPlanner,
    Task,
    TaskPriority,
    TaskDependency,
)
from chromadb.utils.blender_agents.improvement_engine import (
    ImprovementEngine,
    QualityMetric,
    FeedbackLoop,
    ImprovementSuggestion,
)
from chromadb.utils.blender_agents.orchestrator import (
    AgentOrchestrator,
    Pipeline,
    PipelineStage,
)
from chromadb.utils.blender_agents.modeling_agent import ModelingAgent
from chromadb.utils.blender_agents.material_agent import MaterialAgent
from chromadb.utils.blender_agents.render_agent import RenderAgent
from chromadb.utils.blender_agents.animation_agent import AnimationAgent
from chromadb.utils.blender_agents.scene_agent import SceneAgent
from chromadb.utils.blender_agents.blender_bridge import BlenderBridge

__all__ = [
    # Core framework
    "BlenderAgent",
    "AgentCapability",
    "AgentState",
    "TaskResult",
    "TaskStatus",
    # Task planning
    "TaskPlanner",
    "Task",
    "TaskPriority",
    "TaskDependency",
    # Continuous improvement
    "ImprovementEngine",
    "QualityMetric",
    "FeedbackLoop",
    "ImprovementSuggestion",
    # Orchestration
    "AgentOrchestrator",
    "Pipeline",
    "PipelineStage",
    # Concrete agents
    "ModelingAgent",
    "MaterialAgent",
    "RenderAgent",
    "AnimationAgent",
    "SceneAgent",
    # Bridge
    "BlenderBridge",
]
