#!/usr/bin/env python3
"""
Blender Automation Agents - Working Examples
=============================================

Demonstrates the full autonomous agent system for Blender automation,
including self-task creation, continuous improvement, and multi-agent
coordination.

Requirements:
    - Blender 4.0+ installed (apt install blender)
    - Python 3.9+

Usage:
    python3 examples/blender_agents_example.py
"""

import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chromadb.utils.blender_agents import (
    BlenderBridge,
    ModelingAgent,
    MaterialAgent,
    RenderAgent,
    AnimationAgent,
    SceneAgent,
    AgentOrchestrator,
    TaskPlanner,
    ImprovementEngine,
)
from chromadb.utils.blender_agents.improvement_engine import MetricType
from chromadb.utils.blender_agents.task_planner import TaskPriority


def example_1_bridge_basics():
    """Example 1: Direct Blender Bridge operations."""
    print("=" * 70)
    print("Example 1: Blender Bridge - Direct Operations")
    print("=" * 70)

    bridge = BlenderBridge()
    print(f"Blender: {bridge.blender_version}")
    print(f"Output: {bridge.output_dir}")

    # Create a mesh
    result = bridge.create_mesh(mesh_type="monkey", name="Suzanne")
    print(f"\nCreate mesh: success={result.success}")
    print(f"  Data: {result.data}")

    # Get scene info
    info = bridge.get_scene_info()
    print(f"\nScene info: {info.data.get('object_count', 0)} objects")

    print("\n[OK] Bridge works!\n")


def example_2_modeling_agent():
    """Example 2: Modeling agent with self-improvement."""
    print("=" * 70)
    print("Example 2: Modeling Agent - Autonomous 3D Creation")
    print("=" * 70)

    bridge = BlenderBridge()
    agent = ModelingAgent(bridge=bridge, auto_improve=True)

    print(f"\nAgent: {agent.name} ({agent.agent_id})")
    print(f"Capabilities: {[c.value for c in agent.capabilities]}")
    print(f"Available tasks: {[t['name'] for t in agent.get_available_tasks()]}")

    # Create primitives
    print("\n--- Creating geometry ---")
    r1 = agent.run_task("create_primitive", mesh_type="cube", name="MyCube")
    print(f"Cube: success={r1.succeeded}, quality={r1.quality_score:.2f}")

    r2 = agent.run_task("create_primitive", mesh_type="sphere",
                        location=(3, 0, 0), name="MySphere")
    print(f"Sphere: success={r2.succeeded}, quality={r2.quality_score:.2f}")

    # Procedural generation
    print("\n--- Procedural spiral ---")
    r3 = agent.run_task("create_procedural", pattern="spiral", count=20, spacing=3.0)
    print(f"Spiral: success={r3.succeeded}, quality={r3.quality_score:.2f}")

    # Check agent status
    status = agent.get_status()
    print(f"\nAgent status:")
    print(f"  Tasks completed: {status['tasks_completed']}")
    print(f"  Success rate: {status['success_rate']}")
    print(f"  Average quality: {status['average_quality']}")
    print(f"  Improvements: {status['improvements_count']}")

    # Autonomous suggestions
    suggestions = agent.suggest_next_tasks()
    print(f"\nAuto-suggested tasks: {len(suggestions)}")
    for s in suggestions:
        print(f"  - {s['task']}: {s['rationale']}")

    print("\n[OK] Modeling agent works!\n")


def example_3_material_agent():
    """Example 3: Material agent with presets."""
    print("=" * 70)
    print("Example 3: Material Agent - Shaders & Presets")
    print("=" * 70)

    bridge = BlenderBridge()
    agent = MaterialAgent(bridge=bridge)

    # Create object first, then apply materials
    bridge.create_mesh(mesh_type="monkey", name="MaterialTest")

    print("\n--- Applying material presets ---")
    r1 = agent.run_task("apply_preset", object_name="MaterialTest", preset="gold")
    print(f"Gold preset: success={r1.succeeded}, quality={r1.quality_score:.2f}")

    # Create emissive material
    bridge.create_mesh(mesh_type="sphere", name="GlowSphere", location=(3, 0, 0))
    r2 = agent.run_task("create_emission",
                        object_name="GlowSphere",
                        color=(0, 0.5, 1, 1), strength=10.0)
    print(f"Emission: success={r2.succeeded}, quality={r2.quality_score:.2f}")

    # Create gradient
    bridge.create_mesh(mesh_type="cube", name="GradientCube", location=(-3, 0, 0))
    r3 = agent.run_task("create_gradient",
                        object_name="GradientCube",
                        color1=(1, 0, 0, 1), color2=(0, 0, 1, 1))
    print(f"Gradient: success={r3.succeeded}, quality={r3.quality_score:.2f}")

    print(f"\nAgent quality: {agent.get_status()['average_quality']}")
    print("\n[OK] Material agent works!\n")


def example_4_scene_and_render():
    """Example 4: Scene setup and rendering."""
    print("=" * 70)
    print("Example 4: Scene + Render Agents - Full Pipeline")
    print("=" * 70)

    bridge = BlenderBridge()
    scene_agent = SceneAgent(bridge=bridge)
    render_agent = RenderAgent(bridge=bridge)

    # Setup scene
    print("\n--- Scene setup ---")
    r1 = scene_agent.run_task("clear_scene", keep_camera=True)
    print(f"Clear scene: {r1.succeeded}")

    r2 = scene_agent.run_task("lighting_preset", preset="studio_3point")
    print(f"Studio lighting: {r2.succeeded}")

    r3 = scene_agent.run_task("setup_camera",
                              location=(5, -5, 4), rotation=(1.1, 0, 0.8))
    print(f"Camera: {r3.succeeded}")

    r4 = scene_agent.run_task("setup_world", background_color=(0.02, 0.02, 0.04))
    print(f"World: {r4.succeeded}")

    # Create something to render
    bridge.create_mesh(mesh_type="monkey", name="Suzanne")
    bridge.apply_material("Suzanne", "Gold", (1.0, 0.766, 0.336, 1.0), 1.0, 0.3)

    # Render
    print("\n--- Rendering ---")
    output = os.path.join(bridge.output_dir, "test_render")
    r5 = render_agent.run_task("render_preset", preset="preview", output_path=output)
    print(f"Render: success={r5.succeeded}, quality={r5.quality_score:.2f}")
    print(f"  Time: {r5.execution_time:.1f}s")

    print("\n[OK] Scene and render works!\n")


def example_5_task_planner():
    """Example 5: Autonomous task planning and self-task creation."""
    print("=" * 70)
    print("Example 5: Task Planner - Autonomous Self-Task Creation")
    print("=" * 70)

    planner = TaskPlanner(enable_auto_planning=True)

    print(f"\nAvailable goals: {planner.get_all_goals()}")

    # Decompose a goal into tasks
    print("\n--- Planning 'create_3d_scene' ---")
    tasks = planner.plan_from_goal("create_3d_scene")
    for task in tasks:
        deps = [d.task_id for d in task.dependencies]
        print(f"  [{task.priority.name}] {task.name} "
              f"(agent: {task.agent_type}, deps: {len(deps)})")

    # Get next executable tasks
    ready = planner.get_next_tasks(count=5)
    print(f"\nReady to execute: {[t.name for t in ready]}")

    # Simulate completing tasks
    for task in ready[:2]:
        planner.complete_task(task.task_id)
        print(f"  Completed: {task.name}")

    # Check for auto-generated follow-ups
    summary = planner.get_plan_summary()
    print(f"\nPlan summary:")
    print(f"  Pending: {summary['pending_tasks']}")
    print(f"  Completed: {summary['completed_tasks']}")
    print(f"  Ready: {summary['ready_tasks']}")
    print(f"  Auto-generated: {summary['auto_generated']}")

    # Plan product visualization
    print("\n--- Planning 'product_visualization' ---")
    pv_tasks = planner.plan_from_goal("product_visualization")
    for task in pv_tasks:
        print(f"  [{task.priority.name}] {task.name}: {task.description}")

    print("\n[OK] Task planner works!\n")


def example_6_improvement_engine():
    """Example 6: Continuous improvement and feedback loops."""
    print("=" * 70)
    print("Example 6: Improvement Engine - Continuous Learning")
    print("=" * 70)

    engine = ImprovementEngine()

    # Register metrics
    engine.register_metric("render_quality", MetricType.RENDER_QUALITY, target=0.85)
    engine.register_metric("modeling_quality", MetricType.GEOMETRY_QUALITY, target=0.8)
    engine.register_metric("speed", MetricType.PERFORMANCE, target=0.7)

    # Simulate quality history
    import random
    random.seed(42)
    for i in range(10):
        engine.record_metric("render_quality", 0.5 + i * 0.04 + random.random() * 0.1)
        engine.record_metric("modeling_quality", 0.6 + i * 0.03 + random.random() * 0.08)
        engine.record_metric("speed", 0.7 + random.random() * 0.2)

    # Create feedback loop
    loop = engine.create_feedback_loop("render_settings")
    loop.record({"samples": 64, "resolution": 720}, 0.6)
    loop.record({"samples": 128, "resolution": 1080}, 0.75)
    loop.record({"samples": 256, "resolution": 1080}, 0.9)
    loop.record({"samples": 512, "resolution": 2160}, 0.85)

    print(f"\nOptimal render params: {loop.suggest_params()}")
    print(f"Correlations: {loop.correlations}")

    # Generate suggestions
    suggestions = engine.generate_suggestions()
    print(f"\nImprovement suggestions ({len(suggestions)}):")
    for s in suggestions[:5]:
        print(f"  [{s.priority}] {s.description}")
        print(f"    Rationale: {s.rationale}")

    # Create experiment
    engine.create_experiment(
        name="optimal_samples",
        parameter="samples",
        values=[32, 64, 128, 256, 512],
        metric_to_track="render_quality",
    )
    engine.record_experiment_result("optimal_samples", 128, 0.78)
    engine.record_experiment_result("optimal_samples", 256, 0.91)

    # Report
    report = engine.get_improvement_report()
    print(f"\nImprovement report:")
    print(f"  Aggregate quality: {report['aggregate_quality']}")
    for name, m in report["metrics"].items():
        print(f"  {name}: current={m['current']}, trend={m['trend']}, target_met={m['meets_target']}")

    print("\n[OK] Improvement engine works!\n")


def example_7_orchestrator():
    """Example 7: Full multi-agent orchestration."""
    print("=" * 70)
    print("Example 7: Agent Orchestrator - Full Autonomous Pipeline")
    print("=" * 70)

    bridge = BlenderBridge()
    orchestrator = AgentOrchestrator(bridge=bridge, auto_improve=True)
    orchestrator.register_default_agents()

    print(f"\nRegistered agents: {list(orchestrator.agents.keys())}")
    print(f"Available goals: {orchestrator.planner.get_all_goals()}")

    # Create and execute a custom pipeline
    print("\n--- Custom Pipeline ---")
    pipeline = orchestrator.create_pipeline("demo_scene")
    pipeline.add_stage("clear", "scene", "clear_scene", {"keep_camera": True})
    pipeline.add_stage("lighting", "scene", "lighting_preset", {"preset": "sunset"})
    pipeline.add_stage("camera", "scene", "setup_camera",
                       {"location": (6, -5, 4), "focal_length": 35})
    pipeline.add_stage("model", "modeling", "create_primitive",
                       {"mesh_type": "monkey", "name": "Hero"})
    pipeline.add_stage("material", "material", "apply_preset",
                       {"object_name": "Hero", "preset": "gold"})
    pipeline.add_stage("render", "render", "render_preset",
                       {"preset": "preview"})

    summary = orchestrator.execute_pipeline(pipeline)
    print(f"\nPipeline result:")
    print(f"  Status: {summary['status']}")
    print(f"  Stages: {summary['stages_completed']}/{summary['stages_total']}")
    print(f"  Progress: {summary['progress']}")
    print(f"  Avg quality: {summary['average_quality']:.2f}")
    print(f"  Total time: {summary['total_time']:.1f}s")

    # Full orchestrator status
    status = orchestrator.get_status()
    print(f"\nOrchestrator status:")
    for agent_name, agent_status in status["agents"].items():
        print(f"  {agent_name}: completed={agent_status['tasks_completed']}, "
              f"quality={agent_status['average_quality']}")

    print(f"\nImprovement report:")
    report = status["improvement"]
    print(f"  Aggregate quality: {report['aggregate_quality']}")
    print(f"  Active suggestions: {report['active_suggestions']}")

    print("\n[OK] Orchestrator works!\n")


def example_8_autonomous_run():
    """Example 8: Fully autonomous operation."""
    print("=" * 70)
    print("Example 8: Fully Autonomous Operation")
    print("=" * 70)

    bridge = BlenderBridge()
    orchestrator = AgentOrchestrator(bridge=bridge, auto_improve=True, auto_plan=True)
    orchestrator.register_default_agents()

    # Seed initial tasks
    orchestrator.planner.create_task(
        name="create_primitive", agent_type="modeling",
        params={"mesh_type": "cube", "name": "AutoCube"},
        priority=TaskPriority.HIGH,
    )
    orchestrator.planner.create_task(
        name="lighting_preset", agent_type="scene",
        params={"preset": "studio_3point"},
        priority=TaskPriority.MEDIUM,
    )

    # Run autonomously (limited iterations for demo)
    print("\n--- Starting autonomous operation ---")
    results = orchestrator.run_autonomous(max_iterations=3)

    print(f"\nAutonomous results:")
    print(f"  Iterations: {results['iterations']}")
    print(f"  Tasks completed: {results['tasks_completed']}")
    print(f"  Tasks failed: {results['tasks_failed']}")
    print(f"  Average quality: {results['average_quality']:.2f}")

    print("\n[OK] Autonomous operation works!\n")


def main():
    print("\n" + "*" * 70)
    print("  CHROMA BLENDER AUTOMATION AGENTS")
    print("  Autonomous 3D Creation with Self-Improvement")
    print("*" * 70 + "\n")

    examples = [
        ("Bridge Basics", example_1_bridge_basics),
        ("Modeling Agent", example_2_modeling_agent),
        ("Material Agent", example_3_material_agent),
        ("Scene & Render", example_4_scene_and_render),
        ("Task Planner", example_5_task_planner),
        ("Improvement Engine", example_6_improvement_engine),
        ("Orchestrator", example_7_orchestrator),
        ("Autonomous Run", example_8_autonomous_run),
    ]

    passed = 0
    failed = 0

    for name, fn in examples:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"\n[FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed out of {len(examples)}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
