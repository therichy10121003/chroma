"""
Render Agent - Autonomous scene rendering with quality optimization.

Handles render setup, execution, and quality assessment with
continuous improvement of render settings.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from chromadb.utils.blender_agents.base_agent import (
    BlenderAgent, AgentCapability, TaskResult, TaskStatus,
)

logger = logging.getLogger(__name__)


class RenderAgent(BlenderAgent):
    """
    Agent specialized in rendering operations.

    Capabilities:
    - Full scene rendering (still images and animations)
    - Render engine configuration (Cycles, EEVEE, Workbench)
    - Quality vs. speed optimization
    - Multi-pass rendering
    - Output format management
    """

    RENDER_PRESETS = {
        "preview": {"samples": 32, "resolution": (960, 540), "engine": "BLENDER_EEVEE"},
        "draft": {"samples": 64, "resolution": (1280, 720), "engine": "CYCLES"},
        "standard": {"samples": 128, "resolution": (1920, 1080), "engine": "CYCLES"},
        "high_quality": {"samples": 256, "resolution": (1920, 1080), "engine": "CYCLES"},
        "production": {"samples": 512, "resolution": (3840, 2160), "engine": "CYCLES"},
    }

    TASK_REGISTRY = {
        "render_still": "Render a still image",
        "render_animation": "Render an animation sequence",
        "render_preset": "Render using a quality preset",
        "setup_render_settings": "Configure render settings",
        "render_turntable": "Render a 360-degree turntable",
    }

    def __init__(self, bridge: Optional[Any] = None, **kwargs: Any):
        super().__init__(
            name="RenderAgent",
            capabilities=[AgentCapability.RENDERING],
            bridge=bridge,
            **kwargs,
        )

    def get_available_tasks(self) -> List[Dict[str, Any]]:
        tasks = [
            {"name": name, "description": desc}
            for name, desc in self.TASK_REGISTRY.items()
        ]
        tasks.append({
            "name": "presets",
            "description": f"Available presets: {list(self.RENDER_PRESETS.keys())}",
        })
        return tasks

    def execute_task(self, task_name: str, **params: Any) -> TaskResult:
        if not self.bridge:
            return TaskResult(
                task_id="", status=TaskStatus.FAILED,
                error="No BlenderBridge configured",
            )

        handlers = {
            "render_still": self._render_still,
            "render_animation": self._render_animation,
            "render_preset": self._render_preset,
            "setup_render_settings": self._setup_render_settings,
            "render_turntable": self._render_turntable,
        }

        handler = handlers.get(task_name)
        if not handler:
            return TaskResult(
                task_id="", status=TaskStatus.FAILED,
                error=f"Unknown task: {task_name}",
            )
        return handler(**params)

    def assess_quality(self, result: TaskResult) -> float:
        if not result.succeeded:
            return 0.0

        score = 0.5
        data = result.output if isinstance(result.output, dict) else {}

        # Higher samples = better quality
        samples = data.get("samples", 0)
        if samples >= 256:
            score += 0.3
        elif samples >= 128:
            score += 0.2
        elif samples >= 64:
            score += 0.1

        # Higher resolution = better
        res = data.get("resolution", [0, 0])
        if isinstance(res, list) and len(res) >= 2:
            pixels = res[0] * res[1]
            if pixels >= 3840 * 2160:
                score += 0.15
            elif pixels >= 1920 * 1080:
                score += 0.1

        # Reasonable render time
        if result.execution_time < 60:
            score += 0.05

        return min(score, 1.0)

    def _render_still(self, **params: Any) -> TaskResult:
        output_path = params.get("output_path")
        resolution = params.get("resolution", (1920, 1080))
        samples = params.get("samples", 128)
        engine = params.get("engine", "CYCLES")
        file_format = params.get("file_format", "PNG")

        result = self.bridge.render_scene(
            output_path=output_path,
            resolution=resolution,
            samples=samples,
            engine=engine,
            file_format=file_format,
        )

        return TaskResult(
            task_id="", status=TaskStatus.COMPLETED if result.success else TaskStatus.FAILED,
            output=result.data, error=result.error,
            metadata={"task_type": "render_still", "params": params},
        )

    def _render_preset(self, **params: Any) -> TaskResult:
        preset_name = params.get("preset", "standard")
        output_path = params.get("output_path")

        if preset_name not in self.RENDER_PRESETS:
            return TaskResult(
                task_id="", status=TaskStatus.FAILED,
                error=f"Unknown preset: {preset_name}",
            )

        preset = self.RENDER_PRESETS[preset_name]
        return self._render_still(
            output_path=output_path,
            resolution=preset["resolution"],
            samples=preset["samples"],
            engine=preset["engine"],
        )

    def _render_animation(self, **params: Any) -> TaskResult:
        output_dir = params.get("output_dir", self.bridge.output_dir)
        frame_start = params.get("frame_start", 1)
        frame_end = params.get("frame_end", 250)
        engine = params.get("engine", "BLENDER_EEVEE")
        samples = params.get("samples", 64)

        script = f"""
scene = bpy.context.scene
scene.render.engine = "{engine}"
scene.frame_start = {frame_start}
scene.frame_end = {frame_end}
scene.render.filepath = "{output_dir}/frame_"
scene.render.image_settings.file_format = "PNG"

if "{engine}" == "CYCLES":
    scene.cycles.samples = {samples}

bpy.ops.render.render(animation=True)

_result["data"]["output_dir"] = "{output_dir}"
_result["data"]["frames"] = {frame_end - frame_start + 1}
_result["data"]["engine"] = "{engine}"
"""
        result = self.bridge.execute_script(script)
        return TaskResult(
            task_id="", status=TaskStatus.COMPLETED if result.success else TaskStatus.FAILED,
            output=result.data, error=result.error,
            metadata={"task_type": "render_animation"},
        )

    def _setup_render_settings(self, **params: Any) -> TaskResult:
        engine = params.get("engine", "CYCLES")
        samples = params.get("samples", 128)
        resolution = params.get("resolution", (1920, 1080))

        script = f"""
scene = bpy.context.scene
scene.render.engine = "{engine}"
scene.render.resolution_x = {resolution[0]}
scene.render.resolution_y = {resolution[1]}

if "{engine}" == "CYCLES":
    scene.cycles.samples = {samples}
    scene.cycles.use_denoising = True

_result["data"]["engine"] = "{engine}"
_result["data"]["samples"] = {samples}
_result["data"]["resolution"] = [{resolution[0]}, {resolution[1]}]
"""
        result = self.bridge.execute_script(script)
        return TaskResult(
            task_id="", status=TaskStatus.COMPLETED if result.success else TaskStatus.FAILED,
            output=result.data, error=result.error,
            metadata={"task_type": "setup_render_settings"},
        )

    def _render_turntable(self, **params: Any) -> TaskResult:
        frames = params.get("frames", 36)
        output_dir = params.get("output_dir", self.bridge.output_dir)
        target = params.get("target", "Cube")
        radius = params.get("radius", 5.0)

        script = f"""
import math

scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = {frames}

# Create empty to orbit around
target = bpy.data.objects.get("{target}")
if target is None:
    _result["success"] = False
    _result["error"] = "Target object not found"
else:
    cam = bpy.data.objects.get("Camera")
    if cam is None:
        bpy.ops.object.camera_add()
        cam = bpy.context.active_object
    scene.camera = cam

    # Animate camera rotation
    for frame in range(1, {frames} + 1):
        scene.frame_set(frame)
        angle = (2 * math.pi * frame) / {frames}
        cam.location.x = target.location.x + {radius} * math.cos(angle)
        cam.location.y = target.location.y + {radius} * math.sin(angle)
        cam.location.z = target.location.z + 3
        cam.keyframe_insert(data_path="location")

        # Point at target
        direction = target.location - cam.location
        cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
        cam.keyframe_insert(data_path="rotation_euler")

    scene.render.filepath = "{output_dir}/turntable_"
    scene.render.image_settings.file_format = "PNG"
    scene.render.engine = "BLENDER_EEVEE"

    bpy.ops.render.render(animation=True)

    _result["data"]["frames"] = {frames}
    _result["data"]["output_dir"] = "{output_dir}"
    _result["data"]["target"] = "{target}"
"""
        result = self.bridge.execute_script(script)
        return TaskResult(
            task_id="", status=TaskStatus.COMPLETED if result.success else TaskStatus.FAILED,
            output=result.data, error=result.error,
            metadata={"task_type": "render_turntable"},
        )

    def _get_related_tasks(self, task_type: str) -> List[str]:
        return {
            "render_still": ["render_animation"],
            "setup_render_settings": ["render_still", "render_animation"],
            "render_preset": ["render_turntable"],
        }.get(task_type, [])
