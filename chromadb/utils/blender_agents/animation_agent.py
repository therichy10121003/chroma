"""
Animation Agent - Autonomous keyframe animation and motion creation.

Handles keyframe animation, motion paths, and procedural animation
with quality-driven improvement.
"""

import logging
from typing import Any, Dict, List, Optional

from chromadb.utils.blender_agents.base_agent import (
    BlenderAgent, AgentCapability, TaskResult, TaskStatus,
)

logger = logging.getLogger(__name__)


class AnimationAgent(BlenderAgent):
    """
    Agent specialized in animation operations.

    Capabilities:
    - Keyframe animation (position, rotation, scale)
    - Procedural motion (bounce, spin, wave, orbit)
    - Timeline management
    - Animation curve editing
    """

    MOTION_PRESETS = {
        "bounce": "Object bouncing up and down",
        "spin": "Object spinning around axis",
        "wave": "Sinusoidal wave motion",
        "orbit": "Orbital circular motion",
        "grow": "Scale from zero to full size",
        "shake": "Random shaking/vibration",
        "slide": "Linear slide from A to B",
    }

    TASK_REGISTRY = {
        "animate_transform": "Animate object position/rotation/scale",
        "procedural_motion": "Apply procedural motion preset",
        "set_timeline": "Configure animation timeline",
        "animate_visibility": "Animate object visibility",
    }

    def __init__(self, bridge: Optional[Any] = None, **kwargs: Any):
        super().__init__(
            name="AnimationAgent",
            capabilities=[AgentCapability.ANIMATION],
            bridge=bridge,
            **kwargs,
        )

    def get_available_tasks(self) -> List[Dict[str, Any]]:
        tasks = [{"name": n, "description": d} for n, d in self.TASK_REGISTRY.items()]
        tasks.append({
            "name": "motion_presets",
            "description": f"Available: {list(self.MOTION_PRESETS.keys())}",
        })
        return tasks

    def execute_task(self, task_name: str, **params: Any) -> TaskResult:
        if not self.bridge:
            return TaskResult(task_id="", status=TaskStatus.FAILED, error="No bridge")

        handlers = {
            "animate_transform": self._animate_transform,
            "procedural_motion": self._procedural_motion,
            "set_timeline": self._set_timeline,
            "animate_visibility": self._animate_visibility,
        }

        handler = handlers.get(task_name)
        if not handler:
            return TaskResult(task_id="", status=TaskStatus.FAILED, error=f"Unknown: {task_name}")
        return handler(**params)

    def assess_quality(self, result: TaskResult) -> float:
        if not result.succeeded:
            return 0.0
        score = 0.6
        data = result.output if isinstance(result.output, dict) else {}
        keyframes = data.get("keyframes", 0)
        if keyframes >= 10:
            score += 0.2
        elif keyframes >= 5:
            score += 0.1
        if result.execution_time < 5.0:
            score += 0.1
        return min(score, 1.0)

    def _animate_transform(self, **params: Any) -> TaskResult:
        object_name = params.get("object_name", "Cube")
        keyframes = params.get("keyframes", [])
        # keyframes: [{"frame": 1, "location": (0,0,0)}, {"frame": 30, "location": (5,0,0)}]

        if not keyframes:
            keyframes = [
                {"frame": 1, "location": (0, 0, 0)},
                {"frame": 30, "location": (5, 0, 0)},
                {"frame": 60, "location": (5, 5, 0)},
                {"frame": 90, "location": (0, 0, 0)},
            ]

        kf_lines = []
        for kf in keyframes:
            frame = kf.get("frame", 1)
            kf_lines.append(f"    scene.frame_set({frame})")
            if "location" in kf:
                kf_lines.append(f"    obj.location = {kf['location']}")
                kf_lines.append(f'    obj.keyframe_insert(data_path="location")')
            if "rotation" in kf:
                kf_lines.append(f"    obj.rotation_euler = {kf['rotation']}")
                kf_lines.append(f'    obj.keyframe_insert(data_path="rotation_euler")')
            if "scale" in kf:
                kf_lines.append(f"    obj.scale = {kf['scale']}")
                kf_lines.append(f'    obj.keyframe_insert(data_path="scale")')

        script = f"""
obj = bpy.data.objects.get("{object_name}")
scene = bpy.context.scene
if obj is None:
    _result["success"] = False
    _result["error"] = "Object not found"
else:
{chr(10).join(kf_lines)}
    _result["data"]["object_name"] = obj.name
    _result["data"]["keyframes"] = {len(keyframes)}
"""
        result = self.bridge.execute_script(script)
        return TaskResult(
            task_id="", status=TaskStatus.COMPLETED if result.success else TaskStatus.FAILED,
            output=result.data, error=result.error,
            metadata={"task_type": "animate_transform"},
        )

    def _procedural_motion(self, **params: Any) -> TaskResult:
        object_name = params.get("object_name", "Cube")
        motion_type = params.get("motion_type", "bounce")
        frames = params.get("frames", 60)
        amplitude = params.get("amplitude", 2.0)

        motion_scripts = {
            "bounce": f"""
import math
obj = bpy.data.objects.get("{object_name}")
scene = bpy.context.scene
if obj is None:
    _result["success"] = False
else:
    for f in range(1, {frames} + 1):
        scene.frame_set(f)
        t = f / {frames}
        obj.location.z = abs(math.sin(t * math.pi * 4)) * {amplitude}
        obj.keyframe_insert(data_path="location")
    _result["data"]["motion"] = "bounce"
    _result["data"]["keyframes"] = {frames}
""",
            "spin": f"""
import math
obj = bpy.data.objects.get("{object_name}")
scene = bpy.context.scene
if obj is None:
    _result["success"] = False
else:
    for f in range(1, {frames} + 1):
        scene.frame_set(f)
        obj.rotation_euler.z = (f / {frames}) * math.pi * 4
        obj.keyframe_insert(data_path="rotation_euler")
    _result["data"]["motion"] = "spin"
    _result["data"]["keyframes"] = {frames}
""",
            "wave": f"""
import math
obj = bpy.data.objects.get("{object_name}")
scene = bpy.context.scene
if obj is None:
    _result["success"] = False
else:
    start_x = obj.location.x
    for f in range(1, {frames} + 1):
        scene.frame_set(f)
        t = f / {frames}
        obj.location.z = math.sin(t * math.pi * 6) * {amplitude}
        obj.location.x = start_x + t * {amplitude * 3}
        obj.keyframe_insert(data_path="location")
    _result["data"]["motion"] = "wave"
    _result["data"]["keyframes"] = {frames}
""",
            "orbit": f"""
import math
obj = bpy.data.objects.get("{object_name}")
scene = bpy.context.scene
if obj is None:
    _result["success"] = False
else:
    for f in range(1, {frames} + 1):
        scene.frame_set(f)
        angle = (2 * math.pi * f) / {frames}
        obj.location.x = math.cos(angle) * {amplitude}
        obj.location.y = math.sin(angle) * {amplitude}
        obj.keyframe_insert(data_path="location")
    _result["data"]["motion"] = "orbit"
    _result["data"]["keyframes"] = {frames}
""",
            "grow": f"""
obj = bpy.data.objects.get("{object_name}")
scene = bpy.context.scene
if obj is None:
    _result["success"] = False
else:
    scene.frame_set(1)
    obj.scale = (0.01, 0.01, 0.01)
    obj.keyframe_insert(data_path="scale")
    scene.frame_set({frames})
    obj.scale = ({amplitude}, {amplitude}, {amplitude})
    obj.keyframe_insert(data_path="scale")
    _result["data"]["motion"] = "grow"
    _result["data"]["keyframes"] = 2
""",
        }

        if motion_type not in motion_scripts:
            return TaskResult(
                task_id="", status=TaskStatus.FAILED,
                error=f"Unknown motion: {motion_type}",
            )

        result = self.bridge.execute_script(motion_scripts[motion_type])
        return TaskResult(
            task_id="", status=TaskStatus.COMPLETED if result.success else TaskStatus.FAILED,
            output=result.data, error=result.error,
            metadata={"task_type": "procedural_motion", "params": params},
        )

    def _set_timeline(self, **params: Any) -> TaskResult:
        frame_start = params.get("frame_start", 1)
        frame_end = params.get("frame_end", 250)
        fps = params.get("fps", 24)

        script = f"""
scene = bpy.context.scene
scene.frame_start = {frame_start}
scene.frame_end = {frame_end}
scene.render.fps = {fps}
_result["data"]["frame_start"] = {frame_start}
_result["data"]["frame_end"] = {frame_end}
_result["data"]["fps"] = {fps}
_result["data"]["duration_seconds"] = ({frame_end} - {frame_start} + 1) / {fps}
"""
        result = self.bridge.execute_script(script)
        return TaskResult(
            task_id="", status=TaskStatus.COMPLETED if result.success else TaskStatus.FAILED,
            output=result.data, error=result.error,
            metadata={"task_type": "set_timeline"},
        )

    def _animate_visibility(self, **params: Any) -> TaskResult:
        object_name = params.get("object_name", "Cube")
        appear_frame = params.get("appear_frame", 1)
        disappear_frame = params.get("disappear_frame")

        script = f"""
obj = bpy.data.objects.get("{object_name}")
scene = bpy.context.scene
if obj is None:
    _result["success"] = False
    _result["error"] = "Object not found"
else:
    scene.frame_set({appear_frame} - 1)
    obj.hide_render = True
    obj.hide_viewport = True
    obj.keyframe_insert(data_path="hide_render")
    obj.keyframe_insert(data_path="hide_viewport")

    scene.frame_set({appear_frame})
    obj.hide_render = False
    obj.hide_viewport = False
    obj.keyframe_insert(data_path="hide_render")
    obj.keyframe_insert(data_path="hide_viewport")
    kf_count = 2
"""
        if disappear_frame:
            script += f"""
    scene.frame_set({disappear_frame})
    obj.hide_render = True
    obj.hide_viewport = True
    obj.keyframe_insert(data_path="hide_render")
    obj.keyframe_insert(data_path="hide_viewport")
    kf_count = 3
"""
        script += """
    _result["data"]["object_name"] = obj.name
    _result["data"]["keyframes"] = kf_count
"""
        result = self.bridge.execute_script(script)
        return TaskResult(
            task_id="", status=TaskStatus.COMPLETED if result.success else TaskStatus.FAILED,
            output=result.data, error=result.error,
            metadata={"task_type": "animate_visibility"},
        )

    def _get_related_tasks(self, task_type: str) -> List[str]:
        return {
            "animate_transform": ["procedural_motion"],
            "procedural_motion": ["set_timeline"],
            "set_timeline": ["animate_transform"],
        }.get(task_type, [])
