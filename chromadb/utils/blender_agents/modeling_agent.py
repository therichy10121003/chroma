"""
Modeling Agent - Autonomous 3D mesh creation and modification.

Handles geometry creation, sculpting, modifiers, and mesh operations
with self-improvement through quality tracking.
"""

import logging
from typing import Any, Dict, List, Optional

from chromadb.utils.blender_agents.base_agent import (
    BlenderAgent, AgentCapability, TaskResult, TaskStatus,
)

logger = logging.getLogger(__name__)


class ModelingAgent(BlenderAgent):
    """
    Agent specialized in 3D modeling operations.

    Capabilities:
    - Primitive mesh creation (cube, sphere, cylinder, etc.)
    - Modifier application (subdivision, mirror, array, etc.)
    - Mesh transformations (scale, rotate, translate)
    - Boolean operations (union, difference, intersect)
    - Procedural generation patterns
    """

    TASK_REGISTRY = {
        "create_primitive": "Create a basic mesh primitive",
        "add_modifier": "Apply a modifier to an object",
        "transform_object": "Transform object position/rotation/scale",
        "boolean_operation": "Perform boolean operation between objects",
        "create_procedural": "Create procedural geometry",
        "duplicate_object": "Duplicate and position objects",
        "smooth_mesh": "Apply smoothing to mesh",
        "create_array": "Create an array of objects",
    }

    def __init__(self, bridge: Optional[Any] = None, **kwargs: Any):
        super().__init__(
            name="ModelingAgent",
            capabilities=[
                AgentCapability.MODELING,
                AgentCapability.UV_MAPPING,
            ],
            bridge=bridge,
            **kwargs,
        )

    def get_available_tasks(self) -> List[Dict[str, Any]]:
        return [
            {"name": name, "description": desc}
            for name, desc in self.TASK_REGISTRY.items()
        ]

    def execute_task(self, task_name: str, **params: Any) -> TaskResult:
        if not self.bridge:
            return TaskResult(
                task_id="", status=TaskStatus.FAILED,
                error="No BlenderBridge configured",
            )

        handlers = {
            "create_primitive": self._create_primitive,
            "add_modifier": self._add_modifier,
            "transform_object": self._transform_object,
            "boolean_operation": self._boolean_operation,
            "create_procedural": self._create_procedural,
            "duplicate_object": self._duplicate_object,
            "smooth_mesh": self._smooth_mesh,
            "create_array": self._create_array,
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

        score = 0.5  # Base score for success

        data = result.output if isinstance(result.output, dict) else {}

        # More vertices/faces generally means more detail
        vertices = data.get("vertices", 0)
        if vertices > 100:
            score += 0.2
        elif vertices > 10:
            score += 0.1

        # Modifier application is a quality signal
        if data.get("modifier_name"):
            score += 0.15

        # Fast execution is good
        if result.execution_time < 5.0:
            score += 0.15

        return min(score, 1.0)

    def _create_primitive(self, **params: Any) -> TaskResult:
        mesh_type = params.get("mesh_type", "cube")
        location = params.get("location", (0, 0, 0))
        scale = params.get("scale", (1, 1, 1))
        name = params.get("name")

        result = self.bridge.create_mesh(
            mesh_type=mesh_type, location=location, scale=scale, name=name,
        )
        return TaskResult(
            task_id="", status=TaskStatus.COMPLETED if result.success else TaskStatus.FAILED,
            output=result.data, error=result.error,
            metadata={"task_type": "create_primitive", "approach": {"mesh_type": mesh_type}},
        )

    def _add_modifier(self, **params: Any) -> TaskResult:
        object_name = params.get("object_name", "Cube")
        modifier_type = params.get("modifier_type", "SUBSURF")
        settings = params.get("settings", {})

        result = self.bridge.add_modifier(object_name, modifier_type, settings)
        return TaskResult(
            task_id="", status=TaskStatus.COMPLETED if result.success else TaskStatus.FAILED,
            output=result.data, error=result.error,
            metadata={"task_type": "add_modifier"},
        )

    def _transform_object(self, **params: Any) -> TaskResult:
        object_name = params.get("object_name", "Cube")
        location = params.get("location")
        rotation = params.get("rotation")
        scale = params.get("scale")

        transform_lines = []
        if location:
            transform_lines.append(f"obj.location = {location}")
        if rotation:
            transform_lines.append(f"obj.rotation_euler = {rotation}")
        if scale:
            transform_lines.append(f"obj.scale = {scale}")

        script = f"""
obj = bpy.data.objects.get("{object_name}")
if obj is None:
    _result["success"] = False
    _result["error"] = "Object not found"
else:
    {chr(10).join('    ' + line for line in transform_lines)}
    _result["data"]["object_name"] = obj.name
    _result["data"]["location"] = list(obj.location)
    _result["data"]["rotation"] = list(obj.rotation_euler)
    _result["data"]["scale"] = list(obj.scale)
"""
        result = self.bridge.execute_script(script)
        return TaskResult(
            task_id="", status=TaskStatus.COMPLETED if result.success else TaskStatus.FAILED,
            output=result.data, error=result.error,
            metadata={"task_type": "transform_object"},
        )

    def _boolean_operation(self, **params: Any) -> TaskResult:
        target = params.get("target", "Cube")
        tool = params.get("tool", "Cube.001")
        operation = params.get("operation", "DIFFERENCE")

        script = f"""
target = bpy.data.objects.get("{target}")
tool = bpy.data.objects.get("{tool}")
if target is None or tool is None:
    _result["success"] = False
    _result["error"] = "Object not found"
else:
    mod = target.modifiers.new(name="Boolean", type="BOOLEAN")
    mod.operation = "{operation}"
    mod.object = tool
    bpy.context.view_layer.objects.active = target
    bpy.ops.object.modifier_apply(modifier="Boolean")
    _result["data"]["target"] = target.name
    _result["data"]["operation"] = "{operation}"
"""
        result = self.bridge.execute_script(script)
        return TaskResult(
            task_id="", status=TaskStatus.COMPLETED if result.success else TaskStatus.FAILED,
            output=result.data, error=result.error,
            metadata={"task_type": "boolean_operation"},
        )

    def _create_procedural(self, **params: Any) -> TaskResult:
        pattern = params.get("pattern", "grid")
        count = params.get("count", 5)
        spacing = params.get("spacing", 2.0)

        if pattern == "grid":
            script = f"""
import math
for i in range({count}):
    for j in range({count}):
        bpy.ops.mesh.primitive_cube_add(
            location=(i * {spacing}, j * {spacing}, 0),
            scale=(0.4, 0.4, 0.4)
        )
_result["data"]["pattern"] = "grid"
_result["data"]["total_objects"] = {count * count}
"""
        elif pattern == "circle":
            script = f"""
import math
for i in range({count}):
    angle = (2 * math.pi * i) / {count}
    x = math.cos(angle) * {spacing}
    y = math.sin(angle) * {spacing}
    bpy.ops.mesh.primitive_cube_add(location=(x, y, 0), scale=(0.3, 0.3, 0.3))
_result["data"]["pattern"] = "circle"
_result["data"]["total_objects"] = {count}
"""
        elif pattern == "spiral":
            script = f"""
import math
for i in range({count}):
    angle = (4 * math.pi * i) / {count}
    radius = {spacing} * (i / {count})
    x = math.cos(angle) * radius
    y = math.sin(angle) * radius
    z = i * 0.2
    bpy.ops.mesh.primitive_uv_sphere_add(
        location=(x, y, z), scale=(0.2, 0.2, 0.2)
    )
_result["data"]["pattern"] = "spiral"
_result["data"]["total_objects"] = {count}
"""
        else:
            return TaskResult(
                task_id="", status=TaskStatus.FAILED,
                error=f"Unknown pattern: {pattern}",
            )

        result = self.bridge.execute_script(script)
        return TaskResult(
            task_id="", status=TaskStatus.COMPLETED if result.success else TaskStatus.FAILED,
            output=result.data, error=result.error,
            metadata={"task_type": "create_procedural", "params": params},
        )

    def _duplicate_object(self, **params: Any) -> TaskResult:
        object_name = params.get("object_name", "Cube")
        offset = params.get("offset", (2, 0, 0))

        script = f"""
obj = bpy.data.objects.get("{object_name}")
if obj is None:
    _result["success"] = False
    _result["error"] = "Object not found"
else:
    new_obj = obj.copy()
    new_obj.data = obj.data.copy()
    new_obj.location.x += {offset[0]}
    new_obj.location.y += {offset[1]}
    new_obj.location.z += {offset[2]}
    bpy.context.collection.objects.link(new_obj)
    _result["data"]["new_object"] = new_obj.name
"""
        result = self.bridge.execute_script(script)
        return TaskResult(
            task_id="", status=TaskStatus.COMPLETED if result.success else TaskStatus.FAILED,
            output=result.data, error=result.error,
            metadata={"task_type": "duplicate_object"},
        )

    def _smooth_mesh(self, **params: Any) -> TaskResult:
        object_name = params.get("object_name", "Cube")
        iterations = params.get("iterations", 2)

        result = self.bridge.add_modifier(
            object_name, "SUBSURF",
            {"levels": iterations, "render_levels": iterations + 1},
        )
        return TaskResult(
            task_id="", status=TaskStatus.COMPLETED if result.success else TaskStatus.FAILED,
            output=result.data, error=result.error,
            metadata={"task_type": "smooth_mesh"},
        )

    def _create_array(self, **params: Any) -> TaskResult:
        object_name = params.get("object_name", "Cube")
        count = params.get("count", 5)
        offset = params.get("offset", (2, 0, 0))

        result = self.bridge.add_modifier(
            object_name, "ARRAY",
            {"count": count, "relative_offset_displace[0]": offset[0]},
        )
        return TaskResult(
            task_id="", status=TaskStatus.COMPLETED if result.success else TaskStatus.FAILED,
            output=result.data, error=result.error,
            metadata={"task_type": "create_array"},
        )

    def _get_related_tasks(self, task_type: str) -> List[str]:
        relations = {
            "create_primitive": ["add_modifier", "smooth_mesh"],
            "add_modifier": ["smooth_mesh", "transform_object"],
            "create_procedural": ["add_modifier"],
            "boolean_operation": ["smooth_mesh"],
        }
        return relations.get(task_type, [])
