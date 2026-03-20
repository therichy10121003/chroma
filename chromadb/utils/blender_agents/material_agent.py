"""
Material Agent - Autonomous material and shader creation.

Handles PBR materials, node-based shaders, and texture management
with quality-driven improvement.
"""

import logging
from typing import Any, Dict, List, Optional

from chromadb.utils.blender_agents.base_agent import (
    BlenderAgent, AgentCapability, TaskResult, TaskStatus,
)

logger = logging.getLogger(__name__)


class MaterialAgent(BlenderAgent):
    """
    Agent specialized in material and shader operations.

    Capabilities:
    - PBR material creation (metallic, roughness, color)
    - Preset material libraries (glass, metal, wood, etc.)
    - Node-based shader setup
    - Texture assignment and UV control
    """

    MATERIAL_PRESETS = {
        "gold": {"color": (1.0, 0.766, 0.336, 1.0), "metallic": 1.0, "roughness": 0.3},
        "silver": {"color": (0.972, 0.960, 0.915, 1.0), "metallic": 1.0, "roughness": 0.2},
        "copper": {"color": (0.955, 0.637, 0.538, 1.0), "metallic": 1.0, "roughness": 0.35},
        "glass": {"color": (0.95, 0.95, 0.95, 0.1), "metallic": 0.0, "roughness": 0.0},
        "wood": {"color": (0.545, 0.353, 0.169, 1.0), "metallic": 0.0, "roughness": 0.7},
        "plastic_red": {"color": (0.8, 0.1, 0.1, 1.0), "metallic": 0.0, "roughness": 0.4},
        "plastic_blue": {"color": (0.1, 0.2, 0.8, 1.0), "metallic": 0.0, "roughness": 0.4},
        "rubber": {"color": (0.05, 0.05, 0.05, 1.0), "metallic": 0.0, "roughness": 0.9},
        "concrete": {"color": (0.5, 0.5, 0.5, 1.0), "metallic": 0.0, "roughness": 0.85},
        "marble": {"color": (0.9, 0.88, 0.85, 1.0), "metallic": 0.0, "roughness": 0.15},
        "ceramic": {"color": (0.95, 0.92, 0.88, 1.0), "metallic": 0.0, "roughness": 0.2},
        "brushed_steel": {"color": (0.7, 0.7, 0.72, 1.0), "metallic": 1.0, "roughness": 0.45},
    }

    TASK_REGISTRY = {
        "create_material": "Create a PBR material",
        "apply_preset": "Apply a preset material",
        "create_emission": "Create an emissive/glowing material",
        "create_gradient": "Create a gradient material using nodes",
        "assign_material": "Assign material to object",
    }

    def __init__(self, bridge: Optional[Any] = None, **kwargs: Any):
        super().__init__(
            name="MaterialAgent",
            capabilities=[
                AgentCapability.MATERIALS,
                AgentCapability.TEXTURING,
            ],
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
            "description": f"Available presets: {list(self.MATERIAL_PRESETS.keys())}",
        })
        return tasks

    def execute_task(self, task_name: str, **params: Any) -> TaskResult:
        if not self.bridge:
            return TaskResult(
                task_id="", status=TaskStatus.FAILED,
                error="No BlenderBridge configured",
            )

        handlers = {
            "create_material": self._create_material,
            "apply_preset": self._apply_preset,
            "create_emission": self._create_emission,
            "create_gradient": self._create_gradient,
            "assign_material": self._assign_material,
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

        score = 0.6  # Base score

        data = result.output if isinstance(result.output, dict) else {}

        if data.get("material_name"):
            score += 0.2
        if result.execution_time < 3.0:
            score += 0.1
        if data.get("nodes_count", 0) > 2:
            score += 0.1  # More complex shader = higher quality

        return min(score, 1.0)

    def _create_material(self, **params: Any) -> TaskResult:
        object_name = params.get("object_name", "Cube")
        material_name = params.get("material_name", "Material")
        color = params.get("color", (0.8, 0.8, 0.8, 1.0))
        metallic = params.get("metallic", 0.0)
        roughness = params.get("roughness", 0.5)

        result = self.bridge.apply_material(
            object_name, material_name, color, metallic, roughness,
        )
        return TaskResult(
            task_id="", status=TaskStatus.COMPLETED if result.success else TaskStatus.FAILED,
            output=result.data, error=result.error,
            metadata={"task_type": "create_material", "params": params},
        )

    def _apply_preset(self, **params: Any) -> TaskResult:
        object_name = params.get("object_name", "Cube")
        preset_name = params.get("preset", "gold")

        if preset_name not in self.MATERIAL_PRESETS:
            return TaskResult(
                task_id="", status=TaskStatus.FAILED,
                error=f"Unknown preset: {preset_name}. "
                      f"Available: {list(self.MATERIAL_PRESETS.keys())}",
            )

        preset = self.MATERIAL_PRESETS[preset_name]
        result = self.bridge.apply_material(
            object_name=object_name,
            material_name=f"{preset_name}_material",
            color=preset["color"],
            metallic=preset["metallic"],
            roughness=preset["roughness"],
        )

        return TaskResult(
            task_id="", status=TaskStatus.COMPLETED if result.success else TaskStatus.FAILED,
            output=result.data, error=result.error,
            metadata={"task_type": "apply_preset", "params": params},
        )

    def _create_emission(self, **params: Any) -> TaskResult:
        object_name = params.get("object_name", "Cube")
        color = params.get("color", (1.0, 0.5, 0.0, 1.0))
        strength = params.get("strength", 5.0)

        script = f"""
obj = bpy.data.objects.get("{object_name}")
if obj is None:
    _result["success"] = False
    _result["error"] = "Object not found"
else:
    mat = bpy.data.materials.new(name="Emission")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = {color}
    emission.inputs["Strength"].default_value = {strength}
    links.new(emission.outputs[0], output.inputs[0])

    obj.data.materials.append(mat)
    _result["data"]["material_name"] = mat.name
    _result["data"]["strength"] = {strength}
    _result["data"]["nodes_count"] = len(nodes)
"""
        result = self.bridge.execute_script(script)
        return TaskResult(
            task_id="", status=TaskStatus.COMPLETED if result.success else TaskStatus.FAILED,
            output=result.data, error=result.error,
            metadata={"task_type": "create_emission"},
        )

    def _create_gradient(self, **params: Any) -> TaskResult:
        object_name = params.get("object_name", "Cube")
        color1 = params.get("color1", (1.0, 0.0, 0.0, 1.0))
        color2 = params.get("color2", (0.0, 0.0, 1.0, 1.0))

        script = f"""
obj = bpy.data.objects.get("{object_name}")
if obj is None:
    _result["success"] = False
    _result["error"] = "Object not found"
else:
    mat = bpy.data.materials.new(name="Gradient")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    mix = nodes.new("ShaderNodeMixRGB")
    gradient = nodes.new("ShaderNodeTexGradient")
    coord = nodes.new("ShaderNodeTexCoord")

    mix.inputs["Color1"].default_value = {color1}
    mix.inputs["Color2"].default_value = {color2}

    links.new(coord.outputs["Generated"], gradient.inputs["Vector"])
    links.new(gradient.outputs["Fac"], mix.inputs["Fac"])
    links.new(mix.outputs[0], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs[0], output.inputs[0])

    obj.data.materials.append(mat)
    _result["data"]["material_name"] = mat.name
    _result["data"]["nodes_count"] = len(nodes)
"""
        result = self.bridge.execute_script(script)
        return TaskResult(
            task_id="", status=TaskStatus.COMPLETED if result.success else TaskStatus.FAILED,
            output=result.data, error=result.error,
            metadata={"task_type": "create_gradient"},
        )

    def _assign_material(self, **params: Any) -> TaskResult:
        object_name = params.get("object_name", "Cube")
        material_name = params.get("material_name", "Material")

        script = f"""
obj = bpy.data.objects.get("{object_name}")
mat = bpy.data.materials.get("{material_name}")
if obj is None or mat is None:
    _result["success"] = False
    _result["error"] = "Object or material not found"
else:
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    _result["data"]["object_name"] = obj.name
    _result["data"]["material_name"] = mat.name
"""
        result = self.bridge.execute_script(script)
        return TaskResult(
            task_id="", status=TaskStatus.COMPLETED if result.success else TaskStatus.FAILED,
            output=result.data, error=result.error,
            metadata={"task_type": "assign_material"},
        )

    def _get_related_tasks(self, task_type: str) -> List[str]:
        relations = {
            "create_material": ["assign_material"],
            "apply_preset": ["create_emission"],
        }
        return relations.get(task_type, [])
