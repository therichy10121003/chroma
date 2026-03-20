"""
Scene Agent - Autonomous scene composition, lighting, and environment setup.

Handles scene-level operations including camera placement, lighting rigs,
world settings, and environment configuration.
"""

import logging
from typing import Any, Dict, List, Optional

from chromadb.utils.blender_agents.base_agent import (
    BlenderAgent, AgentCapability, TaskResult, TaskStatus,
)

logger = logging.getLogger(__name__)


class SceneAgent(BlenderAgent):
    """
    Agent specialized in scene composition and environment.

    Capabilities:
    - Camera placement and configuration
    - Lighting rigs (studio, outdoor, dramatic, etc.)
    - World/environment setup (sky, HDRI)
    - Scene organization and cleanup
    """

    LIGHTING_PRESETS = {
        "studio_3point": "Classic 3-point studio lighting",
        "outdoor_sun": "Outdoor sunlight with sky",
        "dramatic": "High-contrast dramatic lighting",
        "soft_ambient": "Soft ambient fill lighting",
        "sunset": "Golden hour sunset lighting",
        "neon": "Colorful neon lighting",
    }

    TASK_REGISTRY = {
        "setup_camera": "Position and configure camera",
        "setup_lighting": "Set up scene lighting",
        "lighting_preset": "Apply a lighting preset",
        "setup_world": "Configure world/environment",
        "clear_scene": "Clear all objects from scene",
        "organize_scene": "Organize scene into collections",
    }

    def __init__(self, bridge: Optional[Any] = None, **kwargs: Any):
        super().__init__(
            name="SceneAgent",
            capabilities=[
                AgentCapability.SCENE_SETUP,
                AgentCapability.LIGHTING,
            ],
            bridge=bridge,
            **kwargs,
        )

    def get_available_tasks(self) -> List[Dict[str, Any]]:
        return [{"name": n, "description": d} for n, d in self.TASK_REGISTRY.items()]

    def execute_task(self, task_name: str, **params: Any) -> TaskResult:
        if not self.bridge:
            return TaskResult(task_id="", status=TaskStatus.FAILED, error="No bridge")

        handlers = {
            "setup_camera": self._setup_camera,
            "setup_lighting": self._setup_lighting,
            "lighting_preset": self._lighting_preset,
            "setup_world": self._setup_world,
            "clear_scene": self._clear_scene,
            "organize_scene": self._organize_scene,
        }

        handler = handlers.get(task_name)
        if not handler:
            return TaskResult(task_id="", status=TaskStatus.FAILED, error=f"Unknown: {task_name}")
        return handler(**params)

    def assess_quality(self, result: TaskResult) -> float:
        if not result.succeeded:
            return 0.0
        score = 0.7
        if result.execution_time < 3.0:
            score += 0.15
        data = result.output if isinstance(result.output, dict) else {}
        if data.get("lights_count", 0) >= 3:
            score += 0.15
        return min(score, 1.0)

    def _setup_camera(self, **params: Any) -> TaskResult:
        location = params.get("location", (7, -6, 5))
        rotation = params.get("rotation", (1.1, 0, 0.8))
        focal_length = params.get("focal_length", 50.0)

        result = self.bridge.setup_camera(location, rotation, focal_length)
        return TaskResult(
            task_id="", status=TaskStatus.COMPLETED if result.success else TaskStatus.FAILED,
            output=result.data, error=result.error,
            metadata={"task_type": "setup_camera"},
        )

    def _setup_lighting(self, **params: Any) -> TaskResult:
        light_type = params.get("light_type", "SUN")
        location = params.get("location", (5, 5, 10))
        energy = params.get("energy", 5.0)
        color = params.get("color", (1.0, 1.0, 1.0))

        result = self.bridge.setup_lighting(light_type, location, energy, color)
        return TaskResult(
            task_id="", status=TaskStatus.COMPLETED if result.success else TaskStatus.FAILED,
            output=result.data, error=result.error,
            metadata={"task_type": "setup_lighting"},
        )

    def _lighting_preset(self, **params: Any) -> TaskResult:
        preset = params.get("preset", "studio_3point")

        preset_scripts = {
            "studio_3point": """
# Key light
bpy.ops.object.light_add(type="AREA", location=(4, -3, 5))
key = bpy.context.active_object
key.name = "Key_Light"
key.data.energy = 200
key.data.size = 3
key.rotation_euler = (0.8, 0, 0.5)

# Fill light
bpy.ops.object.light_add(type="AREA", location=(-3, -2, 3))
fill = bpy.context.active_object
fill.name = "Fill_Light"
fill.data.energy = 80
fill.data.size = 4
fill.rotation_euler = (0.6, 0, -0.7)

# Rim light
bpy.ops.object.light_add(type="AREA", location=(0, 4, 4))
rim = bpy.context.active_object
rim.name = "Rim_Light"
rim.data.energy = 150
rim.data.size = 2
rim.rotation_euler = (-0.5, 0, 3.14)

_result["data"]["preset"] = "studio_3point"
_result["data"]["lights_count"] = 3
""",
            "outdoor_sun": """
bpy.ops.object.light_add(type="SUN", location=(0, 0, 10))
sun = bpy.context.active_object
sun.name = "Sun"
sun.data.energy = 5
sun.data.color = (1.0, 0.95, 0.85)
sun.rotation_euler = (0.8, 0.2, 0)

_result["data"]["preset"] = "outdoor_sun"
_result["data"]["lights_count"] = 1
""",
            "dramatic": """
bpy.ops.object.light_add(type="SPOT", location=(3, -2, 5))
spot = bpy.context.active_object
spot.name = "Dramatic_Spot"
spot.data.energy = 500
spot.data.spot_size = 0.5
spot.data.color = (1.0, 0.9, 0.7)
spot.rotation_euler = (0.8, 0, 0.3)

bpy.ops.object.light_add(type="POINT", location=(-2, 1, 2))
fill = bpy.context.active_object
fill.name = "Subtle_Fill"
fill.data.energy = 20
fill.data.color = (0.6, 0.7, 1.0)

_result["data"]["preset"] = "dramatic"
_result["data"]["lights_count"] = 2
""",
            "sunset": """
bpy.ops.object.light_add(type="SUN", location=(10, 0, 2))
sun = bpy.context.active_object
sun.name = "Sunset_Sun"
sun.data.energy = 3
sun.data.color = (1.0, 0.6, 0.3)
sun.rotation_euler = (1.4, 0, 0)

bpy.ops.object.light_add(type="AREA", location=(-5, 0, 5))
sky_fill = bpy.context.active_object
sky_fill.name = "Sky_Fill"
sky_fill.data.energy = 30
sky_fill.data.color = (0.4, 0.5, 0.8)

_result["data"]["preset"] = "sunset"
_result["data"]["lights_count"] = 2
""",
            "neon": """
colors = [(1, 0, 0.5), (0, 0.5, 1), (0, 1, 0.5)]
positions = [(3, 0, 2), (-3, 0, 2), (0, 3, 2)]
for i, (color, pos) in enumerate(zip(colors, positions)):
    bpy.ops.object.light_add(type="POINT", location=pos)
    light = bpy.context.active_object
    light.name = f"Neon_{i}"
    light.data.energy = 100
    light.data.color = color

_result["data"]["preset"] = "neon"
_result["data"]["lights_count"] = 3
""",
        }

        if preset not in preset_scripts:
            return TaskResult(
                task_id="", status=TaskStatus.FAILED,
                error=f"Unknown preset: {preset}. Available: {list(preset_scripts.keys())}",
            )

        result = self.bridge.execute_script(preset_scripts[preset])
        return TaskResult(
            task_id="", status=TaskStatus.COMPLETED if result.success else TaskStatus.FAILED,
            output=result.data, error=result.error,
            metadata={"task_type": "lighting_preset", "params": {"preset": preset}},
        )

    def _setup_world(self, **params: Any) -> TaskResult:
        bg_color = params.get("background_color", (0.05, 0.05, 0.05))
        use_sky = params.get("use_sky", False)

        if use_sky:
            script = f"""
world = bpy.context.scene.world
if world is None:
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
world.use_nodes = True
nodes = world.node_tree.nodes
links = world.node_tree.links
nodes.clear()

output = nodes.new("ShaderNodeOutputWorld")
bg = nodes.new("ShaderNodeBackground")
sky = nodes.new("ShaderNodeTexSky")
sky.sky_type = "NISHITA"
links.new(sky.outputs[0], bg.inputs["Color"])
links.new(bg.outputs[0], output.inputs[0])

_result["data"]["world"] = "sky"
_result["data"]["sky_type"] = "NISHITA"
"""
        else:
            script = f"""
world = bpy.context.scene.world
if world is None:
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes.get("Background")
if bg:
    bg.inputs["Color"].default_value = ({bg_color[0]}, {bg_color[1]}, {bg_color[2]}, 1)

_result["data"]["world"] = "solid_color"
_result["data"]["color"] = {list(bg_color)}
"""
        result = self.bridge.execute_script(script)
        return TaskResult(
            task_id="", status=TaskStatus.COMPLETED if result.success else TaskStatus.FAILED,
            output=result.data, error=result.error,
            metadata={"task_type": "setup_world"},
        )

    def _clear_scene(self, **params: Any) -> TaskResult:
        keep_camera = params.get("keep_camera", True)
        keep_lights = params.get("keep_lights", False)

        script = f"""
for obj in list(bpy.data.objects):
    if {keep_camera} and obj.type == "CAMERA":
        continue
    if {keep_lights} and obj.type == "LIGHT":
        continue
    bpy.data.objects.remove(obj, do_unlink=True)

_result["data"]["remaining_objects"] = len(bpy.data.objects)
"""
        result = self.bridge.execute_script(script)
        return TaskResult(
            task_id="", status=TaskStatus.COMPLETED if result.success else TaskStatus.FAILED,
            output=result.data, error=result.error,
            metadata={"task_type": "clear_scene"},
        )

    def _organize_scene(self, **params: Any) -> TaskResult:
        script = """
# Create collections by object type
type_collections = {}
for obj in bpy.data.objects:
    coll_name = f"{obj.type.title()}s"
    if coll_name not in type_collections:
        coll = bpy.data.collections.new(coll_name)
        bpy.context.scene.collection.children.link(coll)
        type_collections[coll_name] = coll

    # Unlink from current collection and relink
    for c in obj.users_collection:
        c.objects.unlink(obj)
    type_collections[coll_name].objects.link(obj)

_result["data"]["collections"] = list(type_collections.keys())
_result["data"]["total_objects"] = len(bpy.data.objects)
"""
        result = self.bridge.execute_script(script)
        return TaskResult(
            task_id="", status=TaskStatus.COMPLETED if result.success else TaskStatus.FAILED,
            output=result.data, error=result.error,
            metadata={"task_type": "organize_scene"},
        )

    def _get_related_tasks(self, task_type: str) -> List[str]:
        return {
            "setup_camera": ["setup_lighting"],
            "setup_lighting": ["setup_world"],
            "lighting_preset": ["setup_camera", "setup_world"],
            "clear_scene": ["setup_camera", "setup_lighting"],
        }.get(task_type, [])
