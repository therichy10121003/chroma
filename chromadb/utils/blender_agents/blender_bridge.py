"""
Blender Bridge - Interface for executing Blender operations headlessly.

Provides a clean Python API to execute Blender commands via subprocess,
handling script generation, execution, and result parsing.
"""

import subprocess
import tempfile
import json
import os
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class BlenderResult:
    """Result from a Blender operation."""

    success: bool
    output: str
    error: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    render_path: Optional[str] = None
    blend_file: Optional[str] = None
    execution_time: float = 0.0


class BlenderBridge:
    """
    Bridge between Python and Blender's headless mode.

    Executes Blender Python scripts in headless mode, enabling
    automated 3D operations without a GUI.
    """

    def __init__(
        self,
        blender_path: str = "blender",
        output_dir: Optional[str] = None,
        timeout: int = 300,
    ):
        """
        Initialize the Blender bridge.

        Args:
            blender_path: Path to Blender executable
            output_dir: Directory for output files (renders, exports)
            timeout: Maximum execution time in seconds
        """
        self.blender_path = blender_path
        self.output_dir = output_dir or tempfile.mkdtemp(prefix="chroma_blender_")
        self.timeout = timeout

        # Verify Blender is accessible
        self._verify_blender()

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def _verify_blender(self) -> None:
        """Verify Blender is installed and accessible."""
        try:
            result = subprocess.run(
                [self.blender_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Blender returned error: {result.stderr}"
                )
            version_line = result.stdout.strip().split("\n")[0]
            logger.info(f"Blender bridge initialized: {version_line}")
            self.blender_version = version_line
        except FileNotFoundError:
            raise RuntimeError(
                f"Blender not found at '{self.blender_path}'. "
                "Install with: apt-get install blender"
            )

    def execute_script(
        self,
        script: str,
        blend_file: Optional[str] = None,
        background: bool = True,
        extra_args: Optional[List[str]] = None,
    ) -> BlenderResult:
        """
        Execute a Blender Python script.

        Args:
            script: Python script to execute in Blender
            blend_file: Optional .blend file to open first
            background: Run in background (headless) mode
            extra_args: Additional command-line arguments

        Returns:
            BlenderResult with output and status
        """
        import time

        # Write script to temporary file
        script_path = os.path.join(self.output_dir, "_temp_script.py")

        # Wrap script to capture structured output
        wrapped_script = f'''
import bpy
import json
import sys
import traceback

_result = {{"success": True, "data": {{}}}}

try:
{self._indent_script(script)}
except Exception as e:
    _result["success"] = False
    _result["error"] = str(e)
    _result["traceback"] = traceback.format_exc()
    print(f"BLENDER_ERROR: {{str(e)}}", file=sys.stderr)

# Output structured result
print("BLENDER_RESULT_START")
print(json.dumps(_result))
print("BLENDER_RESULT_END")
'''

        with open(script_path, "w") as f:
            f.write(wrapped_script)

        # Build command
        cmd = [self.blender_path]
        if background:
            cmd.append("--background")
        if blend_file:
            cmd.append(blend_file)
        cmd.extend(["--python", script_path])
        if extra_args:
            cmd.extend(extra_args)

        # Execute
        start_time = time.time()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            execution_time = time.time() - start_time

            # Parse structured result
            result_data = self._parse_result(proc.stdout)

            return BlenderResult(
                success=proc.returncode == 0 and result_data.get("success", False),
                output=proc.stdout,
                error=proc.stderr if proc.returncode != 0 else result_data.get("error", ""),
                data=result_data.get("data", {}),
                execution_time=execution_time,
            )

        except subprocess.TimeoutExpired:
            return BlenderResult(
                success=False,
                output="",
                error=f"Blender script timed out after {self.timeout}s",
                execution_time=self.timeout,
            )
        finally:
            # Cleanup temp script
            if os.path.exists(script_path):
                os.remove(script_path)

    def _indent_script(self, script: str) -> str:
        """Indent a script for embedding in try block."""
        lines = script.split("\n")
        return "\n".join(f"    {line}" for line in lines)

    def _parse_result(self, output: str) -> Dict[str, Any]:
        """Parse structured result from Blender output."""
        try:
            start_marker = "BLENDER_RESULT_START"
            end_marker = "BLENDER_RESULT_END"

            start_idx = output.find(start_marker)
            end_idx = output.find(end_marker)

            if start_idx >= 0 and end_idx >= 0:
                json_str = output[start_idx + len(start_marker):end_idx].strip()
                return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass

        return {"success": True, "data": {}}

    def create_mesh(
        self,
        mesh_type: str = "cube",
        location: tuple = (0, 0, 0),
        scale: tuple = (1, 1, 1),
        name: Optional[str] = None,
    ) -> BlenderResult:
        """
        Create a basic mesh object.

        Args:
            mesh_type: Type of mesh (cube, sphere, cylinder, cone, torus, plane, monkey)
            location: (x, y, z) position
            scale: (x, y, z) scale
            name: Optional object name
        """
        mesh_ops = {
            "cube": "bpy.ops.mesh.primitive_cube_add",
            "sphere": "bpy.ops.mesh.primitive_uv_sphere_add",
            "cylinder": "bpy.ops.mesh.primitive_cylinder_add",
            "cone": "bpy.ops.mesh.primitive_cone_add",
            "torus": "bpy.ops.mesh.primitive_torus_add",
            "plane": "bpy.ops.mesh.primitive_plane_add",
            "monkey": "bpy.ops.mesh.primitive_monkey_add",
            "ico_sphere": "bpy.ops.mesh.primitive_ico_sphere_add",
        }

        if mesh_type not in mesh_ops:
            return BlenderResult(
                success=False,
                output="",
                error=f"Unknown mesh type: {mesh_type}. Available: {list(mesh_ops.keys())}",
            )

        name_str = f'"{name}"' if name else f'"{mesh_type.title()}"'

        script = f"""
{mesh_ops[mesh_type]}(location={location}, scale={scale})
obj = bpy.context.active_object
obj.name = {name_str}
_result["data"]["object_name"] = obj.name
_result["data"]["mesh_type"] = "{mesh_type}"
_result["data"]["location"] = list(obj.location)
_result["data"]["scale"] = list(obj.scale)
_result["data"]["vertices"] = len(obj.data.vertices)
_result["data"]["faces"] = len(obj.data.polygons)
"""
        return self.execute_script(script)

    def apply_material(
        self,
        object_name: str,
        material_name: str,
        color: tuple = (0.8, 0.8, 0.8, 1.0),
        metallic: float = 0.0,
        roughness: float = 0.5,
    ) -> BlenderResult:
        """
        Apply a material to an object.

        Args:
            object_name: Name of the target object
            material_name: Name for the material
            color: RGBA color tuple
            metallic: Metallic factor (0-1)
            roughness: Roughness factor (0-1)
        """
        script = f"""
obj = bpy.data.objects.get("{object_name}")
if obj is None:
    _result["success"] = False
    _result["error"] = "Object '{object_name}' not found"
else:
    mat = bpy.data.materials.new(name="{material_name}")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = {color}
    bsdf.inputs["Metallic"].default_value = {metallic}
    bsdf.inputs["Roughness"].default_value = {roughness}
    obj.data.materials.append(mat)
    _result["data"]["material_name"] = mat.name
    _result["data"]["object_name"] = obj.name
"""
        return self.execute_script(script)

    def render_scene(
        self,
        output_path: Optional[str] = None,
        resolution: tuple = (1920, 1080),
        samples: int = 128,
        engine: str = "CYCLES",
        file_format: str = "PNG",
    ) -> BlenderResult:
        """
        Render the current scene.

        Args:
            output_path: Path for render output
            resolution: (width, height) in pixels
            samples: Render sample count
            engine: Render engine (CYCLES, BLENDER_EEVEE, BLENDER_WORKBENCH)
            file_format: Output format (PNG, JPEG, EXR, TIFF)
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, "render_output")

        script = f"""
scene = bpy.context.scene
scene.render.engine = "{engine}"
scene.render.resolution_x = {resolution[0]}
scene.render.resolution_y = {resolution[1]}
scene.render.image_settings.file_format = "{file_format}"
scene.render.filepath = "{output_path}"

if "{engine}" == "CYCLES":
    scene.cycles.samples = {samples}
    scene.cycles.device = "CPU"

bpy.ops.render.render(write_still=True)

_result["data"]["output_path"] = "{output_path}"
_result["data"]["resolution"] = [{resolution[0]}, {resolution[1]}]
_result["data"]["engine"] = "{engine}"
_result["data"]["samples"] = {samples}
_result["render_path"] = "{output_path}"
"""
        return self.execute_script(script)

    def save_blend(
        self,
        filepath: Optional[str] = None,
    ) -> BlenderResult:
        """
        Save the current scene to a .blend file.

        Args:
            filepath: Output .blend file path
        """
        if filepath is None:
            filepath = os.path.join(self.output_dir, "scene.blend")

        script = f"""
bpy.ops.wm.save_as_mainfile(filepath="{filepath}")
_result["data"]["filepath"] = "{filepath}"
_result["blend_file"] = "{filepath}"
"""
        return self.execute_script(script)

    def get_scene_info(self) -> BlenderResult:
        """Get information about the current scene."""
        script = """
scene = bpy.context.scene
objects = []
for obj in scene.objects:
    obj_info = {
        "name": obj.name,
        "type": obj.type,
        "location": list(obj.location),
        "scale": list(obj.scale),
        "rotation": list(obj.rotation_euler),
    }
    if obj.type == "MESH":
        obj_info["vertices"] = len(obj.data.vertices)
        obj_info["faces"] = len(obj.data.polygons)
        obj_info["materials"] = [m.name for m in obj.data.materials if m]
    objects.append(obj_info)

_result["data"]["scene_name"] = scene.name
_result["data"]["frame_start"] = scene.frame_start
_result["data"]["frame_end"] = scene.frame_end
_result["data"]["fps"] = scene.render.fps
_result["data"]["object_count"] = len(objects)
_result["data"]["objects"] = objects
_result["data"]["render_engine"] = scene.render.engine
_result["data"]["resolution"] = [scene.render.resolution_x, scene.render.resolution_y]
"""
        return self.execute_script(script)

    def setup_camera(
        self,
        location: tuple = (7, -6, 5),
        rotation: tuple = (1.1, 0, 0.8),
        focal_length: float = 50.0,
    ) -> BlenderResult:
        """Set up the camera."""
        script = f"""
cam = bpy.data.objects.get("Camera")
if cam is None:
    bpy.ops.object.camera_add(location={location}, rotation={rotation})
    cam = bpy.context.active_object
else:
    cam.location = {location}
    cam.rotation_euler = {rotation}

cam.data.lens = {focal_length}
bpy.context.scene.camera = cam

_result["data"]["camera_name"] = cam.name
_result["data"]["location"] = list(cam.location)
_result["data"]["focal_length"] = cam.data.lens
"""
        return self.execute_script(script)

    def setup_lighting(
        self,
        light_type: str = "SUN",
        location: tuple = (5, 5, 10),
        energy: float = 5.0,
        color: tuple = (1.0, 1.0, 1.0),
    ) -> BlenderResult:
        """Set up scene lighting."""
        script = f"""
bpy.ops.object.light_add(type="{light_type}", location={location})
light = bpy.context.active_object
light.data.energy = {energy}
light.data.color = {color}

_result["data"]["light_name"] = light.name
_result["data"]["light_type"] = "{light_type}"
_result["data"]["energy"] = {energy}
"""
        return self.execute_script(script)

    def add_modifier(
        self,
        object_name: str,
        modifier_type: str,
        settings: Optional[Dict[str, Any]] = None,
    ) -> BlenderResult:
        """
        Add a modifier to an object.

        Args:
            object_name: Target object name
            modifier_type: Modifier type (SUBSURF, MIRROR, ARRAY, SOLIDIFY, etc.)
            settings: Dict of modifier settings
        """
        settings_script = ""
        if settings:
            for key, value in settings.items():
                if isinstance(value, str):
                    settings_script += f'    mod.{key} = "{value}"\n'
                else:
                    settings_script += f"    mod.{key} = {value}\n"

        script = f"""
obj = bpy.data.objects.get("{object_name}")
if obj is None:
    _result["success"] = False
    _result["error"] = "Object '{object_name}' not found"
else:
    mod = obj.modifiers.new(name="{modifier_type}", type="{modifier_type}")
{settings_script}
    _result["data"]["modifier_name"] = mod.name
    _result["data"]["modifier_type"] = "{modifier_type}"
    _result["data"]["object_name"] = obj.name
"""
        return self.execute_script(script)

    def export_scene(
        self,
        filepath: Optional[str] = None,
        file_format: str = "glb",
    ) -> BlenderResult:
        """
        Export scene to various formats.

        Args:
            filepath: Output file path
            file_format: Export format (glb, gltf, obj, fbx, stl)
        """
        if filepath is None:
            filepath = os.path.join(self.output_dir, f"export.{file_format}")

        export_ops = {
            "glb": f'bpy.ops.export_scene.gltf(filepath="{filepath}", export_format="GLB")',
            "gltf": f'bpy.ops.export_scene.gltf(filepath="{filepath}", export_format="GLTF_SEPARATE")',
            "obj": f'bpy.ops.wm.obj_export(filepath="{filepath}")',
            "fbx": f'bpy.ops.export_scene.fbx(filepath="{filepath}")',
            "stl": f'bpy.ops.export_mesh.stl(filepath="{filepath}")',
        }

        if file_format not in export_ops:
            return BlenderResult(
                success=False,
                output="",
                error=f"Unknown format: {file_format}. Available: {list(export_ops.keys())}",
            )

        script = f"""
{export_ops[file_format]}
_result["data"]["filepath"] = "{filepath}"
_result["data"]["format"] = "{file_format}"
"""
        return self.execute_script(script)
