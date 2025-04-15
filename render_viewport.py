import bpy
import os

def setup_opengl_render_high_quality():
    bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.image_settings.color_depth = '16'
    bpy.context.scene.render.film_transparent = True  

    eevee = bpy.context.scene.eevee
    eevee.taa_render_samples = 256
    for area in bpy.context.window.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':

                    space.shading.type = 'MATERIAL'
                    space.shading.use_scene_lights = True
                    space.shading.use_scene_world = True
                    space.shading.render_pass = 'COMBINED'
                    space.shading.color_type = 'MATERIAL'
#                    space.shading.background_type = 'TRANSPARENT'

                    space.overlay.show_overlays = False
                    space.show_gizmo = False
                    space.overlay.show_floor = False
                    space.overlay.show_axis_x = False
                    space.overlay.show_axis_y = False
                    space.overlay.show_cursor = False
                    space.overlay.show_outline_selected = False
                    space.overlay.show_object_origins = False
def set_3d_view_camera(cam):
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.region_3d.view_perspective = 'CAMERA'
                        space.camera = cam
def render_viewport_sequence(output_path, camera_path):
    setup_opengl_render_high_quality()
    for cam in camera_path:
        bpy.context.scene.camera = cam
        set_3d_view_camera(cam)
        bpy.context.view_layer.update()
        bpy.context.scene.frame_set(0)

        filepath = os.path.join(output_path, f"{cam.name}.png")
        bpy.context.scene.render.filepath = filepath

        bpy.ops.render.opengl(write_still=True)
        print(f"Rendered {filepath}")
 

camera_list = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']


output_directory = r"C:\moqiyinlun\3DPrinter\Bike2"

render_viewport_sequence(output_directory, camera_list)
