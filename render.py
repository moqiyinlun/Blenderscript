import bpy
import math

def setup_transparent_render():
    # Set the render engine to 'CYCLES' (you can change this to 'EEVEE' if needed)
    bpy.context.scene.render.engine = 'CYCLES'
    
    # Set the output format to 'PNG'
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_depth = '16'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'  # Make sure alpha is included
    
    # Enable transparency in the film settings
    bpy.context.scene.render.film_transparent = True
def render_sequence_with_all_cameras(start_frame, end_frame, output_path):
    setup_transparent_render()
    
    for frame in range(start_frame, end_frame + 1):
        bpy.context.scene.frame_set(frame)
        for obj in bpy.data.objects:
            if obj.type == 'CAMERA':
                bpy.context.scene.camera = obj
                bpy.context.scene.render.filepath = f"{output_path}/{obj.name}_frame_{frame}.png"
                bpy.ops.render.render(write_still=True)

# Set your start and end frames, and the directory where you want to save the renders.
start_frame = 1
end_frame = 100
output_directory = r"D:\blenderdata\monkey_fire"  # Replace with your desired directory path

render_sequence_with_all_cameras(start_frame, end_frame, output_directory)