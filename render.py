import bpy

def setup_transparent_render():
    # Set the render engine to 'CYCLES' 
    bpy.context.scene.render.engine = 'CYCLES'
    
    # Set the output format to 'PNG'
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_depth = '16'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'  # Make sure alpha is included
    # Optimize render settings
    cycles = bpy.context.scene.cycles
    cycles.samples = 64  # Reduce the number of samples. Adjust according to your needs.
    cycles.use_adaptive_sampling = True  # Use adaptive sampling
    cycles.adaptive_threshold = 0.005  # Adjust according to your needs.
    # Enable transparency in the film settings
    bpy.context.scene.render.film_transparent = True


    # Set Cycles to use GPU
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.preferences.addons['cycles'].preferences.get_devices()
    # Ensure that the CUDA compute device type is used (for NVIDIA cards)
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    
    # Enable the GPU device (assuming only one GPU is available)
    bpy.context.preferences.addons['cycles'].preferences.devices[0].use = True

def render_sequence_with_all_cameras(start_frame, end_frame, output_path,camera_path):
    setup_transparent_render()
    
    for frame in range(start_frame, end_frame + 1):
        bpy.context.scene.frame_set(frame)
        for obj in camera_path:
            bpy.context.scene.camera = obj
            bpy.context.scene.render.filepath = f"{output_path}/{frame}/{obj.name}.png"
            print(f"{output_path}/{frame}/{obj.name}.png")
            bpy.ops.render.render(write_still=True)

# Set your start and end frames, and the directory where you want to save the renders.
camera_list = []
for obj in bpy.data.objects:
    if obj.type == 'CAMERA':
        camera_list.append(obj)
print("finish")
start_frame = 1
end_frame = 20
output_directory = r"D:\blenderdata\monkey_fire"

render_sequence_with_all_cameras(start_frame, end_frame, output_directory,camera_list)