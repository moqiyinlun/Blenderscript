import bpy
import os

# 设置输出目录

def enable_normal_render(base_path="output"):
    bpy.context.scene.use_nodes = True
    bpy.data.scenes["Scene"].view_layers["ViewLayer"].use_pass_normal = True
    bpy.data.scenes["Scene"].view_layers["ViewLayer"].pass_alpha_threshold = 0

    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links
    nodes.clear()

    render_node = nodes.new("CompositorNodeRLayers")

    output_node1 = nodes.new("CompositorNodeSepRGBA")
    links.new(render_node.outputs["Normal"], output_node1.inputs[0])

    def map_range_node(input_socket):
        node = nodes.new("CompositorNodeMapRange")
        node.inputs[1].default_value = -1
        node.inputs[2].default_value = 1
        node.inputs[3].default_value = 0
        node.inputs[4].default_value = 1
        node.use_clamp = True
        links.new(input_socket, node.inputs[0])
        return node

    nodeR = map_range_node(output_node1.outputs[0])
    nodeG = map_range_node(output_node1.outputs[1])
    nodeB = map_range_node(output_node1.outputs[2])

    # combine RGB
    comb_node = nodes.new("CompositorNodeCombRGBA")
    links.new(nodeR.outputs[0], comb_node.inputs[0])
    links.new(nodeG.outputs[0], comb_node.inputs[1])
    links.new(nodeB.outputs[0], comb_node.inputs[2])

    # output file
    output_node = nodes.new("CompositorNodeOutputFile")
    output_node.format.file_format = "PNG"
    output_node.base_path = base_path
    links.new(comb_node.outputs[0], output_node.inputs[0])
    
    return output_node


def render_all_camera_normals(output_base_path="normal_outputs"):
    os.makedirs(output_base_path, exist_ok=True)
    
    scene = bpy.context.scene
    original_camera = scene.camera
    
    for cam in [obj for obj in bpy.data.objects if obj.type == 'CAMERA']:
        print(f"Rendering normal map for camera: {cam.name}")
        scene.camera = cam
        output_node = enable_normal_render(base_path=output_base_path)
        output_node.file_slots[0].path = f"{cam.name}"
        bpy.ops.render.render(write_still=True)
    scene.camera = original_camera


def config_cycles_and_gpu():
    bpy.data.scenes[0].render.engine = "CYCLES"
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    bpy.data.scenes["Scene"].cycles.preview_samples = 32
    bpy.data.scenes["Scene"].cycles.samples = 64
    bpy.data.scenes["Scene"].cycles.use_denoising = False

output_dir = r"D:\blenderdata\normal_renders"
os.makedirs(output_dir, exist_ok=True)
config_cycles_and_gpu()
render_all_camera_normals(output_dir)