import bpy
import math

def create_cameras_around_object(target_object, radius=5.0, num_cameras_per_circle=10, num_circles=5, height_step=1.0):
    """
    Creates cameras in circles around a target object.

    Parameters:
    - target_object: the object around which the cameras will be created.
    - radius: distance of the cameras from the object.
    - num_cameras_per_circle: number of cameras in one circle.
    - num_circles: number of circles of cameras.
    - height_step: the vertical distance between two circles.
    """

    # Calculate angles between cameras in one circle
    angle_step = 2 * math.pi / num_cameras_per_circle
    
    for circle in range(num_circles):
        for i in range(num_cameras_per_circle):
            # Polar coordinates to Cartesian
            x = target_object.location.x + radius * math.cos(i * angle_step)
            y = target_object.location.y + radius * math.sin(i * angle_step)
            z = target_object.location.z + circle * height_step
            
            # Create a new camera
            bpy.ops.object.camera_add(location=(x, y, z))
            cam = bpy.context.object
            cam.name = f"Cam_Circle_{circle+1}_Num_{i+1}"
            
            # Make the camera look at the target object
            direction = target_object.location - cam.location
            # Point the camera to the target
            cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

    # Update the scene
    bpy.context.view_layer.update()

target_obj = bpy.context.scene.objects["Suzanne"]  # Replace 'YourTargetObjectName' with your object's name
create_cameras_around_object(target_obj)