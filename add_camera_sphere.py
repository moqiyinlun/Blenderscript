import bpy
import math

def create_cameras_on_sphere(target_object, radius=1.0, num_cameras_longitude=20, num_cameras_latitude=1):
    """
    Creates cameras on a sphere around a target object.

    Parameters:
    - target_object: the object around which the cameras will be created.
    - radius: distance of the cameras from the object.
    - num_cameras_longitude: number of cameras along the longitude of the sphere.
    - num_cameras_latitude: number of cameras along the latitude of the sphere.
    """

    # Calculate angles between cameras
    long_angle_step = 2 * math.pi / num_cameras_longitude
    lat_angle_step = math.pi / (num_cameras_latitude + 1)  # +1 to avoid camera on poles

    for lat in range(1, num_cameras_latitude + 1):
        for long in range(num_cameras_longitude):
            theta = long * long_angle_step  # Longitude angle
            phi = lat * lat_angle_step  # Latitude angle
            
            # Spherical coordinates to Cartesian
            x = target_object.location.x + radius * math.sin(phi) * math.cos(theta)
            y = target_object.location.y + radius * math.sin(phi) * math.sin(theta)
            z = target_object.location.z + radius * math.cos(phi)
            
            # Create a new camera
            bpy.ops.object.camera_add(location=(x, y, z))
            cam = bpy.context.object
            cam.name = f"Cam_Lat_{lat}_Long_{long}"
            
            # Make the camera look at the target object
            direction = target_object.location - cam.location
            cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

    # Update the scene
    bpy.context.view_layer.update()

target_obj = bpy.context.scene.objects["Cube.001"]
create_cameras_on_sphere(target_obj)