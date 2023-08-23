"""
Python script that can be used by blender's engine to render a pointcloud.

Example usage:
    <blender binary> --python_file <this_file> --background <a .blend file> -- -pointcloud_file
    <a file storing the pointcloud> (see argparse below for more options)

    realization:
    /snap/blender/current/blender --background ~/Git_Repos/from_others/blender_render/blender_files/point_clouds.blend
    --python ./blender_pointcloud.py -- -pointcloud_file ~/DATA/OUT/temp_pc.npy --out_file ~/DATA/OUT/temp_rend.png
    --swap_yz_axes True --default_color 223 254 128 --azimuth_phi 90

NOTE. Tested with blender blender-2.93.4-linux-x64
Installed on a linux box with 16.04.7 LTS (Xenial Xerus) like this:
    sudo snap install blender --channel=2.93lts/stable --classic

Code modified from:
    https://github.com/ianhuang0630/blender_render/blob/main/blender_pointcloud.py


TODO: Fix that when you do a trajectory p020_t292_r005, p020_t315_r005 add a BAD reflection. (suspect: adapt .blend)
"""

import sys
import bpy
import bmesh
import argparse
import numpy as np
import os.path as osp


# This python script is expected to run from the python interpreter associated with the installed blender, so we are in
# the path this package explicitly:
sys.path.append('/home/panos/Git_Repos/changeit3d')
from changeit3d.external_tools.blender_based_visualization.camera_handling import obj_centered_camera_pos, camPosToQuaternion
from changeit3d.external_tools.blender_based_visualization.utils import str2bool, check_rgb_value_type
from changeit3d.external_tools.blender_based_visualization.utils import (swap_axes_of_pointcloud,
                                                                        rotate_x_axis_by_degrees,
                                                                        rotate_y_axis_by_degrees,
                                                                        rotate_z_axis_by_degrees,
                                                                        center_in_unit_sphere)

from changeit3d.external_tools.blender_based_visualization.utils import trim_content_after_last_dot, create_dir


##############################
# Handling Hyper-parameters
##############################
parser = argparse.ArgumentParser(description='Hyper parameters for using Blender to render pointclouds.')

parser.add_argument('-pointcloud_file', type=str, required=True, help='numpy file storing a pointcloud with N points and shape'
                                                                      '(N,3) or (N,6). If 6 dimensions, the last 3 are color'
                                                                      'values.')

parser.add_argument('--point_radius', type=float, default=0.015, help='size of each rendered point')

parser.add_argument('--x_rotation', type=float, default=0, help='degrees to rotate clockwise the pointcloud in '
                                                                'x-dimension')

parser.add_argument('--y_rotation', type=float, default=0, help='degrees to rotate clockwise the pointcloud in '
                                                                'y-dimension')

parser.add_argument('--z_rotation', type=float, default=0, help='degrees to rotate clockwise the pointcloud in '
                                                                'z-dimension')

parser.add_argument('--swap_yz_axes', type=str2bool, default=False, help='swap axes of input pointclouds')

parser.add_argument('--default_color', type=int, nargs=3, default=[128, 128, 128],
                    help='RGB values, to use for all points if no explicit per point color is passed in the '
                         '-pointcloud_file')


parser.add_argument('--center_in_unit_sphere', type=str2bool, default=False, help='place/stretch the pointcloud to '
                                                                                  'have diameter 1, and be centered '
                                                                                  'around the axis.')

parser.add_argument('--scale_up_or_down', type=float, default=1.0, help='multiply input pointcloud by this value. '
                                                                        'if >1 or <1 the effect is '
                                                                        'zooming-in or zooming-out')

parser.add_argument('--out_file', type=str, default='pc_rendered.png', help='filename to save the resulting image')

parser.add_argument('--render_trajectory', type=str2bool, default=False, help='generate more images, around the '
                                                                              'input pointcloud')

args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])  # you need this hack, see:
                                                               # https://www.linuxtut.com/en/9b6ddacd451d8e24bf72/
check_rgb_value_type(args.default_color)


########################
# Load pointcloud data
########################

pts = np.load(args.pointcloud_file)
pts_shape = pts.shape[1]

if pts_shape == 6:
    pt_colors = pts[:, 3:]
    pts = pts[:, :3]
elif pts_shape == 3:
    pt_colors = (np.array(args.default_color) / 255) * np.ones_like(pts)  # use default_color
else:
    raise ValueError('Input pointcloud should have 3 or 6 trailing dimensions.')

##
# Optionally transform the pointcloud
##

if args.swap_yz_axes:
    pts = swap_axes_of_pointcloud(pts, [0, 2, 1])

if args.x_rotation != 0:
    pts = rotate_x_axis_by_degrees(pts, args.x_rotation)

if args.y_rotation != 0:
    pts = rotate_y_axis_by_degrees(pts, args.y_rotation)

if args.z_rotation != 0:
    pts = rotate_z_axis_by_degrees(pts, args.z_rotation)

if args.center_in_unit_sphere:
    pts = center_in_unit_sphere(pts) # TODO see normalize_pointcloud (i.e., is it OK? shall we use bbox normalization to make things work with camera_handling.py)?
    # pts = normalize_pointcloud(pts, norm="bbox")

if args.scale_up_or_down != 1:
    pts *= args.scale_up_or_down


##############################################
# Start blender manipulations to render.
##############################################

if 'object' in bpy.data.objects:
    bpy.data.objects.remove(bpy.data.objects['object'], do_unlink=True)

if 'sphere' in bpy.data.meshes:
    bpy.data.meshes.remove(bpy.data.meshes['sphere'], do_unlink=True)

sphere_mesh = bpy.data.meshes.new('sphere')
sphere_bmesh = bmesh.new()
bmesh.ops.create_icosphere(sphere_bmesh, subdivisions=2, diameter=args.point_radius*2)
sphere_bmesh.to_mesh(sphere_mesh)
sphere_bmesh.free()

sphere_verts = np.array([[v.co.x, v.co.y, v.co.z] for v in sphere_mesh.vertices])
sphere_faces = np.array([[p.vertices[0], p.vertices[1], p.vertices[2]] for p in sphere_mesh.polygons])

verts = (np.expand_dims(sphere_verts, axis=0) + np.expand_dims(pts, axis=1)).reshape(-1, 3)
faces = (np.expand_dims(sphere_faces, axis=0) +
         (np.arange(pts.shape[0]) * sphere_verts.shape[0]).reshape(-1, 1, 1)).reshape(-1, 3)

vert_colors = np.repeat(pt_colors, sphere_verts.shape[0], axis=0).astype(dtype='float64')
vert_colors = vert_colors[faces.reshape(-1), :]


min_z = verts.min(axis=0)[2]
verts[:, 2] -= min_z  # object minimum z is 0.0   ( # TODO -check consequences for render_trajectory)
# verts = center_in_unit_sphere(verts)
# print(verts.shape, faces.shape, vert_colors.shape)

verts = verts.tolist()
faces = faces.tolist()
vert_colors = vert_colors.tolist()

scene = bpy.context.scene
mesh = bpy.data.meshes.new('object')

mesh.from_pydata(verts, [], faces)
mesh.validate()
mesh.vertex_colors.new(name='Col')  # named 'Col' by default
mesh_vert_colors = mesh.vertex_colors['Col']

for i, c in enumerate(mesh.vertex_colors['Col'].data):
    c.color = vert_colors[i] + [1.0]

obj = bpy.data.objects.new('object', mesh)
obj.data.materials.append(bpy.data.materials['sphere_material'])

bpy.context.collection.objects.link(obj)
scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = args.out_file
bpy.ops.render.render(write_still=True)


if args.render_trajectory:
    out_file_basis = trim_content_after_last_dot(args.out_file)
    out_folder = create_dir(out_file_basis)

    # TODO you could customize those externally
    rhos = [5]  # distance from object
    theta_delta_deg = 22.5
    theta_degs = [x * theta_delta_deg for x in range(0, 16)]
    phi_degs = [20]

    combinations = []
    for rho in rhos:
        for phi_deg in phi_degs:
            for theta_deg in theta_degs:
                combinations.append((rho, phi_deg, theta_deg))
    # combinations.append((5, 80, 0))  # top view

    camera = bpy.data.objects['Camera']
    for rho, phi_deg, theta_deg in combinations:
        cx, cy, cz = obj_centered_camera_pos(rho, phi_deg, theta_deg)
        q = camPosToQuaternion(cx, cy, cz)
        camera.location[0] = cx
        camera.location[1] = cy
        camera.location[2] = cz
        camera.rotation_mode = 'QUATERNION'
        camera.rotation_quaternion[0] = q[0]
        camera.rotation_quaternion[1] = q[1]
        camera.rotation_quaternion[2] = q[2]
        camera.rotation_quaternion[3] = q[3]

        out_file = osp.join(out_folder, 'p%03d_t%03d_r%03d.png' % (phi_deg, theta_deg, rho))
        scene.render.filepath = out_file
        bpy.ops.render.render(write_still=True)
