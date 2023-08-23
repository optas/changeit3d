blender_binary=/snap/blender/current/blender
blender_background=~/Git_Repos/from_others/blender_render/blender_files/point_clouds.blend
top_input_pc_dir=~/DATA/OUT/scribble2pcd/data_for_visualization_5

for example_name in 1_1 3_1 5_3
do
  $blender_binary\
  --background $blender_background\
  --python ./blender_pointcloud.py\
  -- -pointcloud_file $top_input_pc_dir/$example_name.npy\
  --out_file $top_input_pc_dir/$example_name.png\
  --swap_yz_axes False\
  --x_rotation 270\
  --z_rotation 180\
  --center_in_unit_sphere True\
  --render_trajectory True\
  --point_radius 0.010
done



