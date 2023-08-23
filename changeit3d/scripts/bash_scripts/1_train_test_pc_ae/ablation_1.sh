### below we are using the default (per argparse) parameters for
### --encoder_conv_layers and --decoder_fc_neurons to train the AE

code_top_dir=/home/panos/Git_Repos/changeit3d/changeit3d                 # replace with your own
data_top_dir=/home/panos/Git_Repos/changeit3d/changeit3d/data

random_seed=2022
n_pc_points=4096
gpu_id=0

# below assumes you have downloaded the ShapeTalk data (pointclouds/scaled_to_align_rendering)
python_script=$code_top_dir/scripts/train_test_pc_ae.py
log_dir=$data_top_dir/experiments/pc_autoencoders
split_file=$data_top_dir/shapetalk/language/misc/unary_split_rs_$random_seed.csv
pc_top_dir=$data_top_dir/shapetalk/point_clouds/scaled_to_align_rendering            
experiment_tag=pointnet/rs_$random_seed/points_$n_pc_points/all_classes/scaled_to_align_rendering


python $python_script\
  -log_dir $log_dir\
  -data_dir $pc_top_dir\
  -split_file $split_file\
  --encoder_net pointnet\
  --batch_size 32\
  --n_pc_points $n_pc_points\
  --random_seed $random_seed\
  --gpu $gpu_id\
  --experiment_tag $experiment_tag
