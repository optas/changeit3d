### below we are using the default parameters for
### --encoder_conv_layers and --decoder_fc_neurons

python_script=/home/panos/Git_Repos/changeit3d/changeit3d/scripts/train_test_pc_ae.py
split_file=/home/panos/Git_Repos/changeit3d/changeit3d/data/shapetalk/language/misc/unary_split_rs_2022.csv
pc_top_dir=/home/panos/Git_Repos/changeit3d/changeit3d/data/shapetalk/point_clouds/scaled_to_align_rendering
pretrained_model_file=/home/panos/Git_Repos/changeit3d/changeit3d/data/pretrained/pc_autoencoders/pointnet/rs_2022/points_4096/all_classes/scaled_to_align_rendering/08-07-2022-22-23-42/best_model.pt
log_dir=./test_pc_ae

random_seed=2022
n_pc_points=4096

python $python_script\
  -log_dir $log_dir\
  -data_dir $pc_top_dir\
  -split_file $split_file\
  --encoder_net pointnet\
  --batch_size 32\
  --n_pc_points $n_pc_points\
  --random_seed $random_seed\
  --gpu 0\
  --do_training False\
  --load_pretrained_model True\
  --pretrained_model_file $pretrained_model_file\
  --extract_latent_codes False