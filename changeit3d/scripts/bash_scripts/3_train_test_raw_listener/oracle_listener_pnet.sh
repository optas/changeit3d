##
## Default arguments per in_out/arguments/parse_train_test_raw_listener_arguments will be used
## 

random_seed=2023
script_file=/home/ubuntu/Git_Repos/changeit3d/changeit3d/scripts/train_test_raw_pc_listener.py
top_data_dir=/home/ubuntu/Git_Repos/changeit3d/changeit3d/data
shape_talk_file=$top_data_dir/shapetalk/language/misc/shapetalk_preprocessed_public_utters_for_listening_oracle_version_0.csv
vocab_file=$top_data_dir/shapetalk/language/vocabulary.pkl
top_pc_dir=$top_data_dir/shapetalk/point_clouds/scaled_to_align_rendering

top_log_dir=$top_data_dir/pretrained/listeners/oracle_listener/all_shapetalk_classes/rs_$random_seed/listener_pnet_based/ablation1

if true # first attempt
then
  python $script_file\
   -top_raw_pc_dir $top_pc_dir\
   -shape_talk_file $shape_talk_file\
   -vocab_file $vocab_file\
   --log_dir $top_log_dir\
   --use_timestamp False\
   --random_seed $random_seed\
   --gpu 0\
   --listening_model pointnet-ablation1
fi

top_log_dir=$top_data_dir/pretrained/listeners/oracle_listener/all_shapetalk_classes/rs_$random_seed/listener_pnet_based/ablation1/more_training_epochs
if false # attempt to train for more epochs the above pre-trained system to see if performance improves
then
  python $script_file\
   -top_raw_pc_dir $top_pc_dir\
   -shape_talk_file $shape_talk_file\
   -vocab_file $vocab_file\
   --log_dir $top_log_dir\
   --use_timestamp False\
   --random_seed $random_seed\
   --gpu 0\
   --listening_model pointnet-ablation1\
   --pretrained_model_file $top_data_dir/pretrained/listeners/oracle_listener/all_shapetalk_classes/rs_$random_seed/listener_pnet_based/ablation1/best_model.pt
fi
