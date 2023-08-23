##
## Default arguments per in_out/arguments/parse_train_test_raw_listener_arguments will be used
## 

random_seed=2023
script_file=/home/panos/Git_Repos/changeit3d/changeit3d/scripts/train_test_raw_pc_listener.py
top_data_dir=/home/panos/Git_Repos/changeit3d/changeit3d/data
shape_talk_file=$top_data_dir/shapetalk/language/misc/shapetalk_preprocessed_public_utters_for_listening_oracle_version_0.csv
vocab_file=$top_data_dir/shapetalk/language/vocabulary.pkl
top_pc_dir=$top_data_dir/shapetalk/point_clouds/scaled_to_align_rendering
top_log_dir=$top_data_dir/pretrained/listeners/oracle_listener/all_shapetalk_classes/rs_$random_seed/listener_dgcnn_based/ablation1/

if true
then
  python $script_file\
   -top_raw_pc_dir $top_pc_dir\
   -shape_talk_file $shape_talk_file\
   -vocab_file $vocab_file\
   --log_dir $top_log_dir\
   --use_timestamp False\
   --random_seed $random_seed\
   --gpu 0\
   --weight_decay 0.0001\
   --batch_size 96\
   --listening_model dgcnn-ablation1   
fi
