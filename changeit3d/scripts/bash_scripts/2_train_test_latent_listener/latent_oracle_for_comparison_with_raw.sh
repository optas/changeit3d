##
## Default arguments per in_out/arguments/parse_train_test_latent_listener_arguments will be used
## 

### This is not used in our paper. However, it is meant to compare the performance of the end2end pointclouds-based oracle 
### new listener, against a listener trained on the same train/test splits but using a latent representation. 
### The expectation is that the end2end oracle should be better performing.

script_file=/home/panos/Git_Repos/changeit3d/changeit3d/scripts/train_test_latent_listener.py

top_data_dir=/home/panos/Git_Repos/changeit3d/changeit3d/data
shape_talk_file=$top_data_dir/shapetalk/language/misc/shapetalk_preprocessed_public_utters_for_listening_oracle_version_0.csv
vocab_file=$top_data_dir/shapetalk/language/vocabulary.pkl

random_seed=2023
top_log_dir=$top_data_dir/pretrained/listeners/oracle_listener/latent_based/all_shapetalk_classes/rs_$random_seed/single_utter/transformer_based


## 1. PC-AE, trained with pointclouds scaled to be aligned with rendering images
latents=$top_data_dir/pretrained/shape_latents/pcae_latent_codes.pkl
log_dir=$top_log_dir/latent_pcae_based

if false
then
  python $script_file\
   -latent_codes_file $latents\
   -shape_talk_file $shape_talk_file\
   -vocab_file $vocab_file\
   --log_dir $log_dir\
   --use_timestamp False\
   --random_seed $random_seed\
   --gpu 1
fi

## 2. SGF based (gradient) latents
latents=$top_data_dir/pretrained/shape_latents/sgf_latent_codes.pkl
log_dir=$top_log_dir/latent_sgf_based

if false
then
  python $script_file\
   -latent_codes_file $latents\
   -shape_talk_file $shape_talk_file\
   -vocab_file $vocab_file\
   --log_dir $log_dir\
   --use_timestamp False\
   --random_seed $random_seed\
   --gpu 0
fi

## 3. ImNet based (implicit) latents
latents=$top_data_dir/pretrained/shape_latents/imnet_latent_codes.pkl
log_dir=$top_log_dir/latent_imnet_based

if false
then
  python $script_file\
   -latent_codes_file $latents\
   -shape_talk_file $shape_talk_file\
   -vocab_file $vocab_file\
   --log_dir $log_dir\
   --use_timestamp False\
   --random_seed $random_seed\
   --gpu 0
fi



