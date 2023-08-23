##
## Default arguments per in_out/arguments/parse_train_test_latent_listener_arguments will be used
## 

script_file=/home/panos/Git_Repos/changeit3d/changeit3d/scripts/train_test_latent_listener.py

top_data_dir=/home/panos/Git_Repos/changeit3d/changeit3d/data
shape_talk_file=$top_data_dir/shapetalk/language/misc/shapetalk_preprocessed_public_merged_utters_version_0.csv
vocab_file=$top_data_dir/shapetalk/language/vocabulary.pkl

random_seed=2022
top_log_dir=$top_data_dir/pretrained/listeners/all_shapetalk_classes/rs_$random_seed/multi_utter/transformer_based
gpu_id=0


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
   --gpu $gpu_id
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
   --gpu $gpu_id
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
   --gpu $gpu_id
fi


## 4. ResNet34/101 based (image) latents
# we increase the weight-decay for these, since these are 512D (or 2048D) instead of 256D 
resnet=101
latents=$top_data_dir/pretrained/shape_latents/resnet"$resnet"_latent_codes.pkl
log_dir=$top_log_dir/latent_resnet"$resnet"_based

if false
then
  python $script_file\
   -latent_codes_file $latents\
   -shape_talk_file $shape_talk_file\
   -vocab_file $vocab_file\
   --log_dir $log_dir\
   --use_timestamp False\
   --random_seed $random_seed\
   --gpu $gpu_id\
   --weight_decay 0.005
fi


## 5. OpenAI CLIP
latents=$top_data_dir/pretrained/shape_latents/openai_clip-vit-large-patch14_latent_codes.pkl
log_dir=$top_log_dir/latent_openai_clip-vit-large-patch14_based

if true
then
  python $script_file\
   -latent_codes_file $latents\
   -shape_talk_file $shape_talk_file\
   -vocab_file $vocab_file\
   --log_dir $log_dir\
   --use_timestamp False\
   --random_seed $random_seed\
   --gpu $gpu_id\
   --weight_decay 0.003
fi


## 6. Open-CLIP
latents=$top_data_dir/pretrained/shape_latents/laion_CLIP-ViT-H-14-laion2B-s32B-b79K_latent_codes.pkl
log_dir=$top_log_dir/latent_laion_CLIP-ViT-H-14-laion2B-s32B-b79K_based


if false
then
  python $script_file\
   -latent_codes_file $latents\
   -shape_talk_file $shape_talk_file\
   -vocab_file $vocab_file\
   --log_dir $log_dir\
   --use_timestamp False\
   --random_seed $random_seed\
   --gpu $gpu_id\
   --weight_decay 0.003
fi