script_file=/home/panos/Git_Repos/changeit3d/changeit3d/scripts/train_change_it_3d.py

top_data_dir=/home/panos/Git_Repos/changeit3d/changeit3d/data
shape_talk_file=$top_data_dir/shapetalk/language/shapetalk_preprocessed_public_version_0.csv
vocab_file=$top_data_dir/shapetalk/language/vocabulary.pkl

latent_backbone=pcae
random_seed=2022
gpu_id=1
ablations=('decoupling_mag_direction' 'coupled')
latent_codes_file=$top_data_dir/pretrained/shape_latents/"$latent_backbone"_latent_codes.pkl
pretrained_listener_file=$top_data_dir/pretrained/listeners/all_shapetalk_classes/rs_"$random_seed"/single_utter/transformer_based/latent_"$latent_backbone"_based/best_model.pkl
top_log_dir=$top_data_dir/pretrained/changers/"$latent_backbone"_based/all_shapetalk_classes


for identity_penalty in 0.01 0.025 0.05 0.075 0.1
do
  for net_ablation in ${ablations[@]};
  do    
    for self_contrast in True False
    do            
      experiment_tag=$net_ablation/idpen_"$identity_penalty"_sc_$self_contrast
      echo $experiment_tag
      
      python $script_file\
      -shape_talk_file $shape_talk_file\
      -vocab_file $vocab_file\
      -latent_codes_file $latent_codes_file\
      -pretrained_listener_file $pretrained_listener_file\
      --log_dir $top_log_dir\
      --random_seed $random_seed\
      --gpu $gpu_id\
      --shape_editor_variant $net_ablation\
      --identity_penalty $identity_penalty\
      --experiment_tag $experiment_tag\
      --self_contrast $self_contrast\
      --use_timestamp False      
    done
  done
done