script_file=/home/panos/Git_Repos/changeit3d/changeit3d/scripts/train_test_monolithic_changeit3d_baseline.py
top_data_dir=/home/panos/Git_Repos/changeit3d/changeit3d/data
shape_talk_file=$top_data_dir/shapetalk/language/misc/shapetalk_preprocessed_public_merged_utters_version_0.csv
vocab_file=$top_data_dir/shapetalk/language/vocabulary.pkl
top_pc_dir=$top_data_dir/shapetalk/pointclouds/scaled_to_align_rendering


log_dir=$top_data_dir/pretrained/changers/monolithic_ablation/with_pointclouds/chair_table_lamp_only
if true
then
    python $script_file\
        -shape_talk_file $shape_talk_file\
        -vocab_file $vocab_file\
        -top_pc_dir $top_pc_dir\
        --log_dir $log_dir\
        --restrict_shape_class chair table lamp
fi

if true
then
    log_dir=$log_dir/results
    python $script_file\
        -shape_talk_file $shape_talk_file\
        -vocab_file $vocab_file\
        -top_pc_dir $top_pc_dir\
        --test True\
        --train False\
        --log_dir $log_dir\
        --restrict_shape_class chair table lamp\
        --pretrained_model $top_data_dir/pretrained/changers/monolithic_ablation/with_pointclouds/chair_table_lamp_only/best_model.pt\
        --pretrained_shape_classifier $top_data_dir/pretrained/pc_classifiers/rs_2022/all_shapetalk_classes/best_model.pkl\
        --shape_part_classifiers_top_dir $top_data_dir/pretrained/part_predictors/shapenet_core_based\
        --pretrained_oracle_listener $top_data_dir/pretrained/listeners/oracle_listener/all_shapetalk_classes/rs_2023/listener_dgcnn_based/ablation1/best_model.pkl
fi

# train with all shape classes
log_dir=$top_data_dir/pretrained/changers/monolithic_ablation/with_pointclouds/all_shapetalk_classes
if false
then
    python $script_file\
        -shape_talk_file $shape_talk_file\
        -vocab_file $vocab_file\
        -top_pc_dir $top_pc_dir\
        --log_dir $log_dir 
fi


if true
then
    log_dir=$log_dir/results
    python $script_file\
        -shape_talk_file $shape_talk_file\
        -vocab_file $vocab_file\
        -top_pc_dir $top_pc_dir\
        --test True\
        --train False\
        --log_dir $log_dir\
        --restrict_shape_class chair table lamp\
        --pretrained_model $top_data_dir/pretrained/changers/monolithic_ablation/with_pointclouds/all_shapetalk_classes/best_model.pt\
        --pretrained_shape_classifier $top_data_dir/pretrained/pc_classifiers/rs_2022/all_shapetalk_classes/best_model.pkl\
        --shape_part_classifiers_top_dir $top_data_dir/pretrained/part_predictors/shapenet_core_based\
        --pretrained_oracle_listener $top_data_dir/pretrained/listeners/oracle_listener/all_shapetalk_classes/rs_2023/listener_dgcnn_based/ablation1/best_model.pkl
fi
