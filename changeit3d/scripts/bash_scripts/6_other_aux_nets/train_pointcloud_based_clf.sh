##
## Last trained and publicly shared model has a test accuracy of 90.3% on the underlying (unary) train/test split of ShapeTalk.
##
## Note. We use the default listed arguments to build this clf. See: scripts/train_test_pc_clf.py

script_file=/home/panos/Git_Repos/changeit3d/changeit3d/scripts/train_test_pc_clf.py
top_pc_dir=/home/panos/Git_Repos/changeit3d/changeit3d/data/shapetalk/point_clouds/scaled_to_align_rendering
random_seed=2022
split_file=/home/panos/Git_Repos/changeit3d/changeit3d/data/shapetalk/language/misc/unary_split_rs_"$random_seed".csv
log_dir=/home/panos/Git_Repos/changeit3d/changeit3d/data/pretrained/pc_classifiers/rs_$random_seed/all_shapetalk_classes

python $script_file\
    -data_dir $top_pc_dir\
    -split_file $split_file\
    --log_dir $log_dir\
    --random_seed $random_seed\
    --gpu 2\
    --use_timestamp False