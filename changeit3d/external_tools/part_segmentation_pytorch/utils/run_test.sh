homedir=/home/ir0/data/language_faders/language_faders/external/language_changes/language_changes/external_tools/part_segmentation_pytorch/
dataset=$homedir'data'
all_objects='Chair Lamp Table Knife Laptop Mug'

for object in $all_objects; do
    printf "\n\n\n\n\n!!!!!!!!!!!!!!!!!!\nTesting $object\n\n\n"
    CUDA_VISIBLE_DEVICES=0 python test_segmentation.py \
        --dataset $dataset \
        --class_choice $object \
        --vis
done