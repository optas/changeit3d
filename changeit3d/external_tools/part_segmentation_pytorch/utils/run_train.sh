batchSize=32
workers=4
nepoch=300
dataset=/home/ir0/data/language_faders/language_faders/external/language_changes/language_changes/external_tools/part_segmentation_pytorch/data
all_objects='Chair Lamp Table Knife Laptop Mug'

for object in $all_objects; do
    printf "\n\n\n\n\n!!!!!!!!!!!!!!!!!!\nTraining $object\n\n\n"
    CUDA_VISIBLE_DEVICES=0 python train_segmentation.py \
        --batchSize $batchSize \
        --workers $workers \
        --nepoch $nepoch \
        --dataset $dataset \
        --class_choice $object
done