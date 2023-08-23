homedir=/home/ir0/data/language_faders/language_faders/external/language_changes/language_changes/external_tools/part_segmentation_pytorch/
dataset=$homedir'data'
all_objects='Chair Lamp Table Knife Laptop Mug'
#Trained model epoch index to use
declare -A model_idxs
model_idxs[Chair]=267
model_idxs[Lamp]=295
model_idxs[Table]=142
model_idxs[Knife]=76
model_idxs[Laptop]=272
model_idxs[Mug]=272
#Index of 3D object model to call
declare -A input_idxs
input_idxs[Chair]=267
input_idxs[Lamp]=50
input_idxs[Table]=142
input_idxs[Knife]=76
input_idxs[Laptop]=10
input_idxs[Mug]=272

for object in $all_objects; do
    idx=${model_idxs[$object]}
    model=$homedir'utils/seg/seg_model_'$object'_'$idx'.pth'
    printf "\n\n\n\n\n!!!!!!!!!!!!!!!!!!\nTraining $object $idx $dataset\n\n\n"
    CUDA_VISIBLE_DEVICES=0 python show_seg.py \
        --model $model \
        --idx ${input_idxs[$object]} \
        --dataset $dataset \
        --class_choice $object
done