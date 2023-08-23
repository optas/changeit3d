### Pretrained Nets:
### ( .zip file of 4.6GB )

## These include the following:

##   - Extracted latent codes (vectors) of shapes based on shape AutoEncoders or other pretrained networks e.g., PC-AE, SGF, ResNet, etc.
##          See /shape_latents

##   - Latent-based neural listeners trained with the above, latent codes
##          See /listeners/all_shapetalk_classes/

##   - Pretrained PC-AE 
##          See /pc_autoencoders

##   - Oracle neural listener for LAB
##          See listeners/oracle_listener

##   - PoinNet-based point cloud shape classifier for Class-Distortion and Frechet Distance
##          See /pc_classifiers

##   - Shape-part-based classifier for localized-Geometric Distance (LGD)
##          See /part_predictors

##   - 40 ChangeIt3D-Nets ablating the AE backbone and other hyper-parameters affecting the design of the networks
##          See /changers


TOP_OUT_DIR="../../data/"
CURDIR="$PWD"


mkdir -p $TOP_OUT_DIR
cd $TOP_OUT_DIR
echo "**** Starting to download pretrained nets at ${TOP_OUT_DIR} ****"

wget https://shapetalk-public.s3.amazonaws.com/pretrained.zip
unzip pretrained.zip
rm -rf pretrained.zip

cd $CURDIR
echo '**** Done ****'
