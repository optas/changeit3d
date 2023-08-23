# ShapeTalk: A Language Dataset and Framework for 3D Shape Edits and Deformations

![representative](https://changeit3d.github.io/img/qualitative_teaser.jpeg)

## Introduction

This codebase accompamnies our <a href="https://openaccess.thecvf.com/content/CVPR2023/papers/Achlioptas_ShapeTalk_A_Language_Dataset_and_Framework_for_3D_Shape_Edits_CVPR_2023_paper.pdf">CVPR-2023<a> paper.
### Related Works
- [PartGlot, CVPR22](https://arxiv.org/abs/2112.06390): Discovering the 3D/shape **part-structure** automatically via referential language.
- [LADIS, EMNLP22](https://arxiv.org/abs/2212.05011): **Disentangling** 3D/shape edits when using ShapeTalk.
- [ShapeGlot, ICCV19](https://ai.stanford.edu/~optas/shapeglot): Building discriminative **listeners and speakers** for 3D shapes.
### Citation

If you find this work useful in your research, please consider citing:
	
	@inproceedings{achlioptas2023shapetalk,
        title={{ShapeTalk}: A Language Dataset and Framework for 3D Shape Edits and Deformations},
        author={Achlioptas, Panos and Huang, Ian and Sung, Minhyuk and
                Tulyakov, Sergey and Guibas, Leonidas},
        booktitle=Conference on Computer Vision and Pattern Recognition (CVPR)
        year={2023}
    }


## Installation
Optional, create **first** a clean environment. E.g.,

```#/usr/bin/bash
 conda create -n changeit3d python=3.8
 conda activate changeit3d 
 conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch
```

Then,

```#/usr/bin/bash
 git clone https://github.com/optas/changeit3d
 cd changeit3d
 pip install -e .
```

Last, if you want to train pointcloud autoencoders or run some of our introduced evaluation metrics, consider installing a fast (GPU-based) implementation of **Chamfer's Loss**:

```#/usr/bin/bash
git submodule add https://github.com/ThibaultGROUEIX/ChamferDistancePytorch changeit3d/losses/ChamferDistancePytorch
```

- Please see ```setup.py``` for all required packages. We left the versions of most of these packages unspecified for an easier and more broadly compatible installation. However, if you want to replicate **precisely** all our experiments, use the versions indicated in the `environment.yml` (e.g., `conda env create -f environment.yml`).
- See F.A.Q. at the bottom of this page for suggestions regarding common installation issues.

### Basic structure of this repository
```bash
./changeit3d	
├── evaluation 				# routines for evaluating shape edits via language
├── models 				# neural-network definitions
├── in_out 				# routines related to I/O operations
├── language 				# tools used to process text (tokenization, spell-check, etc.)
├── external_tools 			# utilities to integrate code from other repos (ImNet, SGF, 3D-shape-part-prediction)
├── scripts 				# various Python scripts
│   ├── train_test_pc_ae.py   	        # Python script to train/test a 3D point-cloud shape autoencoder
│   ├── train_test_latent_listener.py   # Python script to train/test a neural listener based on some previously extracted latent shape-representation
│   ├── ...
│   ├── bash_scripts                    # wrappers of the above (python-based) scripts to run in batch mode with a bash terminal
├── notebooks 				# jupyter notebook versions of the above scripts for easier deployment (and more)	
```

## ShapeTalk Dataset ( :rocket: )

Our work introduces a large-scale visio-linguistic dataset -- [ShapeTalk](http://5.78.48.181:8502/).

First, consider [downloading ShapeTalk](https://docs.google.com/forms/d/e/1FAIpQLSdOouzvK0zmjvmBoiQhbfnhe1Kac72XNmHXzshn6_KUEjw8QQ/viewform?usp=send_form) and then quickly read its [manual](https://github.com/optas/changeit3d/blob/main/assets/README_shapetalk.md) to understand its structure.

### Exploring ShapeTalk ( :microscope: )

Assuming you downloaded ShapeTalk, you should see at the top the downloaded directory subfolders:

| Subfolder                        | Content-explanation |
|:--------------------------------------|:-----------|
|**images**| 2D renderings used for contrasting 3D shapes and collecting referential language via Amazon Mech. Turk|
|**pointclouds**| pointclouds extracted from the surface of the underlying 3D shapes -- used e.g. for training a PCAE & evaluating edits|
|**language**| files capturing the collected language: see [ShapeTalk' manual](https://github.com/optas/changeit3d/blob/main/assets/README_shapetalk.md) if you haven't done it yet|

:arrow_right: To **familiarize yourself** with ShapeTalk you can run this [notebook](./changeit3d/notebooks/analysis/data_set/1_ShapeTalk_basic_statistics.ipynb) to compute basic statistics about it.

:arrow_right: To make a more _fine-grained_ analysis of ShapeTalk w.r.t. its language please run this [notebook](./changeit3d/notebooks/analysis/data_set/2_ShapeTalk_specialized_language_usage.ipynb).

## Neural Listeners ( :ear: )

You can train and evaluate our neural listening architectures with different configurations using this [python script](./changeit3d/scripts/train_test_latent_listener.py) or its equivalent [notebook](./changeit3d/notebooks/discriminative_nets/train_test_latent_listener.ipynb).

<!-- You can also [download](./changeit3d/scripts/bash_scripts/download_pretrained_nets.sh) and use our pre-trained neural listeners trained with (also pretrained) latent-space-based shape representations. -->

Our attained accuracies are given below:

| Shape Backbone                        | Modality   | Overall   | Easy   | Hard   | First   | Last   | Multi-utter<br/>  Trans. vs. (LSTM) |
|:--------------------------------------|:-----------|:----------|:-------|:-------|:------|:-------|:--------|
| [ImNet-AE](https://arxiv.org/abs/1812.02822)                                 | implicit   | 68.0%     | 72.6%  | 63.4%  | 72.4% | 64.9%  | 73.2% (78.4%)  |
| [SGF-AE](https://arxiv.org/abs/2008.06520)                                   | pointcloud | 70.7%     | 75.3%  | 66.1%  | 74.9% | 68.0%  | 76.5%  (79.9%) |
| [PC-AE](https://arxiv.org/abs/1707.02392)                                 | pointcloud | 71.3%     | 75.4%  | 67.2%  | 75.2% | 70.4%  | 75.3%  (81.5%) |
| [ResNet-101](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet101.html)                             | image      | 72.9%     | 75.7%  | 70.1%  | 76.9% | 68.7%  | 79.8%  (84.3%) |
| [ViT (L/14/CLIP)](https://huggingface.co/openai/clip-vit-large-patch14)        | image      | 73.4%     | 76.6%  | 70.2%  | 77.0% | 70.7%  | 79.6%  (84.5%) |
| [ViT (H/14/OpenCLIP)](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K) | image      | **75.5%**     | **78.5%**  | **72.4%**  | **79.5%** | **72.2%**  | **82.3%** (**85.8%**)  |


For the meaning of the sub-populations (easy, hard, etc.), please see our paper, Table 5. 

All reported numbers above concern the _transformer-based_ baseline presented in our paper; the exception is the 
numbers inside parenthesis ("Multi" (LSTM)) which are based on our LSTM baseline. The LSTM baseline performs better **only** in this "Multi" scenario, possibly because our transformer struggles to self-attend well to all concatenated input utterances. 

If you have **new results**, please reach out to [Panos Achlioptas](https://optas.github.io/) to include in our [competition page](https://changeit3d.github.io/benchmarks.html).


## ChangeIt3DNet ( neural 3D editing via language :hammer: )

The algorithmic approach we propose and follow in this work to train a language/3D-shape editor such as the ChangeIt3DNet is to break down the process into **three** steps:  

- **Step1.** Train a **latent-shape** representation network, e.g., a shape AutoEncoder like [PC-AE](https://arxiv.org/abs/1707.02392) or [SGF](https://arxiv.org/abs/2008.06520).
- **Step2.** Use the derived shape "latent-codes" from Step-1 to train a latent-based **neural-listener** following the language and
  listening train/test splits of ShapeTalk.
- **Step3.** Keep the above two networks **frozen**; and build a low-complexity editing network that learns to _move inside the latent space_ of Step-1 in a way that changes the shape of an input object that increases its compatibility with the input language, e.g., to make the output have `thinner legs' as the pre-trained neural listener of step 2 understands this text/shape compatibility.

### Specific details on how to execute the above steps. 
* Step1. [train a generative pipeline for 3D shapes, e.g., an AE]
	* **PC-AE**.  To train a point-cloud-based autoencoder, please take a look at the scripts/train_test_pc_ae.py.
	* **IMNet-AE**. See our [customized repo](https://github.com/ianhuang0630/IM-Net-ShapeTalkPretraining) and the instructions there, which will guide you on how to extract implicit fields for shapes of 3D objects like those of ShapeTalk and train from scratch an ImNet-based autoencoder (Imnet-AE), or to save time, re-use our pre-trained IMNet-AE backbone. To integrate an Imnet-AE in this (changeit3d) repo, see further into `external_tools/imnet/loader.py`. If you have questions about the ImNet-related utilities, please contact [Ian Huang](https://ianhuang0630.github.io/me/).
 	* **Shape-gradient-fields (SGF)**. See our slightly [customized repo](https://github.com/optas/ShapeGF) and the instructions there, which will help you to download and load the pre-trained weights for an AE architecture based on SGF (SGF-AE), which was also trained with ShapeTalk's shapes and our shape (unary) splits. To integrate an SGF-AE to this (changeit3d) repo, see also `external_tools/sgf/loader.py`.

    Note: for quickly testing your integration of either ImNet-AE or SGF-AE in the changeit3d repo you can also use these notebooks: [IMNet-AE-porting](changeit3d/notebooks/_esoteric/6_testing_porting_of_pretrained_imnet.ipynb) and [SGF-AE-porting](changeit3d/notebooks/_esoteric/5_testing_porting_of_pretrained_sgf.ipynb).

* Step2. [train neural listeners]
  	* Given the extracted latent codes of the shapes of an AE system (Step.1) and the data contained in the downloaded ShapeTalk: [```shapetalk_preprocessed_public_version_0.csv```](https://github.com/optas/changeit3d/blob/main/assets/README_shapetalk.md)(e.g., listening-oriented train/test splits, tokenized sentences, the provided [vocabulary object](changeit3d/language/vocabulary.py), etc.) you can train a neural listener using this python [script](changeit3d/scripts/train_test_latent_listener.py).
  	
  	- On average, this will take less than 30 minutes on a modern GPU.
  	
  	- The underlying code is versatile and can be used with many input variations affecting performance. For instance, you can use an LSTM vs. a Trasnformer (default) to encode the language, different shape-latent representations, etc. Please consult the help docs on the input processing function: def [parse_train_test_latent_listener_arguments](changeit3d/in_out/arguments.py).
 
  	- Please also take a look below for our listeners' pre-trained weights.

* Step3. [train networks for language-assisted 3D shape editing]
	* The main Python script you need to train such a network (ChangeIt3D-Net) is [this](./changeit3d/scripts/train_change_it_3d.py) one.
  		- To understand the meaning of the different input parameters, please see the help docs of the [`parse_train_changeit3d_arguments`](./changeit3d/in_out/arguments.py) function.
  
  	* To use our metrics to evaluate your trained network, please use [this](./changeit3d/scripts/evaluate_change_it_3d.py) Python script (see next Section for more details on the evaluation metrics).
  	 
	* We release the weights of 40 different ChangeIt3D networks ablating the effect of the design "hyper-parameters" of a) `self-contrast` b) `identity-penalty` c) 
`decoupling-the-magnitude-of-the-edit` with the 3D shape AE backbones of PCAE and SGF (for the performance effect of these choices see [here](./changeit3d/notebooks/analysis/changeit3d/analyze_change_it_3d_results.ipynb)).


## Pretained Weights and Networks
You can download a large pool of pre-trained models using this [bash script](./changeit3d/scripts/bash_scripts/download_pretrained_nets.sh). 

The structure of the downloaded folders is assumed to be as downloaded by the rest of this codebase to work seamlessly. However, please update (whenever prompted) the `config.json.txt` files to point to the downloaded directories of the pretrained networks in **your** local hard drive!

:exclamation: The provided weights are _not_ identical to those used in the CVPR paper. Unfortunately, we recently had to retrain all our major networks due to a hard drive failure. _Fortunately, the resulting networks' performance is either very close to those attained by those presented in the CVPR manuscript or in some cases **noticeably improved**._ Please see [here](./changeit3d/evaluation/metrics_on_publicly_shared_listeners.md) for the attained accuracies of the shared neural listeners, or run [this](./changeit3d/notebooks/analysis/changeit3d/analyze_change_it_3d_results.ipynb) notebook to analyze the performance of the ChangeIt3D networks with the better-performing (pre-trained and shared) evaluation networks. If you have any questions, please do not hesitate to contact [Panos Achlioptas](https://optas.github.io/). :exclamation: 


## Metrics for Evaluating Editing Modules ( :triangular_ruler: )
To run the metrics introduced in our paper (LAB, l-GD, etc.) to evaluate shape editing systems such as the ChangeIt3D, please use [this](changeit3d/scripts/evaluate_change_it_3d.py) Python script. 

To see the expected input of this script, please read the help of its argument parsing function [`parse_evaluate_changeit3d_arguments`](./changeit3d/in_out/arguments.py). 

You can customize your run to a subset of the available metrics. 
 
 
 ### Details:
 
 - **Listening Association Boost (_LAB_)**:
   - The pre-trained `Oracle` neural listener we use for LAB is trained with _raw_ 3D point cloud input shape data (see the training [script](./changeit3d/scripts/train_test_raw_pc_listener.py)). I.e., it does _not_ operate in the latent space of an AutoEncoder. This choice allows the application of LAB in various changeit3D systems even when they work with different modalities: e.g., _this listener can be applied to compare Meshes, Occupancy Voxels, etc._, as long as you convert your output shape representation to a 3D surface point cloud. 
   - This __evaluating__ listener is trained with a different `split` of ShapeTalk than the one we use for all other reported neural listeners (random_seed=**2023**, see [notebook](changeit3d/notebooks/_esoteric/4_prepare_listening_split_for_LAB_oracle.ipynb)) and uses the [DGCNN](./changeit3d/models/dgcnn.py) backbone for encoding the input point clouds.
   - Its accuracy across the 30 classes of ShapeTalk is **77.03%**. For the three classes (chair, table, lamp) we use for our language-assisted editing experiments, it is **78.37%**.
   - Its pretrained weights are under: <top_download_dir>/pretrained/listeners/oracle_listener/all_shapetalk_classes/rs_2023/listener_dgcnn_based/ablation1/
  
- **Class-Distrotion (_CD_) & Frechet PointCloud Distance (_FPD_)**:
	- You need an object/shape classifier to assign the shape-class probability to a given input point cloud for these two metrics. 
 	- Such a pre-trained classifier is provided and uses a simple PointNet encoder, trained on ShapeTalks `unary` splits (see ShapeTalk [docs](https://github.com/optas/changeit3d/blob/main/assets/README_shapetalk.md)), attaining a test accuracy of 90.3%.
	- You can load to evaluate this classifier or train one from scratch using [this](./changeit3d/scripts/train_test_pc_clf.py) Python script.
 	- Its pretrained weights are under: <top_download_dir>/pretrained/pc_classifiers/rs_2022/all_shapetalk_classes

- **_localized_ Geometric Distance (_l-GD_)**:

  	- To use this metric, one needs a pre-trained shape-**parts** classifier. The metric is implemented [here](https://github.com/optas/changeit3d/blob/main/changeit3d/evaluation/all_metrics.py) (see "Test-6: Part-Based (Localized) Metrics").
  	- We provide pre-trained part-based classifiers for chairs, tables, and lamps.
  	- If you use our pre-trained part classifier, please make sure you properly preprocess the input point clouds (see comments in the above implementation).
  	- The weights can be found at <top_download_dir>/pretrained/part_predictors/shapenet_core_based/
 
  	  NOTE: (Subtle but reasonable question) _Why is the Chamfer-based **l-GD bigger** than the naive Chamfer distance when we apply them between the input point clouds and our edited outputs?_ After all, isn't the l-GD supposed to compensate (by removing) the shape parts that are most likely to be _changed_ given the input text prompt?
  	  	- Answer: L-GD tends to remove many points from the compared point clouds since the referred parts are frequently large (think of the top, or the legs, of tables). Chamfer distances are quite sensitive to the _density_ of the compared point clouds, with smaller densities resulting, on average, in larger distances. As a matter of fact, if you apply the naive Chamfer distance between our input/output at randomly sampled densities (points) that are equal to the corresponding comparators of the l-GD, **naive Chamfer is much larger than l-GD**.
 
## Frequently Asked Questions (FAQ)

1. **[Installation] I cannot install the suggested CUDA-based Chamfer Loss**

    The suggested CUDA-based implementation of Chamfer Loss (submodule) requires a ninja build system installed.

    If you do not have it you can install it like this:

    ```#/usr/bin/bash
    wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
    sudo unzip ninja-linux.zip -d /usr/local/bin/
    sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force
    ```

    Please also see the Troubleshooting section of the original [implementation](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch)

    If you fail to install CD you may want to try the  (~10x) slower provided implementation of Chamfer in losses/nn_distance/chamfer_loss

2. **[ShapeTalk, class-names] ShapeTalk's object _classes_ often have different names compared to their original names in repositories like ShapeNet. Why?**

    For instance, the ShapeTalk object class "flowerpot aggregates objects from the class "pot" of ShapeNet, along with objects from the class "flower_pot" of ModelNet.The objects of those two classes have the same semantics despite their different names. ShapeTalk's "flowerpot" _unifies_ them. For more information, please see the [mixing code](changeit3d/in_out/datasets/shape_talk.py).

3. **[Redoing experiments post-CVPR] addressing our hard drive failure and how this affected our shared pre-trained models**
   
   - a. The shared Oracle neural listener used for evaluating LAB based on DGCNN has an improved accuracy of 78.37 (instead of 76.25%) on the chair, table, and lamp classes.
   - b. The shared PointNet-based shape classifier used for evaluating Class-Distortion has an improved 90.3% accuracy (instead of 89.0%).
   - c. The latent-based neural listeners used (frozen) for training the shared ChangeIt3DNets have slightly different performances (within limits of random-seed/initialization fluctuations). The new performances are reported [here](./changeit3d/evaluation/metrics_on_publicly_shared_listeners.md). 
   - d. The above changes naturally affect our metrics for evaluating the ChangeIt3DNet architectures. You can find their performance for the shared editing networks [here](./changeit3d/notebooks/analysis/changeit3d/analyze_change_it_3d_results.ipynb).


 
