| Shape Backbone                        | Modality   | Overall   | Easy   | Hard   | 1st-Utter   | Last-Utter   | Multi-Utter Trans. vs. (LSTM) |
|:--------------------------------------|:-----------|:----------|:-------|:-------|:------------|:-------------|:--------------|
| [SGF-AE](https://arxiv.org/abs/2008.06520) | pointcloud | 69.7%     | 73.8%  | 65.6%  | 73.9%       | 66.8%        | 75.1% (79.0%)|
| [PC-AE](https://arxiv.org/abs/1707.02392)  | pointcloud | 71.4%     | 75.3%  | 67.4%  | 75.7%       | 68.8%        | 74.9% (80.8%)|
| [ResNet-101](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet101.html) | image      | 72.9%     | 75.8%  | 69.9%  | 76.9%       | 69.6%        | 80.2% (84.7%)|
| [ViT (L/14/CLIP)](https://huggingface.co/openai/clip-vit-large-patch14) | image | 73.6%     | 76.7%  | 70.5%  | 77.4%       | 71.5%        | 80.1% (83.9%)|
| [ViT (H/14/OpenCLIP)](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K) | image | **75.1%**     | **78.2%**  | **72.0%**  | **79.6%** | **72.1%** | **82.2% (86.1%)**|

For the meaning of the sub-populations (easy, hard, etc.) please see our paper, Table 5. 

All reported numbers above concern the _transformer-based_ baseline presented in our paper; the exception is the 
numbers inside parenthesis ("Multi" (LSTM)) which are based on our LSTM baseline. The LSTM baseline performs better **only** in this "Multi" scenario, possibly because our transformer struggles to self-attend well to all concatenated input utterances. 

If you have **new results**, please reach out to [Panos Achlioptas](https://optas.github.io/) to include in our [competition page] (https://changeit3d.github.io/benchmarks.html).
