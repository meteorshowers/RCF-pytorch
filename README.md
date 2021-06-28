### Richer Convolutional Features for Edge Detection
Thanks to <a href="https://github.com/yun-liu">yun-liu's</a> help.
Created by XuanyiLi, if you have any problem in using it, please contact:xuanyili.edu@gmail.com.
The best result of my pytorch model is 0.808 ODS F-score now.
#### my model result
the following are the side outputs and the prediction example
![prediction example](https://github.com/meteorshowers/RCF-pytorch/blob/master/doc/326025.jpg)
### Citation
If you find our work useful in your research, please consider citing:

@article{RcfEdgePami2019,
  author = {Yun Liu and Ming-Ming Cheng and Xiaowei Hu and Jia-Wang Bian and Le Zhang and Xiang Bai and Jinhui Tang},
  title = {Richer Convolutional Features for Edge Detection},
  year  = {2019},
  journal= {IEEE Trans. Pattern Anal. Mach. Intell.},
  volume={}, 
  number={}, 
  pages={}, 
  doi = {},
}

@inproceedings{RCFEdgeCVPR2017,
  title={Richer Convolutional Features for Edge Detection},
  author={Yun Liu and Ming-Ming Cheng, Xiaowei Hu and K Wang and X Bai},
  booktitle={IEEE CVPR},
  year={2017},
}
### online demo(upload your own image):😋
<a href="http://mc.nankai.edu.cn/edge">online demo link</a> 

### Video demo:😋
this is the edge version of movie Titanic:
<a href="https://www.youtube.com/channel/UC_6UOBTYzBzA6s0EZSeTh1g">youtube video link</a> 
![Titanic example](https://github.com/meteorshowers/RCF-pytorch/blob/master/doc/testw.gif)
### Introduction
I implement the edge detection model according to the <a href="https://github.com/yun-liu/rcf">RCF</a>  model in pytorch. 

the result of my pytorch model will be released in the future

| Method |ODS F-score on BSDS500 dataset |ODS F-score on NYU Depth dataset|
|:---|:---:|:---:|
|ours| 0.808 | *** |
| Reference[1]| 0.811 | ***  |


### Installation

Install <a href="https://pytorch.org/">pytorch</a>. The code is tested under 0.4.1 GPU version and Python 3.6  on Ubuntu 16.04. There are also some dependencies for a few Python libraries for data processing and visualizations like `cv2` etc. It's highly recommended that you have access to GPUs.

### Usage

#### image edge detection

To train a RCF model on BSDS500:

        python train_RCF.py

After training, to evaluate:

        python evaluate.py (for further work)

<i>Side Note:</i>  Hello mingyang, I love you

### License
Our code is released under MIT License (see LICENSE file for details).

### Updates

### To do 
* Add support for multi-gpu training for the edge detetion task.
* Improve the performance to 0.806/0.811 in the original paper.
* Add a gpu version of edge-eval code to accelerate the evaluation process.
* Add pami version of RCF.
### source：
*  To download the pretrained model, please click https://drive.google.com/open?id=1TupHeoBKawrniDka0Hc64m3BG4OKG8nM
(This pretrained model is not the best model, just for communicating)
*  To download the vgg16 pretrained model which is used for the backbone. please click https://drive.google.com/file/d/1lUhPKKj-BSOH7yQL0mOIavvrUbjydPp5/view?usp=sharing.
### Related Projects
[1] <a href="https://github.com/yun-liu/rcf">Richer Convolutional Features for Edge Detection</a> 

[2] <a href="https://github.com/s9xie/hed">HED</a> 

[3] <a href="https://github.com/zeakey/hed">HED</a> created by <a href="https://github.com/zeakey">zeakey's</a>

[4] <a href="https://github.com/godman2016/ContourNet">ContourNet</a>

