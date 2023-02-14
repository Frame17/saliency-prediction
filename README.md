# Task-driven Saliency Prediction on Information Visualisation

## Introduction

The project is implemented in scope of the **Machine Perception and Learning** course. 
The goal of the project is to develop a neural network that predicts points of interest on diagrams under task-specific conditions.
The model architecture is based on the **Task-driven Webpage Saliency** paper [[1]](#1) .

## Dataset

Multiple datasets are used for the model training. 

Task-free part of the model is trained with the joint dataset of **What Makes a Visualization Memorable?** [[2]](#2) 
and **Beyond Memorability: Visualization Recognition and Recall**. [[3]](#3)

Task-specific part of the model is trained with the dataset from **Exploring Visual Attention and Saliency Modeling for Task-Based Visual Analysis**. [[4]](#4)

## Model

The model utilizes **FCN semantic image segmentation** [[5]](#5) encoder-decoder architecture for both task-free and task-specific branches.

An attempt was made to exploit the **Xception: Deep Learning with Depthwise Separable Convolutions** [[6]](#6) encoder together with the **Predicting Visual Importance Across Graphic Design Types** [[7]](#7) decoder. 
This approach demonstrated successful results for the task-free saliency prediction, but needs to be adapted for the task-specific part.

## Results

The comparison between the ground truth (left image) and the model prediction (right image) is presented below.
As a similarity measure, pearson correlation coefficient between the images is computed.


<h3 align="center">Good predictions</h3>

<p align="center">
    <img src="https://github.com/Frame17/saliency-prediction/blob/main/prediction_examples/good_prediction_1_%200.924.png?raw=true"/><br>
    <em>CC = 0.924</em>
</p>
<p align="center">
    <img src="https://github.com/Frame17/saliency-prediction/blob/main/prediction_examples/good_prediction_2_0.935.png?raw=true"/><br>
    <em>CC = 0.935</em>
</p>

<h3 align="center">Less accurate predictions</h3>

<p align="center">
    <img src="https://github.com/Frame17/saliency-prediction/blob/main/prediction_examples/bad_prediction_1_0.796.png?raw=true"><br>
    <em>CC = 0.796</em>
</p>
<p align="center">
    <img src="https://github.com/Frame17/saliency-prediction/blob/main/prediction_examples/bad_prediction_2_0.864.png?raw=true"><br>
    <em>CC = 0.864</em>
</p>

## References
<a id="1">[1]</a>
Zheng, Quanlong & Jiao, Jianbo & Cao, Ying & Lau, Rynson. (2018). Task-Driven Webpage Saliency: 15th European Conference, Munich, Germany, September 8–14, 2018, Proceedings, Part XIV. 10.1007/978-3-030-01264-9_18.

<a id="2">[2]</a>
Borkin, Michelle & Vo, Azalea & Bylinskii, Zoya & Isola, Phillip & Sunkavalli, Shashank & Oliva, Aude & Pfister, Hanspeter. (2013). What Makes a Visualization Memorable?. IEEE transactions on visualization and computer graphics. 19. 2306-15. 10.1109/TVCG.2013.234.

<a id="3">[3]</a>
Borkin, Michelle & Bylinskii, Zoya & Kim, Nam & Bainbridge, Constance & Yeh, Chelsea & Borkin, Daniel & Pfister, Hanspeter & Oliva, Aude. (2015). Beyond Memorability: Visualization Recognition and Recall. IEEE transactions on visualization and computer graphics. 22. 10.1109/TVCG.2015.2467732. 

<a id="4">[4]</a>
Polatsek, Patrik & Waldner, Manuela & Viola, Ivan & Kapec, Peter & Benesova, Wanda. (2018). Exploring Visual Attention and Saliency Modeling for Task-Based Visual Analysis. Computers & Graphics. 72. 10.1016/j.cag.2018.01.010. 

<a id="5">[5]</a>
Long, Jonathan & Shelhamer, Evan & Darrell, Trevor. (2014). Fully Convolutional Networks for Semantic Segmentation. Arxiv. 79. 

<a id="6">[6]</a>
Chollet, Francois. (2017). Xception: Deep Learning with Depthwise Separable Convolutions. 1800-1807. 10.1109/CVPR.2017.195. 

<a id="7">[7]</a>
Fosco, Camilo & Casser, Vincent & Bedi, Amish & O'Donovan, Peter & Hertzmann, Aaron & Bylinskii, Zoya. (2020). Predicting Visual Importance Across Graphic Design Types. 
