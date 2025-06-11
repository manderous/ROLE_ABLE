# Reasoning-Oriented and Analogy-Based Methods for Locating and Editing in Zero-Shot Event-Relational Reasoning (ROLE and ABLE)
# 基于推理编辑和类比编辑的零样本事件关系推理方法

This repository contains the implementation for the research work published at **COLING 2025** (The 31st International Conference on Computational Linguistics).

该工程为论文**基于推理编辑和类比编辑的零样本事件关系推理方法**的代码，该文章发表至COLING 2025会议。

https://aclanthology.org/2025.coling-main.181/

<!-- [![Colab MEMIT Demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kmeng01/memit/blob/main/notebooks/memit.ipynb) -->

![image](logo_cover_photo/ROLE_ABLE_Cover_photo.jpg)

## Table of Contents
- [环境配置 Enviroment Setup](#环境配置 Enviroment Setup)
- [ROLE](##ROLE)
- [ABLE](##ABLE)

## 环境配置（Enviroment Setup）
```
CUDA: 11.4
conda install python=3.8
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
conda install transformers==4.41.2
conda install scikit-learn
conda install -c conda-forge sentencepiece
conda install -c conda-forge accelerate
```

## 推理导向的定位和编辑方法（ROLE）
### 推理导向的定位（Reasoning-oriented locating method）
```
python ./experiments/ROLE_locating_(...)No_ext.py
python ./experiments/ROLE_locating_(...)NoW_No_cla.py
python ./experiments/ROLE_locating_(...)NoW_No_ext.py
python ./experiments/ROLE_locating_(...)W_cla.py
python ./experiments/ROLE_locating_(...)W_ext.py
python ./experiments/ROLE_locating_(...)W_Yes_cla.py
python ./experiments/ROLE_locating_(...)W_Yes_ext.py
```
该代码用于计算特定任务（因果、时序和子事件的分类和抽取）对特定样本在特定位置（网络层数，token位置）的因果效应值，来进行推理导向的定位。
另外，  
括号(...)：填入causal、subevent或者temporal；  
cla和ext：分别表示分类和抽取任务；  
(...)W：表示True Positive样本中，Yes的预测概率变化；  
(...)W_Yes：表示False Positive样本中，Yes的预测概率变化；  
(...)No：表示True Negative样本中，No的预测概率变化；  
(...)NoW_No：表示False Negative样本中，No的预测概率变化。  

Analyze locating results:
```
python ./experiments/ROLE_results_analysis.py
python ./experiments/ROLE_results_analysis_chinese.py
```
该代码用于整体分析6种任务在不同（网络层数，token位置）下的因果效应值，实现推理导向的定位：  
对于论文中的图3：选择“T5-Encoder画图, 热力图”部分的代码；  
对于论文中的图4，选择“T5-Decoder画图, 折线图”部分的代码。  

### 推理导向的编辑（Reasoning-oriented editing method）
以下代码用于推理导向的编辑，编辑T5的编码器和解码器，来实现对特定任务的优化。
For encoder's MLP module （编辑编码器的MLP模块）:
```
python ROLE_editing_encoder.py
```
For decoder's Cross-attention module （编辑解码器的交互注意力模块）:
```
python ROLE_editing_decoder.py
```

## 基于类比的定位和编辑方法（ABLE）
（1）用于事件因果关系抽取的数据集：
```
python ABLE_causal_ext_to_subevent_dec.py  # 编辑解码器，类比子事件任务
python ABLE_causal_ext_to_subevent_enc.py  # 编辑编码器，类比子事件任务
python ABLE_causal_ext_to_temporal_dec.py  # 编辑解码器，类比时序任务
python ABLE_causal_ext_to_temporal_enc.py  # 编辑编码器，类比时序任务
```
（2）用于事件因果关系分类的数据集：
```
python ABLE_causal_cla_to_subevent_dec.py  # 编辑解码器，类比子事件任务
python ABLE_causal_cla_to_subevent_enc.py  # 编辑编码器，类比子事件任务
python ABLE_causal_cla_to_temporal_dec.py  # 编辑解码器，类比时序任务
python ABLE_causal_cla_to_temporal_enc.py  # 编辑编码器，类比时序任务
```
（3）用于子事件关系抽取的数据集：
```
python ABLE_subevent_ext_to_causal_dec.py  # 编辑解码器，类比因果任务
python ABLE_subevent_ext_to_causal_enc.py  # 编辑编码器，类比因果任务
python ABLE_subevent_ext_to_temporal_dec.py  # 编辑解码器，类比子事件任务
python ABLE_subevent_ext_to_temporal_enc.py  # 编辑编码器，类比子事件任务
```
（4）分析关键位置的可类比性（Analysis of the analogicality of location）
```
python ./experiments/ABLE_anal_location.py
```
（5）分析编辑幅度的可类比性（Analysis of the analogicality of editing magnitude）
```
python ABLE_anal_edit.py
```

## 模型保存（Saved Models）
ABLE_para_store中保存的模型：https://drive.google.com/drive/folders/1E2RtfE5l6xGL41rWgY443_j5vsut_uQi?usp=drive_link
