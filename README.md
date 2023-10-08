# Jittor 风景图片生成风景比赛

![](./selects/8215750919_fd4ab65209_b.jpg)

## 简介

本项目包含了第三届计图挑战赛---风景图片生成风景比赛的代码实现。本项目首先采用了SPADE网络根据语义分割图生成风景图片，进一步使用TSIT网络将生成的风景图片使用指定的风格图片。

## 安装 

本项目在1张 3090 上运行，训练时间约为2天。

#### 运行环境

- ubuntu 20.04 LTS
- python >= 3.7
- jittor >= 1.3.0

#### 安装依赖

执行以下命令安装 python 依赖

```
pip install -r requirements.txt
```

#### 预训练模型

* resnet101.pkl 在./pretrained/deeplab_jittor/pretrained文件夹下，是jittor的imagenet预训练模型，可以通过`wget https://cg.cs.tsinghua.edu.cn/jittor/assets/build/checkpoints/resnet101.pkl`指令下载到本地
* deeplab语义分割预训练模型 ./pretrained文件夹下的Epoch_40.pkl文件，可以通过运行./deeplab/deeplab_jittor文件夹下的train.py文件到40个epoch左右生成。其中用到的数据集即为比赛提供的训练集，我们将数据集分为9200张的训练集和800张的验证集。如果想要正常训练，可以在deeplab下构造一个datasets文件夹，其中再构造train和val两个文件夹，并分别再构造imgs和labels两个文件夹，分别放入训练集和验证集的原图和语义分割图。也可以使用我们预训练好的模型，下载或训练好预训练模型后置于./pretrained文件夹中。[下载链接](https://cloud.tsinghua.edu.cn/d/26efda16d6d942339b98/)
参考自https://cg.cs.tsinghua.edu.cn/jittor/tutorial/2020-3-17-09-55-segmentation/
* vgg预训练模型 在训练过程中用到了vgg_loss，其中的vgg网络在构造时将pretrained设置为true，会自动下载jittor预训练好的vgg网络。



## 数据预处理

数据集文件夹应命名为`datasets`，并置于与final_gaugan和final_tsit同级目录下

```
datasets
    - train
        - imgs   训练集真实图片
        - labels 训练集语义分割图
    - val  B榜测试集语义分割图
    - label_to_img.json
```



## 训练

在gaugan文件夹中运行`bash train.sh`进行SPADE模型的训练,之后在tsit中运行`bash train.sh`进行TSIT模型的训练,在train.sh中可通过`--input_path`指定训练的数据集路径

## 推理

1.在gaugan运行`bash test.sh`,该步将SPADE模型生成的风景图片放到`./tsit/datasets/gaugan_results`文件夹下

2.运行tsit下的`test.sh`,该步将最终B榜1000张图的结果存放在output_path下，并将挑选的3张图片放在`./tsit/selects`下

注：tsit和gaugan的test.sh中均需要指定input_path,在final_tsit的test.sh中还需指定output_path