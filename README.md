# License-Plate-Recogonition
Course project of SJTU-AU335-Computer-Vision



### Structure

- [`resources/`](./resources/)
  - [`easy/`](./resources/easy/)：简单任务图片
  - [`medium/`](./resources/medium/): 中等任务图片
  - [`difficult/`](./resources/difficult/): 困难任务图片
- [`code/`](./code)
  - [`traditional_method/`](./code/traditional_method)：传统方法代码
    - [`template_data/`](./code/traditional_method/template_data)：模板匹配使用的模板
    - [`easy.py`](./code/traditional_method/easy.py)：easy难度任务代码
    - [`medium.py`](./code/traditional_method/medium.py)：medium难度任务代码
    - [`difficult.py`](./code/traditional_method/difficult.py)：difficult难度任务代码
  - [`deep_learning/`](./code/deep_learning/)：深度学习方法代码
    - [`train.py`](./code/deep_learning/train.py)：训练代码
    - [`test.py`](./code/deep_learning/test.py)：测试代码
    - [`models.py`](./code/deep_learning/models.py)：模型代码
    - [`data_process.py`](./code/deep_learning/data_process.py)：CCPD数据集处理代码，得到实验数据集



### Requirements

- opencv-python==4.5.4
- PyTorch
- numpy
- matplotlib
- Pillow



### How to Run

**1. 传统方法**

```bash
cd ./code/traditional_method
```

以easy任务中1-1为例

```bash
python easy.py --task_id 1 --verbose
```

`--verbose`：是否显示车牌识别中间结果



**2. 深度学习方法**

```bash
cd ./code/deep_learning
```

**仅测试结果：**

```bash
python test.py
```

**训练模型：**

下载CCPD数据集，生成实验数据集

```bash
python data_process.py --image_dir data_path/ccpd_base
```

`data_path`为CCPD数据集下载地址，可以`--train_dir`和`--val_dir`指定训练集和验证集位置

```bash
python train.py --num_epochs 80 --batch_size 128
```

`--train_data`和`--val_data`指定训练集和验证集位置



### Reference

[CCPD](https://github.com/detectRecog/CCPD)

