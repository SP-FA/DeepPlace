# DeepPlace

>### Implementation of NeurIPS 2021 paper "On Joint Learning for Solving Placement and Routing in Chip Design"
>An end-to-end learning approach DeepPlace for placement problem with two stages. The deep reinforcement learning (DRL) agent places the macros sequentially, followed by a gradient-based optimization placer to arrange millions of standard cells. We use [PPO](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) for all the experiments implemented with Pytorch, and the GPU version of
[DREAMPlace](https://github.com/limbo018/DREAMPlace) is adopted as gradient based optimization placer for arranging standard cells.
> 
> [原论文链接](https://arxiv.org/abs/2111.00234)
> 
> [原项目链接](https://github.com/Thinklab-SJTU/EDA-AI/tree/main/DeepPlace)
> 
> [文章速通](https://blog.csdn.net/SP_FA/article/details/134083867?csdn_share_tail=%7B%22type%22%3A%22blog%22%2C%22rType%22%3A%22article%22%2C%22rId%22%3A%22134083867%22%2C%22source%22%3A%22SP_FA%22%7D)

<br>

# 关于此项目

本项目的目的有两点：

1. 使代码更易使用：原项目似乎在安装环境、运行代码方面有诸多问题，而且不方便更换数据集，不方便调整超参。这个项目致力于修复这些问题，使得大家可以快速运行 DeepPlace 代码，而不是像我一样调试好几天。
2. 在 [MaskPlace: Fast Chip Placement via Reinforced Visual Representation Learning](https://arxiv.org/abs/2211.13382) 这篇文章中提出了一个调整 DeepPlace 架构的方式 DeepPlace-no-Overlap，也就是禁止 DeepPlace 放置时出现重叠的情况，这是一个很好的思路，于是我通过自己的方式将其实现。

<br>

## 1. Requirements
安装环境的过程中会出现很多很多很多的问题，我会尽量找到规避问题的方法，并且整理出来。注：只能在 linux 系统上运行。

1. 安装 PyTorch
    ```bash
    conda install pytorch torchvision -c soumith
    ```
2. 安装 baselines
    ```bash
    git clone https://github.com/openai/baselines.git
    cd baselines
    pip install -e .
    ```
3. 安装其他 requirements
    ```bash
    pip install -r requirements.txt
    ```
4. 安装 dgl
    ```bash
    conda install -c dglteam dgl-cuda10.2
    ```
5. 安装 DreamPlace：[参考文章](https://blog.csdn.net/SP_FA/article/details/134887441?spm=1001.2014.3001.5501)（本项目已自带，无需安装）

<br>

## 2. Training

在原项目中支持 Macro Placement、Joint Macro/Standard cell Placement 以及 Validation。目前本项目进度处于整理完成 Macro Placement 部分。

### 2.1 Macro Placement

```bash
python main.py --task "place" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 1 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01 --grid-num 84 --overlap --benchmark "adaptec3"
```

### Macro Placement no Overlap

```bash
python main.py --task "place" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 1 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01 --grid-num 84 --benchmark "adaptec3"
```

**参数说明**

- `num-steps`：该参数已被删除，因为可以计算得出，方式为：`num-steps = num-mini-batch * num-nodes`，其中 num-nodes 是 benchmark 中 macro 的数量
- `save-interval`：该参数已被删除，改为自动保存效果最好的模型
- `grid-num`：默认值为 84，表示用于放置 macro 的 canvas 边长，此处与原文章略有不同，在下文中会详细解释。
- `overlap`：是否允许 macro 之间重叠。
- `benchmark`：使用哪个数据集

### 2.2 Validation
```bash
python validation.py --task "place" --num-processes 1 --num-mini-batch 1 --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --entropy-coef 0.01 --grid-num 84 --overlap --benchmark "adaptec3"
```

### Validation no Overlap

```bash
python validation.py --task "place" --num-processes 1 --num-mini-batch 1 --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --entropy-coef 0.01 --grid-num 84 --benchmark "adaptec3"
```

**参数说明**
- 同上


[//]: # (### Joint Macro/Standard cell Placement)

[//]: # ()
[//]: # (```bash)

[//]: # (python DeepPlace/main.py --task "fullplace" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 1 --num-steps 2840 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01)

[//]: # (```)

<br>

## 3. 关于数据集的处理方式

由于原项目的数据集存在很多问题，而且作者并没有对这些问题给予答复，因此我只能按照自己的方式处理数据集，并在此列出有关的问题。

**处理方式：**
1. 使用 maskplace 文章中的方法读取 benchmark 信息，并且按照 macro 从大到小的顺序作为放置顺序。
2. 关于 netlist 文件，作者并没有说明如何生成该文件，于是我按照 .nets 文件里的信息建双向图，并且去掉了重复的边。但是在此过程中有一个问题：使用作者的 netlist 文件时，训练的 reward 会很大，一般在 [-100, 10] 之间，而用自己的数据集训练时，reward 则大于 -1000。另外，原作者的数据集只选择了 710 个 macro（一共有 723 个），而且 netlist 里边的数量远少于实际的边数量。结合这两点来看，合理推测原作者刻意忽略掉了很多的 netlist 数据，从而提升训练效果。因此，在使用我的代码训练时，可能会出现收敛困难的情况，但是这是正常现象。

<br>

## 4. 关于 DeepPlace-no-Overlap 的实现
在实现过程中对代码做出了如下改动：
1. 由于需要避免 macro 重叠，所以 canvas 要足够大。在原项目中，canvas 边长为 32，observation 边长为 84，canvas 通过上采样的方式与 observation 对齐。本项目将这两个的大小进行统一，都通过 grid-num 参数进行控制，默认为 84。
2. 由于没有了 macro 之间的重叠，因此计算 reward 的时候也就可以不用考虑 Congestion，同时为了和其他文章统一起来，计算 reward 时使用 macro 上的 pin 的位置（源代码中使用的是 macro 的位置）
3. 为了保证正确性，我尽可能少的在原代码上进行修改，因此代码可能非常冗余，请多包含。

<br>

# Todo list
1. 总结环境配置方法以及问题
2. 整理 Joint Macro/Standard cell Placement 的代码

<br>

# References

```
@article{cheng2021joint,
  title={On Joint Learning for Solving Placement and Routing in Chip Design},
  author={Cheng, Ruoyu and Yan, Junchi},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={16508--16519},
  year={2021}
}
```

```
@misc{lai2022maskplace,
      title={MaskPlace: Fast Chip Placement via Reinforced Visual Representation Learning}, 
      author={Yao Lai and Yao Mu and Ping Luo},
      year={2022},
      eprint={2211.13382},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
