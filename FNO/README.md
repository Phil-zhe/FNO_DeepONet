# Burgers-FNO-Small

本项目先通过数值方法生成一维黏性 Burgers 方程数据，再训练 FNO1d 学习从初始条件 `u0(x)` 到终止时刻解 `u(x,T)` 的算子映射。

## 运行顺序

```bash
pip install -r requirements.txt
python scripts/generate_burgers_data.py --config configs/data.yaml
python scripts/train_fno.py --config configs/fno1d.yaml
python scripts/evaluate_fno.py --config configs/fno1d.yaml
python scripts/plot_results.py --config configs/fno1d.yaml
```

## 文件功能

- `configs/`: 数据生成与 FNO1d 训练配置。
- `data/`: 保存生成的数据，默认使用 `data/processed/`。
- `results/`: 保存 checkpoint、图像和表格结果。
- `scripts/`: 数据生成、训练、评估、绘图入口脚本。
- `src/solvers/`: 一维 Burgers 方程伪谱数值求解器。
- `src/data/`: 数据集划分与标准化工具。
- `src/models/`: FNO1d 模型。
- `src/losses/`: 训练损失函数。
- `src/metrics/`: 物理空间评估指标。

## 数据格式

生成的 `.npz` 文件包含：

- `a`: `[N, Nx]`，初始条件 `u0(x)`。
- `u`: `[N, Nx]`，终止时刻解 `u(x,T)`。
- `x`: `[Nx]`，空间坐标。
- `nu`: 黏性系数。
- `T`: 终止时间。
- `dt`: 时间步长。

## 模型结构

FNO1d 输入为 `a(x)` 与坐标 `x`，拼接为两个通道：

```text
输入: a(x) 与 x
lifting: 2 -> width
Fourier Block x depth
projection: width -> fc_dim -> 1
输出: u(x,T)
```

## 默认参数

```text
nu = 0.01
T = 1.0
nx = 256
n_samples = 1200
dt = 0.002
k_max = 8
modes = 16
width = 32
depth = 4
fc_dim = 128
batch_size = 32
epochs = 300
lr = 1e-3
```
