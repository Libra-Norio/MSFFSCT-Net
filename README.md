# MSFFSCT-Net：基于多源特征门控融合与轻量Transformer的非侵入式负荷分解方法
本仓库是论文《基于多源特征融合与轻量Transformer的非侵入式负荷分解方法研究》的官方开源代码。
# 项目简介
本文提出了一种基于门控自适应融合与轻量Transformer的序列到点负荷分解模型MSFFSCT-Net。
该模型通过一维CBAM提取局部瞬态特征，利用单层轻量Transformer捕获长程依赖，并结合物理统计特征（均值、标准差、过零率等），
通过门控机制动态调节多源特征权重。在极低参数量（0.15M）下，实现了高精度的负荷分解与状态识别
（UK-DALE F1 达 82.69%，REDD F1 达 53.16%）， 适用于智能电表等边缘设备的轻量化部署。
# 代码目录结构
```
MSFFSCT-Net/
├── README.md                                      # 项目说明文档
├── requirements.txt                               # 项目依赖库
│
├── 【预处理与配置】                                
│   ├── config.py                                  # 全局配置文件（路径、超参数）
│   ├── NILM_Dataset.py                            # NILM数据集加载相关
│   ├── Pretrain_Dataset.py                        # 预训练数据集相关
│   ├── UKDALE_Parser.py                           # UK-DALE 数据集解析
│   └── REDD_Parser.py                             # REDD 数据集解析
│
├── 【模型结构定义】                                
│   ├── model_structure_s2p_cbam_transformer.py    # ★ 本文主模型 MSFFSCT-Net 结构
│   ├── model_structure_s2p_cbam_transformer_no_gate.py # 无门控消融模型结构
│   ├── model_structure_s2p_cbam.py                # S2P+CBAM 消融模型结构
│   ├── model_structure_s2p.py                     # S2P 对比模型结构
│   ├── model_structure_s2p_1d.py                  # 1D S2P 消融模型结构
│   ├── model_structure_seq2point_balanced.py      # Seq2Point-Balanced 对比模型结构
│   ├── model_structure_auglpn.py                  # AugLPN-NILM 对比模型结构
│   ├── model_structure_regularized_lstm.py        # 双向LSTM 对比模型结构
│   ├── model_structure_hparam.py                  # 超参数敏感性实验专用模型结构
│   └── edge_tcnformer_tf.py                       # EdgeTCNFormer 对比模型结构
│
├── 【训练脚本 - UK-DALE】                         
│   ├── train_transformer_lr0.001.py               # ★ 训练主模型 MSFFSCT-Net
│   ├── train_sct_net_no_gate.py                   # 消融实验：无门控融合
│   ├── train_s2p_cbam_lr0.001.py                  # 消融实验：S2P+CBAM
│   ├── train_s2p_lr0.001.py                       # 训练 S2P
│   ├── train_s2p_lr0.001_1d.py                    # 消融实验：1D S2P
│   ├── train_s2p_balanced.py                      # 训练 S2P-Balanced
│   ├── train_auglpn.py                            # 训练 AugLPN
│   ├── train_regularized_lstm_ukdale_fixed.py     # 训练 LSTM
│   ├── train_edge_tcnformer.py                    # 训练 EdgeTCNFormer
│   ├── train_hparam_sensitivity.py                # 超参数敏感性实验
│   └── train_small_sample_all.py                  # 数据量敏感性实验
│
└── 【训练脚本 - REDD】                            
    ├── train_redd_transformer.py                  # 训练主模型 MSFFSCT-Net (REDD)
    ├── train_s2p_redd.py                          # 训练 S2P (REDD)
    ├── train_seq2point_balanced_redd.py           # 训练 S2P-Balanced (REDD)
    ├── train_auglpn_redd.py                       # 训练 AugLPN (REDD)
    ├── train_regularized_lstm_redd.py             # 训练 LSTM (REDD)
    └── train_edge_tcnformer_redd.py               # 训练 EdgeTCNFormer (REDD)
```
# 实验环境
本代码在以下环境中开发和测试:
```
操作系统: Ubuntu 20.04 (AutoDL 容器)
Python: 3.8.10
TensorFlow: 2.9.0
GPU: NVIDIA RTX 4090
CPU: 16 vCPU Intel(R) Xeon(R) Platinum 8358P CPU @ 2.60GHz
```
# 依赖库安装命令：
```
pip install -r requirements.txt
```
# 数据准备
由于原始数据集体积较大且受GitHub容量限制，本仓库不直接提供原始数据。
我们将经过统一预处理和划分的数据（可直接用于训练和测试）上传至夸克，请按以下步骤操作：
1. 安装夸克APP
2. 点击下载 https://pan.quark.cn/s/3855be053fb3
3. 下载完成将data文件夹保存至MSFFSCT-Net目录下即可

目录结构如下：
```
MSFFSCT-Net/
├── data/           <-- 将下载的数据解压到这里
├── train_transformer_lr0.001.py
├── model_structure_s2p_cbam_transformer.py
└── ...
```
# 快速开始
在 UK-DALE 数据集上训练本文提出的 MSFFSCT-Net 模型：
```
python train_transformer_lr0.001.py
```
在 REDD 数据集上训练本文提出的 MSFFSCT-Net 模型：
```
python train_redd_transformer.py
```

## 预期实验结果

运行训练脚本后，MSFFSCT-Net 模型在测试集上的预期性能如下：

| 数据集 | MAE (W) | NEP | F1 (%) |
| :--- | :--- | :--- | :--- |
| UK-DALE | 9.77 | 0.46 | 82.69 |
| REDD | 24.85 | 0.78 | 53.16 |

注：模型参数量为0.15M，每个电器模型大小约为2.53MB。

## ⚠️ 复现指南（重要）

为了获得与论文完全一致的实验结果，请务必严格遵循以下说明：

1. **使用提供的预处理数据**：请使用夸克网盘提供的 `data` 文件夹，**不要自行下载原始数据并重新预处理**。归一化统计量（均值/标准差）和滑动窗口切分的微小差异，均可能导致性能显著下降。
2. **环境版本**：请严格使用 `requirements.txt` 中指定的库版本（特别是 TensorFlow 2.9）。
3. **随机种子**：我们在训练脚本中固定了 Python、NumPy 和 TensorFlow 的全局随机种子（SEED = 2024）。由于 GPU 底层 cuDNN 算子在默认配置下具有一定的非确定性，不同硬件上跑出的结果可能存在极微小的数值波动。
4. **硬件差异说明**：即使固定了随机种子，不同硬件（如GPU架构、cuDNN版本）仍可能导致微小的结果波动（F1在±1%以内均属正常）。如果出现性能大幅下降，请检查数据放置路径或联系作者。

## 📬 联系方式
如有任何问题，请邮件联系：745051742@qq.com