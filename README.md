# AOR: Anatomical Ontology-Guided Reasoning for Medical Large Multimodal Model in Chest X-Ray Interpretation


## Updates

* **[2025/05/05]** We released our research paper on [arXiv](https://arxiv.org/abs/2505.02830).

## TODO List

- [ ] Release Full training code
- [ ] Release AOR-Instruction data
- [ ] Implementation Guide


## Install
1. Clone the `AOR`
```
git clone https://github.com/Liqq1/AOR
cd AOR
```

2. Create the env
```
conda create -n aor python=3.10 -y
conda activate aor
pip install --upgrade pip  # enable PEP 660 support
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121  # install pytorch
pip install setuptools_scm
pip install --no-cache-dir -e .
```
3. Install the `flash-attn` package
```
pip install ninja
pip install flash-attn --no-build-isolation
```


## Training

AOR is trained on 4 NVIDIA A100 GPUs with the following code.

#### Explanation of Environment Variables

```
ONLY_SPI: Whether train spi module (region feature extractor) only.

CLIP: Use openai/CLIP instead of BioCLIP.

V15: Use LLaVA v1.5 instead of LLaVA v1.
```

#### STAGE 1
```
bash train_stage1.sh
```
#### STAGE 2

```
bash train_stage2.sh
```

#### STAGE 3

```
bash train_stage3.sh
```

## Acknowledgement

- [GPT4RoI](https://github.com/jshilong/GPT4RoI)
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [VoCoT](https://github.com/RupertLuo/VoCoT)
