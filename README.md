# Mitigating Temporal Hallucinations in Video Understanding Models through Dynamic Temporal Probing

The official repository of our paper **"Mitigating Temporal Hallucinations in Video Understanding Models through Dynamic Temporal Probing".**

## Model Overview

![model overview](https://anonymous.4open.science/r/DTP-9B1C/figs/model_overview.png)

**Anonymous Submission**

[project page](https://anonymous.4open.science/r/DTP-9B1C)

## Highlights

- We empirically demonstrate the existence of static bias in video understanding tasks and propose strategies to exploit this bias to mitigate computational overhead.
- We innovatively  proposed the DTP model to accelerate computation using static bias and can be seamlessly integrated with existing large-scale models. A processing paradigm for multimodal tasks has been elaborately designed based on it, offering new perspectives and methods for relevant research and applications.

## Abstract

Long video understanding is essential for applications like autonomous driving and motion recognition, where capturing spatiotemporal dynamics is critical for decision-making. However, processing long-duration videos remains computationally challenging due to the need to model temporal relationships between frames. Existing methods, such as frame pooling or segment-based analysis, reduce complexity but often overlook important spatiotemporal dependencies, leading to information loss. In this paper, we demonstrate that for certain problem requirements, only a subset of frames are crucial for task completion, which is referred to as static bias, and propose a novel method for accelerating computations by leveraging static bias. We propose a Dynamic Temporal Probing (DTP) framework, which adaptively chooses whether to select a single frame from the frame sequence to answer the question or keep the frame sequence to answer the question according to the scoring mechanism, avoiding redundant computation in some processes. By integrating visual and textual features through a self-supervised image-language model (e.g. CLIP), DTP enhances frame-level comprehension while maintaining efficiency. Our method reduces computational overhead and improves temporal modeling, achieving faster long video analysis without sacrificing key spatiotemporal information.

## Demo Videos

The demo videos are provided here [demo.zip](https://anonymous.4open.science/r/DTP-9B1C/demo_videos/demo.zip).

**Comparison with MA-LMM on Video Recognition Task (on LVU Dataset)**

<video src="https://anonymous.4open.science/r/DTP-9B1C/demo_videos/Comparison_with_MA-LMM_on_Video_Question_Answering_Task(on_MSVD_Dataset)"></video>

**Comparison with MA-LMM on Video Question Answering Task (on MSVD Dataset)**

<video src="https://anonymous.4open.science/r/DTP-9B1C/demo_videos/Comparison_with_MA-LMM_on_Video_Recognition_Task_.mp4(on_LVU_Dataset)"></video>

## Installation

### **Clone our repository:**

Directly download by clicking [here](https://github.com/Soughtlin/DTP/archive/refs/heads/main.zip)

### Requirements

You can install the conda environment by running:
```bash
cd DTP
pip install -e .
```

If you are running the code on Apple Silicon, you need to use `eva-decord` instead of `decord`. Here is the modification in the `requirements.txt` file you should do:

```text
contexttimer
eva-decord
einops>=0.4.1
fairscale==0.4.4
...
```

Before running `pip install -e .`, ensure you have the correct requirements.

### Feature Extraction

Our key results in the paper are mainly reported using [OpenAI's CLIP](https://github.com/openai/CLIP) as the frozen image-language encoder (ViT-B32). For feature processing, add the following dependencies:

```sh
cd CLIP  # ... make your modifications
pip install -e .  # inside the CLIP/ folder; updates clip API
```

## Dataset

You can download videos for each dataset through the script provided here (lavis/datasets/download_scripts). For LVU/MSVD datasets, please download the original videos through the official link provided above.

Then extract video frames of each video with fps=10. Example preprocess code is provided here [extract_frames.py](https://anonymous.4open.science/r/Diff-LMM-5F42/data/extract_frames.py). Since different FFMPEG versions are used, the actual extracted frame lengths can be slightly inconsistent. You may need to update the actual frame_length for each video in the annotation file.

```
 ├── data
     └── lvu
         ├── annotation
         ├── frames
         ├── videos
     └── msvd
         ├── annotation
         ├── frames
         ├── videos
```

## Running

### Download Pre-trained LLM

Following [MA-LMM](https://github.com/boheumd/MA-LMM), we use Vicuna-7b as our pre-trained LLM weights, you can download from this [link](https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md) as arrange in this format.

```
├── llm
     ├── vicuna-7b
```

### Download Pre-trained DiT

We use DiT-XL/2 as our pre-trained DiT weights, you can download from this [link](https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt) and place it in the [dit](https://anonymous.4open.science/r/Diff-LMM-5F42/dit) directory. At the same time, you need to make sure to download the pre-trained VAE weights from this [link](https://huggingface.co/stabilityai/sd-vae-ft-ema) and place it in the [vae-ft-ema](https://anonymous.4open.science/r/Diff-LMM-5F42/vae-ft-ema) directory.

### Finetuning on Downstreaming Tasks

Our training process occurred on four H100 GPUs. If you would like to fine-tune the model for various video datasets, please run the following command:

**LVU:**

```bash
bash run_scripts/${dataset}/train.sh
```

**msvd_qa:**

```bash
bash run_scripts/${dataset}/train_qa.sh
```

#### LVU dataset

```bash
    # Please choose the task from the following list
    # ['director', 'genre', 'relationship', 'scene', 'way_speaking', 'writer', 'year']
    datasets.lvu_cls.task ${task}
```

## Acknowledgement

We referenced the repo below for the code

- [MA-LMM](https://github.com/boheumd/MA-LMM)