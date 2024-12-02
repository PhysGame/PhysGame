<p align="center" width="100%">
<a target="_blank"><img src="example/PhysGame_logo.png" alt="PhysGame" style="width: 50%; min-width: 150px; display: block; margin: auto;"></a>
</p>
<h2 align="center"> <a href="https://">PhysGame: Uncovering Physical Commonsense Violations in Gameplay Videos</a></h2>

<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

<h5 align=center>

[![hf](https://img.shields.io/badge/🤗-Hugging%20Face-blue.svg)](https://huggingface.co/PhysGame)
[![arXiv](https://img.shields.io/badge/Arxiv-2311.08046-b31b1b.svg?logo=arXiv)](https://)
[![License](https://img.shields.io/badge/Code%20License-Apache2.0-yellow)](https://github.com/PhysGame/PhysGame/blob/main/LICENSE)

## Benchmark Evaluation :bar_chart:
To evaluate with our **PhysGame** Benchmark, please following the instructions below

## Installation 🛠️
**Note** that Qwen-2 requires torch >= 2.1.2 and LLaVA-Onevision requires transformers >= 4.45

For implementation, we use torch==2.1.2 + cu11.8 with transformers==4.45.1

Git clone our repository, create a Python environment, and activate it via the following command
```bash
git clone https://github.com/PhysGame/PhysGame.git
cd PhysGame
conda create --name physvlm python=3.10
conda activate physvlm
pip install -r requirement.txt
```
## Demo 🤗
Feel free to ask PhysVLM about game physics!!!

Please download PhysVLM weights from [PhysVLM-DPO](https://huggingface.co/PhysGame/PhysVLM-DPO) and [PhysVLM-SFT](https://huggingface.co/PhysGame/PhysVLM-SFT) first. Then, run the gradio demo:
```
python demo_gradio.py --ckpt-path /path/to/PhysVLM-DPO --gpu-id 0
```
You can also run the demo with only text outputs:
```
python demo.py --ckpt-path /path/to/PhysVLM-DPO --gpu-id 0
```
## Acknowledgement 👍
Our code is built upon [PPLLaVA](https://github.com/farewellthree/PPLLaVA) and [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT), thanks for their excellent works!
## Citation ✏️
If you find the code and paper useful for your research, please consider staring this repo and citing our paper:

```
@article{liu2024ppllava,
  title={PPLLaVA: Varied Video Sequence Understanding With Prompt Guidance},
  author={Liu, Ruyang and Tang, Haoran and Liu, Haibo and Ge, Yixiao and Shan, Ying and Li, Chen and Yang, Jiankun},
  journal={arXiv preprint arXiv:2411.02327},
  year={2024}
}
```
<!--
**PhysGame/PhysGame** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
