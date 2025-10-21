# NeurIPS 2025 FANS: A Flatness-Aware Network Structure for Generalization in Offline Reinforcement Learning
# RUN FANS

```bash
Please run td3_fans.py in offline
```
All of the baseline algorithms come from the code library $\mathcal{CORL}$: https://github.com/tinkoff-ai/CORL. All experiments run on a server equipped with an Intel® Xeon® Gold 6254 CPU @ 3.10GHz and NVIDIA GeForce RTX 3090 GPU.

## Getting started

```bash
git clone https://github.com/tinkoff-ai/CORL.git && cd CORL
pip install -r requirements/requirements_dev.txt

# alternatively, you could use docker
docker build -t <image_name> .
docker run --gpus=all -it --rm --name <container_name> <image_name>
```
