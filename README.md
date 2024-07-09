# SpikeLM
This is the implementation of our paper "SpikeLM: Towards General Spike-Driven Language Modeling via Elastic Bi-Spiking Mechanisms" in ICML 2024. 

 ### A new spiking large language model with 7~70 billion parameters following SpikeLM is here ([SpikeLLM](https://arxiv.org/pdf/2407.04752))

Towards energy-efficient artificial intelligence similar to the human brain, the bio-inspired spiking neural networks (SNNs) have advantages of biological plausibility, event-driven sparsity, and binary activation. Recently, large-scale language models exhibit promising generalization capability, making it a valuable issue to explore more general spike-driven models. However, the binary spikes in existing SNNs fail to encode adequate semantic information, placing technological challenges for generalization. This work proposes the first fully spiking mechanism for general language tasks, including both discriminative and generative ones. Different from previous spikes with {0,1} levels, we propose a more general spike formulation with bi-directional, elastic amplitude, and elastic frequency encoding, while still maintaining the addition nature of SNNs. In a single time step, the spike is enhanced by direction and amplitude information; in spike frequency, a strategy to control spike firing rate is well designed. We plug this elastic bi-spiking mechanism in language modeling, named SpikeLM. It is the first time to handle general language tasks with fully spike-driven models, which achieve much higher accuracy than previously possible. SpikeLM also greatly bridges the performance gap between SNNs and ANNs in language modeling.

<div align=center>
<img width=60% src="https://github.com/Xingrun-Xing/SpikeLM/blob/main/main.png"/>
</div>

## Run

### 1. Requirements:
* We pretrain SpikeLM with a single Nvidia A800 Node (8 GPUs).
* python3, pytorch, transformers, ... are required.

### 2. Data:
* Prepare pretraining data (Wikipedia and BookCorpus) the same as BERT with the max length 128. The pretraining data is also the same as [BiPFT](https://github.com/Xingrun-Xing/BiPFT) for the BERT architecture.

### 3. Steps to run:
* Change directory `cd spikeLM-BERT`
* Run `sh bert_base.sh`
* Following, pretraining on Wikipedia and BookCorpus, and finetuning on GLUE are proformed.
