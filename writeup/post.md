## Overview
**Abstract**:  In 2019, researchers at DeepMind published a suite of reinforcement learning environments called *Behavior Suite for Reinforcement Learning*, or *bsuite*. Each environment is designed to directly test a core capability of a general reinforcement learning agent, such as its ability to generalize from past experience or handle delayed rewards. The authors claim that *bsuite* can be used to benchmark agents and bridge the gap between theoretical and applied reinforcement learning understanding. In this blog post, we extend their work by providing specific examples of how *bsuite* can address common challenges faced by reinforcement learning practitioners during the development process. Our work offers pragmatic guidance to researchers and highlights future research directions in reproducible reinforcement learning.

**Intended Audience**: We expect that the reader has a basic understanding of deep learning, reinforcement learning, and common deep reinforcement learning algorithms (e.g. DQN, A2C, PPO). Readers without such knowledge may not fully appreciate this work.

**Goals**: The reader should grasp the fundamentals of *bsuite*, understand our motivation for bridging the gap between theoretical and applied reinforcement learning, identify various ideas for incorporating *bsuite* into the reinforcement learning development cycle, and observe possible research directions in reproducible reinforcement learning. 

**Reading Time**: ~30-40 Minutes

**Important Links**: [Paper (PDF)](https://openreview.net/pdf?id=rygf-kSYwH), [OpenReview](https://openreview.net/forum?id=rygf-kSYwH), [GitHub](https://github.com/deepmind/bsuite)

**Table of Contents**:

0. [Introduction](#0-introduction)

    0.1 [Background](#01-background)

    0.2 [BSuite Summary](#02-bsuite-summary)

    0.3 [Motivation](#03-motivation)

    0.4 [Contribution](#04-contribution)
1. [Initial Model Choice](#1-initial-model-choice)
2. [Preprocessing Selection](#2-preprocessing-selection)
3. [Hyperparameter Tuning](#3-hyperparameter-tuning)
4. [Testing and Debugging](#4-testing-and-debugging)
5. [Model Improvement](#5-model-improvement)
6. [Conclusion](#6-conclusion)<br />

## 0. Introduction
...

### 0.1 Background
The current state of reinforcement learning (RL) theory notably lags progress in practice, especially in non-trivial problems. There are examples of programs learning to play Go from scratch at the level past a professional (cite), learning to navigate diverse video games from raw pixels (cite), and (robotics example). While these algorithms have some foundational roots in theory (gradient descent, TD learning, Q-learning watkins?(cite convergence)), the authors point out a fact quite apparent to practitioners in the field: "The current theory of deep RL is still in its infancy" (cite).  Such theory is prized, since a strong theory can help provide insight and direction for improving known algorithms, while providing hints of where to focus new research.

Fortunately, deep learning can provide a blueprint of the interaction between theoretical and practical improvements. During the 'neural network winter', deep learning techniques were disregarded in favor of more theoretically sound convex loss methods. It was only until the creation of benchmark problems mainly for image recognition where deep learning methods proved more powerful that deep learning became a mainline focus of research. Consequently, a renewed interested in deep learning theory followed shortly after, bolstered by the wealth of applied research. Due to the lack of theory in RL, one possible avenue is to follow such blueprint and create well-defined and vetted benchmarks for the understanding of RL algorithms.

To this end, the trend of RL benchmarks has been to increase in complexity and perhaps publicity of the problem. The earliest such benchmarks were simple MDPs that took non-trivial algorithms to solve, such as Cartpole (cite) and MountainCar (cite). Other environments proved to be more diagnostic by directly testing characteristics such as exploration (Riverswim) and temporal abstraction (Taxi). More modern environments have usually proved somewhat difficult (e.g. not achieve perfect play) for humans such as the Atari Learning Environment (ALE cite) and board games such as Go (cite). The corresponding achievements were highly publicized due to the superhuman performance of the algorithms, prompting superhuman performance to be a benchmark threshold in its own light.

### 0.2 BSuite Summary

BSuite goes against the grain of the current benchmark trend, and instead of chasing prized superhuman performance, it provides a complementary approach by removing confounding factors in environments and testing isolated core capabilities of RL agents. These core capabilities include exploration, basic understanding, memory, generalization, noise, scale, and credit assignment. Many current benchmarks are a culmination of many, and possibly all of these confounding capabilities, while BSuite is a suite of environments that separately gauge prowess of these capabilities. Before going too far too quickly, a highlight of BSuite is the output of a radar chart after running an algorithm through all experiments; a sample of the DQN implementation of stable baselines is shown. For an example algorithm... (explain deep sea). As shown in the example, Deep Sea is dependent on N and is therefore scalable. (explain all factors here)
The entirety of BSuite is a collection of 23 experiments with 16-22 possible 'levels?' Each experiment tests one or more core capabilities simultaneously, usually 2 or fewer at a time. We provide a brief summary on the novelty of BSuite with this constrasting table of qualities. As we noted above, BSuite is most akin to the diagnostic environments of river run and taxi. 

| Quality     | BSuite | Traditional Benchmark |
|-------------| ----------- | -----------------|
| Targeted    | performance corresponds to key issue on task       |     performance corresponds to real-life goal             |
| Simple      | Removes confounding factors        |    Many confounding factors              |
| Challenging | Pushes agents beyond normal range        |   Requires performance in many areas but not necessarily extreme in one area               |
| Scalable    | Discerns when algorithms fail through spectrum         | Compares against other algorithms and not against environment                   |
| Fast        | Well-defined episode and experiment lengths with low observation complexity (e.g. high-level features only)        |    Compute intensive for both number of timesteps (long episodes) and observations (e.g. image data)              |


### 0.3 Motivation

A primary motivation for BSuite was that "Our aim is that these experiments can help provide a bridge between theory and practice, with benefits to both sides".  As cogitated in the background section, the creation of clear benchmarks can yield progress in the theoretical realm after applied progress is made. BSuite is highly prospective in this way since the experiments are so directed and that directed experiments allow for hypotheses that eventually turn into "provable guarantees". As such, it is especially important that the applied side is emphasized through the adoption and diverse use of applied DRL practitioners. 

The examples in the paper are rather meagre: there are 2 examples of algorithm comparison on two environments and three comparisons of algorithms, optimizers, and ensemble sizes in the appendix. These examples give hints of how to use bsuite (along with comments in section 2?), but the reasons aren't made clear about the purpose of these experiments (e.g. why these algorithms were being compared) aside from the general notion of testing some algorithms on bsuite. Looking at the paper reviews (cite openreview), reviewer 1 mentioned how there was no explicit conclusion from the evaluation, and reviewer 3 mentioned that more real examples of diagnostic use  would be important; the authors rebutted that the appendices hold some. Furthermore, reviewer 2 mentioned how they wanted to see traction within with the RL community, and the PCs mentioned how success or failure can rely on community acceptance.

### 0.4 Contribution

**Contribution Statement**: To (i) help bridge the gap between theory and practice, (ii) promote community acceptance, (iii) help applied practitioners, and (iv) present research directions for reproducible RL, we showcase 15 explicit and use cases with experimental illustration of bsuite that directly aim to answer specific questions in the RL development cycle. 

We separate our examples into 5 categories of model selection, preprocessing, hyperparameter tuning, debugging, and model improvement, each section with 3 examples. Most examples use Stable Baselines 3 for DRL code, and are runnable with instructions from our (cite-Github). Furthermore, we created a subset of bsuite, which we will refer to as mini-bsuite or mbsuite in this work out of computational necessity. We kept the exponential scaling nature of BSuite in tact in mbsuite, and tried to keep a diversity of topics, this reduced experiments to X and experiments/tag to X. BSuite never claimed to be finished or complete, so we remark that even though our cutoff is arbitrary, so was the one in the original paper, and it can highlight the effectiveness flexibility of not having to run the entire bsuite to elicit insights. Since we are not discussing architecture, only ideas, we turn the interested reader to our codebase and the wonderful tutorials in the paper and in the relevant colab notebooks. 

We stress that these examples are not meant to 'wow' the reader or produce SOTA research in and of themselves, that would merit a paper in its own right. The main product of this work are the practicality and diversity of ideas in the examples, while the experiments are meant to (spur on more thought). Moreover, these experiments use low compute power and highlight the effectiveness of BSuite and diagnotic RL in general in the low-data regime. We highlight how our examples can save developement time, compute time, increase performance, and lessen frustration on the part of the practitioner (note how we didn't say prevent entirely - this is impossible in the field of DRL). Discussion of these savings and discussion on the categories are relegated to the individual categories themselves, and to maintain any sense of brevity, we now jump in to the examples.

## 1. Initial Model Choice

## 2. Preprocessing Selection

## 3. Hyperparameter Tuning

## 4. Testing and Debugging

## 5. Model Improvement

## 6. Conclusion

## References

[Practice](https://en.wikipedia.org/wiki/Hobbit#Lifestyle)