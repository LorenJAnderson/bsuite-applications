## Overview

**Authors**: {Redacted for peer-review}

**Abstract**:  In 2019, researchers at DeepMind published a suite of reinforcement learning environments called *Behavior Suite for Reinforcement Learning*, or *bsuite*. Each environment is designed to directly test a core capability of a general reinforcement learning agent, such as its ability to generalize from past experience or handle delayed rewards. The authors claim that *bsuite* can be used to benchmark agents and bridge the gap between theoretical and applied reinforcement learning understanding. In this blog post, we extend their work by providing specific examples of how *bsuite* can address common challenges faced by reinforcement learning practitioners during the development process. Our work offers pragmatic guidance to researchers and highlights future research directions in reproducible reinforcement learning.

**Intended Audience**: We expect that the reader has a basic understanding of deep learning, reinforcement learning, and common deep reinforcement learning algorithms (e.g. DQN, A2C, PPO). Readers without such knowledge may not fully appreciate this work.

**Goals**: The reader should grasp the fundamentals of *bsuite*, understand our motivation for bridging the gap between theoretical and applied reinforcement learning, identify various ideas for incorporating *bsuite* into the reinforcement learning development cycle, and observe possible research directions in reproducible reinforcement learning.

**ICLR Paper:** [Osband, Ian, et al. "Behaviour Suite for Reinforcement Learning." 8th International Conference on Learning Representations, 2020.](https://iclr.cc/virtual_2020/poster_rygf-kSYwH.html)

**Important Links**: [Paper](https://openreview.net/pdf?id=rygf-kSYwH), [OpenReview](https://openreview.net/forum?id=rygf-kSYwH), [GitHub](https://github.com/deepmind/bsuite), [Colab Intro Tutorial](https://colab.research.google.com/drive/1rU20zJ281sZuMD1DHbsODFr1DbASL0RH), [Colab Analysis Tutorial](https://colab.research.google.com/github/deepmind/bsuite/blob/master/bsuite/analysis/results.ipynb), 

**Reading Time**: ~30-40 Minutes

**Table of Contents**:

0. [Introduction](#0-introduction)

    0.1 [Background](#01-background)

    0.2 [BSuite Summary](#02-bsuite-summary)

    0.3 [Motivation](#03-motivation)

    0.4 [Contribution](#04-contribution)
1. [Initial Model Choice](#1-initial-model-choice)

   1.1 [Comparing Baseline Algorithms](#11-comparing-baseline-algorithms)

   1.2 [Comparing Off-the-Shelf Implementations](#12-comparing-off-the-shelf-implementations)

   1.3 [Gauging Diminishing Returns of Agent Complexity](#13-gauging-diminishing-returns-of-agent-complexity)

   1.4 [Summary and Future Work](#14-summary-and-future-work)
2. [Preprocessing Selection](#2-preprocessing-selection)
   
   2.1 [Choosing a Better Model vs. Preprocessing](#21-choosing-a-better-model-vs-preprocessing)

   2.2 [Verification of Preprocessing](#22-verification-of-preprocessing)

   2.3 [Other](#23-other)

   2.4 [Summary and Future Work](#24-summary-and-future-work)
3. [Hyperparameter Tuning](#3-hyperparameter-tuning)
   
   3.1 [Unintuitive Hyperparameters](#31)
   
   3.2 [Promising Ranges of Hyperparamters](#32-promising-ranges-of-hyperparameters)

   3.3 [Pace of Annealing Hyperparameters](#33-pace-of-annealing-hyperparameters)

   3.4 [Summary and Future Work](#34-summary-and-future-work)
4. [Testing and Debugging](#4-testing-and-debugging)

   4.1 [Missing Add-on](#41-missing-add-on)

   4.2 [Incorrect Constant](#42-incorrect-constant)

   4.3 [OTS Algorithm Testing](#43-ots-algorithm-testing)

   4.4 [Summary and Future Work](#44-summary-and-future-work)
5. [Model Improvement](#5-model-improvement)
   
   5.1 [Increasing Network Complexity](#51-increasing-network-complexity)

   5.2 [Decoupling or Adding Confidence](#52-decoupling-or-adding-confidence)

   5.3 [Determining Necessary Improvements](#53-determining-necessary-improvements)

   5.4 [Summary and Future Work](#54-summary-and-future-work)
6. [Conclusion](#6-conclusion)

   6.1 [Summary](#61-summary)

   6.2 [Green Computing Statement](#62-green-computing-statement)

   6.3 [Inclusive Computing Statement](#63-inclusive-computing-statement)



## 0. Introduction

Excitement Paragraph

This introduction section provides the necessary background and motivation to understand the importance of our contribution. The background describes how deep learning provides a blueprint for bridging theory to practice, and then discusses traditional reinforcement learning benchmarks. The *bsuite* summary section provides a high-level overview of the core capabilities tested by *bsuite*, an example output (radar plot), an example environment, and a comparison against traditional benchmark environments. The information from these first two sections was primarily distilled from the original *bsuite* publication. In the motivation section presents arguments for increasing the wealth of documented *bsuite* examples, with references to the paper and the reviews. Finally, the contribution section showcases four distinct contributions of our work and provides our rationale for the experiment setups and the content of the remainder of the paper.   

### 0.1 Background
The current state of reinforcement learning (RL) theory notably lags progress in practice, especially in challenging problems. There are examples of deep reinforcement learning (DRL) agents learning to play Go from scratch at the professional level ([Silver et al., 2016](https://www.nature.com/articles/nature16961)), learning to navigate diverse video games from raw pixels ([Mnih et al., 2015](https://www.nature.com/articles/nature14236)), and learning to manipulate objects with robotic hands ([Andrychowicz et al., 2020](https://journals.sagepub.com/doi/10.1177/0278364919887447)). While these algorithms have some foundational roots in theory, including gradient descent ([Bottou, 2010](https://link.springer.com/chapter/10.1007/978-3-7908-2604-3_16)), TD learning ([Sutton, 1988](https://link.springer.com/article/10.1007/BF00115009)), and Q-learning ([Watkins, 1992](https://link.springer.com/article/10.1007/BF00992698)), the authors of *bsuite* acknowledge that, "The current theory of deep reinforcement learning is still in its infancy" ([Osband et al., 2020](https://iclr.cc/virtual_2020/poster_rygf-kSYwH.html)).  A strong theory is prized since it can help provide insight and direction for improving known algorithms, while hinting at future research directions.

Fortunately, deep learning (DL) provides a blueprint of the interaction between theoretical and practical improvements. During the 'neural network winter', deep learning techniques were disregarded in favor of more theoretically sound convex loss methods ([Cortes & Vapnik, 1995](https://link.springer.com/article/10.1007/BF00994018)), even though the main ideas and successful demonstrations existed many years previously ([Rosenblatt, 1958](https://psycnet.apa.org/record/1959-09865-001); [Ivakhenko, 1968](https://en.wikipedia.org/wiki/Alexey_Ivakhnenko); [Fukushima, 1979](https://en.wikipedia.org/wiki/Kunihiko_Fukushima)). It was only until the creation of benchmark problems, mainly for image recognition ([Krizhevsky et al., 2012](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)), deep learning earned the research spotlight due to better scores on the relevant benchmarks. Consequently, a renewed interested in deep learning theory followed shortly after ([Kawaguchi, 2016](https://proceedings.neurips.cc/paper/2016/hash/f2fc990265c712c49d51a18a32b39f0c-Abstract.html); [Bartlett et al., 2017](https://proceedings.neurips.cc/paper/2017/hash/b22b257ad0519d4500539da3c8bcf4dd-Abstract.html); [Belkin et al., 2019](https://www.pnas.org/doi/abs/10.1073/pnas.1903070116)), bolstered by the considerable wealth of applied research. Due to the lack of theory in DRL and the proximity of the DL and DRL research fields, <span style="color: red;">one enticing avenue to accelerate progress in reinforcement learning research is to follow the blueprint laid out by deep learning research and create well-defined and vetted benchmarks for the understanding of deep reinforcement learning algorithms</span>.

To this end, the trend of RL benchmarks has seen an increase in overall complexity and perhaps the publicity potential. The earliest such benchmarks were simple MDPs that served as basic testbeds with fairly obvious solutions, such as *Cartpole* ([Barto et al., 1983](https://ieeexplore.ieee.org/abstract/document/6313077)) and *MountainCar* ([Moore, 1990](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-209.html)). Other benchmarks proved to be more diagnostic by targeting certain capabilities such as *RiverSwim* ([Strehl & Littman, 2008](https://www.sciencedirect.com/science/article/pii/S0022000008000767)) for exploration and *Taxi* ([Dietterich, 2000](https://www.jair.org/index.php/jair/article/view/10266)) for temporal abstraction. Modern benchmarks such as the *ATARI Learning Environment* ([Bellemare et al., 2013](https://www.jair.org/index.php/jair/article/view/10819)) and board games such as *Chess*, *Go*, and *Shogi* are more complex and prove difficult for humans, with even the best humans unable to achieve perfect play. The corresponding achievements were highly publicized ([Silver et al., 2016](https://www.nature.com/articles/nature16961); [Mnih et al., 2015](https://www.nature.com/articles/nature14236)) due to the superhuman performance of the agents, with the agents taking actions that were not even considered by their human counterparts. Consequently, this surge in publicity has vaulted the notion of superhuman performance to be the de facto prize on numerous benchmarks ([Vinyals et al., 2019](https://www.nature.com/articles/s41586-019-1724-z); [Silver et al., 2019](https://www.science.org/doi/abs/10.1126/science.aar6404); [Perolat et al., 2022](https://www.science.org/doi/abs/10.1126/science.add4679); [Ecoffet et al., 2021](https://www.nature.com/articles/s41586-020-03157-9); [Bakhtin et al., 2022](https://www.science.org/doi/abs/10.1126/science.ade9097)).

### 0.2 Summary of *bsuite*

The *bsuite* benchmark ([Osband et al. , 2020.](https://iclr.cc/virtual_2020/poster_rygf-kSYwH.html)) goes against the grain of the current benchmark trend of increasing complexity and publicity. Instead of chasing superhuman performance, it acts as a complement to existing benchmarks by creating 23 environments with minimial confounding factors to test 7 isolated core capabilities of RL agents, as follows: **basic competency**, **exploration**, **memory**, **generalization**, **noise**, **scale**, and **credit assignment**.  Current benchmarks often contain most of these capabilities within a single environment, whereas *bsuite* tailors its environments to target one or a few of these capabilities. Each *bsuite* environment is scalable and has 16 to 22 levels of difficulty, allowing for a more precise analysis of the corresponding capabilities. With respect to the benchmarks described in the preceding paragraph, *bsuite* is most akin to the diagnostic benchmarks of *RiverSwim* ([Strehl & Littman, 2008](https://www.sciencedirect.com/science/article/pii/S0022000008000767)) for and *Taxi* ([Dietterich, 2000](https://www.jair.org/index.php/jair/article/view/10266)). 

As a quick illustration, the *bsuite* evaluation of an agent yields a radar chart that displays the agent's score from 0 to 1 on all seven capabilities. Scores near 0 indicate poor performance, often akin to an agent acting randomly, while scores near 1 indicate mastery of all environment difficulties. A central premise of *bsuite* is that <span style="color: red;">if an agent achieves high scores on certain environments, then it is much more likely to exhibit the associated core capabilities due to the targeted nature of the environments. Therefore, the agent will more likely perform better on a challenging environment that contains many of the capabilities than one with lower scores on *bsuite*</span>. The targeted nature of *bsuite* can provide insights such as eliciting bottlenecks and revealing scaling properties that are opaque in traditional benchmarks. By removing confounding factors, *bsuite* aims to provide more concrete insights of conceptual ideas. This premise is corroborated by current research that shows how insights on small-scale environments can still hold true on large-scale environments ([Ceron et al., 2021](https://proceedings.mlr.press/v139/ceron21a.html)). Due to its high-quality codebase and clear experiments, *bsuite* provides a high-quality benchmark that aids research in RL reproducibility.

<div style="text-align: center;">

![](/home/loren/PycharmProjects/blogpost/writeup/images/radar01.png)

*Figure 1. Radar chart of DQN with 7 core capabilities of bsuite.*

</div>

An example environment is *deep sea* that is targeted towards assessing exploration power. As shown in the picture, *deep sea* is and $N \times N$ grid with starting state at cell $(1, 1)$ and treasure at $(N, N)$. The agent has two actions, left and right, and the goal is to reach the treasure and receive a reward of $1$ by always moving to the right. A reward of $0$ is given to the agent for moving left at a timestep; a reward $-0.01/N$ is given for moving to the right. Note that a human can spot an optimal policy (always move right) nearly instantaneously, while we will show in ([1.1](#11-comparing-baseline-algorithms)) that standard DRL agents fail miserably at this task.

<div style="text-align: center;">

![](/home/loren/PycharmProjects/blogpost/writeup/images/radar02.png)

*Figure 2. Illustration of Deep Sea environment taken from [Osband et al. , 2020](https://iclr.cc/virtual_2020/poster_rygf-kSYwH.html).*

</div>

The **challenge** of *deep sea* is that an agent must choose to explore and receive some negative rewards by going to the right in order to reach the large positive reward of reaching the treasure. This environment **targets** the exploration power of the agent through its **simplistic** implementation by ensuring that a successful agent will explore the state space and not 'give in' to the greedy action of moving left after a couple of episodes. Furthermore, this environment can provide a non-binary score of exploration by **scaling** the environment size $N$ and determining where the agent starts to fail. Finally, the implementation of the environment yields **fast** computation, allowing multiple, quick runs. These 5 aforementioned key qualities are encompassed by all *bsuite* environments, and we contrast such environments against traditional benchmark environments in the below table.

| Key Quality     |   Traditional Benchmark Environment  | *bsuite* Environment                                                                               |
|-----------------|-------------------|----------------------------------------------------------------------------------------------------|
| **Targeted**    | Performance on environment subtly related to many or all core capabilities. | Performance on environment is directly related with one or few core capabilities.                  |
| **Simple**      | Exhibits many confounding factors related to performance. | Removes confounding factors related to performance.                                                |
| **Challenging** | Requires competency in many core capabilities but not necessarily past normal range in any capability. | Pushes agents beyond normal range in one or few core capabilities.                                 |
| **Scalable**    | Discerns agent's power through comparing against other algorithms and human performance. | Discerns agent's competency of core capabilities through increasingly more difficult environments. |
| **Fast**        | Long episodes with computationally-intensive observations. | Relatively small episode and experiment lengths with low observation complexity.                   |


### 0.3 Motivation

The authors of *bsuite* stated, "Our aim is that these experiments can help provide a bridge between theory and practice, with benefits to both sides" ([Osband et al. , 2020.](https://iclr.cc/virtual_2020/poster_rygf-kSYwH.html)).  As discussed in the background section, establishing clear benchmarks can yield progress in the theoretical realm after applied progress is made in the field. The use of BSuite seems highly prospective in this fashion since its environments are targeted, which allow for hypothesis testing and eventual formalization into provable guarantees. As such, <span style="color: red;">it is instrumental that the applied aspect of *bsuite* is emphasized through the adoption and diverse use of applied DRL practitioners</span>. 

The applied examples in the published paper are rather meagre: there are two examples of algorithm comparison on two specific environments and three example comparisons of algorithms, optimizers, and ensemble sizes across the entire *bsuite* gamut in the appendix. The examples on the specific environments showcase how *bsuite* can be used for directed algorithm improvement, but the experiments in the appendices only discuss the general notion of algorithm comparison using *bsuite* scores. In addition, some comments throughout the paper provide hints regarding the applied usage of *bsuite*.

Looking at the [paper reviews](https://openreview.net/forum?id=rygf-kSYwH), [reviewer #1](https://openreview.net/forum?id=rygf-kSYwH&noteId=rkxk2BR3YH) mentioned how there was no explicit conclusion from the evaluation, and [reviewer #3](https://openreview.net/forum?id=rygf-kSYwH&noteId=rJxjmH6otS) mentioned that examples of diagnostic use and concrete examples would help support the paper. Furthermore, [reviewer #2](https://openreview.net/forum?id=rygf-kSYwH&noteId=SJgEVpbAFr) encouraged its publication at a top venue to see traction within with the RL research community, and the [program chairs](https://openreview.net/forum?id=rygf-kSYwH&noteId=7x_6G9OVWG) mentioned how success or failure can rely on community acceptance. Considering that it received a spotlight presentation at ICLR 2020 and has amassed over 100 citations in the relatively small field of RL reproducibility in the past few years, *bsuite* has all intellectual merit and some community momentum to reach the level of a top-tier and timeless benchmark in RL research. <span style="color: red;">To elevate *bsuite* to the status of a top-tier RL benchmark and to help bridge the theoretical and applied sides of RL, we beleive that it is necesssary to develop and present concrete *bsuite* examples that display its diagnostic and insightful applied nature</span>.   

### 0.4 Contribution

**Contribution Statement**: This blog post extends the work of *bsuite* by showcasing 15 explicit use cases with experimental illustration that directly address specific questions in the RL development cycle to (i) help bridge the gap between theory and practice, (ii) promote community acceptance, (iii) aid applied practitioners, and (iv) highlight potential research directions for reproducible RL. 

We separate our examples into 5 categories of **model choice**, **preprocessing selection**, **hyperparameter tuning**, **debugging**, and **model improvement**. Each category answers a specific question during the reinforcement learning developement cycle and provides 3 illustrative examples. Most examples use Stable-Baselines3 (SB3) ([Raffin et al., 2021](https://dl.acm.org/doi/abs/10.5555/3546258.3546526)) for training the DRL agents due to its clarity and simplicity. We provide code and instructions for each experiment in our GitHub codebase (cite). Since the focus of this blog is the discussion of diverse example use cases, not architectural considerations or implementation details, we refer the reader to the [paper appendix](https://openreview.net/pdf?id=rygf-kSYwH#page=13) and the [colab analysis tutorial](https://colab.research.google.com/github/deepmind/bsuite/blob/master/bsuite/analysis/results.ipynb) for more information about the environments and to the [colab intro tutorial](https://colab.research.google.com/drive/1rU20zJ281sZuMD1DHbsODFr1DbASL0RH) and our own codebase (cite) for instructions and examples of the *bsuite* development cycle. 

Due to computational necessity, we created a subset of *bsuite*, which we will refer to as *mini-bsuite* or *mbsuite* in this work that reduced the number of experiments from X to Y and reduced the number of environments per core capability from W to Z. We designed *mbsuite* to mirror the general scaling pattern of each *bsuite* environment and diversity of core capabilities among all *bsuite* environments; a complete description of *mbsuite* can be found in our GitHub codebase (cite). Since *bsuite* was a single `bsuite2019` release and meant to evolve over time, the selection for number and diversity of environments seemed to have an arbitrary threshold; therefore, we don't hesitate to create our own arbitrary threshold resulting in *mbsuite*, and we feel that running experiments on a subset of *bsuite* highlights the strength and flexibility of using a targeted diagnostic benchmark to elicit insights.

We stress that the below examples are not meant to amaze the reader or exhibit state-of-the-art research. <span style="color: red;">The main products of this work are the practicality and diversity of ideas in the examples</span>, while the examples are primarily for basic validation and illustrative purposes. Moreover, these experiments use modest compute power and showcase the effectiveness of *bsuite* in the low-compute regime. Each example has a benefit such as saving development time, shorten compute time, increase performance, and lessen frustration of the practitioner, among other benefits. Discussion of these savings are relegated to the individual categories, and to maintain any sense of brevity, we now begin discussion of the examples.

## 1. Initial Model Selection
The reinforcement learining development cycle typically begins with selecting or being given an underlying environment. Perhaps the first question in the cycle is as follows, "*Which underlying RL model should I choose to best tackle this environment, given my resources*?" Resources can range from the hardware (e.g model size on the GPU), to temporal constraints, to availability of off-the-shelf algorithms ([Liang et al., 2018](https://proceedings.mlr.press/v80/liang18b); [Raffin et al., 2021](https://dl.acm.org/doi/abs/10.5555/3546258.3546526)), to maximimum difficulty of agent implementation. In this section, we illustrate that, while optimally answering the above question may remain out of reach, *bsuite* can be used to provide quantitative answers to those questions.

### 1.1 Comparing Baseline Algorithms

### 1.2 Comparing Off-the-Shelf Implementations

### 1.3 Gauging Diminishing Returns of Agent Complexity

### 1.4 Summary and Future Work

## 2. Preprocessing Choice
Many environments come with various complexities, such as high-dimensional, unscaled observations, unscaled rewards, unnecessary actions, and partially-observable Markov Decision Process (POMDP) dynamics. A natural question to ask is, "*What environment preprocessing techniques will best help my agent attain its goal in this environment*?" While environments sometimes come proprocessed 'out-of-the-box', the classic benchmarking and evaluation paper on *ATARI* ([Machado et al., 2018](https://www.jair.org/index.php/jair/article/view/11182)) states that preprocessing is considered part of the underlying algorithm and is indeed a choice of the practitioner. In this section, we show how *bsuite* can provide insight when selecting the preprocessing methods.

### 2.1 Choosing a Better Model vs. Preprocessing

### 2.2 Verification of Preprocessing

### 2.3 Other

### 2.4 Summary and Future Work

## 3. Hyperparameter Tuning
After selecting a model and determining any preprocessing of the environment, the next step is to train the agent on the environment and gauge its competency. During the training process, initial choices of hyperparameters can play a large role in the agent performance ([Andrychowicz et al., 2021](https://arxiv.org/abs/2006.05990)), ranging from how to explore, how quickly the model should learn from experience, and the length of time that actions are considered to influence rewards. Due to their importance, a question is, "*How can I choose hyperparameters to yield the best performance, given a model?*" In this section, we show how *bsuite* can be used for validation and efficiency of tuning hyperparameters.

### 3.1 Unintuitive Hyperparameters

### 3.2 Promising Ranges of Hyperparameters

### 3.3 Pace of Annealing Hyperparameters

### 3.4 Summary and Future Work

## 4. Testing and Debugging
Known to every practitioner, testing and debugging a program is neraly unavoidable. A common question in the RL development cycle is, "*What tests can I perform to verify that my agent is running as intended?*" Due to the prevalence of silent bugs in RL code and long runtimes, quick unit tests can be invaluable for the practitioner, as shown in successor work to *bsuite* ([Rajan & Hutter, 2019](https://ml.informatik.uni-freiburg.de/wp-content/uploads/papers/19-NeurIPS-Workshop-MDP_Playground.pdf)). In this section, we show how *bsuite* can be used as a sanity check the expectations and assumptions of the implementation, which was mentioned as a use case of *bsuite* in the paper.

### 4.1 Missing Add-on

### 4.2 Incorrect Constant

### 4.3 OTS Algorithm Testing

### 4.4 Summary and Future Work

## 5. Model Improvement
A natural milestone in the RL development cycle is getting an algorithm running bug-free with notable signs of learning. A common follow-up question to ask is "*How can I improve my model to yield better performance?*" The practitioner may consider choosing an entirely new model and repeating some of the above steps; usually, a more enticing option is directly improving the existing model by reusing its core structure and only making minor additions or modifications, an approach taken in the state-of-the-art RAINBOW DQN algorithm ([Hessel et al., 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11796)). In this section, we discuss ideas regarding the improvement of pre-existing somewhat competent models.

### 5.1 Increasing Network Complexity

### 5.2 Decoupling or Adding Confidence

### 5.3 Determining Necessary Improvements

### 5.4 Summary and Future Work

## 6. Conclusion
The above sections complete the main ideas of this paper. We now provide a hindsight summary of our work. Afterwards, we supply statements on green and inclusive computing regarding our contribution. 

### 6.1 Summary

Traditional RL benchmarks contain many confounding variables, which makes post-analysis of agent performance somewhat opaque. In contrast, *bsuite*  provides targeted environments that are meant to gauge agent prowess in one or few core capabilities. The goal of *bsuite* is meant to bridge the gap between practical theory and practical algorithms, yet there currently is no database or list of example use cases for the practitioner. Furthermore, *bsuite* is poised to be a standard RL benchmark for years to come due to its acceptance in a top-tier venue, well-structured codebase, multiple tutorials, and over 100 citations in the past few years in a relatively small field. 

Our work extends *bsuite* by providing 15 concrete examples of its use, with 3 examples in 5 categories. Each category section provides at least one possible avenue of related future work or research. We aim to help propel *bsuite*, and more generally methodical and reproducible RL research, into the mainstream through our explicit examples with simple code. With a diverse set of examples to choose from, we intend applied practitioners to understand more use cases, apply and document the use of *bsuite* in their experiments, and ultimately help bridge the gap between practical theory and practical algorithms. 

### 6.2 Green Computing Statement

The use of *bsuite* can help find directed improvements in algorithms, from high-level model selection and improvement to lower-level debugging, testing, and hyperparameter tuning. Due to the current climate crisis, we feel that thoroughly-tested and accessible ideas that can greatly reduce computational cost should be promoted to a wide audience of researchers.

### 6.3 Inclusive Computing Statement

Many of the ideas in *bsuite* and this post are most helpful in the areas of low compute resources, due to more directed areas of improvment and selection. Due to the seemingly-increasing gap between compute power of various research teams, we feel that thoroughly-tested and accessible ideas that can greatly benefit teams with meagre compute power should be promoted to a wide audience of researchers.

## Acknowledgements
{Redacted for peer-review}

## References

[Silver, David, et al. "Mastering the game of Go with deep neural networks and tree search." nature 529.7587 (2016): 484-489.](https://www.nature.com/articles/nature16961)

[Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." nature 518.7540 (2015): 529-533.](https://www.nature.com/articles/nature14236)

[Andrychowicz, OpenAI: Marcin, et al. "Learning dexterous in-hand manipulation." The International Journal of Robotics Research 39.1 (2020): 3-20.](https://journals.sagepub.com/doi/10.1177/0278364919887447)

[Bottou, Léon. "Large-scale machine learning with stochastic gradient descent." Proceedings of COMPSTAT'2010: 19th International Conference on Computational Statistics, Paris France, August 22-27, 2010 Keynote, Invited and Contributed Papers. Physica-Verlag HD, 2010.](https://link.springer.com/chapter/10.1007/978-3-7908-2604-3_16)

[Sutton, Richard S. "Learning to predict by the methods of temporal differences." Machine learning 3 (1988): 9-44.](https://link.springer.com/article/10.1007/BF00115009)

[Watkins, Christopher JCH, and Peter Dayan. "Q-learning." Machine learning 8 (1992): 279-292.](https://link.springer.com/article/10.1007/BF00992698)

[Osband, Ian, et al. "Behaviour Suite for Reinforcement Learning." 8th International Conference on Learning Representations, 2020.](https://iclr.cc/virtual_2020/poster_rygf-kSYwH.html)

[Cortes, Corinna, and Vladimir Vapnik. "Support-vector networks." Machine learning 20 (1995): 273-297.](https://link.springer.com/article/10.1007/BF00994018)

[Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Proceedings of the 25th International Conference on Neural Information Processing Systems - Volume 1, 2012](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)

[Kawaguchi, Kenji. "Deep learning without poor local minima." Advances in neural information processing systems 29 (2016).](https://proceedings.neurips.cc/paper/2016/hash/f2fc990265c712c49d51a18a32b39f0c-Abstract.html)

[Bartlett, Peter L., Dylan J. Foster, and Matus J. Telgarsky. "Spectrally-normalized margin bounds for neural networks." Advances in neural information processing systems 30 (2017).](https://proceedings.neurips.cc/paper/2017/hash/b22b257ad0519d4500539da3c8bcf4dd-Abstract.html)

[Belkin, Mikhail, et al. "Reconciling modern machine-learning practice and the classical bias–variance trade-off." Proceedings of the National Academy of Sciences 116.32 (2019): 15849-15854.](https://www.pnas.org/doi/abs/10.1073/pnas.1903070116)

[Barto, Andrew G., Richard S. Sutton, and Charles W. Anderson. "Neuronlike adaptive elements that can solve difficult learning control problems." IEEE transactions on systems, man, and cybernetics 5 (1983): 834-846.](https://ieeexplore.ieee.org/abstract/document/6313077)

[Moore, Andrew William. Efficient memory-based learning for robot control. No. UCAM-CL-TR-209. University of Cambridge, Computer Laboratory, 1990.](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-209.html)

[Strehl, Alexander L., and Michael L. Littman. "An analysis of model-based interval estimation for Markov decision processes." Journal of Computer and System Sciences 74.8 (2008): 1309-1331.](https://www.sciencedirect.com/science/article/pii/S0022000008000767)

[Dietterich, Thomas G. "Hierarchical reinforcement learning with the MAXQ value function decomposition." Journal of artificial intelligence research 13 (2000): 227-303.](https://www.jair.org/index.php/jair/article/view/10266)

[Bellemare, Marc G., et al. "The arcade learning environment: An evaluation platform for general agents." Journal of Artificial Intelligence Research 47 (2013): 253-279.](https://www.jair.org/index.php/jair/article/view/10819)

[Silver, David, et al. "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play." Science 362.6419 (2018): 1140-1144.](https://www.science.org/doi/abs/10.1126/science.aar6404)

[Rosenblatt, Frank. "The perceptron: a probabilistic model for information storage and organization in the brain." Psychological review 65.6 (1958): 386.](https://psycnet.apa.org/record/1959-09865-001)

[Ivakhnenko, A. G. "The group method of data handling A rival of stochastic approximation." Soviet Automatic Control 13 (1968): 43-55.](https://en.wikipedia.org/wiki/Alexey_Ivakhnenko)

[Fukushima, Kunihiko. "Neural network model for a mechanism of pattern recognition unaffected by shift in position-neocognitron." IEICE Technical Report, A 62.10 (1979): 658-665.](https://en.wikipedia.org/wiki/Kunihiko_Fukushima)

[Bakhtin, Anton, et al. "Human-level play in the game of Diplomacy by combining language models with strategic reasoning." Science 378.6624 (2022): 1067-1074.](https://www.science.org/doi/abs/10.1126/science.ade9097)

[Perolat, Julien, et al. "Mastering the game of Stratego with model-free multiagent reinforcement learning." Science 378.6623 (2022): 990-996.](https://www.science.org/doi/abs/10.1126/science.add4679)

[Vinyals, Oriol, et al. "Grandmaster level in StarCraft II using multi-agent reinforcement learning." Nature 575.7782 (2019): 350-354.](https://www.nature.com/articles/s41586-019-1724-z)

[Ecoffet, Adrien, et al. "First return, then explore." Nature 590.7847 (2021): 580-586.](https://www.nature.com/articles/s41586-020-03157-9)

[Ceron, Johan Samir Obando, and Pablo Samuel Castro. "Revisiting rainbow: Promoting more insightful and inclusive deep reinforcement learning research." International Conference on Machine Learning. PMLR, 2021.](https://proceedings.mlr.press/v139/ceron21a.html)

[Raffin, Antonin, et al. "Stable-baselines3: Reliable reinforcement learning implementations." The Journal of Machine Learning Research 22.1 (2021): 12348-12355.](https://dl.acm.org/doi/abs/10.5555/3546258.3546526)

[Machado, Marlos C., et al. "Revisiting the arcade learning environment: Evaluation protocols and open problems for general agents." Journal of Artificial Intelligence Research 61 (2018): 523-562.](https://www.jair.org/index.php/jair/article/view/11182)

[Andrychowicz, Marcin, et al. "What matters in on-policy reinforcement learning? a large-scale empirical study." arXiv preprint arXiv:2006.05990 (2020).](https://arxiv.org/abs/2006.05990)

[Rajan, Raghu, and Frank Hutter. "Mdp playground: Meta-features in reinforcement learning." NeurIPS Deep RL Workshop. 2019.](https://ml.informatik.uni-freiburg.de/wp-content/uploads/papers/19-NeurIPS-Workshop-MDP_Playground.pdf)

[Liang, Eric, et al. "RLlib: Abstractions for distributed reinforcement learning." International Conference on Machine Learning. PMLR, 2018.](https://proceedings.mlr.press/v80/liang18b)

[Hessel, Matteo, et al. "Rainbow: Combining improvements in deep reinforcement learning." Proceedings of the AAAI conference on artificial intelligence. Vol. 32. No. 1. 2018.](https://ojs.aaai.org/index.php/AAAI/article/view/11796)