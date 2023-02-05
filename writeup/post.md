## Overview
**Abstract**:  In 2019, researchers at DeepMind published a suite of reinforcement learning environments called *Behavior Suite for Reinforcement Learning*, or *bsuite*. Each environment is designed to directly test a core capability of a general reinforcement learning agent, such as its ability to generalize from past experience or handle delayed rewards. The authors claim that *bsuite* can be used to benchmark agents and bridge the gap between theoretical and applied reinforcement learning understanding. In this blog post, we extend their work by providing specific examples of how *bsuite* can address common challenges faced by reinforcement learning practitioners during the development process. Our work offers pragmatic guidance to researchers and highlights future research directions in reproducible reinforcement learning.

**Intended Audience**: We expect that the reader has a basic understanding of deep learning, reinforcement learning, and common deep reinforcement learning algorithms (e.g. DQN, A2C, PPO). Readers without such knowledge may not fully appreciate this work.

**Goals**: The reader should grasp the fundamentals of *bsuite*, understand our motivation for bridging the gap between theoretical and applied reinforcement learning, identify various ideas for incorporating *bsuite* into the reinforcement learning development cycle, and observe possible research directions in reproducible reinforcement learning.

**Reading Time**: ~30-40 Minutes

**Important Links**: [Paper (PDF)](https://openreview.net/pdf?id=rygf-kSYwH), [OpenReview](https://openreview.net/forum?id=rygf-kSYwH), [GitHub](https://github.com/deepmind/bsuite), [Colab Intro Tutorial](https://colab.research.google.com/drive/1rU20zJ281sZuMD1DHbsODFr1DbASL0RH), [Colab Analysis Tutorial](https://colab.research.google.com/github/deepmind/bsuite/blob/master/bsuite/analysis/results.ipynb), 

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
The current state of reinforcement learning (RL) theory notably lags progress in practice, especially in challenging problems. There are examples of deep reinforcement learning (DRL) agents learning to play Go from scratch at the professional level ([Silver et al., 2016](https://www.nature.com/articles/nature16961)), learning to navigate diverse video games from raw pixels ([Mnih et al., 2015](https://www.nature.com/articles/nature14236)), and learning to manipulate objects with robotic hands ([Andrychowicz et al., 2020](https://journals.sagepub.com/doi/10.1177/0278364919887447)). While these algorithms have some foundational roots in theory, including gradient descent ([Bottou, 2010](https://link.springer.com/chapter/10.1007/978-3-7908-2604-3_16)), TD learning ([Sutton, 1988](https://link.springer.com/article/10.1007/BF00115009)), and Q-learning ([Watkins, 1992](https://link.springer.com/article/10.1007/BF00992698)), the authors of *bsuite* point out that, "The current theory of deep reinforcement learning is still in its infancy" ([Osband et al., 2020](https://iclr.cc/virtual_2020/poster_rygf-kSYwH.html)).  A strong theory is prized since it can help provide insight and direction for improving known algorithms, while hinting at future research directions.

Fortunately, the plain deep learning (DL) provides a blueprint of the interaction between theoretical and practical improvements. During the 'neural network winter', deep learning techniques were disregarded in favor of more theoretically sound convex loss methods ([Cortes & Vapnik, 1995](https://link.springer.com/article/10.1007/BF00994018)), even though the main ideas and successful demonstrations existed many years previously ([Rosenblatt, 1958](https://psycnet.apa.org/record/1959-09865-001); [Ivakhenko, 1968](https://en.wikipedia.org/wiki/Alexey_Ivakhnenko); [Fukushima, 1979](https://en.wikipedia.org/wiki/Kunihiko_Fukushima)). It was only until the creation of benchmark problems mainly for image recognition ([Krizhevsky et al., 2012](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)) when deep learning methods proved more powerful that deep learning earned the research spotlight. Consequently, a renewed interested in deep learning theory followed shortly after ([Kawaguchi, 2016](https://proceedings.neurips.cc/paper/2016/hash/f2fc990265c712c49d51a18a32b39f0c-Abstract.html); [Bartlett et al., 2017](https://proceedings.neurips.cc/paper/2017/hash/b22b257ad0519d4500539da3c8bcf4dd-Abstract.html); [Belkin et al., 2019](https://www.pnas.org/doi/abs/10.1073/pnas.1903070116)), bolstered by the considerable wealth of applied research. Due to the lack of theory in DRL and the proximity of the DL and DRL research fields, <span style="color: red;">one possible avenue is to follow the blueprint laid out by deep learning reserach and create well-defined and vetted benchmarks for the understanding of DRL algorithms</span>.

To this end, the trend of RL benchmarks has been an increase in overall complexity and perhaps the publicity potential. The earliest such benchmarks were simple MDPs that served as basic testbeds with fairly obvious solutions, such as *Cartpole* ([Barto et al., 1983](https://ieeexplore.ieee.org/abstract/document/6313077)) and *MountainCar* ([Moore, 1990](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-209.html)). Other environments proved to be more diagnostic by directly testing characteristics such as *RiverSwim* ([Strehl & Littman, 2008](https://www.sciencedirect.com/science/article/pii/S0022000008000767)) for exploration and *Taxi* ([Dietterich, 2000](https://www.jair.org/index.php/jair/article/view/10266)) for temporal abstraction. More modern environments such as the *ATARI Learning Environment* ([Bellemare et al., 2013](https://www.jair.org/index.php/jair/article/view/10819)) and board games such as *Chess*, *Go*, and *Shogi* are complex and prove difficult for humans, with even the best humans unable to achieve perfect play. The corresponding achievements were highly publicized ([Silver et al., 2016](https://www.nature.com/articles/nature16961); [Mnih et al., 2015](https://www.nature.com/articles/nature14236)) due to the superhuman performance of the agents, vaulting superhuman performance on virtually any environment to be a standard benchmark in its own light ([Vinyals et al., 2019](https://www.nature.com/articles/s41586-019-1724-z); [Silver et al., 2019](https://www.science.org/doi/abs/10.1126/science.aar6404); [Perolat et al., 2022](https://www.science.org/doi/abs/10.1126/science.add4679); [Ecoffet et al., 2021](https://www.nature.com/articles/s41586-020-03157-9); [Bakhtin et al., 2022](https://www.science.org/doi/abs/10.1126/science.ade9097)).

### 0.2 BSuite Summary

The *bsuite* benchmark ([Osband et al. , 2020.](https://iclr.cc/virtual_2020/poster_rygf-kSYwH.html)) goes against the grain of the current benchmark trend of increasing complexity and publicity. Instead of chasing superhuman performance, it acts as a complement to existing benchmarks by creating 23 environments with minimial confounding factors to test 7 isolated core capabilities of RL agents, as follows: **basic competency**, **exploration**, **memory**, **generalization**, **noise**, **scale**, and **credit assignment**.  Current benchmarks often contain most of these confounding capabilities within a single environment, whereas BSuite creates separate environments to gauge prowess of these capabilities through making the environments scalable, with anywhere from 16 to 22 levels of difficulty. With respect to traditional benchmark environments described above, *bsuite* is most akin to the diagnostic environments of *RiverSwim* ([Strehl & Littman, 2008](https://www.sciencedirect.com/science/article/pii/S0022000008000767)) for and *Taxi* ([Dietterich, 2000](https://www.jair.org/index.php/jair/article/view/10266)). 

As a quick illustration, a *bsuite* evaluation of an agent yields a radar chart that displays the agent's score from 0 to 1 on all seven capabilities. Scores near 0 indicate poor performance, often akin to an agent acting randomly, while scores near 1 indicate mastery of all environment difficulties. A central premise of *bsuite* is that <span style="color: red;">if an agent scores high on certain environments, then it is much more likely to exhibit the associated core capabilities due to the targeted nature of the environments</span>. Therefore, the agent will more likely perform better on a challenging environment that contains many of the capabilities than one with lower scores on bsuite. The targeted nature of *bsuite* can provide insights such as eliciting bottlenecks and revealing scaling properties that are opaque in traditional benchmarks. By removing confounding factors, *bsuite* aims to provide more concrete insights of conceptual ideas. This premise is corroborated by current research that shows how insights on small-scale environments can still hold true on large-scale environments ([Ceron et al., 2021](https://proceedings.mlr.press/v139/ceron21a.html)). Due to its high-quality codebase and clear experiments, *bsuite* provides a benchmark that aids research in RL reproducibility.

An example algorithm is Deep Sea that is targeted towards assessing exploration power. As shown in the picture, Deep Sea is and $N \times N$ grid with starting state at cell $(1, 1)$ and treasure at $(N, N)$. The agent has two actions, left and right, and the goal is to reach the treasure and receive a reward of $1$ by always moving to the right. A reward of $0$ is given to the agent for moving left at a timestep; a reward $-0.01/N$ is given for moving to the right. Note that a human can spot an optimal policy (always move right) nearly instantaneously, while we will show in Section X that traditional benchmark DRL agents can fail miserably. 

The **challenge** of this environment is that an agent must choose to explore and receive some negative rewards by going to the right in order to reach the large positive reward of reaching the treasure. This environment **targets** the exploration power of the agent through its **simplistic** implementation by ensuring that a successful agent will explore the state space and not 'give in' to the greedy action of moving left after a couple of episodes. Furthermore, this environment can provide a non-binary score of exploration by **scaling** the environment size $N$ and determining where the agent starts to fail. Finally, the implementation of the environment yields **fast** computation, allowing multiple, quick runs. These five aforementioned qualities are encompassed by all *bsuite* environments, and we contrast such environments with traditional benchmark environments in the below table.

| Quality         | BSuite Environment                                                                                 | Traditional Benchmark Environment                                                                      |
|-----------------|----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| **Targeted**    | Performance on environment is directly related with one or few core capabilities.                  | Performance on environment subtly related to many or all core capabilities.                            |
| **Simple**      | Removes confounding factors related to performance.                                                | Exhibits many confounding factors related to performance.                                              |
| **Challenging** | Pushes agents beyond normal range in one or few core capabilities.                                 | Requires competency in many core capabilities but not necessarily past normal range in any capability. |
| **Scalable**    | Discerns agent's competency of core capabilities through increasingly more difficult environments. | Discerns agent's power through comparing against other algorithms and human performance.               |
| **Fast**        | Relatively small episode and experiment lengths with low observation complexity.                   | Long episodes with computationally-intensive observations.                                             |


### 0.3 Motivation

The authors of *bsuite* stated, "Our aim is that these experiments can help provide a bridge between theory and practice, with benefits to both sides" ([Osband et al. , 2020.](https://iclr.cc/virtual_2020/poster_rygf-kSYwH.html)).  As discussed in the background section, establishing clear benchmarks can yield progress in the theoretical realm after applied progress is made in the field. The use of BSuite seems highly prospective in this fashion since its environments are targeted, which allow for hypothesis testing and eventual formalization into provable guarantees. As such, <span style="color: red;">it is instrumental that the applied aspect of *bsuite* is emphasized through the adoption and diverse use of applied DRL practitioners</span>. 

The applied examples in the published paper are rather meagre: there are two examples of algorithm comparison on two specific environments and three example comparisons of algorithms, optimizers, and ensemble sizes across the entire *bsuite* gamut in the appendix. The examples on the specific environments showcase how *bsuite* can be used for directed algorithm improvement, but the experiments in the appendices only discuss the general notion of algorithm comparison using *bsuite* scores. In addition, some comments throughout the paper provide hints regarding the applied usage of *bsuite*.

Looking at the [paper reviews](https://openreview.net/forum?id=rygf-kSYwH), [reviewer #1](https://openreview.net/forum?id=rygf-kSYwH&noteId=rkxk2BR3YH) mentioned how there was no explicit conclusion from the evaluation, and [reviewer #3](https://openreview.net/forum?id=rygf-kSYwH&noteId=rJxjmH6otS) mentioned that examples of diagnostic use and concrete examples would help support the paper. Furthermore, [reviewer #2](https://openreview.net/forum?id=rygf-kSYwH&noteId=SJgEVpbAFr) encouraged its publication at a top venue to see traction within with the RL research community, and the [program chairs](https://openreview.net/forum?id=rygf-kSYwH&noteId=7x_6G9OVWG) mentioned how success or failure can rely on community acceptance. Considering that it received a spotlight presentation at ICLR 2020 and has amassed over 100 citations in the relatively small field of RL reproducibility in the past few years, *bsuite* has all intellectual merit and some community momentum to reach the level of a top-tier and timeless benchmark in RL research. <span style="color: red;">To elevate *bsuite* to the status of a top-tier RL benchmark and to help bridge the theoretical and applied sides of RL, we beleive that it is necesssary to develop and present concrete *bsuite* examples that display its diagnostic and insightful applied nature</span>.   

### 0.4 Contribution

**Contribution Statement**: This blog post extends the work of *bsuite* by showcasing 15 explicit use cases with experimental illustration that directly address specific questions in the RL development cycle to (i) help bridge the gap between theory and practice, (ii) promote community acceptance, (iii) aid applied practitioners, and (iv) highlight potential research directions for reproducible RL. 

We separate our examples into 5 categories of model selection, preprocessing, hyperparameter tuning, debugging, and model improvement, each section with 3 examples. Most examples use Stable Baselines 3 for DRL code, and are runnable with instructions from our (cite-Github). Furthermore, we created a subset of bsuite, which we will refer to as mini-bsuite or mbsuite in this work out of computational necessity. We kept the exponential scaling nature of BSuite in tact in mbsuite, and tried to keep a diversity of topics, this reduced experiments to X and experiments/tag to X. BSuite never claimed to be finished or complete, so we remark that even though our cutoff is arbitrary, so was the one in the original paper, and it can highlight the effectiveness flexibility of not having to run the entire bsuite to elicit insights. Since we are not discussing architecture, only ideas, we turn the interested reader to our codebase and the wonderful tutorials in the paper and in the relevant colab notebooks. To maintain any notion of brevity in this blog post, we refer the reader to the [paper appendix](https://openreview.net/pdf?id=rygf-kSYwH#page=13) and one of the [colab tutorials](https://colab.research.google.com/github/deepmind/bsuite/blob/master/bsuite/analysis/results.ipynb) for more information about the environments. 

We stress that these examples are not meant to 'wow' the reader or produce SOTA research in and of themselves, that would merit a paper in its own right. The main product of this work are the practicality and diversity of ideas in the examples, while the experiments are meant to (spur on more thought). Moreover, these experiments use low compute power and highlight the effectiveness of BSuite and diagnotic RL in general in the low-data regime. We highlight how our examples can save developement time, compute time, increase performance, and lessen frustration on the part of the practitioner (note how we didn't say prevent entirely - this is impossible in the field of DRL). Discussion of these savings and discussion on the categories are relegated to the individual categories themselves, and to maintain any sense of brevity, we now jump in to the examples.

## 1. Initial Model Choice

## 2. Preprocessing Selection

## 3. Hyperparameter Tuning

## 4. Testing and Debugging

## 5. Model Improvement

## 6. Conclusion

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