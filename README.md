# In Search of Robust Measures of Generalization

**Gintare Karolina Dziugaite**, **Alexandre Drouin**, Brady Neal, Nitarshan Rajkumar, Ethan Caballero, Linbo Wang, Ioannis Mitliagkas, Daniel M. Roy

>One of the principal scientific challenges in deep learning is explaining generalization, i.e., why the particular way the community now trains networks to achieve small training error also leads to small error on held-out data from the same population. It is widely appreciated that some worst-case theories -- such as those based on the VC dimension of the class of predictors induced by modern neural network architectures -- are unable to explain empirical performance. A large volume of work aims to close this gap, primarily by developing bounds on generalization error, optimization error, and excess risk. When evaluated empirically, however, most of these bounds are numerically vacuous. Focusing on generalization bounds, this work addresses the question of how to evaluate such bounds empirically. Jiang et al. (2020) recently described a large-scale empirical study aimed at uncovering potential causal relationships between bounds/measures and generalization. Building on their study, we highlight where their proposed methods can obscure failures and successes of generalization measures in explaining generalization. We argue that generalization measures should instead be evaluated within the framework of distributional robustness.


![Cover figure](https://github.com/nitarshan/robust-generalization-measures/raw/master/paper_graphic.png)

This repository holds the code and data for our NeurIPS 2020 paper.

## Code

We are working on cleaning and updating the code. Stay tuned for an update. If you really cannot wait, you can look at this [anonymized repository](https://github.com/nitarshan/banana-smoothie-recipe-1776), although the code is slightly outdated and hard to read.


## Data

The full data set used in our study will be made available here. We will provide 1) a csv file with all experimental records (model configurations, generalization measures, generalization error), 2) we also expect to be able to release the weights of the trained models (although this is a technical challenge due to the size of the data). Stay tuned.


## Contact us

karolina.dziugaite@elementai.com, alexandre.drouin@elementai.com
