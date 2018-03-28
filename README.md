This repository is for our CS224N Winter 2018 project. The master branch is the starter code provided by the course instructors.
Our separate branches are for the finalized versions of the different models that we tested.
For all of our models we followed: https://arxiv.org/abs/1611.01603

-bidaff_attention implements just the bi-directional attention flow

-CNN_Embedding was a dev branch for implementing the character level word embedding from the paper

-Highway_CE is a full implementation of Bidaff sans the dynamic programming on the output

-SelfAttention adds an additional attention layer similar to: https://arxiv.org/abs/1710.10723

We explored modifying the cross entropy loss to better reflect our desired outputs in two different ways.
One by applying the modification to an entire batch and one on a single sample. These are illustrated in:

-Batch_Mask, Element_Mask

Our final report for the course is at: https://www.overleaf.com/read/fhxrynstdzyk

# cs224n-win18-squad
Code for the Default Final Project (SQuAD) for [CS224n](http://web.stanford.edu/class/cs224n/), Winter 2018

Note: this code is adapted in part from the [Neural Language Correction](https://github.com/stanfordmlgroup/nlc/) code by the Stanford Machine Learning Group.
