# generate-words
Collection of Python scripts implementing various AI algorithms to generate words. Uses PyTorch framework.

These implementations are inspired by/taken from a series of Youtube videos on the topic made by Andrej Karpathy! 

The included *names.txt* dataset, as an example, has the most common 32K names takes from *ssa.gov* for the year 2018, taken from Andrej's GitHub.

Following scrips are at present there:

1. *BigramNNetwork.py* - simple, 1 layer neural network implementing a (very simple) bi-gram language model

2. *MultilayerPerceptron.py* - multi layer neural network implementing the MLP proposed in [*"A Neural Probabilistic Language Model"*, Bengio et al. 2003](https://home.cs.colorado.edu/~mozer/Teaching/syllabi/6622/papers/BengioDucharmeVincentJauvin2003.pdf)
![MLP architecture](<Bengio et al NN architecture.jpg>)

3. *MultilayerPerceptron-BatchNorm-1-discussion.py* - contains discussion of several possible optimizations to the MLP network from the previous step. As result, the Batch Norm optimization is chosen for further implementation

4. *MultilayerPerceptron-BatchNorm-2.py* - adds a Batch Norm layer to our previous MLP network implementation

![BN formulas](<BN article formulas.jpg>)

5. *MultilayerPerceptron-BatchNorm-3.py* - refactors the above implementations adding classes which can be composed to implement a NN architecture, similiar to the classes found in the *PyTorch* framework

6. *MultilayerPerceptron-BatchNorm-4-backprop.py* - some excercises in manual backpropagation on our previous NN to build an understanding of the technique

![backprop excercise](<Batchnorm-4-backprop - Exc.3.jpg>)

7. *MultilayerPerceptron-WaveNet.py* - changes the MLP implementation from step 5 by using WaveNet's idea 
   - don't squash context inito a single vector but use a binary tree!!
   - OPEN TODO::: explain more !!!!!!!!!
   
![VaweNet layers](<WaveNet layers representation.jpg>)

An additional series of implementations starts with:

1. *NanoGpt-Train.py* - a basic bigram model for a new dataset (*tinyshakesperate.txt*) 
    - OPEN TODO:: starting point for GPT reimplementation ???

