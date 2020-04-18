# Julia_notebooks
Jupyter notebooks on [Julia](http://julialang.org/) programming created by [Milton Huang](http://emotrics.com/people/milton/)

## Setup
Julia pulls in external libraries with the `using` command. If you are using a notebook on your local system, you make need to install the related package with `import Pkg` and `Pkg.add("NameOfPackage")`.

## Vizualization
* [Colors](colors.ipynb): basics of how color is used in Julia
* Plots - I have examples of [plotting using the Covid-19](covid/covid.ipynb) data from Johns Hopkins

## Neural nets
* [perceptron](perceptron.ipynb): playing with the elementary unit of a neural net
* [MXNet](mxnet): folder of notebooks using [MXNet](http://mxnet.io/) framework
  * [Multi Layer Perceptron (MLP)](mxnet/mnistMLP.ipynb): basic use of MXNet.jl to create an MLP
  * [ConvNet](mxnet/mnistLenet.ipynb): use of MXNet to create Convolutional Neural Networks
  * Recurrent Neural Net (RNN)

------------------------

[![Creative Commons License](https://i.creativecommons.org/l/by-sa/4.0/88x31.png)](http://creativecommons.org/licenses/by-sa/4.0/) The content of these notebooks is licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).

Julia code is licensed under MIT.
