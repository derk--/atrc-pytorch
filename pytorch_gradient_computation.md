
## Computational Graphs and Gradient Computations in PyTorch

PyTorch has a built in library called autograd to make function differentation and gradient computation painless.


```python
%pylab inline
from __future__ import print_function
import torch
import numpy as np
from torch.autograd import Variable
```

    Populating the interactive namespace from numpy and matplotlib


## Computation Graphs and Variables

A <i>computation graph</i> is a structure that conceptualizes how composite functions are computed. Consider the following:


```python
X = torch.Tensor([[2,2],[2,2]])
W = torch.Tensor([[0.5,0.5],[0.5,0.5]])
a = torch.mm(X,W)
b = torch.Tensor([4])
Y = a + b
print(a, b, Y)
```

    
     2  2
     2  2
    [torch.FloatTensor of size 2x2]
     
     4
    [torch.FloatTensor of size 1]
     
     6  6
     6  6
    [torch.FloatTensor of size 2x2]
    


We can see that Y is a composite function of a, which is a composite function of X and W. We can visualize this computation graph: <img src="https://www.safaribooksonline.com/library/view/deep-learning-with/9781788624336/assets/4bffe8e5-599a-424e-8217-abbdee0cc8b2.png"> (1)

**PyTorch implements backprop** by passing gradients along a computation graph. In some deep learning frameworks (TensorFlow; Keras) one needs to define the graph explicitly. In PyTorch, the computation graph is built behind the scenes, implicitly, according to the order in which you specify operations. 

PyTorch does not automatically through Tensors into a computation graph, however. To tell PyTorch we want a Tensor to be part of a computation graph we wrap it in a Variable( ). Almost all tensors can be wrapped as variables. 


```python
X = Variable(torch.Tensor([[2,2],[2,2]]))
W = Variable(torch.Tensor([[0.5,0.5],[0.5,0.5]]))
a = torch.mm(X,W)
b = Variable(torch.Tensor([4]))
Y = torch.add(a,b)
print(a, b, Y)
```

    Variable containing:
     2  2
     2  2
    [torch.FloatTensor of size 2x2]
     Variable containing:
     4
    [torch.FloatTensor of size 1]
     Variable containing:
     6  6
     6  6
    [torch.FloatTensor of size 2x2]
    


Note that PyTorch automatically placed a and Y into a Variable since its arguments were also variables. 

Variables are very small wrappers of a tensor. They have 2 important attributes: <b>data</b> returns the Tensor wrapped in the variable; <b>grad</b> stores any gradient values computed for this variable.

Note that the interface for a Variable and Tensor is essentially the same: You can do all the same ops on a Variable as you can a Tensor.


```python
print('data wrapped in Y: ', Y.data)
print('gradient stored in Y: ',Y.grad)

X = Y + 5
Z = torch.max(X,0) ##Note that Z is still a variable. 
print('maximum along row entries:', Z)
```

    data wrapped in Y:  
     4  4
     5  5
    [torch.FloatTensor of size 2x2]
    
    gradient stored in Y:  None
    maximum along row entries: (Variable containing:
     10
     10
    [torch.FloatTensor of size 2]
    , Variable containing:
     1
     1
    [torch.LongTensor of size 2]
    )


## Running Gradients

It is super easy to compute the gradient of some variable. Remember that the gradient is the composition of all partial derivitives of the function w.r.t. all of its variable parameters. We call .backward() on the Variable V we want to compute the gradient for, and the gradient of each parameter p with respect to V is stored in p.grad. 

$Y$ is a matrix. Write the components of $Y$ as $$Y_{ij} = f_{ij}(X,W) + g_{ij}(b)$$
where \begin{align} f_{11} &= [XW]_{11} = x_{11}w_{11} + x_{12}w_{12}\\
f_{12} &= [XW]_{12} = x_{11}w_{12} + x_{12}w_{22}\\ 
f_{21} &= [XW]_{21} = x_{21}w_{11} + x_{22}w_{21}\\ 
f_{22} &= [XW]_{22} = x_{21}w_{12} + x_{22}w_{22}\end{align}

backward() requires as input an initial gradient signal to pass through (e.g. an initial gradient value). In the case of the computation graph above, if we call Y.backward(torch.Tensor([[1,1],[1,1]])), that means we will compute the gradient of Y with respect to variables b, X, and W, where $\frac{\partial Y}{\partial f_{ij}} = 1$ and $\frac{\partial Y}{\partial g_{ij}} = 1$ for all $i,j$. We need to specify this initial "gradient signal" so that all the upstream partials can be computed. Remember that in the backprop algorithm, we pass the **same** gradient signal to all upstream functions.

When backward() is not given a parameter but the variable is a scalar, the default value for the initial gradient is 1.

Lets do some quick calculus. Recall the chain rule:
$$\frac{\partial f}{\partial w_i} = \sum_{d=1}^D \frac{\partial f}{\partial g_d}\frac{\partial g_d}{\partial w_i}$$
where $f$ is a function of $g_d$ and $g_d$ is a function of $w_i$.

If the initial gradient signal is $\frac{\partial Y}{\partial f_{ij}} = \frac{\partial Y}{\partial g_{ij}} = 1$, what will the gradients be at the variables $X, W$, and $b$? Let's do $b$ first:

$$\frac{\partial Y}{\partial b} = \sum_{i=1}^2\sum_{j=1}^2 \frac{\partial Y}{\partial g_{ij}}\frac{\partial g_{ij}}{\partial b}$$

since $Y$ is a function of all $g_{ij}$. We have $g_{ij}(b)= b$, so $\frac{\partial g_{ij}}{\partial b} = 1$, so 
$\frac{\partial Y}{\partial b} = 4$.

Now lets do the gradient for $X$. Think of $X$ as really 4 parameters $x_{11}, x_{12}, x_{21}, x_{22}$ whose gradients
we need. We see $Y$ is a function of all $f_{ij}$, and so by the chain rule: 

$$\frac{\partial Y}{\partial x_{11}} = \sum_{d=1}^2 \frac{\partial Y}{\partial f_{1d}}\frac{\partial f_{1d}}{\partial x_{11}} \;\;\;\;\; \frac{\partial Y}{\partial x_{12}} = \sum_{d=1}^2 \frac{\partial Y}{\partial f_{1d}}\frac{\partial f_{1d}}{\partial x_{12}} \;\;\;\;\; \frac{\partial Y}{\partial x_{21}} = \sum_{d=1}^2 \frac{\partial Y}{\partial f_{2d}}\frac{\partial f_{2d}}{\partial x_{21}} \;\;\;\;\; \frac{\partial Y}{\partial x_{22}} = \sum_{d=1}^2 \frac{\partial Y}{\partial f_{2d}}\frac{\partial f_{2d}}{\partial x_{22}}$$

since $Y$ are the functions of $f_{11}$ and $f_{21}$ that are the functions of $x_{11}$, and so forth. The gradient of $Y$ is set to 1 everywhere, and the partials of each $f_{ij}$ are straightforward:

\begin{align} \frac{\partial f_{11}}{\partial x_{11}} = w_{11} \;\;\;\;\; 
              \frac{\partial f_{11}}{\partial x_{12}} = w_{21} \\ 
              \frac{\partial f_{12}}{\partial x_{11}} = w_{12} \;\;\;\;\;
              \frac{\partial f_{12}}{\partial x_{12}} = w_{22} \\
              \frac{\partial f_{21}}{\partial x_{21}} = w_{11} \;\;\;\;\;
              \frac{\partial f_{21}}{\partial x_{22}} = w_{21} \\
              \frac{\partial f_{22}}{\partial x_{21}} = w_{12} \;\;\;\;\;
              \frac{\partial f_{22}}{\partial x_{22}} = w_{22} \\
              \end{align}

All $w_{ij} = 0.5$, so the gradients for every $x_{ij}$ should be equal to 1. Now taking the partial of Y with respect to each $w_{ij}$ instead, you will follow a very similar procedure and find that they are equal to 4 for every $w_{ij}$.

**Exercise**: Make sure you have your partial derivatives and the chain rule down pat. Compute the partial of 
$Y$ with respect to every $w_{ij}$ and verify that they are all equal to 4 when the initial gradient signal is 1.

That was kind of tedious. **It's not in PyTorch!**


```python
X = Variable(torch.Tensor([[2,2],[2,2]]), requires_grad=True)
W = Variable(torch.Tensor([[0.5,0.5],[0.5,0.5]]), requires_grad=True)
a = torch.mm(X,W)
b = Variable(torch.Tensor([4]),requires_grad=True)
Y = torch.add(a,b)

## Starting from the gradient signal provided as input, backprop across the computation graph
## to compute the partials of every variable that Y depends on.
Y.backward(torch.Tensor([[1,1],[1,1]]))

## Check the gradients: 
print(W.grad)
print(X.grad)
print(b.grad)
```

    Variable containing:
     4  4
     4  4
    [torch.FloatTensor of size 2x2]
    
    Variable containing:
     1  1
     1  1
    [torch.FloatTensor of size 2x2]
    
    Variable containing:
     4
    [torch.FloatTensor of size 1]
    


Good! Note that we have set the requires_grad parameter of a Variable to True as we require the variable to have a gradient computed on it. When we get build deep nets we will do so layer wise, and the Variables of the layers will have requires_grad to be True automatically. But by default we have requires_grad to False, since for example the input tensor to a deep network does not need to have a gradient computed. 

Another note: when we build deep nets, we usually do not need to specify a parameter for backward. The gradient that comes into the output layer of the network will be computed with respect to a specified loss function. 

(1) https://www.safaribooksonline.com/library/view/deep-learning-with/9781788624336/04621ce7-b316-427e-b03c-eb65fd40d049.xhtml 

Author: Derek Doran, Dept. of CSE, Wright State University, for ATRC Summer 2018 

Homepage: https://derk--.github.io/
