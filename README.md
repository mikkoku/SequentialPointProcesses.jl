# SequentialPointProcesses

This package defines a few sequential point processes and
implements ```logpdf``` and ```rand``` for them.

## Installation

```julia
import Pkg
Pkg.add("https://github.com/mikkoku/PointPatternStatistics.jl")
Pkg.add("https://github.com/mikkoku/SequentialPointProcesses.jl")
```

## Softcore model

The Softcore model is parametrised by a kernel $f$. The density for the point $y$
given the $k$ earlier points is proportional to $$ \exp(\sum_{i=0}^k f(\|x_i - y\|)$$.
See https://arxiv.org/abs/2005.01517 for more details.


```julia
using SequentialPointProcesses
using Distributions
M1(theta) = Softcore(d -> (2.3/d)^(theta))
pp = rand(M1(4.5), (x=(0,10), y=(0,10)), 10)
logpdf(M1(4.5), pp, 100)
```

## Other models

Mixture and Uniform.


### Implementation of Softcore model

The method for ```rand``` uses simple rejection algorithm with uniform proposals
and thus may fail if the density is generally low.

The method for ```logpdf``` uses grid based numerical integration and the grid
size has to be specified ```logpdf(model, pointpattern, N)``` where `N` is the number
of integration points in the first dimension.

#### Multithreading

It is possible to use threads by using
```julia
logpdf(M1(4.5), pp, SequentialPointProcesses.Options(nx=100, parallel=:threads))
```
and CUDA by using
```julia
using CUDA
M1cuda(theta::T) where T = Softcore(d -> (2.3/d)^(theta), d -> CUDA.pow((T(2.3)/d), theta))
logpdf(M1cuda(4.5), pp, SequentialPointProcesses.Options(nx=100, parallel=:cuda))
logpdf(M1cuda(4.5f0), pp, SequentialPointProcesses.Options(nx=100, parallel=:cuda, cudatype=Float32))
```
where `cudatype` can Float32 or Float64 and `cudabatchsize`
can be increased to decrease the amount of GPU memory required.
Using CUDA might require specifying a CUDA friendly kernel.
