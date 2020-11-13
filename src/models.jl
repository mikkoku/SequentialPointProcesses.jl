
abstract type SequentialPointProcess end
@doc raw"""
  Softcore(kernel, kernel_integral=kernel)

Define a Softcore model with kernel.

The kernel is a positive function of distance. Large values indicate repulsion.

The desity for the next point ``x_k`` is
```math
f(x_k | x_{1:k-1}) \propto \exp \left(-\sum_{i=1}^{k-1} f(\| x_i - x_k \|)\right).
```

```kernel_integral``` can be specified to
use a different kernel for computing the normalization coefficients.
"""
struct Softcore{F,F2} <: SequentialPointProcess
  kernel::F
  kernel_integral::F2
end
Softcore(f) = Softcore(f, f)

"""Uniform()

Uniform density.
"""
struct Uniform <: SequentialPointProcess end
"""Mixture(model1, model2, theta)

Defines a Mixture model with components ```model1``` and ```model2``` with
probibilities ```(1-theta)``` and ```theta```.
"""
struct Mixture{T1, T2} <: SequentialPointProcess
  first::T1
  second::T2
  theta::Float64
end

function normalizedlogdensities(m::Softcore, xy, window, int)
  z = normconstants(xy[1:end-1], m.kernel_integral, window, int)
  for i in eachindex(z)
    if z[i] == 0.0
      z[i] = -Inf
    else
      z[i] = logfk(xy[i+1], view(xy, 1:i), m.kernel) - log(z[i])
    end
  end
  z
end
function normalizedlogdensities(::Uniform, xy, window, int)
  x1, x2 = window.x
  y1, y2 = window.y
  A = (x2-x1)*(y2-y1)
  fill(log(1/A), length(xy)-1)
end
function logdens(m::Mixture, xy, window, cubpar)
  0.0 <= m.theta <= 1.0 || return -Inf
  z1 = normalizedlogdensities(m.first, xy, window, cubpar)
  z2 = normalizedlogdensities(m.second, xy, window, cubpar)
  sum(zip(z1, z2)) do (z1, z2)
    log((1-m.theta) * exp(z1) + m.theta * exp(z2))
  end
end
function logdens(m, xy, window, cubpar)
  z1 = normalizedlogdensities(m, xy, window, cubpar)
  sum(z1)
end
