
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
  function Softcore(f, fint, hastime::Bool)
    f2, f2int =
    if hastime
      f, fint
    else
      (d, i) -> f(d), (d, i) -> fint(d)
    end
    new{typeof(f2), typeof(f2int)}(f2, f2int)
  end
end
function Softcore(f, hastime::Bool=false)
  Softcore(f, f, hastime)
end
function Softcore(f, fint)
  Softcore(f, fint, false)
end

struct Hardcore{THETA, R} <: SequentialPointProcess
  theta::THETA
  R::R
end

struct Softcore2{THETA, PSI, R, PHI, OP} <: SequentialPointProcess
  theta::THETA #k -> c + d*m[k]
  psi::PSI #(S, R) = S >= 1
  R::R #(k) = a + b*m[k]
  phi::PHI #(d, R) = d <= R
  op::OP # = +
end

struct Softcore2v{THETA, PSI, R, PHI, OP} <: SequentialPointProcess
  theta::Vector{THETA} #k -> c + d*m[k]
  psi::PSI #(S, R) = S >= 1
  R::Vector{R} #(k) = a + b*m[k]
  phi::PHI #(d, R) = d <= R
  op::OP # = +
end

Softcore2(m::Softcore2v) = Softcore2(k -> m.theta[k], m.psi, k -> m.R[k], m.phi, m.op)

"""
  OverlappingDiscs(theta, R)

  Define a sequential model where the probability of the next point depends on
  how many discs overlap that location. Each point generates a disc `R(i)` and the
  radius of the disc may depend on the id of the point. The
  parameter `theta(noverlaps, k)` is the density corresponding the number
  of overlaps and the id of the next point.

  theta and R may be specified as numbers, vectors of length equal to the
  number of points.
"""
struct OverlappingDiscs{F, F2} <: SequentialPointProcess
  theta::F
  R::F2
  function OverlappingDiscs(theta, R)
    f, f2 = totheta(theta), to1fun(R)
    new{typeof(f), typeof(f2)}(f, f2)
  end
end
to1fun(x::AbstractVector{<:Real}) = k -> x[k]
to1fun(x::Real) = k -> x
to1fun(x) = x
totheta(x::AbstractVector{<:Real}) = (noverlaps, k) -> ifelse(noverlaps==0, 1-x[k], x[k])
totheta(x::Real) = (noverlaps, k) -> ifelse(noverlaps==0, 1-x, x)
totheta(x) = x

struct SequentialTreeModel{FR, FI} <: SequentialPointProcess
  R::FR
  interaction::FI
end

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

function normalizedlogdensities(m, xy, window, int)
  z = normconstants(m, xy[1:end-1], window, int)
  for i in eachindex(z)
    if z[i] == 0.0
      z[i] = -Inf
    else
      z[i] = logfk(m, xy[i+1], view(xy, 1:i)) - log(z[i])
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
