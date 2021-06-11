struct Options
    nx::Int
    parallel::Union{Bool, Symbol}
    cudatype::Type
    cudabatchsize::Int
end
function Options(;nx::Int, parallel=false, cudatype::Type=Float64, cudabatchsize::Int=1)
    if !(parallel in (false, :threads, :cuda))
        throw(ArgumentError("parallel should be one of false, :threads, :cuda"))
    end
    Options(nx, parallel, cudatype, cudabatchsize)
end


# log unnormalized density
fk(m::Softcore, x, xbefore) = exp(logfk(m, x, xbefore))
function logfk(m::Softcore, x, xbefore)
  d((x1, y1), (x2, y2)) = sqrt((x1-x2)^2 + (y1-y2)^2)
  -sum((m.kernel(d(x,y), i) for (i, y) in enumerate(xbefore)))
end

function quadps(window, nx)
  (x1,x2), (y1,y2) = window.x, window.y
  ny = ceil(Int, nx*(y2-y1)/(x2-x1))
  qx = quadps(x1, x2, nx)
  qy = quadps(y1, y2, ny)

  qx, qy, step(qx)*step(qy)
end
function quadps(x1, x2, n)
  s = (x2-x1)/(2n)
  LinRange(x1+s, x2-s, n)
end

function compute_integral_y!(m::Softcore, Ix, x, xs, qy)
  fill!(Ix, 0.0)
  for y in qy
    s1 = 0.0 #Base.TwicePrecision(0.0)
    for i in eachindex(xs)
      @inbounds x2, y2 = xs[i]
      d = sqrt((x-x2)^2 + (y-y2)^2)
      s1 += m.kernel_integral(d, i)
      @inbounds Ix[i] += exp(-Float64(s1))
    end
  end
  Ix
end

# normconstants(xs, f, window, nx::Int) = normconstants(xs, f, window, (nx=nx,))
function normconstants(m::Union{Softcore,Softcore2,Softcore2v}, xs, window, intpar)
# function normconstants(m::Softcore, xs, window, intpar)
  intpar.nx == 0 && return ones(length(xs))
  qx, qy, w = quadps(window, intpar.nx)
  if intpar.parallel === false
    I = compute_integral(m, qx, qy, xs, (nx=intpar.nx,))
  elseif intpar.parallel === :threads
    I = compute_integral(m, qx, qy, xs, (nx=intpar.nx, threads=true))
  elseif intpar.parallel === :cuda
    if !cuda_available
      throw(ArgumentError("CUDA not loaded."))
    end
    I = compute_integral(m, qx, qy, xs, (nx=intpar.nx, type=intpar.cudatype,
      batchsize=intpar.cudabatchsize))
  else
    throw(ArgumentError("Invalid Options given."))
  end
  I .*= w
end
# Using separate accumulator for each grid line prodives a little better accuracy
# at the cost of doubling the memory requirement.
function compute_integral(m::Union{Softcore,Softcore2,Softcore2v}, qx, qy, xs, ::NamedTuple{(:nx,)})
# function compute_integral(m::Softcore, qx, qy, xs, ::NamedTuple{(:nx,)})
  I = zeros(length(xs))
  Ix = similar(I)
  for x in qx
    compute_integral_y!(m, Ix, x, xs, qy)
    I .+= Ix
  end
  I
end

function compute_integral(m::Union{Softcore,Softcore2,Softcore2v}, qx, qy, xs, ::NamedTuple{(:nx, :threads)})
# function compute_integral(m::Softcore, qx, qy, xs, ::NamedTuple{(:nx, :threads)})
  I = [zeros(length(xs)) for _ in 1:Threads.nthreads()]
  Ix = [similar(I[1]) for _ in 1:Threads.nthreads()]
  Threads.@threads for x in qx
    compute_integral_y!(m, Ix[Threads.threadid()], x, xs, qy)
    I[Threads.threadid()] .+= Ix[Threads.threadid()]
  end
  sum(I)
end
