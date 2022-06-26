struct Grid
  maxwidth::Float64
end
struct GridThreads
  maxwidth::Float64
  num_threads::Int
    end
GridThreads(maxwidth) = GridThreads(maxwidth, Threads.nthreads())
struct GridPolyester
  maxwidth::Float64
  num_threads::Int
end
struct GridCUDA{T}
  maxwidth::Float64
  batchsize::Int
  num_threads::Tuple{Int,Int}
  verbose::Bool
  GridCUDA(T, maxwidth, batchsize=1, num_threads=(256,1), verbose=false) = new{T}(float(maxwidth), batchsize, num_threads, verbose)
end


# log unnormalized density
fk(m::Softcore, x, xbefore) = exp(logfk(m, x, xbefore))
function logfk(m::Softcore, x, xbefore)
  d((x1, y1), (x2, y2)) = sqrt((x1-x2)^2 + (y1-y2)^2)
  -sum((m.kernel(d(x,y), i) for (i, y) in enumerate(xbefore)))
end

function quadps(window, int)
  (x1,x2), (y1,y2) = window.x, window.y
  w = int.maxwidth
  nx = ceil(Int, (x2-x1)/w)
  ny = ceil(Int, (y2-y1)/w)
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
    s1 = 0.0
    for i in eachindex(xs)
      @inbounds x2, y2 = xs[i]
      d = sqrt((x-x2)^2 + (y-y2)^2)
      s1 += m.kernel_integral(d, i)
      @inbounds Ix[i] += exp(-Float64(s1))
    end
  end
  Ix
end
function compute_integral(m::Union{Softcore,Softcore2,Softcore2v}, qx, qy, xs, intpar::GridCUDA)
    if !cuda_available
    error("CUDA not loaded.")
    end
  compute_integral_cuda(m, qx, qy, xs, intpar)
  end

function normconstants(m::Union{Softcore,Softcore2,Softcore2v}, xs, window, intpar)
  qx, qy, w = quadps(window, intpar)
  I = compute_integral(m, qx, qy, xs, intpar)
  I .*= w
end
# Using separate accumulator for each grid line prodives a little better accuracy
# at the cost of doubling the memory requirement.
function compute_integral(m::Union{Softcore,Softcore2,Softcore2v}, qx, qy, xs, ::Grid)
# function compute_integral(m::Softcore, qx, qy, xs, ::NamedTuple{(:nx,)})
  I = zeros(length(xs))
  Ix = similar(I)
  for x in qx
    compute_integral_y!(m, Ix, x, xs, qy)
    I .+= Ix
  end
  I
end

function compute_integral(m::Union{Softcore,Softcore2,Softcore2v}, qx, qy, xs, int::GridThreads)
# function compute_integral(m::Softcore, qx, qy, xs, ::NamedTuple{(:nx, :threads)})
  nthreads = int.num_threads
  if nthreads > Threads.nthreads()
    println("requested number of threads larger than number of available threads")
  end
  I = [Vector{Float64}(undef, length(xs)) for _ in 1:nthreads]
  Ix = [similar(I[1]) for _ in 1:nthreads]
  GC.enable(false)
  

  Threads.@threads for tid in 1:nthreads
    fill!(I[tid], 0.0)
    endidx = fld(length(qx)*tid, nthreads)
    startidx = fld(length(qx)*(tid-1), nthreads)+1
    for qi in startidx:endidx
      compute_integral_y!(m, Ix[tid], qx[qi], xs, qy)
      I[tid] .+= Ix[tid]
    end
  end
  GC.enable(true)
  sum(I)
end
