HardcoreModels = Union{Hardcore1, Hardcore2}
logfk(m::HardcoreModels, p, xbefore) = log(fk(m, p, xbefore))
function fk_bool(m::Hardcore1, p, xbefore)
  x, y = p
  for (i, (x1, y1)) in enumerate(xbefore)
    d2 = (x1-x)^2 + (y1-y)^2
    if d2 <= (m.R(i))^2
      return true
    end
  end
  false
end
function fk(m::Hardcore1, p, xbefore)
  n = length(xbefore)
  theta = m.theta(n+1)
  if fk_bool(m, p, xbefore)
    theta
  else
    1-theta
  end
end
function maximumfk(m::HardcoreModels, n)
  t = m.theta(n)
  max(t, 1-t)
end
function fk(m::Hardcore2, p, xbefore)
  s = 0.0
  x, y = p
  n = length(xbefore)
  for (i, (x1, y1)) in enumerate(xbefore)
    d2 = (x1-x)^2 + (y1-y)^2
    if d2 <= (m.R(n+1, i))^2
      return m.theta(n+1)
    end
  end
  1-m.theta(n+1)
end

function mapcircle!(f, image, x, dx, y, dy, R, temp, init=0)
  y0 = max(ceil(Int, (y - R)/dy + 0.5), 1)
  y1 = min(floor(Int, (y + R)/dy + 0.5), size(image, 2))
  dxs = view(temp, 1:y1-y0+1)
  # 17.2.2022 couldn't remember why these asserts were here.
  # if !(y/dy + 0.5 - y0 <= R/dy + 2eps(y))
  #   throw((;y, dy, y0, y1, R))
  # end
  # @assert y/dy + 0.5 - y0 <= R/dy + 2eps(y)
  # @assert -(y/dy + 0.5 - y1) <= R/dy + 2eps(y)
  # dxs .= y0:y1
  # LoopVectorization.vmap!(dxs, dxs) do gy
  #   @fastmath sqrt(max(0.0, R^2 - (y+0.5-gy)^2))
  # end
  map!(dxs, y0:y1) do gy
    @fastmath sqrt(max(0.0, R^2 - (y-(gy-0.5)*dy)^2))/dx
  end
  s = init
  xdx = x/dx
  @fastmath for i in eachindex(y0:y1)
    @inbounds dgx = dxs[i]
    @inbounds gy = (y0:y1)[i]
    #dx = sqrt(R^2 - (y+0.5-gy)^2)
    # x0 = max(ceil(Int, x - dx + 0.5), 1)
    # x1 = min(floor(Int, x + dx + 0.5), size(image, 1))
    x0 = max(unsafe_trunc(Int, (xdx - dgx) + 1.5), 1)
    x1 = min(unsafe_trunc(Int, (xdx + dgx) + 0.5), size(image, 1))
    s += f(image, x0, x1, gy)
  end
  s
end



mutable struct IntArrayIntegral{T}
  maxwidth::Float64
  data::Union{Nothing, Tuple{Matrix{T}, Vector{Float64}, Vector{Float64}}}
  IntArrayIntegral(T, maxwidth::Number) = new{T}(Float64(maxwidth), nothing)
  IntArrayIntegral(maxwidth::Number) = IntArrayIntegral(Int8, maxwidth)
end

function compute_integral2(m::HardcoreModels, data, window, nx)
  (x1,x2), (y1,y2) = window.x, window.y

function compute_integral2(m::HardcoreModels, data, window, int)
  (x1,x2), (y1,y2) = window.x, window.y
  w = int.maxwidth
  nx = ceil(Int, (x2-x1)/w)
  ny = ceil(Int, (y2-y1)/w)
  dx = (x2-x1)/nx
  dy = (y2-y1)/ny
  A = (x2-x1)*(y2-y1)
  I = compute_integral2(m, data, x1, dx, y1, dy, nx, ny, int)
  # Rounding can cause I > A which is wrong.
  @. I = min(A, I)
  I
end

end

function compute1hardcore(image, x, dx, y, dy, R, tmp)
  mapcircle!(image, x, dx, y, dy, R, tmp, 0) do A, x0, x1, gy
    c = 0
    for gx in x0:x1
      @inbounds a = A[gx, gy]
      if a == 0
        c += 1
      end
      @inbounds A[gx, gy] = 1
    end
    c
  end
end

function compute_integral2(m::Hardcore2, data, x0, dx, y0, dy, (image, I, tmp))
  length(image) == 0 && return fill!(I, zero(eltype(I)))
  cellsize = dx*dy
  for n in 1:(length(data))
    fill!(image, 0)
    counter = 0
    l = length(image)
    for i in 1:n
      x, y = data[i]
      R = m.R(n+1, i)
      if R/dx > sum(size(image))
        counter = l
        break
      end
      counter += compute1hardcore(image, x-x0, dx, y-y0, dy, R, tmp)
    end
    I[n] = counter*cellsize
  end
  I
end

function compute_integral2(m::HardcoreModels, data, x0, dx, y0, dy, nx, ny, int::IntArrayIntegral{T}) where T
  (image, I, tmp) = something(int.data,
    (Matrix{T}(undef, nx, ny), 
    Vector{Float64}(undef, length(data)), Vector{Float64}(undef, ny)))

  if size(image) != (nx, ny)
    image = Matrix{T}(undef, nx, ny)
  end
  fill!(image, zero(T))

  resize!(tmp, ny)
  resize!(I, length(data))
  # T == Bool ? falses(nx, ny) : zeros(T, nx, ny)
  int.data = (image, I, tmp)
  compute_integral2(m, data, x0, dx, y0, dy, int.data)
end

function compute_integral2(m::Hardcore1, data, x0, dx, y0, dy, (image, I, tmp))
  length(image) == 0 && return fill!(I, zero(eltype(I)))
  cellsize = dx*dy
  counter = 0
  l = length(image)
  for n in 1:(length(data))
    x, y = data[n]
    R = m.R(n)
    if R/dx > sum(size(image)) || counter == l
      counter = l
    else
      counter += compute1hardcore(image, x-x0, dx, y-y0, dy, R, tmp)
    end
    I[n] = counter*cellsize
  end
  I
end

function normconstants(m::HardcoreModels, xs, window, int)
  A = PointPatternStatistics.area(window)
  I = compute_integral2(m, xs, window, int)
  for n in 1:length(xs)
    a = I[n]
    theta = m.theta(n+1)
    I[n] = a*theta + (A-a)*(1-theta)
  end
  I
end
