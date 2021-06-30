HardcoreModels = Union{Hardcore1, Hardcore2}
logfk(m::HardcoreModels, p, xbefore) = log(fk(m, p, xbefore))
function fk(m::Hardcore1, p, xbefore)
  s = 0.0
  x, y = p
  n = length(xbefore)
  for (i, (x1, y1)) in enumerate(xbefore)
    d2 = (x1-x)^2 + (y1-y)^2
    if d2 <= (m.R(n+1))^2
      return m.theta(n+1)
    end
  end
  1-m.theta(n+1)
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

function mapcircle!(f, image, x, y, R, temp, init=0)
  y0 = max(ceil(Int, y - R + 0.5), 1)
  y1 = min(floor(Int, y + R + 0.5), size(image, 2))
  dxs = view(temp, 1:y1-y0+1)
  @assert y + 0.5 - y0 <= R + 2eps(y)
  @assert -(y + 0.5 - y1) <= R + 2eps(y)
  # dxs .= y0:y1
  # LoopVectorization.vmap!(dxs, dxs) do gy
  #   @fastmath sqrt(max(0.0, R^2 - (y+0.5-gy)^2))
  # end
  map!(dxs, y0:y1) do gy
    @fastmath sqrt(max(0.0, R^2 - (y+0.5-gy)^2))
  end
  s = init
  @fastmath for i in eachindex(y0:y1)
    @inbounds dx = dxs[i]
    @inbounds gy = (y0:y1)[i]
    #dx = sqrt(R^2 - (y+0.5-gy)^2)
    # x0 = max(ceil(Int, x - dx + 0.5), 1)
    # x1 = min(floor(Int, x + dx + 0.5), size(image, 1))
    x0 = max(unsafe_trunc(Int, x - dx + 1.5), 1)
    x1 = min(unsafe_trunc(Int, x + dx + 0.5), size(image, 1))
    s += f(image, x0, x1, gy)
  end
  s
end




function compute_integral2(m::HardcoreModels, data, window, nx)
  (x1,x2), (y1,y2) = window.x, window.y
  ny = round(Int, nx*(y2-y1)/(x2-x1))
  scale = nx/(x2-x1)
  compute_integral2(m, data, scale, x1, y1, nx, ny)
end

function compute1hardcore(image, x, y, R, tmp)
  mapcircle!(image, x, y, R, tmp, 0) do A, x0, x1, gy
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

function compute_integral2(m::Hardcore2, data, scale, x0, y0, nx, ny)
  nx == 0 && return zeros(length(data))
  cellsize = 1/scale^2
  image = zeros(Int8, nx, ny)
  I = Vector{Float64}(undef, length(data))
  tmp = Vector{Float64}(undef, ny)
  for n in 1:(length(data))
    fill!(image, 0)
    counter = 0
    l = length(image)
    for i in 1:n
      x, y = data[i]
      R = m.R(n+1, i)
      if R*scale > sum(size(image))
        counter = l
        break
      end
      counter += compute1hardcore(image, (x-x0)*scale, (y-y0)*scale,
        R*scale, tmp)
    end
    theta = m.theta(n+1)
    I[n] = counter*cellsize*theta + (l-counter)*cellsize*(1-theta)
  end
  I
end

function compute_integral2(m::Hardcore1, data, scale, x0, y0, nx, ny)
  nx == 0 && return zeros(length(data))
  cellsize = 1/scale^2
  image = zeros(Int8, nx, ny)
  I = Vector{Float64}(undef, length(data))
  tmp = Vector{Float64}(undef, ny)
  counter = 0
  l = length(image)
  for n in 1:(length(data))
    x, y = data[n]
    R = m.R(n+1)
    if R*scale > sum(size(image)) || counter == l
      counter = l
    else
      counter += compute1hardcore(image, (x-x0)*scale, (y-y0)*scale,
        R*scale, tmp)
    end
    theta = m.theta(n+1)
    I[n] = counter*cellsize*theta + (l-counter)*cellsize*(1-theta)
  end
  I
end

function normconstants(m::HardcoreModels, xs, window, int)
  int.parallel === false || throw(ArgumentError("Hardcore model doesn't support parallization"))

  compute_integral2(m, xs, window, int.nx)
end
