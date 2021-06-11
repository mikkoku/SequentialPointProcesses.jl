fk(m::SequentialTreeModel, p, xbefore) = exp(logfk(m, p, xbefore))
function logfk(m::SequentialTreeModel, p, xbefore)
  s = 0.0
  x, y = p
  n = length(xbefore)
  for (i, (x1, y1)) in enumerate(xbefore)
    d2 = (x1-x)^2 + (y1-y)^2
    if d2 <= (m.R(i) + m.R(n+1))^2
      s += m.interaction(n+1, i)
    end
  end
  s
end


function compute_integral(m::SequentialTreeModel, data, window, nx)
  qx, qy, w = quadps(window, nx)
  #cellsize = 1/scale^2
  I = Vector{Float64}(undef, length(data))
  for n in 1:(length(data))
    s = 0.0
    for y in qy
      for x in qx
        s += fk(m, (x, y), view(data, 1:n))
      end
    end
    I[n] = s*w;
  end
  I
end

function addtocircle!(image, x, y, z, R, temp)
  mapcircle!(image, x, y, R, temp) do A, x0, x1, gy
    for gx in x0:x1
      @inbounds A[gx, gy] += z
    end
    0
  end
end


function addtocirclesafe!(image, x, y, z, R)
  for gy in axes(image, 2)
    for gx in axes(image, 1)
      if ((gx-0.5)-x)^2 + ((gy-0.5)-y)^2 <= R^2
        image[gx, gy] += z
      end
    end
  end
end

function compute_integral2(m::SequentialTreeModel, data, window, nx)
  (x1,x2), (y1,y2) = window.x, window.y
  ny = ceil(Int, nx*(y2-y1)/(x2-x1))
  scale = nx/(x2-x1)
  compute_integral2(m, data, scale, x1, y1, nx, ny)
end

function fastsum(f, x, ::Val{false})
  sum(f, x)
end
function fastsum(f, x)
  #fastsum(f, x, Val(loopvectorization_available))
  if loopvectorization_available
    fastsum(f, x, Val(true))
  else
    fastsum(f, x, Val(false))
  end
end

function compute_integral2(m::SequentialTreeModel, data, scale, x0, y0, nx, ny)
  nx == 0 && return zeros(length(data))
  cellsize = 1/scale^2
  image = zeros(Float64, nx, ny)
  I = Vector{Float64}(undef, length(data))
  tmp = Vector{Float64}(undef, ny)
  for n in 1:(length(data))
    fill!(image, 0.0)
    for i in 1:n
      x, y = data[i]
      addtocircle!(image, (x-x0)*scale, (y-y0)*scale,
        m.interaction(n+1, i), (m.R(n+1) + m.R(i))*scale, tmp)
    end
    I[n] = cellsize*fastsum(exp, image)
  end
  I
end


function normconstants(m::SequentialTreeModel, xs, window, nx::Int)
  compute_integral2(m, xs, window, nx)
end
