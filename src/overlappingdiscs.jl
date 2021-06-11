logfk(m::OverlappingDiscs, p, xbefore) = log(fk(m, p, xbefore))
function fk(m::OverlappingDiscs, p, xbefore)
  m.theta(countoverlaps(m, p, xbefore), length(xbefore))
end
function countoverlaps(m::OverlappingDiscs, p, xbefore)
  k = 0
  x, y = p
  for (i, (x1, y1)) in enumerate(xbefore)
    if (x1-x)^2 + (y1-y)^2 <= m.R(i)^2
      k += 1
    end
  end
  k
end

function add1tocircle!(image, ks, x, y, R, temp)
  mapcircle!(image, x, y, R, temp) do A, x0, x1, gy
    for gx in x0:x1
      @inbounds k = A[gx, gy]
      ks[k+1] -= 1
      #k += one(T)
      ks[k+2] += 1
      @inbounds A[gx, gy] = k + 1
    end
    0
  end
end

function compute_integral(m::OverlappingDiscs, data, scale, x0, y0, nx, ny)
  nx == 0 && return ones(length(data))
  cellsize = 1/scale^2
  length(data) + 1 < typemax(Int16) || throw(Error("Too many points."))
  image = zeros(Int16, nx, ny)
  tmp = Vector{Float64}(undef, ny)
  ks = zeros(Int, length(data) + 1)
  # Remember that arrays have 1 based indexing whereas counts start from 0
  ks[1] = length(image)
  I = Vector{Float64}(undef, length(data))
  for (i, (x, y)) in enumerate(data)
    # add1tocircle!(image, ks, 0.5 + x*scale, 0.5 + y*scale, m.R(i)*scale)
    add1tocircle!(image, ks, (x-x0)*scale, (y-y0)*scale, m.R(i)*scale, tmp)
    Ii = 0.0
    for (ki, k) in enumerate(ks)
      Ii += k * m.theta(ki-1, i)
    end
    I[i] = Ii * cellsize
  end
  I
end

function unbiased_normalization_constant_estimator(m::OverlappingDiscs, data, window, n)
  I = Vector{Float64}(undef, length(data))
  A = PointPatternStatistics.area(window)
  fastcount = false
  x = y = I
  if loopvectorization_available
    R0 = m.R(0)
    if all(i -> R0 == m.R(i), eachindex(data))
      fastcount = true
      x = map(x -> x.x, data)
      y = map(x -> x.y, data)
    end
  end
  for (i, _) in enumerate(data)
    maxtheta = maximum(m.theta(k, i) for k in 0:i)
    Mk = 0
    for _ in 1:n
      counter = 0
      while true
        p = randp(window)
        u = rand()
        counter += 1
        k = if fastcount
          countoverlaps(m, p, view(x, 1:i), view(y, 1:i))
        else
          countoverlaps(m, p, view(data, 1:i))
        end
        if u <= m.theta(k, i)/maxtheta
          Mk += counter
          break
        end
      end
    end
    I[i] = A*maxtheta/(Mk/n)
  end
  I
end

function discretization(window, nx::Int)
  (x1,x2), (y1,y2) = window.x, window.y
  ny = round(Int, nx*(y2-y1)/(x2-x1))
  scale = nx/(x2-x1)
  scale, x1, y1, nx, ny
end

function normconstants(m::OverlappingDiscs, xs, window, int)
  int.parallel === false || throw(ArgumentError("Model doesn't support parallization"))
  nx = int.nx

  if nx < 0
    unbiased_normalization_constant_estimator(m, xs, window, -nx)
  else
  compute_integral(m, xs, discretization(window, nx)...)
  end
end
