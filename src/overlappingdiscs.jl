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
function add1tocircle!(image, ks, x, y, R)
  add1tocircle!(image, ks, round(Int, x), round(Int, y), R)
end
function add1tocircle!(image, ks, x::Int, y::Int, R)
  Ri = ceil(Int, R) + 1
  xi = round(Int, x)
  yi = round(Int, y)
  checkbounds(image, xi, yi)
  xr = intersect(xi .+ (-Ri:Ri), axes(image, 1)) .- xi
  yr = intersect(yi .+ (-Ri:Ri), axes(image, 2)) .- yi
  #println((x, y, xr, yr, R, Ri, axes(image, 1), axes(image, 2)))
  for dx in xr, dy in yr
    if dx^2 + dy^2 <= R^2
      k = image[x+dx, y+dy]
      ks[k+1] -= 1
      k += 1
      ks[k+1] += 1
      image[x+dx, y+dy] = k
    end
  end
end

function compute_integral(m::OverlappingDiscs, data, scale, nx, ny)
  cellsize = 1/scale^2
  image = zeros(Int, nx, ny)
  ks = zeros(Int, length(data) + 1)
  # Remember that arrays have 1 based indexing whereas counts start from 0
  ks[1] = length(image)
  I = Vector{Float64}(undef, length(data))
  for (i, (x, y)) in enumerate(data)
    add1tocircle!(image, ks, 0.5 + x*scale, 0.5 + y*scale, m.R(i)*scale)
    Ii = 0.0
    for (ki, k) in enumerate(ks)
      Ii += k * m.theta(ki-1, i)
    end
    I[i] = Ii * cellsize
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
  compute_integral(m, xs, discretization(window, nx)...)
end
