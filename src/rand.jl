Base.rand(m::SequentialPointProcess, window, npoints::Integer; giveup=-1, returnanyway=false) =
  rpattern(m, window, npoints, giveup, returnanyway)
function rpattern(model, window, npoints, giveup=-1, returnanyway=false)
  xy = Vector{NTuple{2,Float64}}(undef, npoints)
  if npoints == 0
    return PointPattern(xy, window)
  end
  xy[1] = randpoint(Uniform(), xy, window, giveup)
  for i in 2:npoints
    prop = randpoint(model, view(xy, 1:i-1), window, giveup)
    if isnothing(prop) # If sampling from the sequential model fails at some point return nothing.
      # This works with ABC, since that kind of parameter configuration is too unlikely.
      # In other contexts, giveup is not used.
      if returnanyway
        return PointPattern(xy[1:i], window)
      else
        return PointPattern(empty(xy), window)
      end
    end
    xy[i] = prop
  end
  PointPattern(xy, window)
end
function acceptpoint(m::Softcore, xyprop, xyold)
  p = fk(m, xyprop, xyold)
  return rand() <= p
end
maximumfk(_, n) = 1.0
function acceptpoint(m::HardcoreModels, xyprop, xyold)
  p = fk(m, xyprop, xyold)/maximumfk(m, length(xyold) + 1)
  @assert p <= 1.0
  return rand() <= p
end
function acceptpoint(m::OverlappingDiscs, xyprop, xyold)
  k = length(xyold) + 1
  p = m.theta(countoverlaps(m, xyprop, xyold), k) / maximum(n -> m.theta(n,k), 0:length(xyold))
  return rand() <= p
end
function acceptpoint(::Uniform, xyprop, xyold)
  true
end
function randpoint(m::Mixture, pattern, window, giveup)
  if rand() < m.theta
    randpoint(m.first, pattern, window, giveup)
  else
    randpoint(m.second, pattern, window, giveup)
  end
end
function randp(window)
  randcoord(x) = window[x][1] + (window[x][2]-window[x][1])*rand()
  randx() = randcoord(:x)
  randy() = randcoord(:y)
  randx(), randy()
end
function randpoint(model, pattern, window, giveup=-1)
  tries = 0
  while true
    xyprop = randp(window)
    if acceptpoint(model, xyprop, pattern)
      return xyprop
    end
    tries += 1
    if giveup != -1 && tries > giveup
      return nothing
    end
  end
  return nothing
end
