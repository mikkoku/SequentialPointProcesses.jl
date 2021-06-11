# log unnormalized density
fk(m::Softcore2v, x, xbefore) = fk(Softcore2(m), x, xbefore)
logfk(m::Union{Softcore2,Softcore2v}, x, xbefore) = log(fk(m, x, xbefore))
function fk(m::Softcore2, x, xbefore)
  d((x1, y1), (x2, y2)) = sqrt((x1-x2)^2 + (y1-y2)^2)
  k = length(xbefore)+1
  R = m.R(k)

  S = mapreduce(m.op, xbefore) do y
    m.phi(d(x,y), R)
  end
  theta = m.theta(k)
  psiS = m.psi(S, R)
  theta * psiS + (one(theta)-theta) * (one(psiS)-psiS)
end


function compute_integral_y!(m::Union{Softcore2, Softcore2v}, Ix, x, xs, qy)
  fill!(Ix, 0.0)
  n = length(xs)
  for y in qy
    s1 = 0.0 #Base.TwicePrecision(0.0)
    for k in 1:n
      Ix[k] += fk(m, (x,y), view(xs, 1:k))
    end
  end
  Ix
end
