loopvectorization_available = true

function fastsum(f, x, ::Val{true})
    LoopVectorization.vmapreduce(f, +, x)
end

function countoverlaps(m::OverlappingDiscs, p, x, y)
  k = 0
  x0, y0 = p
  LoopVectorization.@avx for i in eachindex(x)
    k += (x[i]-x0)^2 + (y[i]-y0)^2 <= m.R(i)^2
  end
  k
end
