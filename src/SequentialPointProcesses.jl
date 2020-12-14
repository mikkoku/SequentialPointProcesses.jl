module SequentialPointProcesses
import PointPatternStatistics.PointPattern
import Distributions
using Requires

include("logfk.jl")
include("models.jl")
include("rand.jl")
include("overlappingdiscs.jl")

Distributions.logpdf(model::SequentialPointProcess, pp::PointPattern, int::Integer) =
    logdens(model, pp.data, pp.window, Options(nx=int))
Distributions.logpdf(model::SequentialPointProcess, pp::PointPattern, int::Options) =
    logdens(model, pp.data, pp.window, int)

export Softcore,Mixture,Uniform, OverlappingDiscs

cuda_available = false
function __init__()
    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" include("cuda.jl")
end

end
