module SequentialPointProcesses
import PointPatternStatistics.PointPattern
import Distributions
using Requires

include("logfk.jl")
include("models.jl")
include("rand.jl")

Distributions.logpdf(model::SequentialPointProcess, pp::PointPattern, int::NamedTuple) =
    logdens(model, pp.data, pp.window, int)

export Softcore,Mixture,Uniform

function __init__()
    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" include("cuda.jl")
end

end
