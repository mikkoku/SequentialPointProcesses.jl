module SequentialPointProcesses
import PointPatternStatistics: PointPattern
import PointPatternStatistics # area
import Distributions
using Requires

include("models.jl")
include("logfk.jl")
include("rand.jl")
include("overlappingdiscs.jl")
include("sequentialtreemodel.jl")
include("softcore2.jl")
include("hardcore.jl")

extractlocations(pp) = [(x,y) for (x,y) in pp.data]
Distributions.logpdf(model::SequentialPointProcess, pp::PointPattern, int::Integer) =
    logdens(model, extractlocations(pp), pp.window, Options(nx=int))
Distributions.logpdf(model::SequentialPointProcess, pp::PointPattern, int::Options) =
    logdens(model, extractlocations(pp), pp.window, int)

export Softcore,Mixture,Uniform, OverlappingDiscs,
    SequentialTreeModel, Softcore2, Softcore2v,
    Hardcore, Hardcore1, Hardcore2

cuda_available = false
loopvectorization_available = false
function __init__()
    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" include("cuda.jl")
    @require LoopVectorization="bdcacae8-1622-11e9-2a5c-532679323890" include("loopvectorization.jl")
end

end
