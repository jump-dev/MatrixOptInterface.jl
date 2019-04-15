using MatrixOptInterface

const MatOI = MatrixOptInterface
const MOI = MatOI.MOI
const MOIU = MatOI.MOIU
const MOIB = MOI.Bridges

raw_lp = MatOI.LPSolverForm{Float64}(
    MOI.MIN_SENSE,
    [4,3],
    [1 2;2 1],
    [3, 3],
    [MatOI.LESS_THAN, MatOI.LESS_THAN],
    [-Inf, -Inf],
    [Inf, Inf]
)

model = MatOI.MatrixOptimizer(raw_lp)

@show model

MOIU.@model(ModelData, (), (),
            (MOI.Zeros, MOI.Nonnegatives),
            (), (), (), (), (MOI.VectorAffineFunction,))

const cache = MOIU.UniversalFallback(ModelData{Float64}())
opt = MatOI.Optimizer()
MOI.empty!(cache)
cached = MOIU.CachingOptimizer(cache, opt)
bridged = MOIB.full_bridge_optimizer(cached, Float64)

MOI.copy_to(bridged, model, copy_names = false)

MOI.optimize!(bridged)

@show opt.data

@show MatOI.LPSolverForm(opt)


