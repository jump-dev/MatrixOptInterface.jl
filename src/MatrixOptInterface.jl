module MatrixOptInterface

using MathOptInterface
const MOI = MathOptInterface
const MOIU = MathOptInterface.Utilities

export MatrixOptimizer

MOIU.@model(MIPInnerModel,
    (MOI.ZeroOne, MOI.Integer),
    (MOI.EqualTo, MOI.GreaterThan, MOI.LessThan, MOI.Interval),
    (),
    (),
    (MOI.SingleVariable),
    (MOI.ScalarAffineFunction),
    (),
    ()
)

const Model = MOIU.UniversalFallback{MIPInnerModel{Float64}}

"""
    Model()
Create an empty instance of MatrixOptInterface.Model.
"""
function Model()
    return MOIU.UniversalFallback(MIPInnerModel{Float64}())
end

function Base.show(io::IO, ::Model)
    print(io, "A MatrixOptInterface Model")
    return
end

# abstract type AbstractLPForm{T} end
struct LPMatrixOptInterfaceForm{T}
    direction::MOI.OptimizationSense
    c::Vector{T}
    A::Matrix{T}
    c_lb::Vector{T}
    c_ub::Vector{T}
    v_lb::Vector{T}
    v_ub::Vector{T}
end
struct LPStandardForm{T}# <: AbstractLPForm{T}
    direction::MOI.OptimizationSense
    c::Vector{T}
    A::Matrix{T}
    b::Vector{T}
end
struct LPCannonicalForm{T}
    direction::MOI.OptimizationSense
    c::Vector{T}
    A::Matrix{T}
    b::Vector{T}
end
struct LPSolverForm{T}
    direction::MOI.OptimizationSense
    c::Vector{T}
    A::Matrix{T}
    b::Vector{T}
    senses::Vector{Char}
    v_lb::Vector{T}
    v_ub::Vector{T}
end
struct MILP{T}
    lp
    variable_type
end

function MatrixOptInterfaceForm(lp::MatrixOptInterfaceForm{T}) where T
    return lp
end
function MatrixOptInterfaceForm(lp::LPStandardForm{T}) where T
    return LPMatrixOptInterfaceForm{T}(
        lp.direction,
        lp.c,
        lp.A,
        lp.b,
        lp.b,
        fill(zero(T), length(lp.c)),
        fill(typemax(T), length(lp.c)),
    )
end
function MatrixOptInterfaceForm(lp::LPCannonicalForm{T}) where T
    return LPMatrixOptInterfaceForm{T}(
        lp.direction,
        lp.c,
        lp.A,
        fill(typemin(T), length(lp.b)),
        lp.b,
        fill(typemin(T), length(lp.c)),
        fill(typemax(T), length(lp.c)),
    )
end
function MatrixOptInterfaceForm(lp::LPSolverForm{T}) where T
    c_lb = fill(typemin(T), length(lp.b))
    c_ub = fill(typemax(T), length(lp.b))
    for i in eachindex(lp.b)
        if lp.senses[i] == '<'
            c_ub[i] = lp.b[i]
        elseif lp.senses[i] == '>'
            c_lb[i] = lp.b[i]
        elseif lp.senses[i] == '='
            c_lb[i] = lp.b[i]
            c_ub[i] = lp.b[i]
        else
            error("invalid sign $(lp.senses[i])")
        end
    end
    return LPMatrixOptInterfaceForm{T}(
        lp.direction,
        lp.c,
        lp.A,
        c_lb,
        c_ub,
        lp.v_lb,
        lp.v_ub,
    )
end

"""
reasobale forms:

A) LP standard form:

opt <c, x>
s.t.
    Ax == b
     x >= 0

B) LP cannonical form:

opt <c, x>
s.t.
    Ax <= b

C) MBP (our standard)
opt <c, x>
s.t.
    c_lb <= Ax <= c_ub
    v_lb <=  x <= v_ub

C) Solver
opt <c, x>
s.t.
    Ax sense c_ub
    v_lb <=  x <= v_ub

Extra
 vartype = {'C','I','B'}
 sense = {'<','>','='}
    
    
"""
function MatrixOptimizer(raw_lp)
    lp = MatrixOptInterfaceForm(raw_lp)
    num_variables = length(lp.c)
    num_constraints = length(lp.c_lb)

    # use caching
    optimizer = Model()

    x = MOI.add_variables(optimizer, num_variables)

    objective_function = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(lp.c, x), 0.0)
    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            objective_function)
    MOI.set(optimizer, MOI.ObjectiveSense(), lp.sense)

    # Add constraints
    for i in 1:num_constraints
        add_constraint(optimizer, 
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(lp.A[i,:], x), 0.0),
            lp.c_lb[i], lp.c_ub[i])
    end

    # Add bounds
    for i in 1:num_variables
        add_constraint(optimizer, 
            MOI.SingleVariable(x[i]), lp.v_lb[i], lp.v_ub[i])
    end

    return optimizer
end

function add_constraint(optimizer, func, lb, ub)
    if lb == ub > -Inf
        MOI.add_constraint(optimizer, func, MOI.EqualTo(lb))
    else
        if lb > -Inf && ub < Inf
            MOI.add_constraint(optimizer, func, MOI.Interval(lb, ub))
        elseif lb > -Inf
            MOI.add_constraint(optimizer, func, MOI.GreaterThan(lb))
        elseif ub < Inf
            MOI.add_constraint(optimizer, func, MOI.LessThan(ub))
        end
    end
end

end