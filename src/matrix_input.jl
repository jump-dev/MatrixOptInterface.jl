
MOIU.@model(MIPInnerModel,
(),
(MOI.EqualTo, MOI.GreaterThan, MOI.LessThan, MOI.Interval),
(),
(),
(MOI.SingleVariable,),
(MOI.ScalarAffineFunction,),
(),
()
)

const Model{T} = MOIU.UniversalFallback{MIPInnerModel{T}}

"""
Model()
Create an empty instance of MatrixOptInterface.Model.
"""
function Model{T}() where T
    return MOIU.UniversalFallback(MIPInnerModel{T}())
end

function Base.show(io::IO, ::Model)
    print(io, "A MatrixOptInterface Model")
    return
end

abstract type AbstractLPForm{T} end
struct LPMatrixOptInterfaceForm{T} <: AbstractLPForm{T}#, V<:AbstractVector{T}, M<:AbstractMatrix{T}}
    direction::MOI.OptimizationSense
    c::Vector{T}
    A::Matrix{T}
    c_lb::Vector{T}
    c_ub::Vector{T}
    v_lb::Vector{T}
    v_ub::Vector{T}
end
struct LPStandardForm{T} <: AbstractLPForm{T}
    direction::MOI.OptimizationSense
    c::Vector{T}
    A::Matrix{T}
    b::Vector{T}
end
struct LPCannonicalForm{T} <: AbstractLPForm{T}
    direction::MOI.OptimizationSense
    c::Vector{T}
    A::Matrix{T}
    b::Vector{T}
end
struct LPSolverForm{T} <: AbstractLPForm{T}
    direction::MOI.OptimizationSense
    c::Vector{T}
    A::Matrix{T}
    b::Vector{T}
    senses::Vector{ConstraintSense}
    v_lb::Vector{T}
    v_ub::Vector{T}
end
struct MILP{T}
    lp
    variable_type
end

function change_form(::Type{F}, lp::F) where {F <: AbstractLPForm{T}} where T
    return lp
end

function change_form(::Type{LPMatrixOptInterfaceForm{T}}, lp::LPStandardForm{T}) where T
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
function change_form(::Type{LPMatrixOptInterfaceForm{T}}, lp::LPCannonicalForm{T}) where T
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
function change_form(::Type{LPMatrixOptInterfaceForm{T}}, lp::LPSolverForm{T}) where T
    c_lb = fill(typemin(T), length(lp.b))
    c_ub = fill(typemax(T), length(lp.b))
    for i in eachindex(lp.b)
        if lp.senses[i] == LESS_THAN
            c_ub[i] = lp.b[i]
        elseif lp.senses[i] == GREATER_THAN
            c_lb[i] = lp.b[i]
        elseif lp.senses[i] == EQUAL_TO
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

function change_form(::Type{LPCannonicalForm{T}}, lp::LPMatrixOptInterfaceForm{T}) where T
    has_c_upper = Int[]
    has_c_lower = Int[]
    sizehint!(has_c_upper, length(lp.c_ub))
    sizehint!(has_c_lower, length(lp.c_ub))
    for i in eachindex(lp.c_ub)
        if lp.c_ub < Inf
            push!(has_c_upper, i)
        end
        if lp.c_lb > -Inf
            push!(has_c_lower, i)
        end
    end
    has_v_upper = Int[]
    has_v_lower = Int[]
    sizehint!(has_v_upper, length(lp.v_ub))
    sizehint!(has_v_lower, length(lp.v_ub))
    for i in eachindex(lp.v_ub)
        if lp.v_ub < Inf
            push!(has_v_upper, i)
        end
        if lp.v_lb > -Inf
            push!(has_v_lower, i)
        end
    end
    Id = Matrix{T}(I, length(lp.c), length(lp.c))
    new_A = vcat(
        lp.A[has_c_upper,:],
        -lp.A[has_c_lower,:],
        Id[has_v_upper,:],
        -Id[has_v_lower,:],
                )
    new_b = vcat(
        lp.c_ub[has_c_upper],
        -lp.c_lb[has_c_lower],
        lp.v_ub[has_v_upper],
        -lp.v_lb[has_v_lower],
    )
    return LPCannonicalForm{T}(
        lp.direction,
        lp.c,
        new_A,
        new_b
    )
end
function change_form(::Type{LPCannonicalForm{T}}, lp::F) where {F <: AbstractLPForm{T}} where T
    temp_lp = change_form(LPMatrixOptInterfaceForm{T}, lp)
    change_form(LPCannonicalForm{T}, lp)
end

function change_form(::Type{LPStandardForm{T}}, lp::LPCannonicalForm{T}) where T
    new_A = hcat(
        lp.A,
        -lp.A,
        Matrix{T}(I, length(lp.b), length(lp.b))
    )
    new_c = vcat(
        lp.c,
        -lp.c,
        fill(0.0, length(lp.b))
    )
    return LPStandardForm{T}(
        lp.direction,
        new_c,
        new_A,
        copy(lp.b)
    )
end
function change_form(::Type{LPStandardForm{T}}, lp::F) where {F <: AbstractLPForm{T}} where T
    temp_lp = change_form(LPMatrixOptInterfaceForm{T}, lp)
    new_lp = change_form(LPCannonicalForm{T}, temp_lp)
    change_form(LPStandardForm{T}, new_lp)
end

function change_form(::Type{LPSolverForm{T}}, lp::LPMatrixOptInterfaceForm{T}) where T
    new_A = copy(lp.A)
    senses = fill(LESS_THAN, length(lp.c_lb))
    new_b = fill(NaN, length(lp.c_lb))
    for i in eachindex(lp.c_lb)
        if lp.c_lb[i] == lp.c_ub[i]
            senses[i] = EQUAL_TO
            new_b[i] = lp.c_lb[i]
        elseif lp.c_lb[i] > -Inf && lp.c_ub[i] < Inf
            senses[i] = GREATER_THAN
            new_b[i] = lp.c_lb[i]
            push!(new_b, lp.c_ub[i])
            push!(sense, LESS_THAN)
            new_A = vcat(new_A, lp.A[i,:])
        elseif lp.c_lb[i] > -Inf
            senses[i] = GREATER_THAN
            new_b[i] = lp.c_lb[i]
        elseif lp.c_ub[i] < Inf
            senses[i] = LESS_THAN
            new_b[i] = lp.c_ub[i]
        end
    end
    return LPMatrixOptInterfaceForm{T}(
        lp.direction,
        lp.c,
        new_A,
        new_b,
        senses,
        lp.v_lb,
        lp.v_ub,
    )
end
function change_form(::Type{LPSolverForm{T}}, lp::F) where {F <: AbstractLPForm{T}} where T
    temp_lp = change_form(LPMatrixOptInterfaceForm{T}, lp)
    change_form(LPSolverForm{T}, temp_lp)
end

"""
Possible forms:

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

D) Solver
opt <c, x>
s.t.
Ax sense c_ub
v_lb <=  x <= v_ub

Extra
vartype = {'C','I','B'}
sense = {'<','>','='}


"""
function MatrixOptimizer(raw_lp::F) where {F <: AbstractLPForm{T}} where T
    lp = change_form(LPMatrixOptInterfaceForm{T}, raw_lp)
    num_variables = length(lp.c)
    num_constraints = length(lp.c_lb)

    # use caching
    optimizer = Model{T}()

    x = MOI.add_variables(optimizer, num_variables)

    objective_function = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(lp.c, x), 0.0)
    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
            objective_function)
    MOI.set(optimizer, MOI.ObjectiveSense(), lp.direction)

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