const CI = MOI.ConstraintIndex
const VI = MOI.VariableIndex

mutable struct MOISolution{R} where R <: Real
    termination_status::MOI.TerminationStatusCode
    primal_status::MOI.ResultStatusCode
    dual_status::MOI.ResultStatusCode
    primal::Vector{R}
    dual::Vector{R}
    slack::Vector{R}
    objval::R
    solve_time::R
    # other params
    objective_bound::R
    relative_mip_gap::R
    iteration_count::Int
    barrier_iterations::Int
    node_count::Int
end
MOISolution(R::DataType) = MOISolution{R}(MOI.OPTIMIZE_NOT_CALLED,
                            MOI.NO_SOLUTION,
                            MOI.NO_SOLUTION,
                            R[], R[], R[], NaN, 0.0,
                            #other
                            NaN, NaN, 1, 1, 0)

# Used to build the data with allocate-load during `copy_to`.
# When `optimize!` is called, a the data is passed to SCS
# using `SCS_solve` and the `ModelData` struct is discarded
mutable struct ModelData
    # problem dimensions
    m::Int # Number of rows/constraints
    n::Int # Number of cols/variables
    # A matrix in sparse form
    I::Vector{Int} # List of rows
    J::Vector{Int} # List of cols
    V::Vector{Float64} # List of coefficients
    # constraint bounds
    b::Vector{Float64} # constants
    sense::Vector{ConstraintSense} # constants
    # objective
    objconstant::Float64 # The objective is min c'x + objconstant
    c::Vector{Float64}
    # variable type
    # v_type::Vector{}
end

# This is tied to SCS's internal representation
mutable struct ConeData
    eq::Int # number of == constraints
    lt::Int # number of <= equality constraints
    # gt::Int # number of >= constraints
    function ConeData()
        new(0, 0)#, 0)
    end
end

mutable struct Optimizer <: MOI.AbstractOptimizer
    solve_function::Function
    cone::ConeData
    maxsense::Bool
    data::Union{Nothing, ModelData} # only non-Void between MOI.copy_to and MOI.optimize!
    sol::MOISolution
    options
    function Optimizer(solve_function = no_solve_function; options...)
        new(ConeData(), false, nothing, MOISolution(), options)
    end
end

function no_solve_function(optimizer::Optimizer)
    error("No solver set.")
end

MOI.get(::Optimizer, ::MOI.SolverName) = "Matrix optimizer"

function MOI.is_empty(optimizer::Optimizer)
    !optimizer.maxsense && optimizer.data === nothing
end
function MOI.empty!(optimizer::Optimizer)
    optimizer.maxsense = false
    optimizer.data = nothing # It should already be nothing except if an error is thrown inside copy_to
    optimizer.sol.ret_val = 0
end

MOIU.supports_allocate_load(::Optimizer, copy_names::Bool) = !copy_names

function MOI.supports(::Optimizer,
                      ::Union{MOI.ObjectiveSense,
                              MOI.ObjectiveFunction{MOI.SingleVariable},
                              MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}})
    return true
end

function MOI.supports(::Optimizer, ::MOI.VariablePrimalStart,
                      ::Type{MOI.VariableIndex})
    return true
end

function MOI.supports(::Optimizer,
                      ::Union{MOI.ConstraintPrimalStart,
                              MOI.ConstraintDualStart},
                      ::Type{<:MOI.ConstraintIndex})
    return true
end

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{<:MOI.VectorAffineFunction{Float64}},
    ::Type{<:Union{MOI.Zeros, MOI.Nonnegatives}})
    # ::Type{MOI.Nonnegatives})
    return true
end

function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike; kws...)
    return MOIU.automatic_copy_to(dest, src; kws...)
end

using Compat.SparseArrays

# Computes cone dimensions
function constroffset(cone::ConeData,
                      ci::CI{<:MOI.AbstractFunction, MOI.Zeros})
    return ci.value
end
#_allocate_constraint: Allocate indices for the constraint `f`-in-`s`
# using information in `cone` and then update `cone`
function _allocate_constraint(cone::ConeData, f, s::MOI.Zeros)
    ci = cone.f
    cone.eq += MOI.dimension(s)
    return ci
end
function constroffset(cone::ConeData,
                      ci::CI{<:MOI.AbstractFunction, MOI.Nonnegatives})
    return cone.eq + ci.value
end
function _allocate_constraint(cone::ConeData, f, s::MOI.Nonnegatives)
    ci = cone.gt
    cone.gt += MOI.dimension(s)
    return ci
end
function constroffset(optimizer::Optimizer, ci::CI)
    return constroffset(optimizer.cone, ci::CI)
end
function MOIU.allocate_constraint(optimizer::Optimizer, f::F, s::S) where {F <: MOI.AbstractFunction, S <: MOI.AbstractSet}
    return CI{F, S}(_allocate_constraint(optimizer.cone, f, s))
end

output_index(t::MOI.VectorAffineTerm) = t.output_index
variable_index_value(t::MOI.ScalarAffineTerm) = t.variable_index.value
variable_index_value(t::MOI.VectorAffineTerm) = variable_index_value(t.scalar_term)
coefficient(t::MOI.ScalarAffineTerm) = t.coefficient
coefficient(t::MOI.VectorAffineTerm) = coefficient(t.scalar_term)
# constrrows: Recover the number of rows used by each constraint.
# When, the set is available, simply use MOI.dimension
constrrows(s::MOI.AbstractVectorSet) = 1:MOI.dimension(s)
# When only the index is available, use the `optimizer.ncone.nrows` field
constrrows(optimizer::Optimizer, ci::CI{<:MOI.AbstractVectorFunction, <:MOI.AbstractVectorSet}) = 1:optimizer.cone.nrows[constroffset(optimizer, ci)]

constraint_senses(s::MOI.Zeros) = fill(EQUAL_TO, MOI.dimension(s))
constraint_senses(s::MOI.Nonnegatives) = fill(GREATER_THAN, MOI.dimension(s))
function MOIU.load_constraint(optimizer::Optimizer, ci, f::MOI.VectorAffineFunction, s::MOI.AbstractVectorSet)
    A = sparse(output_index.(f.terms), variable_index_value.(f.terms), coefficient.(f.terms))
    # sparse combines duplicates with + but does not remove zeros created so we call dropzeros!
    dropzeros!(A)
    I, J, V = findnz(A)
    offset = constroffset(optimizer, ci)
    rows = constrrows(s)
    optimizer.cone.nrows[offset] = length(rows)
    i = offset .+ rows
    b = f.constants
    # Input format is Ax + b >= 0
    # Ouput format is A2x <= b -> -Ax <= b
    # The SCS format is b - Ax ∈ cone
    optimizer.data.b[i] = b
    optimizer.data.sense[i] = constraint_senses(s)
    append!(optimizer.data.I, offset .+ I)
    append!(optimizer.data.J, J)
    append!(optimizer.data.V, -V)
end

function MOIU.allocate_variables(optimizer::Optimizer, nvars::Integer)
    optimizer.cone = ConeData()
    VI.(1:nvars)
end

function MOIU.load_variables(optimizer::Optimizer, nvars::Integer)
    cone = optimizer.cone
    m = cone.eq + cone.gt
    I = Int[]
    J = Int[]
    V = Float64[]
    b = zeros(m)
    sense = fill(EQUAL_TO, m)
    c = zeros(nvars)
    optimizer.data = ModelData(m, nvars, I, J, V, b, sense, 0., c)
    # `optimizer.sol` contains the result of the previous optimization.
    # It is used as a warm start if its length is the same, e.g.
    # probably because no variable and/or constraint has been added.
    optimizer.sol.primal = zeros(nvars)
    optimizer.sol.dual = zeros(m)
    optimizer.sol.slack = zeros(m)
    nothing
end

function MOIU.allocate(::Optimizer, ::MOI.VariablePrimalStart,
                       ::MOI.VariableIndex, ::Union{Nothing, Float64})
end
function MOIU.allocate(::Optimizer, ::MOI.ConstraintPrimalStart,
                       ::MOI.ConstraintIndex,
                       ::Union{Nothing, AbstractVector{Float64}})
end
function MOIU.allocate(::Optimizer, ::MOI.ConstraintDualStart,
                       ::MOI.ConstraintIndex,
                       ::Union{Nothing, AbstractVector{Float64}})
end
function MOIU.allocate(optimizer::Optimizer, ::MOI.ObjectiveSense, sense::MOI.OptimizationSense)
    optimizer.maxsense = sense == MOI.MAX_SENSE
end
function MOIU.allocate(::Optimizer, ::MOI.ObjectiveFunction,
                       ::MOI.Union{MOI.SingleVariable,
                                   MOI.ScalarAffineFunction{Float64}})
end

function MOIU.load(::Optimizer, ::MOI.VariablePrimalStart,
                   ::MOI.VariableIndex, ::Nothing)
end
function MOIU.load(optimizer::Optimizer, ::MOI.VariablePrimalStart,
                   vi::MOI.VariableIndex, value::Float64)
    optimizer.sol.primal[vi.value] = value
end
function MOIU.load(::Optimizer, ::MOI.ConstraintPrimalStart,
                   ::MOI.ConstraintIndex, ::Nothing)
end
function MOIU.load(optimizer::Optimizer, ::MOI.ConstraintPrimalStart,
                   ci::MOI.ConstraintIndex, value)
    offset = constroffset(optimizer, ci)
    rows = constrrows(optimizer, ci)
    optimizer.sol.slack[offset .+ rows] .= value
end
function MOIU.load(::Optimizer, ::MOI.ConstraintDualStart,
                   ::MOI.ConstraintIndex, ::Nothing)
end
function MOIU.load(optimizer::Optimizer, ::MOI.ConstraintDualStart,
                   ci::MOI.ConstraintIndex, value)
    offset = constroffset(optimizer, ci)
    rows = constrrows(optimizer, ci)
    optimizer.sol.dual[offset .+ rows] .= value
end
function MOIU.load(::Optimizer, ::MOI.ObjectiveSense, ::MOI.OptimizationSense)
end
function MOIU.load(optimizer::Optimizer, ::MOI.ObjectiveFunction,
                   f::MOI.SingleVariable)
    MOIU.load(optimizer,
              MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
              MOI.ScalarAffineFunction{Float64}(f))
end
function MOIU.load(optimizer::Optimizer, ::MOI.ObjectiveFunction,
                   f::MOI.ScalarAffineFunction)
    c0 = Vector(sparsevec(variable_index_value.(f.terms), coefficient.(f.terms),
                          optimizer.data.n))
    optimizer.data.objconstant = f.constant
    optimizer.data.c = optimizer.maxsense ? -c0 : c0
    return nothing
end

function MOI.optimize!(optimizer::Optimizer)
    sol = optimizer.solve_function(optimizer)
    sol.objval = (optimizer.maxsense ? -1 : 1) *
                    dot(optimizer.data.c, sol.primal) +
                    optimizer.objconstant
    optimizer.sol = sol
end

function MOI.get(optimizer::Optimizer, ::MOI.SolveTime)
    return optimizer.sol.solve_time
end

function MOI.get(optimizer::Optimizer, ::MOI.TerminationStatus)
    return optimizer.sol.termination_status
end

MOI.get(optimizer::Optimizer, ::MOI.ObjectiveValue) = optimizer.sol.objval

function MOI.get(optimizer::Optimizer, ::MOI.PrimalStatus)
    return optimizer.sol.primal_status
end
function MOI.get(optimizer::Optimizer, ::MOI.VariablePrimal, vi::VI)
    optimizer.sol.primal[vi.value]
end
function MOI.get(optimizer::Optimizer, a::MOI.VariablePrimal, vi::Vector{VI})
    return MOI.get.(optimizer, a, vi)
end
function MOI.get(optimizer::Optimizer, ::MOI.ConstraintPrimal,
                 ci::CI{<:MOI.AbstractFunction, S}) where S <: MOI.AbstractSet
    offset = constroffset(optimizer, ci)
    rows = constrrows(optimizer, ci)
    primal = optimizer.sol.slack[offset .+ rows]
    return primal
end

function MOI.get(optimizer::Optimizer, ::MOI.DualStatus)
    return optimizer.sol.dual_status
end
function MOI.get(optimizer::Optimizer, ::MOI.ConstraintDual,
                 ci::CI{<:MOI.AbstractFunction, S}) where S <: MOI.AbstractSet
    offset = constroffset(optimizer, ci)
    rows = constrrows(optimizer, ci)
    dual = optimizer.sol.dual[offset .+ rows]
    return dual
end

MOI.get(optimizer::Optimizer, ::MOI.ResultCount) = 1