module MatrixOptInterface

using LinearAlgebra
using MathOptInterface
const MOI = MathOptInterface
const MOIU = MathOptInterface.Utilities

export MatrixOptimizer

@enum ConstraintSense EQUAL_TO GREATER_THAN LESS_THAN

@enum VariableType CONTINUOUS INTEGER BINARY

include("matrix_input.jl")
include("allocate_load.jl")

end