# MatrixOptInterface.jl

> [!WARNING]
> Needs the development version of MathOptInterface

[![Build Status](https://github.com/jump-dev/MatrixOptInterface.jl/workflows/CI/badge.svg)](https://github.com/jump-dev/MatrixOptInterface.jl/actions)
[![Coverage](https://codecov.io/gh/jump-dev/MatrixOptInterface.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jump-dev/MatrixOptInterface.jl)

**This package is this in early development, feedback is welcome!**

An interface for optimization problems in matrix forms.
Several matrix forms are represented in subtypes of
[`MathOptInterface (MOI)`](https://github.com/jump-dev/MathOptInterface.jl)
models.
As they implement the MOI API, they can be copied directly to an MOI or JuMP
model.

Here is a simple example copying a linear program represented in its standard
form into JuMP:
```julia
import MatrixOptInterface
const MatOI = MatrixOptInterface
lp = MatOI.LPStandardForm{Float64, Matrix{Float64}}(MOI.MAX_SENSE, [1.0, 0.0], [1.0 1.0], [1.0])

using JuMP
model = Model()
MOI.copy_to(model, lp)
```

Giving some arbitrary names to the variables for pretty printing:
```julia
set_name.(all_variables(model), "x" .* string.(1:2))
print(model)
```
we get the following output:
```
Max x1
Subject to
 x1 + x2 = 1.0
 [x1, x2] ∈ MathOptInterface.Nonnegatives(2)
```

The LP standard form is the the only one implemented in this package.
The 4 different available forms of linear programs are given below.
In each form, `sense` is a `MOI.OptimizationSense` so its values
are either `MOI.FEASIBILITY_SENSE`, `MOI.MAX_SENSE` or `MOI.MIN_SENSE`.

An LP standard form with `LPStandardForm(sense, c, A, b)`:
```
sense ⟨c, x⟩
s.t.
Ax == b
 x >= 0
```

An LP geometric form with `LPGeometricForm(sense, c, A, b)`:
```
sense ⟨c, x⟩
s.t.
Ax <= b
```

A generic LP form with `LPForm(sense, c, A, c_lb, c_ub, v_lb, v_ub)`:
```
sense ⟨c, x⟩
s.t.
c_lb <= Ax <= c_ub
v_lb <=  x <= v_ub
```

An LP Solver form with `LPSolverForm(sense, c, A, b, senses, v_lb, v_ub)`:
```
sense ⟨c, x⟩
s.t.
Ax senses b
v_lb <=  x <= v_ub
```
where `senses` is a vector whose `i`th entry is either
* `MatOI.EQUAL_TO` if `(Ax)_i = b_i`,
* `MatOI.LESS_THAN` if `(Ax)_i <= b_i`,
* `MatOI.GREATER_THAN` if `(Ax)_i >= b_i`.

## Transition from linprog

If you are used to [`MathProgBase`](https://github.com/JuliaOpt/MathProgBase.jl)'s, MATLAB's or scipy's `linprog` function, this package provides an easy transition.
The package does not provide a one-shot function that takes the form as input are returns the solution and status as
in general, solvers may have multiple solutions to report and the status cannot be summarized in a single value.
`MathOptInterface` embraces this complexity and allows to retrieve multiple solutions,
has 4 different solver-independent statuses (`MOI.TerminationStatus`, `MOI.PrimalStatus`, `MOI.DualStatus` and `MOI.RawStatusString`) and also
allows solvers to provide solver independent statuses.
For these reasons, there is no `linprog` functions covering all use cases but using this packages one covering your use cases can be implemented in just a few lines:
1) Create the appropriate structure from the matrix data, e.g. `LPForm`, `LPSolverForm`, ...
2) Copy it to a JuMP model or MOI optimizer.
3) Call `JuMP.optimize!` or `MOI.optimize!`.
4) Retrieve and return solutions and statuses.
