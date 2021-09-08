using SparseArrays, Test

import MatrixOptInterface

const MatOI = MatrixOptInterface
const MOI = MatOI.MOI
const MOIU = MatOI.MOIU
const MOIB = MOI.Bridges
const MOIT = MOI.Test


const ATOL = 1e-4
const RTOL = 1e-4

include("conic_form.jl")

const dense_A = [1.0 2.0
                 3.0 4.0]
@testset "Matrix $(typeof(A))" for A in [
    dense_A, sparse(dense_A)
]
    dense_b = [5.0, 6.0]
    dense_c = [7.0, 8.0]
    @testset "Vector $(typeof(b))" for (b, c) in zip(
        [dense_b, sparsevec(dense_b)], [dense_c, sparsevec(dense_c)]
    )
        @testset "Standard form LP" begin
            s = """
            variables: x1, x2
            cx1: x1 >= 0.0
            cx2: x2 >= 0.0
            c1:  x1 + 2x2 == 5.0
            c2: 3x1 + 4x2 == 6.0
            minobjective: 7x1 + 8x2
            """
            expected = MOIU.Model{Float64}()
            MOIU.loadfromstring!(expected, s)

            var_names = ["x1", "x2"]
            con_names = ["c1", "c2"]
            vcon_names = ["cx1", "cx2"]
            model = MOIU.Model{Float64}()

            sense = MOI.MIN_SENSE
            v_lb = [0.0, 0.0]
            v_ub = [Inf, Inf]

            function test_expected(form)
                MOI.copy_to(MOI.Bridges.Constraint.Scalarize{Float64}(model), form)
                MOI.set(model, MOI.VariableName(), MOI.VariableIndex.(1:2), var_names)
                MOI.set(model, MOI.ConstraintName(), MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}.(1:2), con_names)
                MOIT.util_test_models_equal(model, expected, var_names, con_names)
            end

            @testset "change $(typeof(lp))" for lp in [
                MatOI.LPStandardForm{Float64, typeof(A), typeof(c)}(
                    sense, c, A, b
                ),
                MatOI.LPForm{Float64, typeof(A), typeof(c)}(
                    sense, c, A, b, b, v_lb, v_ub
                ),
                MatOI.LPSolverForm{Float64, typeof(A), typeof(c)}(
                    sense, c, A, b, [MatOI.EQUAL_TO, MatOI.EQUAL_TO], v_lb, v_ub
                )
            ]
                test_expected(lp)
                @testset "to $F" for F in [
                    #MatOI.LPStandardForm{Float64, typeof(A)}, # FIXME doesn't work as the form gets bloated in the conversion
                    MatOI.LPForm{Float64, typeof(A), typeof(c)},
                    MatOI.LPSolverForm{Float64, typeof(A), typeof(c)}
                ]
                    test_expected(MatOI.change_form(F, lp))
                end
            end
        end
        @testset "Geometric form LP" begin
            s = """
            variables: x1, x2
            c1:  x1 + 3x2 <= 7.0
            c2: 2x1 + 4x2 <= 8.0
            maxobjective: 5x1 + 6x2
            """
            expected = MOIU.Model{Float64}()
            MOIU.loadfromstring!(expected, s)

            var_names = ["x1", "x2"]
            con_names = ["c1", "c2"]
            model = MOIU.Model{Float64}()

            sense = MOI.MAX_SENSE
            v_lb = [-Inf, -Inf]
            v_ub = [Inf, Inf]

            function test_expected(form)
                MOI.copy_to(model, form)
                MOI.set(model, MOI.VariableName(), MOI.VariableIndex.(1:2), var_names)
                MOI.set(model, MOI.ConstraintName(), MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}.(1:2), con_names)
                MOIT.util_test_models_equal(model, expected, var_names, con_names)
            end

            @testset "change $(typeof(lp))" for lp in [
                MatOI.LPGeometricForm{Float64, typeof(A'), typeof(c)}(
                    sense, b, A', c
                ),
                MatOI.LPForm{Float64, typeof(A'), typeof(c)}(
                    sense, b, A', [-Inf, -Inf], c, v_lb, v_ub
                ),
                MatOI.LPSolverForm{Float64, typeof(A'), typeof(c)}(
                    sense, b, A', c, [MatOI.LESS_THAN, MatOI.LESS_THAN], v_lb, v_ub
                )
            ]
                test_expected(lp)
                @testset "to $F" for F in [
                    MatOI.LPGeometricForm{Float64, typeof(A), typeof(c)},
                    MatOI.LPForm{Float64, typeof(A), typeof(c)},
                    MatOI.LPSolverForm{Float64, typeof(A), typeof(c)}
                ]
                    test_expected(MatOI.change_form(F, lp))
                end
            end
        end
    end
end
