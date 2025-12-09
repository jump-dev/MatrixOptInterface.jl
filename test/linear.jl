module TestLinear

using SparseArrays, Test
import MathOptInterface as MOI
import MatrixOptInterface as MatOI

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$(name)", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end

function _test_expected(form, expected, var_names, con_names, S)
    model = MOI.Utilities.Model{Float64}()
    MOI.copy_to(
        MOI.Bridges.Constraint.Scalarize{Float64}(model),
        form,
    )
    MOI.set(
        model,
        MOI.VariableName(),
        MOI.VariableIndex.(eachindex(var_names)),
        var_names,
    )
    MOI.set(
        model,
        MOI.ConstraintName(),
        MOI.ConstraintIndex{
            MOI.ScalarAffineFunction{Float64},
            S,
        }.(eachindex(con_names)),
        con_names,
    )
    return MOI.Test.util_test_models_equal(
        model,
        expected,
        var_names,
        con_names,
    )
end


function test_standard_form()
    dense_A = [
        1.0 2.0
        3.0 4.0
    ]
    dense_b = [5.0, 6.0]
    dense_c = [7.0, 8.0]
    s = """
    variables: x1, x2
    cx1: x1 >= 0.0
    cx2: x2 >= 0.0
    c1:  x1 + 2x2 == 5.0
    c2: 3x1 + 4x2 == 6.0
    minobjective: 7x1 + 8x2
    """
    expected = MOI.Utilities.Model{Float64}()
    MOI.Utilities.loadfromstring!(expected, s)

    var_names = ["x1", "x2"]
    con_names = ["c1", "c2"]

    sense = MOI.MIN_SENSE
    v_lb = [0.0, 0.0]
    v_ub = [Inf, Inf]

    @testset "Matrix $(typeof(A))" for A in [dense_A, sparse(dense_A)]
        @testset "Vector $(typeof(b))" for (b, c) in zip(
            [dense_b, sparsevec(dense_b)],
            [dense_c, sparsevec(dense_c)],
        )
            @testset "change $func" for (func, args) in [
                (MatOI.lp_standard_form, (
                    sense,
                    c,
                    A,
                    b,
                )),
                (MatOI.lp_solver_form, (
                    sense,
                    c,
                    A,
                    b,
                    [MatOI.EQUAL_TO, MatOI.EQUAL_TO],
                    v_lb,
                    v_ub,
                )),
            ]
                lp = func(args...)
                _test_expected(lp, expected, var_names, con_names, MOI.EqualTo{Float64})
            end
        end
    end
end

function test_geometric_form()
    dense_A = [
        1.0 2.0
        3.0 4.0
    ]
    dense_b = [5.0, 6.0]
    dense_c = [7.0, 8.0]
    s = """
    variables: x1, x2
    cx1: x1 >= 0.0
    cx2: x2 >= 0.0
    c1: x1 + 2x2 <= 5.0
    c2: 3x1 + 4x2 <= 6.0
    minobjective: 7x1 + 8x2
    """
    expected = MOI.Utilities.Model{Float64}()
    MOI.Utilities.loadfromstring!(expected, s)

    var_names = ["x1", "x2"]
    con_names = ["c1", "c2"]

    sense = MOI.MIN_SENSE

    @testset "Matrix $(typeof(A))" for A in [dense_A, sparse(dense_A)]
        @testset "Vector $(typeof(b))" for (b, c) in zip(
            [dense_b, sparsevec(dense_b)],
            [dense_c, sparsevec(dense_c)],
        )
            lp = MatOI.lp_geometric_form(
                sense,
                c,
                A,
                b,
            )
            _test_expected(lp, expected, var_names, con_names, MOI.LessThan{Float64})
        end
    end
end

function test_form()
    dense_A = [
        1.0 2.0
        3.0 4.0
    ]
    dense_lb = [5.0, 6.0]
    dense_ub = [7.0, 9.0]
    dense_c = [7.0, 8.0]
    s = """
    variables: x1, x2
    cx1: x1 >= 0.0
    cx2: x2 >= 0.0
    c1: x1 + 2x2 in MOI.Interval(5.0, 7.0)
    c2: 3x1 + 4x2 in MOI.Interval(6.0, 9.0)
    minobjective: 7x1 + 8x2
    """
    expected = MOI.Utilities.Model{Float64}()
    MOI.Utilities.loadfromstring!(expected, s)

    var_names = ["x1", "x2"]
    con_names = ["c1", "c2"]

    sense = MOI.MIN_SENSE
    v_lb = [0.0, 0.0]
    v_ub = [Inf, Inf]

    @testset "Matrix $(typeof(A))" for A in [dense_A, sparse(dense_A)]
        @testset "Vector $(typeof(c))" for (c_lb, c_ub, c) in zip(
            [dense_lb, sparsevec(dense_lb)],
            [dense_ub, sparsevec(dense_ub)],
            [dense_c, sparsevec(dense_c)],
        )
            lp = MatOI.lp_form(
                sense,
                c,
                A,
                c_lb,
                c_ub,
                v_lb,
                v_ub,
            )
            _test_expected(lp, expected, var_names, con_names, MOI.Interval{Float64})
        end
    end
end

end

TestLinear.runtests()
