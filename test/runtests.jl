using SparseArrays, Test
using SCS
using ProxSDP
using COSMO

import MatrixOptInterface

const MatOI = MatrixOptInterface
const MOI = MatOI.MOI
const MOIU = MatOI.MOIU
const MOIB = MOI.Bridges


const ATOL = 1e-4
const RTOL = 1e-4

const dense_A = [1.0 2.0
                 3.0 4.0]
@testset "Matrix $(typeof(A))" for A in [
    dense_A, sparse(dense_A)
]
    b = [5, 6]
    c = [7, 8]
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
            MOI.copy_to(MOI.Bridges.Constraint.Scalarize{Float64}(model), form, copy_names = false)
            MOI.set(model, MOI.VariableName(), MOI.VariableIndex.(1:2), var_names)
            MOI.set(model, MOI.ConstraintName(), MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}.(1:2), con_names)
            MOI.set(model, MOI.ConstraintName(), MOI.ConstraintIndex{MOI.SingleVariable, MOI.GreaterThan{Float64}}.(1:2), vcon_names)
            MOIU.test_models_equal(model, expected, var_names, [con_names; vcon_names])
        end

        @testset "change $(typeof(lp))" for lp in [
            MatOI.LPStandardForm{Float64, typeof(A)}(
                sense, c, A, b
            ),
            MatOI.LPForm{Float64, typeof(A)}(
                sense, c, A, b, b, v_lb, v_ub
            ),
            MatOI.LPSolverForm{Float64, typeof(A)}(
                sense, c, A, b, [MatOI.EQUAL_TO, MatOI.EQUAL_TO], v_lb, v_ub
            )
        ]
            test_expected(lp)
            @testset "to $F" for F in [
                #MatOI.LPStandardForm{Float64, typeof(A)}, # FIXME doesn't work as the form gets bloated in the conversion
                MatOI.LPForm{Float64, typeof(A)},
                MatOI.LPSolverForm{Float64, typeof(A)}
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
            MOI.copy_to(model, form, copy_names = false)
            MOI.set(model, MOI.VariableName(), MOI.VariableIndex.(1:2), var_names)
            MOI.set(model, MOI.ConstraintName(), MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}.(1:2), con_names)
            MOIU.test_models_equal(model, expected, var_names, con_names)
        end

        @testset "change $(typeof(lp))" for lp in [
            MatOI.LPGeometricForm{Float64, typeof(A')}(
                sense, b, A', c
            ),
            MatOI.LPForm{Float64, typeof(A')}(
                sense, b, A', [-Inf, -Inf], c, v_lb, v_ub
            ),
            MatOI.LPSolverForm{Float64, typeof(A')}(
                sense, b, A', c, [MatOI.LESS_THAN, MatOI.LESS_THAN], v_lb, v_ub
            )
        ]
            test_expected(lp)
            @testset "to $F" for F in [
                MatOI.LPGeometricForm{Float64, typeof(A)},
                MatOI.LPForm{Float64, typeof(A)},
                MatOI.LPSolverForm{Float64, typeof(A)}
            ]
                test_expected(MatOI.change_form(F, lp))
            end
        end
    end
end

CONIC_OPTIMIZERS = [SCS.Optimizer, ProxSDP.Optimizer, COSMO.Optimizer]

@testset "MOI to MatOI conversion 1" begin
    # _psd1test: https://github.com/jump-dev/MathOptInterface.jl/blob/master/src/Test/contconic.jl#L2417

    for optimizer in CONIC_OPTIMIZERS
        model = MOI.instantiate(optimizer, with_bridge_type=Float64)
        δ = √(1 + (3*√2+2)*√(-116*√2+166) / 14) / 2
        ε = √((1 - 2*(√2-1)*δ^2) / (2-√2))
        y2 = 1 - ε*δ
        y1 = 1 - √2*y2
        obj = y1 + y2/2
        k = -2*δ/ε
        x2 = ((3-2obj)*(2+k^2)-4) / (4*(2+k^2)-4*√2)
        α = √(3-2obj-4x2)/2
        β = k*α

        X = MOI.add_variables(model, 6)
        x = MOI.add_variables(model, 3)

        vov = MOI.VectorOfVariables(X)

        cX = MOI.add_constraint(
            model, 
            MOI.VectorAffineFunction{Float64}(vov), MOI.PositiveSemidefiniteConeTriangle(3)
        )

        cx = MOI.add_constraint(
            model, 
            MOI.VectorAffineFunction{Float64}(MOI.VectorOfVariables(x)), MOI.SecondOrderCone(3)
        )

        c1 = MOI.add_constraint(
            model, 
            MOI.VectorAffineFunction(
                MOI.VectorAffineTerm.(1:1, MOI.ScalarAffineTerm.([1., 1., 1., 1.], [X[1], X[3], X[end], x[1]])), 
                [-1.0]
            ), 
            MOI.Zeros(1)
        )

        c2 = MOI.add_constraint(
            model, 
            MOI.VectorAffineFunction(
                MOI.VectorAffineTerm.(1:1, MOI.ScalarAffineTerm.([1., 2, 1, 2, 2, 1, 1, 1], [X; x[2]; x[3]])), 
                [-0.5]
            ), 
            MOI.Zeros(1)
        )

        objXidx = [1:3; 5:6]
        objXcoefs = 2*ones(5)
        MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([objXcoefs; 1.0], [X[objXidx]; x[1]]), 0.0))
        MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

        MatModel = MatOI.getConicForm(Float64, model, [cX; cx; c1; c2])

        @test MatModel.c' ≈ [2. 2. 2. 0. 2. 2. 1. 0. 0.]
        @test MatModel.b' ≈ [-1.  -0.5  0.   0.   0.   0.   0.   0.   0.   0.   0. ]
        A = MatModel.A
        @test A[1, 1] ≈	-1.0 
        @test A[2, 1] ≈	-1.0
        @test A[6, 1] ≈	-1.0
        @test A[2, 2] ≈	-2.0
        @test A[7, 2] ≈	-1.41421 atol=ATOL rtol=RTOL
        @test A[1, 3] ≈	-1.0
        @test A[2, 3] ≈	-1.0
        @test A[9, 3] ≈	-1.0
        @test A[2, 4] ≈	-2.0
        @test A[8, 4] ≈	-1.41421 atol=ATOL rtol=RTOL
        @test A[2, 5] ≈	-2.0
        @test A[10, 5] ≈ -1.41421 atol=ATOL rtol=RTOL
        @test A[1, 6] ≈	-1.0
        @test A[2, 6] ≈	-1.0
        @test A[11, 6] ≈ -1.0
        @test A[1, 7] ≈	-1.0
        @test A[3, 7] ≈	-1.0
        @test A[2, 8] ≈	-1.0
        @test A[4, 8] ≈	-1.0
        @test A[2, 9] ≈	-1.0
        @test A[5, 9] ≈	-1.0
    end
end

@testset "MOI to MatOI conversion 2" begin
    # find equivalent diffcp program here - https://github.com/AKS1996/jump-gsoc-2020/blob/master/diffcp_sdp_3_py.ipynb
    
    for optimizer in CONIC_OPTIMIZERS
        model = MOI.instantiate(optimizer, with_bridge_type=Float64)

        x = MOI.add_variables(model, 7)
        @test MOI.get(model, MOI.NumberOfVariables()) == 7

        η = 10.0

        c1  = MOI.add_constraint(
            model, 
            MOI.VectorAffineFunction(
                MOI.VectorAffineTerm.(1, MOI.ScalarAffineTerm.(-1.0, x[1:6])),
                [η]
            ), 
            MOI.Nonnegatives(1)
        )
        c2 = MOI.add_constraint(model, MOI.VectorAffineFunction(MOI.VectorAffineTerm.(1:6, MOI.ScalarAffineTerm.(1.0, x[1:6])), zeros(6)), MOI.Nonnegatives(6))
        α = 0.8
        δ = 0.9
        c3 = MOI.add_constraint(model, MOI.VectorAffineFunction(MOI.VectorAffineTerm.([fill(1, 7); fill(2, 5);     fill(3, 6)],
                                                                MOI.ScalarAffineTerm.(
                                                                [ δ/2,       α,   δ, δ/4, δ/8,      0.0, -1.0,
                                                                    -δ/(2*√2), -δ/4, 0,     -δ/(8*√2), 0.0,
                                                                    δ/2,     δ-α,   0,      δ/8,      δ/4, -1.0],
                                                                [x[1:7];     x[1:3]; x[5:6]; x[1:3]; x[5:7]])),
                                                                zeros(3)), MOI.PositiveSemidefiniteConeTriangle(2))
        c4 = MOI.add_constraint(
            model, 
            MOI.VectorAffineFunction(
                MOI.VectorAffineTerm.(1, MOI.ScalarAffineTerm.(0.0, [x[1:3]; x[5:6]])),
                [0.0]
            ), 
            MOI.Zeros(1)
        )

        MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x[7])], 0.0))
        MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)

        MatModel = MatOI.getConicForm(Float64, model, [c1; c2; c3; c4])

        @test MatModel.c ≈ [-0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -1.0]
        @test MatModel.b ≈ [0.0, 10.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 0.0, 0.0]
        A = MatModel.A
        @test A[2 , 1]  ≈  1.0
        @test A[3 , 1]  ≈  -1.0
        @test A[9 , 1]  ≈  -0.45
        @test A[10, 1]  ≈  0.45
        @test A[11, 1]  ≈  -0.45
        @test A[2 , 2]  ≈  1.0
        @test A[4 , 2]  ≈  -1.0
        @test A[9 , 2]  ≈  -0.8
        @test A[10, 2]  ≈  0.318198 atol=ATOL rtol=RTOL
        @test A[11, 2]  ≈  -0.1
        @test A[2 , 3]  ≈  1.0
        @test A[5 , 3]  ≈  -1.0
        @test A[9 , 3]  ≈  -0.9
        @test A[2 , 4]  ≈  1.0
        @test A[6 , 4]  ≈  -1.0
        @test A[9 , 4]  ≈  -0.225
        @test A[2 , 5]  ≈  1.0
        @test A[7 , 5]  ≈  -1.0
        @test A[9 , 5]  ≈  -0.1125
        @test A[10, 5]  ≈  0.1125
        @test A[11, 5]  ≈  -0.1125
        @test A[2 , 6]  ≈  1.0
        @test A[8 , 6]  ≈  -1.0
        @test A[11, 6]  ≈  -0.225
        @test A[9 , 7]  ≈  1.0
        @test A[11, 7]  ≈  1.0
    end
end
