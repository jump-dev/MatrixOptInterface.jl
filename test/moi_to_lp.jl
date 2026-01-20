s = """
variables: x1, x2
cx1: x1 >= 0.0
cx2: x2 >= 0.0
c1:  x1 + 2x2 == 5.0
c2: 3x1 + 4x2 == 6.0
minobjective: 7x1 + 8x2
"""
moi = MOIU.Model{Float64}()
MOIU.loadfromstring!(moi, s)

var_names = ["x1", "x2"]
con_names = ["c1", "c2"]
vcon_names = ["cx1", "cx2"]

sense = MOI.MIN_SENSE
v_lb = [0.0, 0.0]
v_ub = [Inf, Inf]
const dense_A = [
    1.0 2.0
    3.0 4.0
]
dense_b = [5.0, 6.0]
dense_c = [7.0, 8.0]

using SparseArrays
@show lp = MatrixOptInterface.LPForm{Float64, SparseArrays.SparseMatrixCSC{Float64,Int64}, Vector{Float64}}()
@show index_map = MOI.copy_to(lp, moi)


