using QPSReader

qps1 = readqps(joinpath(@__DIR__, "QAFIRO.SIF")) # lower bounds
qps2 = readqps(joinpath(@__DIR__, "HS21.SIF")) # low/upp bounds
qps3 = readqps(joinpath(@__DIR__, "HS52.SIF")) # free bounds
qps4 = readqps(joinpath(@__DIR__, "AFIRO.SIF")) # LP
qps5 = readqps(joinpath(@__DIR__, "HS35MOD.SIF")) # fixed variables

Q = [
  6.0 2.0 1.0
  2.0 5.0 2.0
  1.0 2.0 4.0
]
c = [-8.0; -3; -3]
A = [
  1.0 0.0 1.0
  0.0 2.0 1.0
]
b = [0.0; 3]
l = [0.0; 0; 0]
u = [Inf; Inf; Inf]
