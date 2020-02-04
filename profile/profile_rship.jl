
using PoSH, JuLIP, BenchmarkTools, LinearAlgebra, Test, Random, StaticArrays
using JuLIP
using JuLIP: evaluate, evaluate_d, evaluate!
using JuLIP.Testing

function randR()
   R = rand(JVecF) .- 0.5
   return (0.9 + 2 * rand()) * R/norm(R)
end
randR(N) = [ randR() for n=1:N ], zeros(Int16, N), 0
randcoeffs(B) = rand(length(B)) .* (1:length(B)).^(-2)

trans = PolyTransform(3, 1.0)
fcut = PolyCutoff2s(2, 0.5, 3.0)
B = SHIPBasis(SparseSHIP(6, 12, wL=2.0), trans, fcut)
coeffs = randcoeffs(B)
ship = SHIP(B, coeffs)
rship = RSHIP(ship.J, PoSH.SphericalHarmonics.RSHBasis(ship.SH.maxL), ship.zlist,
              ship.alists, ship.aalists, real.(ship.coeffs))
rship1 = PoSH.convertc2r(ship)
J32 = PoSH.floattype(ship.J, Float32)
SH32 = PoSH.SphericalHarmonics.RSHBasis(ship.SH.maxL, Float32)
rship2 = RSHIP(J32, SH32, rship1.zlist,
               rship1.alists, rship1.aalists,
               (Float32.(rship.coeffs[1]),))

Rs, Zs, z0 = randR(10)
Rs32 = [ Float32.(R) for R in Rs ]
tmp = PoSH.alloc_temp(ship, length(Rs))
rtmp = PoSH.alloc_temp(rship, length(Rs))
rtmp1 = PoSH.alloc_temp(rship1, length(Rs))
rtmp2 = PoSH.alloc_temp(rship2, length(Rs))
@btime evaluate!($tmp, $ship, $Rs, $Zs, $z0)
@btime evaluate!($rtmp, $rship, $Rs, $Zs, $z0)
@btime evaluate!($rtmp1, $rship1, $Rs, $Zs, $z0)
@btime evaluate!($rtmp2, $rship2, $Rs32, $Zs, $z0)
# @btime PoSH.evaluate_new!($rtmp, $rship, $Rs, $Zs, $z0)

# profiling results with Int32 and Float64
# (base) Fuji-2:profile ortner$ j13 profile_rship.jl
# [ Info: Added 5403 basis function(s).
#   71.257 μs (0 allocations: 0 bytes)
#   26.281 μs (0 allocations: 0 bytes)
#   43.119 μs (0 allocations: 0 bytes)
