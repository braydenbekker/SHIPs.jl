

using JuLIP, SHIPs, LinearAlgebra

deg = 10
trans = SHIPs.IdTransform()
J = SHIPs.TransformedJacobi(deg, trans, 0.5, 3.0)
ship4 = SHIPBasis(SparseSHIP(3, 10; wL = 1.5), J)
J = ship4.J

nargs = 3
function testf(Rs)
   f = 1.0
   for R in Rs
      b = SHIPs.evaluate(J, norm(R))
      f *= b[3]
   end
   return f
end

# testf(Rs) = prod(norm.(Rs).^2)

SHIPs.Exp.determine_order(testf, nargs, ship4)
