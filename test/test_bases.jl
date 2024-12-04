using Test
using QuantumInterface: tensor, ⊗, ptrace, reduced, ptrace, permutesystems
using QuantumInterface: GenericBasis, CompositeBasis, SpinBasis, FockBasis

N1 = 6
N2 = (2, 3)
N3 = :foo

b1 = GenericBasis(N1)
b2 = GenericBasis(N2)
b3 = GenericBasis(N3)

@test length(b1) == N1
@test length(b2) == N2
@test length(b3) == N3
@test b1 == b1
@test b2 == b2
@test b3 == b3
@test b1 != b2
@test b2 != b3
@test b1 != b3
@test b1 != FockBasis(2)

b1 = GenericBasis(6)
b2 = SpinBasis(1//2)
b3 = FockBasis(2)

@test length(b1) == 6
@test length(b2) == 2
@test length(b3) == 3
@test b1 == b1
@test b2 == b2
@test b3 == b3
@test b1 != b2
@test b2 != b3
@test b1 != b3

#@test tensor(b1) == b1
comp_b1 = tensor(b1, b2)
comp_uni = b1 ⊗ b2
comp_b2 = tensor(b1, b1, b2, b3)
@test size(comp_b1) == (6,2)
@test size(comp_uni) == (6,2)
@test size(comp_b2) == (6,6,2,3)

@test b1^3 == CompositeBasis(b1, b1, b1)
@test (b1⊗b2)^2 == CompositeBasis(b1, b2, b1, b2)
@test_throws ArgumentError b1^(0)

comp_b1_b2 = tensor(comp_b1, comp_b2)
@test size(comp_b1_b2) == (6,2,6,6,2,3)
@test comp_b1_b2 == CompositeBasis(b1, b2, b1, b1, b2, b3)

@test_throws ArgumentError tensor()
@test comp_b2.bases == tensor(b1, comp_b1, b3).bases
@test comp_b2 == tensor(b1, comp_b1, b3)

@test_throws ArgumentError ptrace(comp_b1, [1, 2])
@test ptrace(comp_b2, [1,4]) == comp_b1
@test ptrace(comp_b2, [1]) == ptrace(comp_b2, [2]) == tensor(comp_b1, b3) == ptrace(comp_b2, 1)
@test ptrace(comp_b2, [1, 2]) == ptrace(comp_b1⊗b3, [1])
@test ptrace(comp_b2, [2, 3]) == ptrace(comp_b1, [2])⊗b3
@test ptrace(comp_b2, [2, 3, 4]) == reduced(comp_b2, [1])
@test_throws ArgumentError reduced(comp_b1, [])

comp1 = tensor(b1, b2, b3)
comp2 = tensor(b2, b1, b3)
@test permutesystems(comp1, [2,1,3]) == comp2
