dagger(a::AbstractOperator) = arithmetic_unary_error("Hermitian conjugate", a)
directsum(a::AbstractOperator...) = reduce(directsum, a)
ptrace(a::AbstractOperator, index) = arithmetic_unary_error("Partial trace", a)
_index_complement(b::CompositeBasis, indices) = complement(nsubsystems(b), indices)
reduced(a, indices) = ptrace(a, _index_complement(basis(a), indices))
traceout!(s::StateVector, i) = ptrace(s,i)
traceout!(op::AbstractOperator, i) = ptrace(op,i)

permutesystems(a::AbstractOperator, perm) = arithmetic_unary_error("Permutations of subsystems", a)
