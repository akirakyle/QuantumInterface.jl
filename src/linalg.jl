dagger(a::AbstractOperator) = arithmetic_unary_error("Hermitian conjugate", a)
directsum(a::AbstractOperator...) = reduce(directsum, a)
ptrace(a::AbstractOperator, index) = arithmetic_unary_error("Partial trace", a)
_index_complement(b::CompositeBasis, indices) = complement(length(bases(b)), indices)
reduced(a, indices) = ptrace(a, _index_complement(basis(a), indices))
traceout!(s::StateVector, i) = ptrace(s,i)

permutesystems(a::AbstractOperator, perm) = arithmetic_unary_error("Permutations of subsystems", a)

nsubsystems(s::AbstractKet) = nsubsystems(basis(s))
nsubsystems(b::CompositeBasis) = length(bases(b))
nsubsystems(b::Basis) = 1
nsubsystems(s::AbstractOperator) = nsubsystems(basis(s))
# TODO: fixme to work with other composite bases like KetBraBasis...
nsubsystems(b::CompositeOperatorBasis) = length(bases(b))
nsubsystems(::Nothing) = 1 # TODO Exists because of QuantumSavory; Consider removing this and reworking the functions that depend on it. E.g., a reason to have it when performing a project_traceout measurement on a state that contains only one subsystem
