dagger(a::AbstractOperator) = arithmetic_unary_error("Hermitian conjugate", a)
directsum(a::AbstractOperator...) = reduce(directsum, a)
ptrace(a::AbstractOperator, index) = arithmetic_unary_error("Partial trace", a)
_index_complement(b::CompositeBasis, indices) = complement(nsubsystems(b), indices)
reduced(a, indices) = ptrace(a, _index_complement(basis(a), indices))
traceout!(s::StateVector, i) = ptrace(s,i)
traceout!(op::AbstractOperator, i) = ptrace(op,i)

permutesystems(a::AbstractOperator, perm) = arithmetic_unary_error("Permutations of subsystems", a)

nsubsystems(b::Basis) = 1
nsubsystems(b::CompositeBasis) = length(b.bases)
nsubsystems(b::SumBasis) = length(b.bases)
nsubsystems(b::CompositeOperatorBasis) = length(b.bases)
nsubsystems(b::KetBraBasis) = length(b.left)
nsubsystems(b::HeisenbergWeylBasis{dims}) where {dims} = length(dims)
nsubsystems(b::PauliBasis{modes}) where {modes} = modes

nsubsystems(s::AbstractKet) = nsubsystems(basis(s))
nsubsystems(s::AbstractOperator) = nsubsystems(basis(s))
nsubsystems(::Nothing) = 1 # TODO Exists because of QuantumSavory; Consider removing this and reworking the functions that depend on it. E.g., a reason to have it when performing a project_traceout measurement on a state that contains only one subsystem

