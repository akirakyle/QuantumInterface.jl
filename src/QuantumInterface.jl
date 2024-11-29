module QuantumInterface

"""
    length(b::Basis)

Total dimension of the Hilbert space.
"""
function length end

"""
    basis(a)

Return the basis of an object.


Returns B where B<:Basis when typeof(a)<:StateVector.
Returns B where B<:OperatorBasis when typeof(a)<:AbstractOperator.
Returns B where B<:SuperOperatorBasis for typeof(a)<:AbstractSuperOperator.
"""
function basis end
#basis(sv::StateVector{B}) where {B} = B
#basis(op::AbstractOperator{OperatorBasis{B,B}}) where {B} = B
#basis(op::AbstractSuperOperator{SuperOperatorBasis{OperatorBasis{B,B}, OperatorBasis{B,B}}}) where {B} = B

function apply! end

function dagger end

function directsum end
const ⊕ = directsum
directsum() = GenericBasis(0)

function dm end

function embed end

function entanglement_entropy end

function expect end

function identityoperator end

function permutesystems end

function projector end

function project! end

function projectrand! end

function ptrace end

function reduced end

"""
    tensor(x, y, z...)

Tensor product of the given objects. Alternatively, the unicode
symbol ⊗ (\\otimes) can be used.
"""
function tensor end
const ⊗ = tensor
tensor() = throw(ArgumentError("Tensor function needs at least one argument."))

function tensor_pow end # TODO should Base.^ be the same as tensor_pow?

function traceout! end
traceout!(s::StateVector, i) = ptrace(s,i)

function variance end

##
# Qubit specific
##

function nqubits end

function projectX! end

function projectY! end

function projectZ! end

function projectXrand! end

function projectYrand! end

function projectZrand! end

function reset_qubits! end

##
# Quantum optics specific
##

function coherentstate end

function thermalstate end

function displace end

function squeeze end

function wigner end


include("abstract_types.jl")
include("bases.jl")

include("linalg.jl")
include("tensor.jl")
include("embed_permute.jl")
include("expect_variance.jl")
include("identityoperator.jl")

include("julia_base.jl")
include("julia_linalg.jl")
include("sparse.jl")

include("show.jl")

include("sortedindices.jl")

end # module
