import Base: ==, +, -, *, /, ^, length, one, exp, conj, conj!, transpose, copy
import LinearAlgebra: tr, ishermitian, norm, normalize, normalize!

basis(a::StateVector) = throw(ArgumentError("basis() is not defined for this type of state vector: $(typeof(a))."))

"""
    tensor(x::AbstractOperator, y::AbstractOperator, z::AbstractOperator...)

Tensor product ``\\hat{x}⊗\\hat{y}⊗\\hat{z}⊗…`` of the given operators.
"""
tensor(a::AbstractOperator, b::AbstractOperator) = arithmetic_binary_error("Tensor product", a, b)
tensor(op::AbstractOperator) = op
tensor(operators::AbstractOperator...) = reduce(tensor, operators)
tensor(state::StateVector) = state
tensor(states::Vector{T}) where T<:StateVector = reduce(tensor, states)

"""
    nsubsystems(a)

Return the number of subsystems of a quantum object in its tensor product
decomposition.

See also [`CompositeBasis`](@ref).
"""
nsubsystems(s::StateVector) = nsubsystems(basis(s))
nsubsystems(s::AbstractOperator) = nsubsystems(basis(s))
nsubsystems(b::CompositeBasis) = length(b.bases)
nsubsystems(b::Basis) = 1
nsubsystems(::Nothing) = 1 # TODO Exists because of QuantumSavory; Consider removing this and reworking the functions that depend on it. E.g., a reason to have it when performing a project_traceout measurement on a state that contains only one subsystem

"""
    nsubspaces(a)

Return the number of subspaces of a quantum object in its direct sum
decomposition.

See also [`SumBasis`](@ref).
"""
nsubspaces(b::SumBasis) = length(b.bases)

function apply! end

function dagger end

function directsum end
const ⊕ = directsum
directsum() = GenericBasis(0)
directsum(x::StateVector...) = reduce(directsum, x)
directsum(a::AbstractOperator...) = reduce(directsum, a)

function dm end

"""
    embed(basis1[, basis2], operators::Dict)

`operators` is a dictionary `Dict{Vector{Int}, AbstractOperator}`. The integer vector
specifies in which subsystems the corresponding operator is defined.
"""
"""
    embed(basis1[, basis2], indices::Vector, operators::Vector)

Tensor product of operators where missing indices are filled up with identity operators.
"""
function embed end

function entanglement_entropy end

"""
    expect(index, op, state)

If an `index` is given, it assumes that `op` is defined in the subsystem specified by this number.
"""

function expect end

function identityoperator end

"""
    permutesystems(a, perm)

Change the ordering of the subsystems of the given object.

For a permutation vector `[2,1,3]` and a given object with basis `[b1, b2, b3]`
this function results in `[b2, b1, b3]`.
"""
function permutesystems end

permutesystems(a::AbstractOperator, perm) = arithmetic_unary_error("Permutations of subsystems", a)

function projector end

function project! end

function projectrand! end

"""
    ptrace(a, indices)

Partial trace of the given basis, state or operator.

The `indices` argument, which can be a single integer or a vector of integers,
specifies which subsystems are traced out. The number of indices has to be
smaller than the number of subsystems, i.e. it is not allowed to perform a
full trace.
"""
function ptrace end

"""
    reduced(a, indices)

Reduced basis, state or operator on the specified subsystems.

The `indices` argument, which can be a single integer or a vector of integers,
specifies which subsystems are kept. At least one index must be specified.
"""
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

"""
    variance(index, op, state)

If an `index` is given, it assumes that `op` is defined in the subsystem specified by this number
"""
function variance end


"""Prepare the identity superoperator over a given space."""
function identitysuperoperator end


"""
    ishermitian(op::AbstractOperator)

Check if an operator is Hermitian.
"""
ishermitian(op::AbstractOperator) = arithmetic_unary_error(ishermitian, op)

"""
    tr(x::AbstractOperator)

Trace of the given operator.
"""
tr(x::AbstractOperator) = arithmetic_unary_error("Trace", x)

"""
    norm(x::StateVector)

Norm of the given bra or ket state.
"""
norm(x::StateVector) = norm(x.data) # FIXME issue #12

"""
    normalize(x::StateVector)

Return the normalized state so that `norm(x)` is one.
"""
normalize(x::StateVector) = x/norm(x)

"""
    normalize!(x::StateVector)

In-place normalization of the given bra or ket so that `norm(x)` is one.
"""
normalize!(x::StateVector) = (normalize!(x.data); x) # FIXME issue #12

"""
    normalize(op)

Return the normalized operator so that its `tr(op)` is one.
"""
normalize(op::AbstractOperator) = op/tr(op)

"""
    normalize!(op)

In-place normalization of the given operator so that its `tr(x)` is one.
"""
normalize!(op::AbstractOperator) = throw(ArgumentError("normalize! is not defined for this type of operator: $(typeof(op)).\n You may have to fall back to the non-inplace version 'normalize()'."))


# Common error messages
arithmetic_unary_error(funcname, x::AbstractOperator) = throw(ArgumentError("$funcname is not defined for this type of operator: $(typeof(x)).\nTry to convert to another operator type first with e.g. dense() or sparse()."))
arithmetic_binary_error(funcname, a::AbstractOperator, b::AbstractOperator) = throw(ArgumentError("$funcname is not defined for this combination of types of operators: $(typeof(a)), $(typeof(b)).\nTry to convert to a common operator type first with e.g. dense() or sparse()."))
addnumbererror() = throw(ArgumentError("Can't add or subtract a number and an operator. You probably want 'op + identityoperator(op)*x'."))

one(x::Union{<:Basis,<:AbstractOperator}) = identityoperator(x)

"""
    transpose(op)

Transpose of the given operator.
"""
transpose(a::AbstractOperator) = arithmetic_unary_error("Transpose", a)

"""
    length(b::Basis)

Total dimension of the Hilbert space.
"""
Base.length(b::Basis) = throw(ArgumentError("Base.length() is not defined for $(typeof(b))"))

"""
    size(b::Basis)

A vector containing the local dimensions of each Hilbert space in its tensor
product decomposition into subsystems.

See also [`nsubsystems`](@ref) and [`CompositeBasis`](@ref).
"""
Base.size(b::Basis) = [length(b)]


##
# States
##

==(a::AbstractKet, b::AbstractBra) = false
==(a::AbstractBra, b::AbstractKet) = false
-(a::T) where {T<:StateVector} = T(basis(a), -a.data) # FIXME issue #12
*(a::StateVector, b::Number) = b*a
copy(a::T) where {T<:StateVector} = T(basis(a), copy(a.data)) # FIXME issue #12
length(a::StateVector) = length(basis(a))::Int

# Array-like functions
Base.size(x::StateVector) = size(x.data) # FIXME issue #12
@inline Base.axes(x::StateVector) = axes(x.data) # FIXME issue #12
Base.ndims(x::StateVector) = 1
Base.ndims(::Type{<:StateVector}) = 1
Base.eltype(x::StateVector) = eltype(x.data) # FIXME issue #12

# Broadcasting
Base.broadcastable(x::StateVector) = x

Base.adjoint(a::StateVector) = dagger(a)
dagger(a::StateVector) = arithmetic_unary_error("Hermitian conjugate", a)


##
# Operators
##

length(a::AbstractOperator) = length(basis_l(a))::Int*length(basis_r(a))::Int
basis(a::AbstractOperator) = (check_samebases(basis_l(a), basis_r(a)); basis_l(a))
basis_l(a::AbstractOperator) = throw(ArgumentError("basis_l() is not defined for this type of operator: $(typeof(a))."))
basis_r(a::AbstractOperator) = throw(ArgumentError("basis_r() is not defined for this type of operator: $(typeof(a))."))

# Ensure scalar broadcasting
Base.broadcastable(x::AbstractOperator) = Ref(x)

# Arithmetic operations
+(a::AbstractOperator, b::AbstractOperator) = arithmetic_binary_error("Addition", a, b)
+(a::Number, b::AbstractOperator) = addnumbererror()
+(a::AbstractOperator, b::Number) = addnumbererror()
+(a::AbstractOperator) = a

-(a::AbstractOperator) = arithmetic_unary_error("Negation", a)
-(a::AbstractOperator, b::AbstractOperator) = arithmetic_binary_error("Subtraction", a, b)
-(a::Number, b::AbstractOperator) = addnumbererror()
-(a::AbstractOperator, b::Number) = addnumbererror()

*(a::AbstractOperator, b::AbstractOperator) = arithmetic_binary_error("Multiplication", a, b)
^(a::AbstractOperator, b::Integer) = Base.power_by_squaring(a, b)

"""
    addible(a, b)

Check if any two subtypes of `StateVector` or `AbstractOperator`
 can be added together.

Spcefically this checks whether the left basis of a is equal
to the left basis of b and whether the right basis of a is equal
to the right basis of b.
"""
addible(a::Union{<:StateVector,<:AbstractOperator},
        b::Union{<:StateVector,<:AbstractOperator}) = false
addible(a::AbstractBra, b::AbstractBra) = (basis(a) == basis(b))
addible(a::AbstractKet, b::AbstractKet) = (basis(a) == basis(b))
addible(a::AbstractOperator, b::AbstractOperator) =
    (basis_l(a) == basis_l(b)) && (basis_r(a) == basis_r(b))


"""
    multiplicable(a, b)

Check if any two subtypes of `StateVector` or `AbstractOperator`,
can be multiplied in the given order.
"""
multiplicible(a::Union{<:StateVector,<:AbstractOperator},
              b::Union{<:StateVector,<:AbstractOperator}) = false
multiplicable(a::AbstractBra, b::AbstractKet) = (basis(a) == basis(b))
multiplicable(a::AbstractOperator, b::AbstractKet) = (basis_r(a) == basis(b))
multiplicable(a::AbstractBra, b::AbstractOperator) = (basis(a) == basis_l(b))
multiplicable(a::AbstractOperator, b::AbstractOperator) = (basis_r(a) == basis_l(b))

"""
    exp(op::AbstractOperator)

Operator exponential.
"""
exp(op::AbstractOperator) = throw(ArgumentError("exp() is not defined for this type of operator: $(typeof(op)).\nTry to convert to dense operator first with dense()."))

Base.size(op::AbstractOperator) = (length(basis_l(op)),length(basis_r(op)))
function Base.size(op::AbstractOperator, i::Int)
    i < 1 && throw(ErrorException("dimension index is < 1"))
    i > 2 && return 1
    i==1 ? length(basis_l(op)) : length(basis_r(op))
end

Base.adjoint(a::AbstractOperator) = dagger(a)
dagger(a::AbstractOperator) = arithmetic_unary_error("Hermitian conjugate", a)

conj(a::AbstractOperator) = arithmetic_unary_error("Complex conjugate", a)
conj!(a::AbstractOperator) = conj(a::AbstractOperator)

ptrace(a::AbstractOperator, index) = arithmetic_unary_error("Partial trace", a)
