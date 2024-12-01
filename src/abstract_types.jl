"""
    Basis{N}

Abstract type for all specialized bases.
The `Basis` type is meant to specify a basis of the Hilbert space of the
studied system. The type parameter must encode all relevant information to
correctly distinguish equivlent/compatible bases as it is used in method dispatch.
For example with an OperatorBasis, the left and right bases are compatible if
and only if an instance is a subtype of OperatorBasis{B,B} where {B <: Basis}.

Furthermore all subtypes must implement `Base.length` which returns the dimension
of the Hilbert space.

Composite systems can be defined with help of the [`CompositeBasis`](@ref) class.

N represents the underlying hilbert space, charicterized by it's dimensionality
Subtypes can add more type parameters as needed...
TODO: specify that it really should represent an orthornormal basis so that N is
both the number of basis elements and size of Hilbert space
"""
abstract type Basis{N} end
Base.:(==)(b1::T, b2::T) where {T<:Basis} = true
Base.:(==)(b1::Basis, b2::Basis) = false
Base.length(b::Basis{N}) where {N} = N

"""
Parametric composite type for all operator bases.
Orthonormal Operator basis is one where inner product is given
by Hilbert-Schmidt or trace inner prouduct product.
TODO: write this condition down explicitly.
Examples include pauli or Heisenberg-Weyl bases as well as
standard "ket-bra" basis

See [TODO: reference operators.md in docs]
"""
#abstract type OperatorBasis{BL<:Basis,BR<:Basis} end
#struct OperatorBasis{BL<:Basis,BR<:Basis} end
abstract type OperatorBasis{N,M} end
abstract type UnitaryOperatorBasis{N,M} <: OperatorBasis{N,M} end
Base.:(==)(b1::T, b2::T) where {T<:OperatorBasis} = true
Base.:(==)(b1::OperatorBasis, b2::OperatorBasis) = false
Base.size(b::OperatorBasis{N,M}) where {N,M} = (N,M)
Base.length(b::OperatorBasis{N,M}) where {N,M} = N*M

"""
Parametric composite type for all superoperator bases.

See [TODO: reference superoperators.md in docs]
"""
abstract type SuperOperatorBasis{N,M} end
Base.:(==)(b1::T, b2::T) where {T<:SuperOperatorBasis} = true
Base.:(==)(b1::SuperOperatorBasis, b2::SuperOperatorBasis) = false

"""
Exception that should be raised for an illegal algebraic operation.
"""
mutable struct IncompatibleBases <: Exception end

"""
Abstract type for `Bra` and `Ket` states.

The state vector type stores an abstract state with respect to a certain
Hilbert space basis.
All deriving types must define the `fullbasis` function which
returns the state vector's underlying `Basis`.

Must implement hash for subspace basis?
"""
abstract type StateVector{B<:Basis} end
abstract type AbstractKet{B} <: StateVector{B} end
abstract type AbstractBra{B} <: StateVector{B} end

"""
Abstract type for all operators which represent linear maps between two
Hilbert spaces with respect to a given basis in each space.

All deriving operator types must define the `fullbasis` function which
returns the operator's underlying `OperatorBasis`.

For fast time evolution also at least the function
`mul!(result::Ket,op::AbstractOperator,x::Ket,alpha,beta)` should be
implemented. Many other generic multiplication functions can be defined in
terms of this function and are provided automatically.

See [TODO: reference operators.md in docs]
"""
abstract type AbstractOperator{B<:OperatorBasis} end

"""
Abstract type for all super-operators which represent linear maps between two
operator spaces with respect to a given basis for each space.

All deriving operator types must define the `fullbasis` function which
returns the operator's underlying `SuperOperatorBasis`.

See [TODO: reference superoperators.md in docs]
```
"""
abstract type AbstractSuperOperator{B<:SuperOperatorBasis} end
