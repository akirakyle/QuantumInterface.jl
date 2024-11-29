"""
    Basis{T}

Abstract type for all specialized bases.
The `Basis` type is meant to specify a basis of the Hilbert space of the
studied system. The type parameter must encode all relevant information to
correctly distinguish equivlent/compatible bases as it is used in method dispatch.
For example with an OperatorBasis, the left and right bases are compatible if
and only if an instance is a subtype of OperatorBasis{B,B} where {B <: Basis}.

Furthermore all subtypes must implement `Base.length` which returns the dimension
of the Hilbert space.

Composite systems can be defined with help of the [`TensorBasis`](@ref) class.
"""
abstract type Basis{T} end
Base.:(==)(b1::Basis{T}, b2::Basis{T}) where {T} = true
Base.:(==)(b1::Basis, b2::Basis) = false

"""
Parametric composite type for all operator bases.

See [TODO: reference operators.md in docs]
"""
#abstract type OperatorBasis{BL<:Basis,BR<:Basis} end
struct OperatorBasis{BL<:Basis,BR<:Basis} end

"""
Parametric composite type for all superoperator bases.

See [TODO: reference superoperators.md in docs]
"""
#abstract type SuperOperatorBasis{BL<:OperatorBasis,BR<:OperatorBasis} end
struct SuperOperatorBasis{BL<:OperatorBasis,BR<:OperatorBasis} end

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
