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

length() represents the underlying hilbert space, charicterized by it's dimensionality
Subtypes can add more type parameters as needed...
It really should represent an orthornormal basis so that length() is
both the number of basis elements and size of Hilbert space

== shoud be equality...

subtypes all have function interface to access all relevant fields of each basis so
downstream should only use that interface and not assume internal layout.
see https://docs.julialang.org/en/v1/manual/style-guide/#Prefer-exported-methods-over-direct-field-access

Composite systems can be defined with help of the [`CompositeBasis`](@ref) class.
"""
abstract type Basis end

"""
Parametric composite type for all operator bases.
Orthonormal Operator basis is one where inner product is given
by Hilbert-Schmidt or trace inner prouduct product.
TODO: write this condition down explicitly.
Examples include pauli or Heisenberg-Weyl bases as well as
standard "ket-bra" basis

Must implement Base.size and which return a
two-tuple describing left and right hilbert space dimensions

== shoud be equality...

See [TODO: reference operators.md in docs]
"""
abstract type OperatorBasis end
abstract type UnitaryOperatorBasis <: OperatorBasis end

"""
Parametric composite type for all superoperator bases.

Must implement Base.size returning a two-tuple of
two-tuples describing left and right operator dimensions

== shoud be equality...

See [TODO: reference superoperators.md in docs]
"""
abstract type SuperOperatorBasis end

"""
Exception that should be raised for an illegal algebraic operation.
"""
mutable struct IncompatibleBases <: Exception end

"""
Abstract type for `Bra` and `Ket` states.

The state vector type stores an abstract state with respect to a certain
Hilbert space basis.
All deriving types must define the `basis` function which
returns the state vector's underlying `Basis`.

Must implement hash for subspace basis?
"""
abstract type StateVector{B<:Basis} end
abstract type AbstractKet{B} <: StateVector{B} end
abstract type AbstractBra{B} <: StateVector{B} end

"""
Abstract type for all operators which represent linear maps between two
Hilbert spaces with respect to a given basis in each space.

All deriving operator types must define the `basis` function which
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

All deriving operator types must define the `basis` function which
returns the operator's underlying `SuperOperatorBasis`.

See [TODO: reference superoperators.md in docs]
```
"""
abstract type AbstractSuperOperator{B<:SuperOperatorBasis} end
