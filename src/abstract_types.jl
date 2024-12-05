"""
Abstract type for all specialized bases of a Hilbert space.

The `Basis` type specifies an orthonormal basis for the Hilbert
space of the studied system. All subtypes must implement `Base.:(==)` along with
`Base.length`, which should return the total dimension of the Hilbert space.

Composite systems can be defined with help of [`CompositeBasis`](@ref).

All relevant properties of subtypes of `Basis` defined in `QuantumInterface`
should be accessed using their documented functions and should not
assume anything about the internal representation of instances of these  
types (i.e. don't access the struct's fields directly).
"""
abstract type Basis end

"""
Abstract base class for `Bra` and `Ket` states.

The state vector class stores the coefficients of an abstract state
in respect to a certain basis. These coefficients are stored in the
`data` field and the basis is defined in the `basis`
field.
"""
abstract type StateVector end
abstract type AbstractKet <: StateVector end
abstract type AbstractBra <: StateVector end

"""
Abstract base class for all operators and super operators.

All deriving operator classes have to define the fields
`basis_l` and `basis_r` defining the left and right side bases.

For fast time evolution also at least the function
`mul!(result::Ket,op::AbstractOperator,x::Ket,alpha,beta)` should be
implemented. Many other generic multiplication functions can be defined in
terms of this function and are provided automatically.
"""
abstract type AbstractOperator end

const AbstractQObjType = Union{<:StateVector,<:AbstractOperator}
