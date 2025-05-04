"""
Abstract type for all state vectors.

This type represents any abstract pure quantum state as given by an element of a
Hilbert space with respect to a certain basis. To be compatible with methods
defined in `QuantumInterface`, all subtypes must implement the `basis` method
which should return a subtype of the abstract [`Basis`](@ref) type.

See also [`AbstractKet`](@ref) and [`AbstractBra`](@ref).
"""
abstract type StateVector end

"""
Abstract type for `Ket` states.

This subtype of [`StateVector`](@ref) is meant to represent `Ket` states which
are related to their dual `Bra` by the conjugate transpose.

See also [`AbstractBra`](@ref).
"""
abstract type AbstractKet <: StateVector end

"""
Abstract type for `Bra` states.

This subtype of [`StateVector`](@ref) is meant to represent `Bra` states which
are related to their dual `Ket` by the conjugate transpose.

See also [`AbstractBra`](@ref).
"""
abstract type AbstractBra <: StateVector end

"""
Abstract type for all operators and super operators.

This type represents any abstract mixed quantum state given by a density
operator (or superoperator) mapping between two Hilbert spaces.  All subtypes
must implement the [`basis_l`](@ref) and [`basis_r`](@ref) methods which return
subtypes of [`Basis`](@ref) representing the left and right bases that the
operator maps between. A subtype is considered compatible with multiplication by
a subtype of [`AbstractBra`](@ref) defined in same left basis as the operator
and a subtype of [`AbstractKet`](@ref) defined in the same right basis as the
operator.

For fast time evolution also at least the function
`mul!(result::Ket,op::AbstractOperator,x::Ket,alpha,beta)` should be
implemented. Many other generic multiplication functions can be defined in terms
of this function and are provided automatically.
"""
abstract type AbstractOperator end

abstract type OperatorVector end
abstract type AbstractSuperKet <: OperatorVector end
abstract type AbstractSuperBra <: OperatorVector end

abstract type AbstractChannel end
abstract type AbstractSuperOperator <: AbstractChannel end
abstract type AbstractChoiState <: AbstractChannel end
abstract type AbstractKraus <: AbstractChannel end
abstract type AbstractStinespring <: AbstractChannel end

const QObject = Union{StateVector, OperatorVector, AbstractOperator, AbstractChannel}

abstract type Space end

"""
Abstract type for all specialized bases of a Hilbert space.

This type specifies an orthonormal basis for the Hilbert space of the given
system. All subtypes must implement `Base.:(==)` and `dimension`, where the
latter should return the total dimension of the Hilbert space.

Composite systems can be defined with help of [`CompositeBasis`](@ref).
Custom subtypes can also define composite systems by implementing
`Base.length` and `Base.getindex`.

All relevant properties of concrete subtypes of `Basis` defined in
`QuantumInterface` should be accessed using their documented functions and
should not assume anything about the internal representation of instances of
these types (i.e. do not access the fields of the structs directly).
"""
abstract type Basis <: Space end
abstract type OperatorBasis <: Space end

function summary(stream::IO, x::AbstractOperator)
    print(stream, "$(typeof(x).name.name)(dim=$(dimension(x.basis_l))x$(dimension(x.basis_r)))\n")
    if multiplicable(x,x)
        print(stream, "  basis: ")
        show(stream, basis(x))
    else
        print(stream, "  basis left:  ")
        show(stream, x.basis_l)
        print(stream, "\n  basis right: ")
        show(stream, x.basis_r)
    end
end

show(stream::IO, x::AbstractOperator) = summary(stream, x)

traceout!(s::StateVector, i) = ptrace(s,i)
