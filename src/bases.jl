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

# According to this I should use singleton type instances instead of types themselves
# https://discourse.julialang.org/t/singleton-types-vs-instances-as-type-parameters/2802/3
# I think I would've needed a `Type{}` to get equality working above with subtypes such as
# abstract type GenericBasis{T} <: Basis{T} end
# GenericBasis(T) = GenericBasis{T}
# Base.length(b::Type{GenericBasis{N}}) where {N} = N
# abstract type TensorBasis{T<:Tuple{Vararg{<:Basis}}} <: Basis{T} end
# TensorBasis(bases::Tuple) = TensorBasis{Tuple{bases...}}
# TensorBasis(bases::Vector) = TensorBasis{Tuple{bases...}}
# TensorBasis(bases...) = TensorBasis{Tuple{bases...}}
# bases(b::Type{TensorBasis{T}}) where {T} = fieldtypes(T)
# Base.length(b::Type{TensorBasis{T}}) where {T} = prod(length.(fieldtypes(T)))

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

"""
    GenericBasis(N)

A general purpose basis of dimension N.

Should only be used rarely since it defeats the purpose of checking that the
bases of state vectors and operators are correct for algebraic operations.
The preferred way is to specify special bases for different systems.
"""
struct GenericBasis{T} <: Basis{T}
    GenericBasis(T) = new{T}()
end
Base.length(b::GenericBasis{N}) where {N} = N

"""
    TensorBasis(b1, b2...)

Basis for composite Hilbert spaces.

Stores the subbases in a tuple. Instead of creating a TensorBasis
directly `tensor(b1, b2...)` or `b1 ⊗ b2 ⊗ …` can be used.
"""
struct TensorBasis{T} <: Basis{T}
    TensorBasis(T) = new{typeassert(T, Tuple{Vararg{<:Basis}})}()
end
TensorBasis(bases::Basis...) = TensorBasis((bases...,))
TensorBasis(bases::Vector) = TensorBasis((bases...,))
bases(b::TensorBasis{T}) where {T} = T
Base.length(b::TensorBasis{T}) where {T} = prod(length.(T))

"""
    tensor(x::Basis, y::Basis, z::Basis...)

Create a [`TensorBasis`](@ref) from the given bases.

Any given TensorBasis is expanded so that the resulting TensorBasis never
contains another TensorBasis.
"""
tensor(b::Basis) = TensorBasis(b)
tensor(b1::Basis, b2::Basis) = TensorBasis(b1, b2)
tensor(b1::TensorBasis, b2::TensorBasis) = TensorBasis(bases(b1)..., bases(b1)...)
tensor(b1::TensorBasis, b2::Basis) = TensorBasis(bases(b1)..., b2)
tensor(b1::Basis, b2::TensorBasis) = TensorBasis(b1, bases(b2)...)
tensor(bases::Basis...) = reduce(tensor, bases)

function Base.:^(b::Basis, N::Integer)
    if N < 1
        throw(ArgumentError("Power of a basis is only defined for positive integers."))
    end
    tensor((b for i=1:N)...)
end

"""
    reduced(a, indices)

Reduced basis, state or operator on the specified subsystems.

The `indices` argument, which can be a single integer or a vector of integers,
specifies which subsystems are kept. At least one index must be specified.
"""
function reduced(b::TensorBasis, indices)
    if length(indices)==0
        throw(ArgumentError("At least one subsystem must be specified in reduced."))
    elseif length(indices)==1
        return bases(b)[indices[1]]
    else
        return TensorBasis(bases(b)[indices])
    end
end

"""
    ptrace(a, indices)

Partial trace of the given basis, state or operator.

The `indices` argument, which can be a single integer or a vector of integers,
specifies which subsystems are traced out. The number of indices has to be
smaller than the number of subsystems, i.e. it is not allowed to perform a
full trace.
"""
function ptrace(b::TensorBasis, indices)
    J = [i for i in 1:length(bases(b)) if i ∉ indices]
    length(J) > 0 || throw(ArgumentError("Tracing over all indices is not allowed in ptrace."))
    reduced(b, J)
end


"""
    permutesystems(a, perm)

Change the ordering of the subsystems of the given object.

For a permutation vector `[2,1,3]` and a given object with basis `[b1, b2, b3]`
this function results in `[b2, b1, b3]`.
"""
function permutesystems(b::TensorBasis, perm)
    @assert length(bases(b)) == length(perm)
    @assert isperm(perm)
    TensorBasis(bases(b)[perm])
end


##
# Common bases
##

"""
    FockBasis(N,offset=0)

Basis for a Fock space where `N` specifies a cutoff, i.e. what the highest
included fock state is. Similarly, the `offset` defines the lowest included
fock state (default is 0). Note that the dimension of this basis is `N+1-offset`.
"""
struct FockBasis{T} <: Basis{T}
    function FockBasis(N, offset=0)
        if N < 0 || offset < 0 || N <= offset
            throw(DimensionMismatch())
        end
        new{(N,offset)}()
    end
end
cutoff(b::FockBasis{T}) where {T} = T[1]
offset(b::FockBasis{T}) where {T} = T[2]
Base.length(b::FockBasis{T}) where {T} = T[1] - T[2] + 1

"""
    NLevelBasis(N)

Basis for a system consisting of N states.
"""
struct NLevelBasis{T} <: Basis{T}
    function NLevelBasis(N::T) where {T}
        if N < 1
            throw(DimensionMismatch())
        end
        new{N}()
    end
end
Base.length(b::NLevelBasis{T}) where {T} = T

"""
    SpinBasis(n)

Basis for spin-n particles.

The basis can be created for arbitrary spinnumbers by using a rational number,
e.g. `SpinBasis(3//2)`. The Pauli operators are defined for all possible
spin numbers.
"""
struct SpinBasis{T} <: Basis{T}
    function SpinBasis(spinnumber::T) where {T<:Rational}
        n = numerator(spinnumber)
        d = denominator(spinnumber)
        if !(d==2 || d==1) || n < 0
            throw(DimensionMismatch())
        end
        new{spinnumber}()
    end
end
SpinBasis(spinnumber) = SpinBasis(convert(Rational{Int}, spinnumber))
spinnumber(b::SpinBasis{T}) where {T} = T
Base.length(b::SpinBasis{T}) where {T} = numerator(T*2 + 1)

"""
    PauliBasis(num_qubits)

Basis for an N-qubit space where `num_qubits` specifies the number of qubits.
The dimension of the basis is 2²ᴺ.
"""
struct PauliBasis{T} <: Basis{T}
    PauliBasis(num_qubits) = new{num_qubits}()
end
Base.length(b::PauliBasis{N}) where {N} = 4^N


"""
    SumBasis(b1, b2...)

Similar to [`TensorBasis`](@ref) but for the [`directsum`](@ref) (⊕)
"""
struct SumBasis{T} <: Basis{T}
    TensorBasis(T) = new{typeassert(T, Tuple{Vararg{<:Basis}})}()
end
SumBasis(bases::Basis...) = SumBasis((bases...,))
SumBasis(bases::Vector) = SumBasis((bases...,))
bases(b::SumBasis{T}) where {T} = T
Base.length(b::SumBasis{T}) where {T} = sum(length.(T))

"""
    directsum(b1::Basis, b2::Basis)

Construct the [`SumBasis`](@ref) out of two sub-bases.
"""
directsum(b::Basis) = TensorBasis(b)
directsum(b1::Basis, b2::Basis) = TensorBasis(b1, b2)
directsum(b1::SumBasis, b2::SumBasis) = TensorBasis(bases(b1)..., bases(b1)...)
directsum(b1::SumBasis, b2::Basis) = TensorBasis(bases(b1)..., b2)
directsum(b1::Basis, b2::SumBasis) = TensorBasis(b1, bases(b2)...)
directsum(bases::Basis...) = reduce(dicectsum, bases)

embed(b::SumBasis, indices, ops) = embed(b, b, indices, ops)

##
# show methods
##

function show(stream::IO, x::GenericBasis)
    if length(length(x)) == 1
        write(stream, "Basis(dim=$(length(x)[1]))")
    else
        s = replace(string(length(x)), " " => "")
        write(stream, "Basis(shape=$s)")
    end
end

function show(stream::IO, x::TensorBasis)
    write(stream, "[")
    for i in 1:length(bases(x))
        show(stream, bases(x)[i])
        if i != length(bases(x))
            write(stream, " ⊗ ")
        end
    end
    write(stream, "]")
end

function show(stream::IO, x::SpinBasis)
    d = denominator(spinnumber(x))
    n = numerator(spinnumber(x))
    if d == 1
        write(stream, "Spin($n)")
    else
        write(stream, "Spin($n/$d)")
    end
end

function show(stream::IO, x::FockBasis)
    if iszero(offset(x))
        write(stream, "Fock(cutoff=$(cutoff(x)))")
    else
        write(stream, "Fock(cutoff=$(cutoff(x)), offset=$(offset(x)))")
    end
end

function show(stream::IO, x::NLevelBasis)
    write(stream, "NLevel(N=$(length(x)))")
end

function show(stream::IO, x::SumBasis)
    write(stream, "[")
    for i in 1:length(bases(x))
        show(stream, bases(x)[i])
        if i != length(bases(x))
            write(stream, " ⊕ ")
        end
    end
    write(stream, "]")
end
