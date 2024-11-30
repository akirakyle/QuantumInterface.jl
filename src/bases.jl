
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
    GenericBasis(N)

A general purpose basis of dimension N.

Should only be used rarely since it defeats the purpose of checking that the
bases of state vectors and operators are correct for algebraic operations.
The preferred way is to specify special bases for different systems.
"""
struct GenericBasis{T} <: Basis
    GenericBasis(T) = new{T}()
end
Base.length(b::GenericBasis{N}) where {N} = N

"""
    TensorBasis(b1, b2...)

Basis for composite Hilbert spaces.

Stores the subbases in a tuple. Instead of creating a TensorBasis
directly `tensor(b1, b2...)` or `b1 ⊗ b2 ⊗ …` can be used.
"""
struct TensorBasis{T<:Tuple{Vararg{Basis}}} <: Basis
    bases
    TensorBasis(x) = new{typeof(x)}(x)
end
TensorBasis(bases::Basis...) = TensorBasis((bases...,))
TensorBasis(bases::Vector) = TensorBasis((bases...,))
Base.length(b::TensorBasis) = prod(length.(b.bases))

"""
    tensor(x::Basis, y::Basis, z::Basis...)

Create a [`TensorBasis`](@ref) from the given bases.

Any given TensorBasis is expanded so that the resulting TensorBasis never
contains another TensorBasis.
"""
tensor(b::Basis) = TensorBasis(b)
tensor(b1::Basis, b2::Basis) = TensorBasis(b1, b2)
tensor(b1::TensorBasis, b2::TensorBasis) = TensorBasis(b1.bases..., b1.bases...)
tensor(b1::TensorBasis, b2::Basis) = TensorBasis(b1.bases..., b2)
tensor(b1::Basis, b2::TensorBasis) = TensorBasis(b1, b2.bases...)
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
        return b.bases[indices[1]]
    else
        return TensorBasis(b.bases[indices])
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
    J = [i for i in 1:length(b.bases) if i ∉ indices]
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
    @assert length(b.bases) == length(perm)
    @assert isperm(perm)
    TensorBasis(b.bases[perm])
end

"""
    SumBasis(b1, b2...)

Similar to [`TensorBasis`](@ref) but for the [`directsum`](@ref) (⊕)
"""
struct SumBasis{T<:Tuple{Vararg{Basis}}} <: Basis
    bases
    SumBasis(x) = new{typeof(x)}(x)
end
SumBasis(bases::Basis...) = SumBasis((bases...,))
SumBasis(bases::Vector) = SumBasis((bases...,))
Base.length(b::SumBasis{T}) where {T} = sum(length.(b.bases))

"""
    directsum(b1::Basis, b2::Basis)

Construct the [`SumBasis`](@ref) out of two sub-bases.
"""
directsum(b::Basis) = TensorBasis(b)
directsum(b1::Basis, b2::Basis) = TensorBasis(b1, b2)
directsum(b1::SumBasis, b2::SumBasis) = TensorBasis(b1.bases..., b1.bases...)
directsum(b1::SumBasis, b2::Basis) = TensorBasis(b1.bases..., b2)
directsum(b1::Basis, b2::SumBasis) = TensorBasis(b1, b2.bases...)
directsum(bases::Basis...) = reduce(dicectsum, bases)

embed(b::SumBasis, indices, ops) = embed(b, b, indices, ops)


##
# Common finite bases
##

"""
    NLevelBasis(N)

Basis for a system consisting of N states.
"""
struct NLevelBasis{T} <: Basis
    function NLevelBasis(N::Integer)
        if N < 1
            throw(DimensionMismatch())
        end
        new{N}()
    end
end
Base.length(b::NLevelBasis{N}) where {N} = N

"""
    SpinBasis(n)

Basis for spin-n particles.

The basis can be created for arbitrary spinnumbers by using a rational number,
e.g. `SpinBasis(3//2)`. The Pauli operators are defined for all possible
spin numbers.
"""
struct SpinBasis{T} <: Basis
    function SpinBasis(spinnumber::Rational)
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
    SubspaceBasis(basisstates)

A basis describing a subspace embedded a higher dimensional Hilbert space.
"""
struct SubspaceBasis{B, H} <: Basis
    basisstates
    function SubspaceBasis(superbasis::Basis, basisstates::Vector{<:AbstractKet})
        for state = basisstates
            if basis(state) != superbasis
                throw(ArgumentError("The basis of the basisstates has to be the superbasis."))
            end
        end
        H = hash([hash(x) for x=basisstates])
        new{superbasis, H}(basisstates)
    end
end
SubspaceBasis(basisstates::Vector) = SubspaceBasis(basis(basisstates[1]), basisstates)
Base.length(b::SubspaceBasis{N}) where {N} = length(b.basisstates)

"""
    ManyBodyBasis(b, occupations)

Basis for a many body system.

The basis has to know the associated one-body basis `b` and which occupation states
should be included. The occupations_hash is used to speed up checking if two
many-body bases are equal.
"""
struct ManyBodyBasis{B,H} <: Basis
    occupations
    ManyBodyBasis(onebodybasis::Basis, occupations) =
        new{onebodybasis, hash(hash.(occupations))}(occupations)
end

"""
    ChargeBasis(ncut) <: Basis

Basis spanning `-ncut, ..., ncut` charge states, which are the fourier modes
(irreducible representations) of a continuous U(1) degree of freedom, truncated
at `ncut`.

The charge basis is a natural representation for circuit-QED elements such as
the "transmon", which has a hamiltonian of the form
```julia
b = ChargeBasis(ncut)
H = 4E_C * (n_g * identityoperator(b) + chargeop(b))^2 - E_J * cosφ(b)
```
with energies periodic in the charge offset `n_g`.
See e.g. https://arxiv.org/abs/2005.12667.
"""
struct ChargeBasis{T} <: Basis
    function ChargeBasis(ncut::Integer)
        if ncut < 0
            throw(DimensionMismatch())
        end
        new{ncut}()
    end
end
Base.length(b::ChargeBasis{ncut}) where {ncut} = 2*ncut + 1

"""
    ShiftedChargeBasis(nmin, nmax) <: Basis

Basis spanning `nmin, ..., nmax` charge states. See [`ChargeBasis`](@ref).
"""
struct ShiftedChargeBasis{T,N} <: Basis
    function ShiftedChargeBasis(nmin::T, nmax::T) where {T<:Integer}
        if nmax <= nmin
            throw(DimensionMismatch())
        end
        new{nmin,nmax}()
    end
end
Base.length(b::ShiftedChargeBasis{N}) where {N} = N[2] - N[1] + 1


##
# Common infinite bases along with finite cutoff versions
##

abstract type ParticleBasis <: Basis end
abstract type InfFockBasis <: ParticleBasis end
abstract type InfCoherentStateBasis <: ParticleBasis end
abstract type InfPositionBasis <: ParticleBasis end
abstract type InfMomentumBasis <: ParticleBasis end

"""
    FockBasis(N,offset=0)

Basis for a Fock space where `N` specifies a cutoff, i.e. what the highest
included fock state is. Similarly, the `offset` defines the lowest included
fock state (default is 0). Note that the dimension of this basis is `N+1-offset`.
"""
struct FockBasis{N,offset} <: InfFockBasis
    function FockBasis(N::T, offset::T=0) where {T<:Integer}
        if N < 0 || offset < 0 || N <= offset
            throw(DimensionMismatch())
        end
        new{N,offset}()
    end
end
cutoff(b::FockBasis{N,O}) where {N,O} = N
offset(b::FockBasis{N,O}) where {N,O} = O
Base.length(b::FockBasis{N,O}) where {N,O} = N - O + 1

"""
    PositionBasis(xmin, xmax, Npoints)
    PositionBasis(b::MomentumBasis)

Basis for a particle in real space.

For simplicity periodic boundaries are assumed which means that
the rightmost point defined by `xmax` is not included in the basis
but is defined to be the same as `xmin`.

When a [`MomentumBasis`](@ref) is given as argument the exact values
of ``x_{min}`` and ``x_{max}`` are due to the periodic boundary conditions
more or less arbitrary and are chosen to be
``-\\pi/dp`` and ``\\pi/dp`` with ``dp=(p_{max}-p_{min})/N``.
"""
struct PositionBasis{xmin,xmax,N} <: InfPositionBasis
    PositionBasis(xmin::F, xmax::F, N::T) where {F<:Real,T<:Integer} =
        new{xmin,xmax,N}()
end
function PositionBasis(xmin::F1, xmax::F2, N) where {F1,F2}
    F = promote_type(F1,F2)
    return PositionBasis(convert(F,xmin), convert(F,xmax), N)
end
Base.length(b::PositionBasis{xmin,xmax,N}) where {xmin,xmax,N} = N

"""
    MomentumBasis(pmin, pmax, Npoints)
    MomentumBasis(b::PositionBasis)

Basis for a particle in momentum space.

For simplicity periodic boundaries are assumed which means that
`pmax` is not included in the basis but is defined to be the same as `pmin`.

When a [`PositionBasis`](@ref) is given as argument the exact values
of ``p_{min}`` and ``p_{max}`` are due to the periodic boundary conditions
more or less arbitrary and are chosen to be
``-\\pi/dx`` and ``\\pi/dx`` with ``dx=(x_{max}-x_{min})/N``.
"""
struct MomentumBasis{pmin,pmax,N} <: InfMomentumBasis
    MomentumBasis(pmin::F, pmax::F, N::T) where {F<:Real, T<:Integer} =
        new{pmin,pmax,N}()
end
function MomentumBasis(pmin::F1, pmax::F2, N) where {F1,F2}
    F = promote_type(F1,F2)
    return MomentumBasis(convert(F,pmin), convert(F,pmax), N)
end
Base.length(b::MomentumBasis{pmin,pmax,N}) where {pmin,pmax,N} = N

# FIXME
PositionBasis(b::MomentumBasis) = (dp = (b.pmax - b.pmin)/b.N; PositionBasis(-pi/dp, pi/dp, b.N))
MomentumBasis(b::PositionBasis) = (dx = (b.xmax - b.xmin)/b.N; MomentumBasis(-pi/dx, pi/dx, b.N))


# TODO CoherentStateBasis?
# TODO GaussianBasis? as operator basis only???

##
# Common operator bases
##

"""
    KetBraBasis(BL,BR)

Typical "Ket-Bra" Basis.
TODO: write more...
"""
struct KetBraBasis{BL<:Basis, BR<:Basis} <: OperatorBasis
    left::BL
    right::BR
    KetBraBasis(bl::Basis, br::Basis) = new{typeof(bl),typeof(br)}(bl,br)
end
Base.length(b::KetBraBasis) = b.left*b.right


"""
    PauliBasis(num_qubits)

Basis for an N-qubit space where `num_qubits` specifies the number of qubits.
The dimension of the basis is 2²ᴺ.
"""
struct PauliBasis{T} <: OperatorBasis
    PauliBasis(num_qubits) = new{num_qubits}()
end
Base.length(b::PauliBasis{N}) where {N} = 4^N
