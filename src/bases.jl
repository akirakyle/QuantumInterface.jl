
# According to this I should use singleton type instances instead of types themselves
# https://discourse.julialang.org/t/singleton-types-vs-instances-as-type-parameters/2802/3
# I think I would've needed a `Type{}` to get equality working above with subtypes such as
# abstract type GenericBasis{T} <: Basis{T} end
# GenericBasis(T) = GenericBasis{T}
# Base.length(b::Type{GenericBasis{N}}) where {N} = N
# abstract type CompositeBasis{T<:Tuple{Vararg{<:Basis}}} <: Basis{T} end
# CompositeBasis(bases::Tuple) = CompositeBasis{Tuple{bases...}}
# CompositeBasis(bases::Vector) = CompositeBasis{Tuple{bases...}}
# CompositeBasis(bases...) = CompositeBasis{Tuple{bases...}}
# bases(b::Type{CompositeBasis{T}}) where {T} = fieldtypes(T)
# Base.length(b::Type{CompositeBasis{T}}) where {T} = prod(length.(fieldtypes(T)))

# TODO: create function interface to access all relevant fields of each basis so
# that downstream code only uses funcitos and not fields...
# see https://docs.julialang.org/en/v1/manual/style-guide/#Prefer-exported-methods-over-direct-field-access

"""
    GenericBasis(N)

A general purpose basis of dimension N.

Should only be used rarely since it defeats the purpose of checking that the
bases of state vectors and operators are correct for algebraic operations.
The preferred way is to specify special bases for different systems.
"""
struct GenericBasis{N} <: Basis{N}
    # Note no type checknig here so this can be abused to put anything that's isbits in
    GenericBasis(N) = new{N}()
end

"""
    CompositeBasis(b1, b2...)

Basis for composite Hilbert spaces.

Stores the subbases in a tuple. Instead of creating a CompositeBasis
directly `tensor(b1, b2...)` or `b1 ⊗ b2 ⊗ …` can be used.
"""
struct CompositeBasis{N, T<:Tuple{Vararg{Basis}}} <: Basis{N}
    bases
    CompositeBasis(x) = new{prod(length.(x)), typeof(x)}(x)
end
CompositeBasis(bases::Basis...) = CompositeBasis((bases...,))
CompositeBasis(bases::Vector) = CompositeBasis((bases...,))

"""
    tensor(x::Basis, y::Basis, z::Basis...)

Create a [`CompositeBasis`](@ref) from the given bases.

Any given CompositeBasis is expanded so that the resulting CompositeBasis never
contains another CompositeBasis.
"""
tensor(b::Basis) = CompositeBasis(b)
tensor(b1::Basis, b2::Basis) = CompositeBasis(b1, b2)
tensor(b1::CompositeBasis, b2::CompositeBasis) = CompositeBasis(b1.bases..., b1.bases...)
tensor(b1::CompositeBasis, b2::Basis) = CompositeBasis(b1.bases..., b2)
tensor(b1::Basis, b2::CompositeBasis) = CompositeBasis(b1, b2.bases...)
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
function reduced(b::CompositeBasis, indices)
    if length(indices)==0
        throw(ArgumentError("At least one subsystem must be specified in reduced."))
    elseif length(indices)==1
        return b.bases[indices[1]]
    else
        return CompositeBasis(b.bases[indices])
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
function ptrace(b::CompositeBasis, indices)
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
function permutesystems(b::CompositeBasis, perm)
    @assert length(b.bases) == length(perm)
    @assert isperm(perm)
    CompositeBasis(b.bases[perm])
end

"""
    SumBasis(b1, b2...)

Similar to [`CompositeBasis`](@ref) but for the [`directsum`](@ref) (⊕)
"""
struct SumBasis{N, T<:Tuple{Vararg{Basis}}} <: Basis{N}
    bases
    SumBasis(x) = new{sum(length.(x)), typeof(x)}(x)
end
SumBasis(bases::Basis...) = SumBasis((bases...,))
SumBasis(bases::Vector) = SumBasis((bases...,))

"""
    directsum(b1::Basis, b2::Basis)

Construct the [`SumBasis`](@ref) out of two sub-bases.
"""
directsum(b::Basis) = CompositeBasis(b)
directsum(b1::Basis, b2::Basis) = CompositeBasis(b1, b2)
directsum(b1::SumBasis, b2::SumBasis) = CompositeBasis(b1.bases..., b1.bases...)
directsum(b1::SumBasis, b2::Basis) = CompositeBasis(b1.bases..., b2)
directsum(b1::Basis, b2::SumBasis) = CompositeBasis(b1, b2.bases...)
directsum(bases::Basis...) = reduce(dicectsum, bases)

embed(b::SumBasis, indices, ops) = embed(b, b, indices, ops)


##
# Common finite bases
##

"""
    NLevelBasis(N)

Basis for a system consisting of N states.
"""
struct NLevelBasis{N} <: Basis{N}
    function NLevelBasis(N::Integer)
        if N < 1
            throw(DimensionMismatch())
        end
        new{N}()
    end
end

"""
    SpinBasis(n)

Basis for spin-n particles.

The basis can be created for arbitrary spinnumbers by using a rational number,
e.g. `SpinBasis(3//2)`. The Pauli operators are defined for all possible
spin numbers.
"""
struct SpinBasis{N,T} <: Basis{N}
    function SpinBasis(spinnumber::Rational)
        n = numerator(spinnumber)
        d = denominator(spinnumber)
        if !(d==2 || d==1) || n < 0
            throw(DimensionMismatch())
        end
        new{numerator(spinnumber*2 + 1), spinnumber}()
    end
end
SpinBasis(spinnumber) = SpinBasis(convert(Rational{Int}, spinnumber))
spinnumber(b::SpinBasis{N,T}) where {N,T} = T

"""
    SubspaceBasis(basisstates)

A basis describing a subspace embedded a higher dimensional Hilbert space.
"""
struct SubspaceBasis{N,B,H} <: Basis{N}
    basisstates
    function SubspaceBasis(superbasis::Basis, basisstates::Vector{<:AbstractKet})
        for state = basisstates
            if basis(state) != superbasis
                throw(ArgumentError("The basis of the basisstates has to be the superbasis."))
            end
        end
        H = hash([hash(x) for x=basisstates])
        new{length(basisstates), superbasis, H}(basisstates)
    end
end
SubspaceBasis(basisstates::Vector) = SubspaceBasis(basis(basisstates[1]), basisstates)

"""
    ManyBodyBasis(b, occupations)

Basis for a many body system.

The basis has to know the associated one-body basis `b` and which occupation states
should be included. The occupations_hash is used to speed up checking if two
many-body bases are equal.
"""
struct ManyBodyBasis{N,B,H} <: Basis{N}
    occupations
    ManyBodyBasis(onebodybasis::Basis, occupations) =
        new{length(occupations), onebodybasis, hash(hash.(occupations))}(occupations)
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
struct ChargeBasis{N,T} <: Basis{N}
    function ChargeBasis(ncut::Integer)
        if ncut < 0
            throw(DimensionMismatch())
        end
        new{n2*ncut + 1, cut}()
    end
end

"""
    ShiftedChargeBasis(nmin, nmax) <: Basis

Basis spanning `nmin, ..., nmax` charge states. See [`ChargeBasis`](@ref).
"""
struct ShiftedChargeBasis{N,T,S} <: Basis{N}
    function ShiftedChargeBasis(nmin::T, nmax::T) where {T<:Integer}
        if nmax <= nmin
            throw(DimensionMismatch())
        end
        new{nmax-nmin+1,nmin,nmax}()
    end
end


##
# Common infinite bases
##

"""
    FockBasis(N,offset=0)

Basis for a Fock space where `N` specifies a cutoff, i.e. what the highest
included fock state is. Similarly, the `offset` defines the lowest included
fock state (default is 0). Note that the dimension of this basis is `N+1-offset`.
"""
struct FockBasis{N,cutoff,offset} <: Basis{N}
    function FockBasis(cutoff::Number, offset::Number=0)
        if isinf(cutoff)
            return new{Inf,Inf,0}()
        end
        if cutoff < 0 || offset < 0 || cutoff <= offset
            throw(DimensionMismatch())
        end
        new{cutoff-offset+1, cutoff, offset}()
    end
end
cutoff(b::FockBasis{N,C,O}) where {N,C,O} = C
offset(b::FockBasis{N,C,O}) where {N,C,O} = O

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
struct PositionBasis{N,xmin,xmax} <: Basis{N}
    PositionBasis(N::Number, xmin::F, xmax::F) where {F<:Real} =
        isinf(N) ? new{-Inf,Inf,Inf}() : new{xmin,xmax,N}()
end
function PositionBasis(xmin::F1, xmax::F2, N) where {F1,F2}
    F = promote_type(F1,F2)
    return PositionBasis(N, convert(F,xmin), convert(F,xmax))
end

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
struct MomentumBasis{N,pmin,pmax} <: Basis{N}
    MomentumBasis(pmin::F, pmax::F, N::T) where {F<:Real, T<:Integer} =
        isinf(N) ? new{-Inf,Inf,Inf}() : new{pmin,pmax,N}()
end
function MomentumBasis(N, pmin::F1, pmax::F2) where {F1,F2}
    F = promote_type(F1,F2)
    return MomentumBasis(N, convert(F,pmin), convert(F,pmax))
end

# FIXME
PositionBasis(b::MomentumBasis) = (dp = (b.pmax - b.pmin)/b.N; PositionBasis(-pi/dp, pi/dp, b.N))
MomentumBasis(b::PositionBasis) = (dx = (b.xmax - b.xmin)/b.N; MomentumBasis(-pi/dx, pi/dx, b.N))

"""
    CoherentStateBasis(Npoints, min, max)

Basis for a particle in phase space with cutoff in x and p of min to max.
elements are ket{alpha}
"""
struct CoherentStateBasis{N,min,max} <: Basis{N}
    CoherentStateBasis(N::Number, min::F, max::F) where {F<:Real} =
        isinf(N) ? new{-Inf,Inf,Inf}() : new{min,max,N}()
end


##
# Common operator bases
##

"""
    CompositeOperatorBasis(BL,BR)

Basis for composite Hilbert spaces.

Stores the subbases in a tuple. Instead of creating a CompositeBasis
directly `tensor(b1, b2...)` or `b1 ⊗ b2 ⊗ …` can be used.
"""
struct CompositeOperatorBasis{N,T<:Tuple{Vararg{OperatorBasis}}} <: OperatorBasis{N}
    bases
    function CompositeBasis(x)
        N,M = reduce(((N1,M1), (N2,M2)) -> (N1*N2, M2*M2), x; init=(1,1))
        new{(N,M),typeof(x)}(x)
    end
end
CompositeOperatorBasis(bases::OperatorBasis...) = CompositeOperatorBasis((bases...,))
CompositeOperatorBasis(bases::Vector) = CompositeOperatorBasis((bases...,))

"""
    tensor(x::Basis, y::Basis, z::Basis...)

Create a [`CompositeBasis`](@ref) from the given bases.

Any given CompositeBasis is expanded so that the resulting CompositeBasis never
contains another CompositeBasis.
"""
tensor(b::OperatorBasis) = CompositeOperatorBasis(b)
tensor(b1::OperatorBasis, b2::OperatorBasis) = CompositeOperatorBasis(b1, b2)
tensor(b1::CompositeOperatorBasis, b2::CompositeOperatorBasis) = CompositeOperatorBasis(b1.bases..., b1.bases...)
tensor(b1::CompositeOperatorBasis, b2::OperatorBasis) = CompositeOperatorBasis(b1.bases..., b2)
tensor(b1::OperatorBasis, b2::CompositeOperatorBasis) = CompositeOperatorBasis(b1, b2.bases...)
tensor(bases::OperatorBasis...) = reduce(tensor, bases)

"""
    KetBraBasis(BL,BR)

Typical "Ket-Bra" Basis.
TODO: write more...
"""
struct KetBraBasis{N,BL<:Basis,BR<:Basis} <: OperatorBasis{N}
    left::BL
    right::BR
    KetBraBasis(bl, br) = new{(length(bl), length(br)), typeof(bl), typeof(br)}(bl, br)
end

tensor(b::KetBraBasis) = b # TODO is this right?
tensor(b1::KetBraBasis, b2::KetBraBasis) = KetBraBasis(b1.left⊗b2.left, b1.right⊗b2.right)
tensor(bases::KetBraBasis...) = reduce(tensor, bases)

"""
    HeisenbergWeylBasis(modes, dim)

Unitary operator basis for a tensor product of modes number of dim_i-dimensional operator space in
the clock-shift matrices.
"""
struct HeisenbergWeylBasis{N,dims} <: UnitaryOperatorBasis{N}
    dims
    HeisenbergWeylBasis(dims::Tuple{Integer}) =
        new{(prod(dims),prod(dims)),dims}(dims)
    # TODO: add ordering? i.e. symplectic form
end
HeisenbergWeylBasis(modes::Integer, dim::Integer) = HeisenbergWeylBasis(ntuple(i->dim, modes))

tensor(b::HeisenbergWeylBasis) = b # TODO is this right?
tensor(b1::HeisenbergWeylBasis, b2::HeisenbergWeylBasis) = HeisenbergWeylBasis(b1.dims..., b2.dims...)
tensor(bases::HeisenbergWeylBasis...) = reduce(tensor, bases)

"""
    PauliBasis(num_qubits)

Basis for an N-qubit space where `num_qubits` specifies the number of qubits.
The dimension of the basis is 2ᴺ times 2ᴺ.
"""
struct PauliBasis{N,modes} <: UnitaryOperatorBasis{N}
    modes
    PauliBasis(modes::Integer) = new{(2^modes,2^modes),modes}(modse)
    # TODO: add ordering? i.e. symplectic form
end

tensor(b::PauliBasis) = b # TODO is this right?
tensor(b1::PauliBasis, b2::PauliBasis) = PauliBasis(b1.modes+b2.modes)
tensor(bases::PauliBasis...) = reduce(tensor, bases)

"""
    GaussianBasis(modes)

Operator basis in terms of Gaussians in phase space. 
cutoffs are number of Gaussian basis elements alllowed to be superposed...
So normal Gaussian state formalism correspnods to one in each mode.
"""
struct GaussianBasis{N,cutoffs} <: OperatorBasis{N}
    cutoffs
    GaussianBasis(cutoffs::Tuple{Integer}) =
        new{(prod(cutoffs),prod(cutoffs)),cutoffs}(cutoffs)
    # TODO: add ordering? i.e. symplectic form
end
GaussianBasis(modes::Integer, cutoff::Integer) = GaussianBasis(ntuple(i->cutoff, modes))

tensor(b::GaussianBasis) = b # TODO is this right?
tensor(b1::GaussianBasis, b2::GaussianBasis) = GaussianBasis(b1.dims..., b2.dims...)
tensor(bases::GaussianBasis...) = reduce(tensor, bases)


##
# Common super-operator bases
##


"""
    KetKetBraBraBasis(BL,BR)

Typical "KetKet-BraBra" SuperOperatorBasis.
TODO: write more...
"""
struct KetKetBraBraBasis{N,BL<:KetBraBasis, BR<:KetBraBasis} <: SuperOperatorBasis{N}
    left::BL
    right::BR
    KetBraBasis(bl, br) = new{(length(bl), length(br)), typeof(bl), typeof(br)}(bl, br)
end

tensor(b::KetKetBraBraBasis) = b # TODO is this right?
tensor(b1::KetKetBraBraBasis, b2::KetKetBraBraBasis) = KetKetBraBraBasis(b1.left⊗b2.left, b1.right⊗b2.right)
tensor(bases::KetKetBraBraBasis...) = reduce(tensor, bases)
