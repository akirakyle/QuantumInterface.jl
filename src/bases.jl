##
# Generic, composite, sum bases
##

"""
    GenericBasis(N)

A general purpose basis of dimension N.

Should only be used rarely since it defeats the purpose of checking that the
bases of state vectors and operators are correct for algebraic operations.
The preferred way is to specify special bases for different systems.
"""
struct GenericBasis{N} <: Basis
    # Note no type checknig here so this can be abused to put anything that's isbits in
    GenericBasis(N) = new{N}()
end

"""
    CompositeBasis(b1, b2...)

Basis for composite Hilbert spaces.

Stores the subbases in a tuple. Instead of creating a CompositeBasis
directly `tensor(b1, b2...)` or `b1 ⊗ b2 ⊗ …` can be used.
"""
struct CompositeBasis{T<:Tuple{Vararg{Basis}}} <: Basis
    bases
    CompositeBasis(x) = new{typeof(x)}(x)
end
CompositeBasis(bases::Basis...) = CompositeBasis((bases...,))
CompositeBasis(bases::Vector) = CompositeBasis((bases...,))
bases(b::CompositeBasis) = b.bases

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
        return bases(b)[indices[1]]
    else
        return CompositeBasis(bases(b)[indices])
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
function permutesystems(b::CompositeBasis, perm)
    @assert length(bases(b)) == length(perm)
    @assert isperm(perm)
    CompositeBasis(bases(b)[perm])
end

"""
    SumBasis(b1, b2...)

Similar to [`CompositeBasis`](@ref) but for the [`directsum`](@ref) (⊕)
"""
struct SumBasis{T<:Tuple{Vararg{Basis}}} <: Basis
    bases
    SumBasis(x) = new{typeof(x)}(x)
end
SumBasis(bases::Basis...) = SumBasis((bases...,))
SumBasis(bases::Vector) = SumBasis((bases...,))
bases(b::SumBasis) = b.bases

##
# Common finite bases
##

"""
    NLevelBasis(N)

Basis for a system consisting of N states.
"""
struct NLevelBasis{N} <: Basis
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
struct SpinBasis{spin} <: Basis
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
spinnumber(b::SpinBasis{spin}) where {spin} = spin

"""
    SubspaceBasis(basisstates)

A basis describing a subspace embedded a higher dimensional Hilbert space.
"""
struct SubspaceBasis{super,hash} <: Basis
    superbasis
    basisstates
    function SubspaceBasis(superbasis::Basis, basisstates::Vector{<:AbstractKet})
        for state = basisstates
            if basis(state) != superbasis
                throw(ArgumentError("The basis of the basisstates has to be the superbasis."))
            end
        end
        H = hash([hash(x) for x=basisstates])
        new{superbasis, H}(superbasis, basisstates)
    end
end
SubspaceBasis(basisstates::Vector) = SubspaceBasis(basis(basisstates[1]), basisstates)
associatedbasis(b::SubspaceBasis) = b.superbasis
basisstates(b::SubspaceBasis) = b.basisstates

"""
    ManyBodyBasis(b, occupations)

Basis for a many body system.

The basis has to know the associated one-body basis `b` and which occupation states
should be included. The occupations_hash is used to speed up checking if two
many-body bases are equal.
"""
struct ManyBodyBasis{obbasis,occ} <: Basis
    onebodybasis
    occupations
    ManyBodyBasis(onebodybasis::Basis, occupations) =
        new{onebodybasis, hash(hash.(occupations))}(onebodybasis, occupations)
end
associatedbasis(b::ManyBodyBasis) = b.onebodybasis
basisstates(b::ManyBodyBasis) = b.occupations

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
struct ChargeBasis{ncut} <: Basis
    function ChargeBasis(ncut::Integer)
        if ncut < 0
            throw(DimensionMismatch())
        end
        new{ncut}()
    end
end
cutoff(b::ChargeBasis{ncut}) where {ncut} = cut

"""
    ShiftedChargeBasis(nmin, nmax) <: Basis

Basis spanning `nmin, ..., nmax` charge states. See [`ChargeBasis`](@ref).
"""
struct ShiftedChargeBasis{nmin,nmax} <: Basis
    function ShiftedChargeBasis(nmin::T, nmax::T) where {T<:Integer}
        if nmax <= nmin
            throw(DimensionMismatch())
        end
        new{nmin,nmax}()
    end
end
cutoff_min(b::ShiftedChargeBasis{nmin,nmax}) where {nmin,nmax} = nmin
cutoff_max(b::ShiftedChargeBasis{nmin,nmax}) where {nmin,nmax} = nmax


##
# Common infinite bases
##

"""
    FockBasis(N,offset=0)

Basis for a Fock space where `N` specifies a cutoff, i.e. what the highest
included fock state is. Similarly, the `offset` defines the lowest included
fock state (default is 0). Note that the dimension of this basis is `N+1-offset`.
"""
struct FockBasis{cut,off} <: Basis
    function FockBasis(cutoff::Number, offset::Number=0)
        if isinf(cutoff)
            return new{Inf,0}()
        end
        if cutoff < 0 || offset < 0 || cutoff <= offset
            throw(DimensionMismatch())
        end
        new{cutoff, offset}()
    end
end
cutoff(b::FockBasis{cut,off}) where {cut,off} = cut
offset(b::FockBasis{cut,off}) where {cut,off} = off

"""
    PositionBasis(Npoints, xmin, xmax)
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
struct PositionBasis{N,xmin,xmax} <: Basis
    PositionBasis(N::Number, xmin::F, xmax::F) where {F<:Real} =
        isinf(N) ? new{Inf,-Inf,Inf}() : new{N,xmin,xmax}()
end
function PositionBasis(N, xmin::F1, xmax::F2) where {F1,F2}
    F = promote_type(F1,F2)
    return PositionBasis(N, convert(F,xmin), convert(F,xmax))
end
cutoff_min(b::PositionBasis{N,xmin,xmax}) where {N,xmin,xmax} = xmin
cutoff_max(b::PositionBasis{N,xmin,xmax}) where {N,xmin,xmax} = xmax

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
struct MomentumBasis{N,pmin,pmax} <: Basis
    MomentumBasis(N::T, pmin::F, pmax::F) where {F<:Real, T<:Integer} =
        isinf(N) ? new{Inf,-Inf,Inf}() : new{N,pmin,pmax}()
end
function MomentumBasis(N, pmin::F1, pmax::F2) where {F1,F2}
    F = promote_type(F1,F2)
    return MomentumBasis(N, convert(F,pmin), convert(F,pmax))
end
cutoff_min(b::MomentumBasis{N,pmin,pmax}) where {N,pmin,pmax} = pmin
cutoff_max(b::MomentumBasis{N,pmin,pmax}) where {N,pmin,pmax} = pmax

# FIXME
PositionBasis(b::MomentumBasis) = (dp = (b.pmax - b.pmin)/b.N; PositionBasis(-pi/dp, pi/dp, b.N))
MomentumBasis(b::PositionBasis) = (dx = (b.xmax - b.xmin)/b.N; MomentumBasis(-pi/dx, pi/dx, b.N))

"""
    CoherentStateBasis(Npoints, min, max)

Basis for a particle in phase space with cutoff in x and p of min to max.
elements are ket{alpha}
"""
struct CoherentStateBasis{N,min,max} <: Basis
    CoherentStateBasis(N::Number, min::F, max::F) where {F<:Real} =
        isinf(N) ? new{Inf,-Inf,Inf}() : new{N,min,max}()
end
cutoff_min(b::CoherentStateBasis{N,min,max}) where {N,min,max} = min
cutoff_max(b::CoherentStateBasis{N,min,max}) where {N,min,max} = max


##
# Common operator bases
##

"""
    CompositeOperatorBasis(BL,BR)

Basis for composite Hilbert spaces.

Stores the subbases in a tuple. Instead of creating a CompositeBasis
directly `tensor(b1, b2...)` or `b1 ⊗ b2 ⊗ …` can be used.
"""
struct CompositeOperatorBasis{T<:Tuple{Vararg{OperatorBasis}}} <: OperatorBasis
    bases
    CompositeBasis(x) = new{typeof(x)}(x)
end
CompositeOperatorBasis(bases::OperatorBasis...) = CompositeOperatorBasis((bases...,))
CompositeOperatorBasis(bases::Vector) = CompositeOperatorBasis((bases...,))
bases(b::CompositeOperatorBasis) = b.bases

"""
    KetBraBasis(BL,BR)

Typical "Ket-Bra" Basis.
nsubsystems(BL) must equal nsubsystems(BR) otherwise doesn't make sense...
TODO: write more... 
"""
struct KetBraBasis{BL<:Basis,BR<:Basis} <: OperatorBasis
    left::BL
    right::BR
    function KetBraBasis(bl, br)
        if nsubsystems(bl) != nsubsystems(br)
            throw(IncompatibleBases())
        end
        new{typeof(bl), typeof(br)}(bl, br)
    end
end
left(b::KetBraBasis) = b.left
right(b::KetBraBasis) = b.right

"""
    HeisenbergWeylBasis(modes, dim)

Unitary operator basis for a tensor product of modes number of dim_i-dimensional operator space in
the clock-shift matrices.
"""
struct HeisenbergWeylBasis{dims} <: UnitaryOperatorBasis
    HeisenbergWeylBasis(dims::Tuple{Integer}) = new{dims}()
    # TODO: add ordering? i.e. symplectic form
end
HeisenbergWeylBasis(modes::Integer, dim::Integer) = HeisenbergWeylBasis(ntuple(i->dim, modes))
dimensions(b::HeisenbergWeylBasis{dims}) where {dims} = dims

"""
    PauliBasis(num_qubits)

Basis for an N-qubit space where `num_qubits` specifies the number of qubits.
The dimension of the basis is 2ᴺ times 2ᴺ.
"""
struct PauliBasis{modes} <: UnitaryOperatorBasis
    PauliBasis(modes::Integer) = new{modes}()
    # TODO: add ordering? i.e. symplectic form
end
# modes accessed with nsubsystems

"""
    GaussianBasis(modes)

Operator basis in terms of Gaussians in phase space. 
cutoffs are number of Gaussian basis elements alllowed to be superposed...
So normal Gaussian state formalism correspnods to one in each mode.
"""
struct GaussianBasis{cutoffs} <: OperatorBasis
    GaussianBasis(cutoffs::Tuple{Integer}) = new{cutoffs}()
    # TODO: add ordering? i.e. symplectic form
end
GaussianBasis(modes::Integer, cutoff::Integer) = GaussianBasis(ntuple(i->cutoff, modes))
cutoffs(b::GaussianBasis{cuts}) where {cuts} = cuts

##
# Common super-operator bases
##

"""
    KetKetBraBraBasis(BL,BR)

Typical "KetKet-BraBra" SuperOperatorBasis.
TODO: write more...
"""
struct KetKetBraBraBasis{BL<:KetBraBasis, BR<:KetBraBasis} <: SuperOperatorBasis
    left::BL
    right::BR
    KetBraBasis(bl, br) = new{typeof(bl), typeof(br)}(bl, br)
end
left(b::KetKetBraBraBasis) = b.left
right(b::KetKetBraBraBasis) = b.right

