function equal_bases(a, b)
    Base.depwarn("`==` should be preferred over `equal_bases`!", :equal_bases)
    if a===b
        return true
    end
    for i=1:length(a)
        if a[i]!=b[i]
            return false
        end
    end
    return true
end

Base.@deprecate PauliBasis(num_qubits) NQubitBasis(num_qubits) false

function samebases(b1::Basis, b2::Basis)
    Base.depwarn("`==` should be preferred over `samebases(b1::Basis, b2::Basis)`!", :samebases)
    b1==b2
end

function samebases(b1::Tuple{Basis, Basis}, b2::Tuple{Basis, Basis})
    Base.depwarn("`==` should be preferred over `samebases(b1::Tuple{Basis, Basis}, b2::Tuple{Basis, Basis})`!", :samebases)
    b1==b2 # for checking superoperators
end

function samebases(a::AbstractOperator)
    Base.depwarn("`issquare` should be preferred over `samebases(a::AbstractOperator)`!", :check_samebases)
    samebases(a.basis_l, a.basis_r)::Bool # FIXME issue #12
end

function samebases(a::AbstractOperator, b::AbstractOperator)
    Base.depwarn("`addible` should be preferred over `samebases(a::AbstractOperator, b::AbstractOperator)`!", :check_samebases)
    samebases(a.basis_l, b.basis_l)::Bool && samebases(a.basis_r, b.basis_r)::Bool # FIXME issue #12
end

function check_samebases(b1, b2)
    Base.depwarn("Depending on context, `check__multiplicable`, `check_addible`, or `check_issquare` should be preferred over `check_samebases`!", :check_samebases)
    if BASES_CHECK[] && !samebases(b1, b2)
        throw(IncompatibleBases())
    end
end

function check_samebases(a::Union{AbstractOperator, AbstractSuperOperator})
    Base.depwarn("`check_issquare` should be preferred over `check_samebases(a::Union{AbstractOperator, AbstractSuperOperator})`!", :check_samebases)
    check_samebases(a.basis_l, a.basis_r) # FIXME issue #12
end

function multiplicable(b1::Basis, b2::Basis)
    Base.depwarn("`==` should be preferred over `multiplicable(b1::Basis, b2::Basis)`!", :multiplicable)
    b1==b2
end

function multiplicable(b1::CompositeBasis, b2::CompositeBasis)
    Base.depwarn("`==` should be preferred over `multiplicable(b1::CompositeBasis, b2::CompositeBasis)`!", :multiplicable)
    if !equal_shape(b1.shape,b2.shape)
        return false
    end
    for i=1:length(b1.shape)
        if !multiplicable(b1.bases[i], b2.bases[i])
            return false
        end
    end
    return true
end

Base.@deprecate @samebases(ex) @compatiblebases(ex) false
