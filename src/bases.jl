"""
    basis(a)

Return the basis of a quantum object.

If it's ambiguous, e.g. if an operator has a different left and right basis, an
[`IncompatibleBases`](@ref) error is thrown.

See [`StateVector`](@ref) and [`AbstractOperator`](@ref)
"""
function basis end

"""
    basis_l(a)

Return the left basis of an operator.
"""
function basis_l end

"""
    basis_r(a)

Return the right basis of an operator.
"""
function basis_r end

"""
Exception that should be raised for an illegal algebraic operation.
"""
mutable struct IncompatibleBases <: Exception end

const BASES_CHECK = Ref(true)

"""
    @compatiblebases

Macro to skip checks for compatible bases. Useful for `*`, `expect` and similar
functions.
"""
macro compatiblebases(ex)
    return quote
        BASES_CHECK[] = false
        local val = $(esc(ex))
        BASES_CHECK[] = true
        val
    end
end

"""
    samebases(b1::Basis, b2::Basis)

Test if two bases are the same. Equivalant to `==`. See
[`check_samebases`](@ref).
"""
samebases(b1::Basis, b2::Basis) = b1==b2

"""
    check_samebases(a, b)

Throw an [`IncompatibleBases`](@ref) error if the bases are not the same. See
[`samebases`](@ref).
"""
function check_samebases(b1, b2)
    if BASES_CHECK[] && !samebases(b1, b2)
        throw(IncompatibleBases())
    end
end

"""
    check_addible(a, b)

Throw an [`IncompatibleBases`](@ref) error if the objects are not addible as
determined by `addible(a, b)`.  Disabled by use of [`@compatiblebases`](@ref)
anywhere further up in the call stack.
"""
function check_addible(a, b)
    if BASES_CHECK[] && !addible(a, b)
        throw(IncompatibleBases())
    end
end

"""
    check_multiplicable(a, b)

Throw an [`IncompatibleBases`](@ref) error if the objects are not multiplicable
as determined by `multiplicable(a, b)`.  Disabled by use of
[`@compatiblebases`](@ref) anywhere further up in the call stack.
"""
function check_multiplicable(a, b)
    if BASES_CHECK[] && !multiplicable(a, b)
        throw(IncompatibleBases())
    end
end
