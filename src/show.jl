import Base: show, summary

function summary(stream::IO, x::AbstractOperator)
    b = basis(x)
    print(stream, "$(typeof(x).name.name)(dim=$(size(b)[1])x$(size(b)[2]))\n")
    if b isa KetBraBasis && bases(b).left == bases(b).right
        print(stream, "  basis: ")
        show(stream, bases(b).left)
    elseif b isa KetBraBasis
        print(stream, "  basis left:  ")
        show(stream, bases(b).left)
        print(stream, "\n  basis right: ")
        show(stream, bases(b).right)
    else
        print(stream, "  basis: ")
        show(stream, basis(b))
    end
end

show(stream::IO, x::AbstractOperator) = summary(stream, x)

function show(stream::IO, x::GenericBasis)
    if length(length(x)) == 1
        write(stream, "Basis(dim=$(length(x)[1]))")
    else
        s = replace(string(length(x)), " " => "")
        write(stream, "Basis(shape=$s)")
    end
end

function show(stream::IO, x::CompositeBasis)
    write(stream, "[")
    for i in 1:nsubsystems(x)
        show(stream, bases(x)[i])
        if i != nsubsystems(x)
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
