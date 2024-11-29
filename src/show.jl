import Base: show, summary

function summary(stream::IO, x::AbstractOperator)
    b = fullbasis(x)
    print(stream, "$(typeof(x).name.name)(dim=$(length(b.left))x$(length(b.right)))\n")
    if samebases(b)
        print(stream, "  basis: ")
        show(stream, basis(b))
    else
        print(stream, "  basis left:  ")
        show(stream, b.left)
        print(stream, "\n  basis right: ")
        show(stream, b.right)
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
