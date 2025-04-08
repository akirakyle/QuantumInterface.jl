import Base: show, summary

function summary(stream::IO, x::AbstractOperator)
    print(stream, "$(typeof(x).name.name)(dim=$(length(x.basis_l))x$(length(x.basis_r)))\n")
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
