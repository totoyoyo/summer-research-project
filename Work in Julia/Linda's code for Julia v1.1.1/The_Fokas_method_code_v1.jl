
# Import Packages
using SymPy
using PyCall
sympy = pyimport("sympy")
using Distributions
using ApproxFun
using Plots
using NBInclude
using QuadGK
import QuadGK.quadgk
using Roots
using Gadfly
using PyPlot
using StrLiterals
import LinearAlgebra
using TickTock

# Declare Global Variables
TOL = 1e-05

DIGITS = 3

INFTY = 10

# Declare Helper Functions
function func(object)
    return object.__class__
end

function check_all(array, condition)
    for x in array
        if !condition(x)
            return false
        end
    end
    return true
end

function check_any(array, condition)
    for x in array
        if condition(x)
            return true
        end
    end
    return false
end

function set_tol(x::Union{Number, Array}, y::Union{Number, Array}; atol = TOL)
    if isa(x, Number) && isa(y, Number)
       return atol * mean([x y])
    elseif isa(x, Array) && isa(y, Array)
        if size(x) != size(y)
            throw(error("Array dimensions do not match"))
        end
        # Avoid InexactError() when taking norm()
        x = convert(Array{Complex}, x)
        y = convert(Array{Complex}, y)
        return atol * (LinearAlgebra.norm(x) + LinearAlgebra.norm(y))
    else
        throw(error("Invalid input"))
    end
end

function evaluate(func::Union{Function, SymPy.Sym, Number}, a::Number)
    if isa(func, Function)
        funcA = func(a)
    elseif isa(func, SymPy.Sym) # SymPy.Sym must come before Number because SymPy.Sym will be recognized as Number
        freeSymbols = free_symbols(func)
        if length(freeSymbols) > 1
            throw(error("func should be univariate"))
        elseif length(freeSymbols) == 1
            t = free_symbols(func)[1,1]
            if isa(a, SymPy.Sym) # if x is SymPy.Sym, do not convert result to Number to preserve pretty printing
                funcA = subs(func, t, a)
            else
                funcA = SymPy.N(subs(func, t, a))
            end
        else
            funcA = func
        end
    else # func is Number
        funcA = func
    end
    return funcA
end

function partition(n::Int)
    if n < 0
        throw(error("Non-negative n required"))
    end
    output = []
    for i = 0:n
        j = n - i
        push!(output, (i,j))
    end
    return output
end

function get_deriv(u::Union{SymPy.Sym, Number}, k::Int)
    if k < 0
        throw(error("Non-negative k required"))
    end
    if isa(u, SymPy.Sym)
        freeSymbols = free_symbols(u)
        if length(freeSymbols) > 1
            throw(error("u should be univariate"))
        elseif length(freeSymbols) == 1
            t = freeSymbols[1]
            y = u
            for i = 1:k
                newY = diff(y, t) # diff is from SymPy
                y = newY
            end
            return y
        else
            if k > 0
                return 0
            else
                return u
            end
        end
    else
        if k > 0
            return 0
        else
            return u
        end
    end
end

function add_func(f::Union{Number, Function}, g::Union{Number, Function})
    function h(x)
        if isa(f, Number)
            if isa(g, Number)
                return f + g
            else
                return f + g(x)
            end
        elseif isa(f, Function)
            if isa(g, Number)
                return f(x) + g
            else
                return f(x) + g(x)
            end
        end
    end
    return h
end

function mult_func(f::Union{Number, Function}, g::Union{Number, Function})
    function h(x)
        if isa(f, Number)
            if isa(g, Number)
                return f * g
            else
                return f * g(x)
            end
        elseif isa(f, Function)
            if isa(g, Number)
                return f(x) * g
            else
                return f(x) * g(x)
            end
        end
    end
    return h
end

function sym_to_func(sym::Union{SymPy.Sym, Number})
    try
        freeSymbols = free_symbols(sym)
        if length(freeSymbols) > 1
            throw(error("sym should be univariate"))
        else
            function func(x)
                if length(freeSymbols) == 0
                    result = SymPy.N(sym)
                else
                    result = SymPy.N(subs(sym, freeSymbols[1], x))
                end
                return result
            end
            return func
        end
    catch
        function func(x)
            return sym
        end
        return func
    end
end

function prettyRound(x::Number; digs::Int = DIGITS)
    if isa(x, Int)
        return x
    elseif isa(x, Real)
        if isa(x, Rational) || isa(x, Irrational) # If x is rational or irrational numbers like e, pi
            return x
        elseif round(abs(x); digits = digs) == floor(abs(x))
            return Int(round(x))
        else
            return round(x; digits = digs)
        end
    elseif isa(x, Complex)
        roundedReal = prettyRound(real(x); digs = digs)
        roundedComplex = prettyRound(imag(x); digs = digs)
        return roundedReal + im*roundedComplex
    else
        return round(x; digits = digs)
    end
end

function prettyPrint(x::Union{Number, SymPy.Sym})
    expr = x
    if isa(expr, SymPy.Sym)
        prettyExpr = expr
        for a in sympy[:preorder_traversal](expr)
            if length(free_symbols(a)) == 0 && length(a.args) == 0
                if !(a in [sympy.E, PI]) && length(intersect(a.args, [sympy.E, PI])) == 0 # keep the transcendental numbers as symbols
                    prettyA = prettyRound.(SymPy.N(a))
                    prettyExpr = subs(prettyExpr, (a, prettyA))
                end
            end
        end
    else
        prettyExpr = prettyRound.(expr)
        prettyExpr = convert(SymPy.Sym, prettyExpr)
    end
    return prettyExpr
end

function is_approxLess(x::Number, y::Number; atol = TOL)
    if isa(x, Sym) == true || isa(y, Sym) == true
        x = convert(Float64, x)
        y = convert(Float64, x)
    end
    return !isapprox(x, y; atol = atol) && x < y
end

function is_approx(x::Number, y::Number; atol = TOL)
    if isa(x, Sym) == true || isa(y, Sym) == true
        x = convert(Float64, x)
        y = convert(Float64, x)
    end
    return isapprox(x, y; atol = atol)
end

function argument(z::Number)
    if angle(z) >= 0 # in [0,pi]
        return angle(z)
    else 
        # angle(z) is in (-pi, 0]
        # Shift it up to (pi,2pi]
        argument = 2pi + angle(z) # This is in (pi,2pi]
        if is_approx(argument, 2pi) # Map 2pi to 0
            return 0
        else
            return argument # This is now in [0,2pi)
        end
    end
end

function trace_contour(a::Number, n::Int, sampleSize::Int; infty = INFTY)
    lambdaVec = []
    for counter = 1:sampleSize
        x = rand(Uniform(-infty,infty), 1, 1)[1]
        y = rand(Uniform(-infty,infty), 1, 1)[1]
        lambda = x + y*im
        if real(a*lambda^n)>0
            append!(lambdaVec, lambda)
        end
    end
    Gadfly.plot(x=real(lambdaVec), y=imag(lambdaVec), Guide.xlabel("Re"), Guide.ylabel("Im"), Coord.Cartesian(ymin=-infty,ymax=infty, xmin=-infty, xmax=infty, fixed = true))
end

function get_distancePointLine(z::Number, theta::Number)
    if theta >= 2pi && theta < 0
        throw(error("Theta must be in [0,2pi)"))
    else
        if is_approx(argument(z), theta)
            return 0
        else
            x0, y0 = real(z), imag(z)
            if is_approx(theta, pi/2) || is_approx(theta, 3pi/2)
                return abs(x0)
            elseif is_approx(theta, 0) || is_approx(theta, 2pi)
                return abs(y0)
            else
                k = tan(theta)
                x = (y0+1/k*x0)/(k+1/k)
                y = k*x
                distance = LinearAlgebra.norm(z-(x+im*y))
                return distance
            end
        end
    end
end

# Structs
struct StructDefinitionError <: Exception
    msg::String
end

struct SymLinearDifferentialOperator
    # Entries in the array should be SymPy.Sym or Number. SymPy.Sym seems to be a subtype of Number, i.e., Array{Union{Number,SymPy.Sym}} returns Array{Number}. But specifying symPFunctions as Array{Number,2} gives a MethodError when the entries are Sympy.Sym objects.
    symPFunctions::Array
    interval::Tuple{Number,Number}
    t::SymPy.Sym
    SymLinearDifferentialOperator(symPFunctions::Array, interval::Tuple{Number,Number}, t::SymPy.Sym) =
    try
        symL = new(symPFunctions, interval, t)
        check_symLinearDifferentialOperator_input(symL)
        return symL
    catch err
        throw(err)
    end
end

function check_symLinearDifferentialOperator_input(symL::SymLinearDifferentialOperator)
    symPFunctions, (a,b), t = symL.symPFunctions, symL.interval, symL.t
    for symPFunc in symPFunctions
        if isa(symPFunc, SymPy.Sym)
            if size(free_symbols(symPFunc)) != (1,) && size(free_symbols(symPFunc)) != (0,)
                throw(StructDefinitionError(:"Only one free symbol is allowed in symP_k"))
            end
        elseif !isa(symPFunc, Number)
            throw(StructDefinitionError(:"symP_k should be SymPy.Sym or Number"))
        end
    end
    return true
end

# symL is an attribute of L that needs to be input by the user. There are checks to make sure symL is indeed the symbolic version of L.
# Principle: Functionalities of Julia Functions >= Functionalities of SymPy. If p_k has no SymPy representation, the only consequence should be that outputs by functions that take L as arugment has no symbolic expression. E.g., we allow L.pFunctions and L.symL.pFunctions to differ.
struct LinearDifferentialOperator
    pFunctions::Array # Array of julia functions or numbers representing constant functions
    interval::Tuple{Number,Number}
    symL::SymLinearDifferentialOperator
    LinearDifferentialOperator(pFunctions::Array, interval::Tuple{Number,Number}, symL::SymLinearDifferentialOperator) =
    try
        L = new(pFunctions, interval, symL)
        check_linearDifferentialOperator_input(L)
        return L
    catch err
        throw(err)
    end
end

# Assume symFunc has only one free symbol, as required by the definition of SymLinearDifferentialOperator. 
# That is, assume the input symFunc comes from SymLinearDifferentialOperator.
function check_func_sym_equal(func::Union{Function,Number}, symFunc, interval::Tuple{Number,Number}, t::SymPy.Sym) # symFunc should be Union{SymPy.Sym, Number}, but somehow SymPy.Sym gets ignored
    (a,b) = interval
    # Randomly sample 1000 points from (a,b) and check if func and symFunc agree on them
    for i = 1:1000
        # Check endpoints
        if i == 1
            x = a
        elseif i == 2
            x = b
        else
            x = rand(Uniform(a,b), 1)[1,1]
        end
        funcEvalX = evaluate.(func, x)
        if isa(symFunc, SymPy.Sym)
            symFuncEvalX = SymPy.N(subs(symFunc,t,x))
            # N() converts SymPy.Sym to Number
            # https://docs.sympy.org/latest/modules/evalf.html
            # subs() works no matter symFunc is Number or SymPy.Sym
        else
            symFuncEvalX = symFunc
        end
        tol = set_tol(funcEvalX, symFuncEvalX)
        if !isapprox(real(funcEvalX), real(symFuncEvalX); atol = real(tol)) ||
            !isapprox(imag(funcEvalX), imag(symFuncEvalX); atol = imag(tol))
            println("x = $x")
            println("symFunc = $symFunc")
            println("funcEvalX = $funcEvalX")
            println("symFuncEvalX = $symFuncEvalX")
            return false
        end
    end
    return true
end

# Check whether the inputs of L are valid.
function check_linearDifferentialOperator_input(L::LinearDifferentialOperator)
    pFunctions, (a,b), symL = L.pFunctions, L.interval, L.symL
    symPFunctions, t = symL.symPFunctions, symL.t
    # domainC = Complex(a..b, 0..0) # Domain [a,b] represented in the complex plane
    p0 = pFunctions[1]
    # p0Chebyshev = Fun(p0, a..b) # Chebysev polynomial approximation of p0 on [a,b]
    if !check_all(pFunctions, pFunc -> (isa(pFunc, Function) || isa(pFunc, Number)))
        throw(StructDefinitionError(:"p_k should be Function or Number"))
    elseif length(pFunctions) != length(symPFunctions)
        throw(StructDefinitionError(:"Number of p_k and symP_k do not match"))
    elseif (a,b) != symL.interval
        throw(StructDefinitionError(:"Intervals of L and symL do not match"))
    # # Assume p_k are in C^{n-k}. Check whether p0 vanishes on [a,b]. 
    # # roots() in IntervalRootFinding doesn't work if p0 is sth like t*im - 2*im. Neither does find_zero() in Roots.
    # # ApproxFun.roots() 
    # elseif (isa(p0, Function) && (!isempty(roots(p0Chebyshev)) || all(x->x>b, roots(p0Chebyshev)) || all(x->x<b, roots(p0Chebyshev)) || p0(a) == 0 || p0(b) == 0)) || p0 == 0 
    #     throw(StructDefinitionError(:"p0 vanishes on [a,b]"))
    elseif !all(i -> check_func_sym_equal(pFunctions[i], symPFunctions[i], (a,b), t), 1:length(pFunctions))
        # throw(StructDefinitionError(:"symP_k does not agree with p_k on [a,b]"))
        warn("symP_k does not agree with p_k on [a,b]") # Make this a warning instead of an error because the functionalities of Julia Functions may be more than those of SymPy objects; we do not want to compromise the functionalities of LinearDifferentialOperator because of the restrictions on SymPy.
    else
        return true
    end
end

struct VectorBoundaryForm
    M::Array # Why can't I specify Array{Number,2} without having a MethodError?
    N::Array # We can also use Matrix{<:Number}, and so we will need to alter the below checker
    VectorBoundaryForm(M::Array, N::Array) =
    try
        U = new(M, N)
        check_vectorBoundaryForm_input(U)
        return U
    catch err
        throw(err)
    end
end

# Check whether the input matrices that characterize U are valid
function check_vectorBoundaryForm_input(U::VectorBoundaryForm)
    # M, N = U.M, U.N
    # Avoid Inexact() error when taking rank()
    M = convert(Array{Complex}, U.M)
    N = convert(Array{Complex}, U.N)
    if !(check_all(U.M, x -> isa(x, Number)) && check_all(U.N, x -> isa(x, Number)))
        throw(StructDefinitionError(:"Entries of M, N should be Number"))
    elseif size(U.M) != size(U.N)
        throw(StructDefinitionError(:"M, N dimensions do not match"))
    elseif size(U.M)[1] != size(U.M)[2]
        throw(StructDefinitionError(:"M, N should be square matrices"))
    elseif LinearAlgebra.rank(hcat(M, N)) != size(M)[1] # rank() throws weird "InexactError()" when taking some complex matrices
        throw(StructDefinitionError(:"Boundary operators not linearly independent"))
    else
        return true
    end
end

# Construct Adjoint Boundary Conditions

function get_L(symL::SymLinearDifferentialOperator)
    symPFunctions, (a,b), t = symL.symPFunctions, symL.interval, symL.t
    if check_all(symPFunctions, x->!isa(x, SymPy.Sym))
        pFunctions = symPFunctions
    else
        pFunctions = sym_to_func.(symPFunctions)
    end
    L = LinearDifferentialOperator(pFunctions, (a,b), symL)
    return L
end

function get_URank(U::VectorBoundaryForm)
    # Avoid InexactError() when taking hcat() and rank()
    M = convert(Array{Complex}, U.M)
    N = convert(Array{Complex}, U.N)
    MHcatN = hcat(M, N)
    return LinearAlgebra.rank(MHcatN)
end

function get_Uc(U::VectorBoundaryForm)
    try
        check_vectorBoundaryForm_input(U)
        n = get_URank(U)
        I = complex(Matrix{Float64}(LinearAlgebra.I, 2n, 2n))
        M, N = U.M, U.N
        MHcatN = hcat(M, N)
        # Avoid InexactError() when taking rank()
        mat = convert(Array{Complex}, MHcatN)
        for i = 1:(2*n)
            newMat = vcat(mat, I[i:i,:])
            newMat = convert(Array{Complex}, newMat)
            if LinearAlgebra.rank(newMat) == LinearAlgebra.rank(mat) + 1
                mat = newMat
            end
        end
        UcHcat = mat[(n+1):(2n),:]
        Uc = VectorBoundaryForm(UcHcat[:,1:n], UcHcat[:,(n+1):(2n)])
        return Uc
    catch err
        return err
    end
end

function get_H(U::VectorBoundaryForm, Uc::VectorBoundaryForm)
    MHcatN = hcat(convert(Array{Complex}, U.M), convert(Array{Complex}, U.N))
    McHcatNc = hcat(convert(Array{Complex}, Uc.M), convert(Array{Complex}, Uc.N))
    H = vcat(MHcatN, McHcatNc)
    return H
end

function get_pDerivMatrix(L::LinearDifferentialOperator; symbolic = false)
    if symbolic
        symL = L.symL
        symPFunctions, t = symL.symPFunctions, symL.t
        n = length(symPFunctions)-1
        symPDerivMatrix = Array{SymPy.Sym}(undef, n,n)
        pFunctionSymbols = symPFunctions
        for i in 0:(n-1)
            for j in 0:(n-1)
                index, degree = i, j
                symP = pFunctionSymbols[index+1]
                # If symP is not a Sympy.Sym object (e.g., is a Number instead), then cannot use get_deriv()
                if !isa(symP, SymPy.Sym)
                    if degree > 0
                        symPDeriv = 0
                    else
                        symPDeriv = symP
                    end
                else
                    symPDeriv = get_deriv(symP, degree)
                end
                symPDerivMatrix[i+1,j+1] = symPDeriv
            end
        end
        return symPDerivMatrix
    else
        symPDerivMatrix = get_pDerivMatrix(L; symbolic = true)
        n = length(L.pFunctions)-1
        pDerivMatrix = sym_to_func.(symPDerivMatrix)
    end
    return pDerivMatrix
end

function get_Bjk(L::LinearDifferentialOperator, j::Int, k::Int; symbolic = false, pDerivMatrix = get_pDerivMatrix(L; symbolic = symbolic))
    n = length(L.pFunctions)-1
    if j <= 0 || j > n || k <= 0 || k > n
        throw("j, k should be in {1, ..., n}")
    end
    sum = 0
    if symbolic
        symPDerivMatrix = get_pDerivMatrix(L; symbolic = true)
        for l = (j-1):(n-k)
            summand = binomial(l, j-1) * symPDerivMatrix[n-k-l+1, l-j+1+1] * (-1)^l
            sum += summand
        end
    else
        for l = (j-1):(n-k)
            summand = mult_func(binomial(l, j-1) * (-1)^l, pDerivMatrix[n-k-l+1, l-j+1+1])
            sum = add_func(sum, summand)
        end
    end
    return sum
end

function get_B(L::LinearDifferentialOperator; symbolic = false, pDerivMatrix = get_pDerivMatrix(L; symbolic = symbolic))
    n = length(L.pFunctions)-1
    B = Array{Union{Function, Number, SymPy.Sym}}(undef, n, n)
    for j = 1:n
        for k = 1:n
            B[j,k] = get_Bjk(L, j, k; symbolic = symbolic, pDerivMatrix = pDerivMatrix)
        end
    end
    return B
end

function get_BHat(L::LinearDifferentialOperator, B::Array)
#     if check_any(B, x->isa(x, SymPy.Sym))
#         throw("Entries of B should be Function or Number")
#     end
    pFunctions, (a,b) = L.pFunctions, L.interval
    n = length(pFunctions)-1
    BHat = Array{Float64, 2}(undef, 2n, 2n)
    BHat = convert(Array{Complex}, BHat)
    BEvalA = evaluate.(B, a)
    BEvalB = evaluate.(B, b)
    BHat[1:n,1:n] = -BEvalA
    BHat[(n+1):(2n),(n+1):(2n)] = BEvalB
    BHat[1:n, (n+1):(2n)] = zeros(n, n)
    BHat[(n+1):(2n), 1:n] = zeros(n, n)
    return BHat
end

function get_J(BHat, H)
    n = size(H)[1]
    H = convert(Array{Complex}, H)
    J = (BHat * inv(H))'
    # J = convert(Array{Complex}, J)
    return J
end

function get_adjointCand(J)
    n = convert(Int, size(J)[1]/2)
    J = convert(Array{Complex}, J)
    PStar = J[(n+1):2n,1:n]
    QStar = J[(n+1):2n, (n+1):2n]
    adjointU = VectorBoundaryForm(PStar, QStar)
    return adjointU
end

function get_xi(L::LinearDifferentialOperator; symbolic = true, xSym= nothing)
    if symbolic
        t = L.symL.t
        n = length(L.pFunctions)-1
        symXi = Array{SymPy.Sym}(undef, n,1)
        if isa(xSym, Nothing)
            throw(error("xSymrequired"))
        else
            for i = 1:n
                symXi[i] = get_deriv(xSym, i-1)
            end
            return symXi
        end
    else
        if isa(xSym, Nothing)
            throw(error("xSym required"))
        elseif !isa(xSym, SymPy.Sym) && !isa(xSym, Number)
            throw(error("xSym should be SymPy.Sym or Number"))
        else
            symXi = get_xi(L; symbolic = true, xSym = xSym)
            xi = sym_to_func.(symXi)
        end
    end
end

function get_Ux(L::LinearDifferentialOperator, U::VectorBoundaryForm, xSym)
    (a, b) = L.interval
    xi = get_xi(L; symbolic = false, xSym = xSym)
    xiEvalA = evaluate.(xi, a)
    xiEvalB = evaluate.(xi, b)
    M, N = U.M, U.N
    Ux = M*xiEvalA + N*xiEvalB
    return Ux
end

function check_adjoint(L::LinearDifferentialOperator, U::VectorBoundaryForm, adjointU::VectorBoundaryForm, B::Array)
    (a, b) = L.interval
    M, N = U.M, U.N
    P, Q = (adjointU.M)', (adjointU.N)'
    # Avoid InexactError() when taking inv()
    BEvalA = convert(Array{Complex}, evaluate.(B, a))
    BEvalB = convert(Array{Complex}, evaluate.(B, b))
    left = M * inv(BEvalA) * P
    right = N * inv(BEvalB) * Q
#     println("left = $left")
#     println("right = $right")
    tol = set_tol(left, right)
    return all(i -> isapprox(left[i], right[i]; atol = tol), 1:length(left)) # Can't use == to deterimine equality because left and right are arrays of floats
end

function get_adjointU(L::LinearDifferentialOperator, U::VectorBoundaryForm, pDerivMatrix=get_pDerivMatrix(L))
    B = get_B(L; pDerivMatrix = pDerivMatrix)
    BHat = get_BHat(L, B)
    Uc = get_Uc(U)
    H = get_H(U, Uc)
    J = get_J(BHat, H)
    adjointU = get_adjointCand(J)
    if check_adjoint(L, U, adjointU, B)
        return adjointU
    else
        throw(error("Adjoint found not valid"))
    end
end

# Approximate roots of exponential polynomial 
x = symbols("x")
sympyAddExpr = 1 + x
sympyMultExpr = 2*x
sympyPowerExpr = x^2
sympyExpExpr = sympy.E^x

# although the function body is the same as "power" and "others", this case is isolated because negative exponents, e.g., factor_list(e^(-im*x)), give PolynomialError('a polynomial expected, got exp(-I*x)',), while factor_list(cos(x)) runs normally
function separate_real_imaginary_exp(expr::SymPy.Sym)
    result = real(expr) + im*imag(expr)
    return result
end

# we won't be dealing with cases like x^(x^x)
function separate_real_imaginary_power(expr::SymPy.Sym)
    result = real(expr) + im*imag(expr)
    return result
end

function separate_real_imaginary_mult(expr::SymPy.Sym)
    terms = expr.args
    result = 1
    # if the expanded expression contains toplevel multiplication, the individual terms must all be exponentials or powers
    for term in terms
        # println("term = $term")
        # if term is exponential
        if func(term) == func(sympyExpExpr)
            termSeparated = separate_real_imaginary_exp(term)
        # if term is power (not sure if this case and the case below overlaps)
        elseif func(term) == func(sympyPowerExpr)
            termSeparated = separate_real_imaginary_power(term)
            # else, further split each product term into indivdual factors (this case also includes the case where term is a number, which would go into the "constant" below)
        else
            termSeparated = term # term is a number
#             (constant, factors) = factor_list(term)
#             termSeparated = constant
#             # separate each factor into real and imaginary parts and collect the product of separated factors
#             for (factor, power) in factors
#                 factor = factor^power
#                 termSeparated = termSeparated * (real(factor) + im*imag(factor))
#             end
        end
        # println("termSeparated = $termSeparated") 
        # collect the product of separated term, i.e., product of separated factors
        result = result * termSeparated
    end
    result = real(result) + im*imag(result)
    return result
end

function separate_real_imaginary_add(expr::SymPy.Sym)
    x = symbols("x")
    # if the expanded expression contains toplevel addition, the individual terms must all be products or symbols
    terms = expr.args
    result = 0
    # termSeparated = 0 # to avoid undefined error if there is no else (case incomplete)
    for term in terms
        # println("term = $term")
        # if term is a symbol
        if func(term) == func(x)
            termSeparated = term
        # if term is exponential
        elseif func(term) == func(sympyExpExpr)
            termSeparated = separate_real_imaginary_exp(term)
        # if term is a power
        elseif func(term) == func(sympyPowerExpr)
            termSeparated = separate_real_imaginary_power(term)
        # if term is a product
        elseif func(term) == func(sympyMultExpr)
            termSeparated = separate_real_imaginary_mult(term)
        # if term is a number
        else
            termSeparated = term
        end
        # println("termSeparated = $termSeparated")
        result = result + termSeparated
    end
    result = real(result) + im*imag(result)
    return result
end

function separate_real_imaginary_power_mult_add(expr::SymPy.Sym)
    if func(expr) == func(sympyPowerExpr)
        result = separate_real_imaginary_power(expr)
    elseif func(expr) == func(sympyMultExpr)
        result = separate_real_imaginary_mult(expr)
        else #if SymPy.func(expr) == SymPy.func(sympyAddExpr)
        result = separate_real_imaginary_add(expr)
#     else
#         result = expr
    end
    return result
end

function separate_real_imaginary_others(expr::SymPy.Sym)
    # if the expanded expression is neither of the above, it must be a single term, e.g., x or cos(2x+1), which is a function wrapping around an expression; in this case, use the helper function to clean up the expression and feed it back to the function
    term = expr.args[1]
    termCleaned = separate_real_imaginary_power_mult_add(term)
    result = subs(expr,expr.args[1],termCleaned)
    result = real(result) + im*imag(result)
    return result
end

function separate_real_imaginary(delta::SymPy.Sym)
    x = symbols("x", real = true)
    y = symbols("y", real = true)
    
    freeSymbols = free_symbols(delta)
    # check if delta has one and only one free symbol (e.g., global variable lambda)
    if length(freeSymbols) == 1
        lambda = freeSymbols[1]
        # substitute lambda with x+iy
        expr = subs(delta, lambda, x+im*y)
        # expand the new expression
        expr = expand(expr)
        
        if func(expr) == func(sympyPowerExpr)
#             println(expr)
#             println("power!")
            result = separate_real_imaginary_power(expr)
#             println("result = $result")
        elseif func(expr) == func(sympyAddExpr)
#             println(expr)
#             println("addition!")
            result = separate_real_imaginary_add(expr)
#             println("result = $result")
        elseif func(expr) == func(sympyMultExpr)
#             println(expr)
#             println("multiplication!")
            result = separate_real_imaginary_mult(expr)
#             println("result = $result")
        else
#             println(expr)
#             println("single term!")
            result = separate_real_imaginary_others(expr)
#             println("result = $result")
        end
        result = expand(result)
        return real(result) + im*imag(result)
        
    else
        throw("Delta has more than one variable")
    end
end

function plot_levelCurves(bivariateDelta::SymPy.Sym; realFunc = real(bivariateDelta), imagFunc = imag(bivariateDelta), xRange = (-INFTY, INFTY), yRange = (-INFTY, INFTY), step = INFTY/1000, width = 1000, height = 800)
    gr(size=(width,height), leg=false)
    freeSymbols = free_symbols(bivariateDelta)
    x = symbols("x", real = true)
    y = symbols("y", real = true)
    
    xGridStep = (xRange[2] - xRange[1])/20
    yGridStep = (yRange[2] - yRange[1])/20
    
    if freeSymbols == [x, y]
        Plots.contour(xRange[1]:step:xRange[2], yRange[1]:step:yRange[2], realFunc, levels=[0], size = (width, height), tickfontsize = 8, seriescolor=:reds, transpose = false, linewidth = 4, linealpha = 1, xticks = xRange[1]:xGridStep:xRange[2], yticks = yRange[1]:yGridStep:yRange[2], grid = true, gridalpha = 0.5)
        Plots.contour!(xRange[1]:step:xRange[2], yRange[1]:step:yRange[2], imagFunc, levels=[0], size = (width, height), tickfontsize = 8, seriescolor=:blues, transpose = false, linewidth = 4, linealpha = 1, xticks = xRange[1]:xGridStep:xRange[2], yticks = yRange[1]:yGridStep:yRange[2], grid = true, gridalpha = 0.5)
    else
        Plots.contour(xRange[1]:step:xRange[2], yRange[1]:step:yRange[2], realFunc, levels=[0], size = (width, height), tickfontsize = 8, seriescolor=:reds, transpose = true, linewidth = 4, linealpha = 1, xticks = xRange[1]:xGridStep:xRange[2], yticks = yRange[1]:yGridStep:yRange[2], grid = true, gridalpha = 0.5)
        Plots.contour!(xRange[1]:step:xRange[2], yRange[1]:step:yRange[2], imagFunc, levels=[0], size = (width, height), tickfontsize = 8, seriescolor=:blues, transpose = true, linewidth = 4, linealpha = 1, xticks = xRange[1]:xGridStep:xRange[2], yticks = yRange[1]:yGridStep:yRange[2], grid = true, gridalpha = 0.5)
    end
end

function check_boundaryConditions(L::LinearDifferentialOperator, U::VectorBoundaryForm, fSym::Union{SymPy.Sym, Number})
    # Checks whether f satisfies the homogeneous boundary conditions
    Ux = get_Ux(L, U, fSym)
    return check_all(Ux, x->is_approx(x, 0))
end

function get_MPlusMinus(adjointU::VectorBoundaryForm; symbolic = false, generic = false)
    # these are numeric matrices
    bStar, betaStar = adjointU.M, adjointU.N
    n = size(bStar)[1]
    if symbolic
        # return MPlus and MMinus as symbolic expressions with (the global variable) lambda as free variable
        lambda = symbols("lambda")
        if generic
            alpha = symbols("alpha")
        else
            alpha = sympy.E^(2*PI*im/n)
        end
        MPlusMat = Array{SymPy.Sym}(undef, n,n)
        for k = 1:n
            for j = 1:n
                sumPlus = 0
                for r = 0:(n-1)
                    summandPlus = (-im*alpha^(k-1)*lambda)^r * bStar[j,r+1]
                    sumPlus += summandPlus
                end
                sumPlus = simplify(sumPlus)
                MPlusMat[k,j] = sumPlus
            end
        end
        MPlusSym = MPlusMat
        MMinusMat = Array{SymPy.Sym}(undef, n,n)
        for k = 1:n
            for j = 1:n
                sumMinus = 0
                for r = 0:(n-1)
                    summandMinus = (-im*alpha^(k-1)*lambda)^r * betaStar[j,r+1]
                    sumMinus += summandMinus
                end
                sumMinus = simplify(sumMinus)
                MMinusMat[k,j] = sumMinus
            end
        end
        MMinusSym = MMinusMat
        return (MPlusSym, MMinusSym)
    else
        alpha = sympy.E^(2*pi*im/n)
        # if not symbolic, return MPlus and MMinus as functions of lambda
        function MPlus(lambda::Number)
            MPlusMat = complex(Matrix{Float64}(LinearAlgebra.I, n, n)) # Initialise as a complex-valued identity matrix
            for k = 1:n
                for j = 1:n
                    sumPlus = 0
                    for r = 0:(n-1)
                        summandPlus = (-im*alpha^(k-1)*lambda)^r * bStar[j,r+1]
                        sumPlus += summandPlus
                    end
                    MPlusMat[k,j] = sumPlus
                end
            end
            return MPlusMat
        end
        function MMinus(lambda::Number)
            MMinusMat = complex(Matrix{Float64}(LinearAlgebra.I, n, n)) # Initialise as a complex-valued identity matrix
            for k = 1:n
                for j = 1:n
                    sumMinus = 0
                    for r = 0:(n-1)
                        summandMinus = (-im*alpha^(k-1)*lambda)^r * betaStar[j,r+1]
                        sumMinus += summandMinus
                    end
                    MMinusMat[k,j] = sumMinus
                end
            end
            return MMinusMat
        end
    end
    return (MPlus, MMinus)
end


function get_M(adjointU::VectorBoundaryForm; symbolic = false, generic = false)
    bStar, betaStar = adjointU.M, adjointU.N
    n = size(bStar)[1]
    if symbolic
        # return M as a symbolic expression with lambda as free variable
        lambda = symbols("lambda")
        if generic
            alpha = symbols("alpha")
        else
            alpha = sympy.E^(2*PI*im/n)
        end
        (MPlusSym, MMinusSym) = get_MPlusMinus(adjointU; symbolic = true, generic = generic)
        MLambdaSym = Array{SymPy.Sym}(undef, n,n)
        for k = 1:n
            for j = 1:n
                MLambdaSym[k,j] = simplify(MPlusSym[k,j] + MMinusSym[k,j] * sympy.E^(-im*alpha^(k-1)*lambda))
            end
        end
        MSym = simplify.(MLambdaSym)
        return MSym
    else
        alpha = sympy.E^(2*pi*im/n)
        function M(lambda::Number)
            (MPlus, MMinus) = get_MPlusMinus(adjointU)
            MPlusLambda, MMinusLambda = MPlus(lambda), MMinus(lambda)
            MLambda = complex(Matrix{Float64}(LinearAlgebra.I, n, n))
            for k = 1:n
                for j = 1:n
                    MLambda[k,j] = MPlusLambda[k,j] + MMinusLambda[k,j] * sympy.E^(-im*alpha^(k-1)*lambda)
                end
            end
            return MLambda
        end
        return M 
    end
end

function get_delta(adjointU::VectorBoundaryForm; symbolic = false, generic = false)
    if symbolic
        MSym = get_M(adjointU; symbolic = true, generic = generic)
        deltaSym = simplify(sympy.det(MSym))
        return deltaSym
    else
       function delta(lambda::Number)
            M = get_M(adjointU; symbolic = false)
            MLambda = convert(Array{Complex}, M(lambda))
            return LinearAlgebra.det(MLambda)
        end
        return delta 
    end
end

function get_Xlj(adjointU::VectorBoundaryForm, l::Number, j::Number; symbolic = false, generic = false)
    bStar, betaStar = adjointU.M, adjointU.N
    n = size(bStar)[1]
    if symbolic
        MSym = get_M(adjointU; symbolic = true, generic = generic)
        MBlockSym = [MSym MSym; MSym MSym]
        XljSym = MBlockSym[(l+1):(l+1+n-2), (j+1):(j+1+n-2)]
        return XljSym
    else
        M = get_M(adjointU; symbolic = false)
        function Xlj(lambda::Number)
            MLambda = M(lambda)
            MLambdaBlock = [MLambda MLambda; MLambda MLambda]
            XljLambda = MLambdaBlock[(l+1):(l+1+n-2), (j+1):(j+1+n-2)]
            return XljLambda
        end
        return Xlj 
    end
end

function get_alpha(m::Int, n::Int)
    if m == 1
        result = (-1)^n
    elseif m == 2
        result = (-1)^(n+1)*n^2
    else
        sum = 0
        for k = 1:(n-m+2)
            product = 1
            for j = k:(m+k-3)
                product *= (n-j)
            end
            sum += binomial(m+k-3, k-1) * product
        end
        result = (-1)^(n+m-1)*2^(m-2)*n*sum
    end
    return result
end

function get_ChebyshevTermIntegral(n::Int; symbolic = true, c = symbols("c"))
    if symbolic
        if c == 0
            if n == 0
                expr = 2
            elseif n == 1
                expr = 0
            else
                expr = ((-1)^(n+1)-1)/(n^2-1)
            end
        else
            expr = 0
            for m = 1:(n+1)
                summand = get_alpha(m, n) * (sympy.E^(im*c)/(im*c)^m  + (-1)^(m+n)*sympy.E^(-im*c)/(im*c)^m)
                expr += summand
            end
        end
        expr = simplify(expr)
        return expr
    else
        function TTilde(c)
            if c == 0
                if n == 0
                    result = 2
                elseif n == 1
                    result = 0
                else
                    result = ((-1)^(n+1)-1)/(n^2-1)
                end
            else
                result = 0
                for i = 1:(n+1)
                    summand = get_alpha(i, n) * (MathConstants.e^(im*c)/(im*c)^i  + (-1)^(i+n)*MathConstants.e^(-im*c)/(im*c)^i)
                    result += summand
                end
            end
            return result
        end
        return TTilde
    end
end

function get_ChebyshevCoefficients(f::Union{Function,Number})
    fCheb = ApproxFun.Fun(f, 0..1) # Approximate f on [0,1] using chebyshev polynomials
    chebCoefficients = ApproxFun.coefficients(fCheb) # get coefficients of the Chebyshev polynomial
    return chebCoefficients
end

function get_ChebyshevIntegral(l::Number, f::Union{Function, Number}; symbolic = false, lambda = nothing, alpha = nothing)
    fChebCoeffs = get_ChebyshevCoefficients(f)
    # Replace coefficients too close to 0 by 0
    # fChebCoeffs = [if is_approx(x, 0) 0 else x end for x in fChebCoeffs]
    if symbolic
        lambda = symbols("lambda")
        c = alpha^(l-1)*lambda/2
        integralSym = 0
        for m = 1:length(fChebCoeffs)
            fChebCoeff = fChebCoeffs[m]
            if is_approx(fChebCoeff, 0)
                continue
            else
            integralSym += fChebCoeffs[m] * get_ChebyshevTermIntegral(m-1; symbolic = true, c = c)
            end
        end
        integralSym = integralSym/(2*sympy.E^(im*c))
        integralSym = simplify(integralSym)
        return integralSym
    else
        if isa(lambda, Nothing) || isa(alpha, Nothing)
            throw("lambda, alpha required")
        else
            c = alpha^(l-1)*lambda/2
            integral = 0
            for m = 1:length(fChebCoeffs)
                fChebCoeff = fChebCoeffs[m]
                if is_approx(fChebCoeff, 0)
                    continue
                else
                    integral += fChebCoeff * get_ChebyshevTermIntegral(m-1; symbolic = false)(c)
                end
            end
            integral = integral/(2*MathConstants.e^(im*c))
            return integral
        end
    end
end

function get_FPlusMinus(adjointU::VectorBoundaryForm; symbolic = false, generic = false)
    bStar, betaStar = adjointU.M, adjointU.N
    n = size(bStar)[1]
    if symbolic
        lambda = symbols("lambda")
        (MPlusSym, MMinusSym) = get_MPlusMinus(adjointU; symbolic = true, generic = generic)
        deltaSym = get_delta(adjointU; symbolic = true, generic = generic)
        if generic
            alpha = symbols("alpha")
            c = symbols("c")
            FT = SymFunction("FT[f]")(c)
            sumPlusSymGeneric = 0
            sumMinusSymGeneric = 0
            for l = 1:n
                summandPlusSymGeneric = 0
                summandMinusSymGeneric = 0
                for j = 1:n
                    XljSym = get_Xlj(adjointU, l, j; symbolic = true, generic = true)
                    integralSymGeneric = subs(FT, c, alpha^(l-1)*lambda)
                    summandPlusSymGeneric += (-1)^((n-1)*(l+j)) * SymPy.det(XljSym) * MPlusSym[1,j] * integralSymGeneric
                    summandMinusSymGeneric += (-1)^((n-1)*(l+j)) * SymPy.det(XljSym) * MMinusSym[1,j] * integralSymGeneric
                end
                sumPlusSymGeneric += summandPlusSymGeneric
                sumMinusSymGeneric += summandMinusSymGeneric
            end
            FPlusSymGeneric = simplify(1/(2*PI*deltaSym)*sumPlusSymGeneric)
            FMinusSymGeneric = simplify((-sympy.E^(-im*lambda))/(2*PI*deltaSym)*sumMinusSymGeneric)
            return (FPlusSymGeneric, FMinusSymGeneric)
        else
            alpha = sympy.E^(2*PI*im/n)
            function FPlusSym(f::Union{Function, Number})
                sumPlusSym = 0
                for l = 1:n
                    summandPlusSym = 0
                    for j = 1:n
                        XljSym = get_Xlj(adjointU, l, j; symbolic = true)
                        integralSym = get_ChebyshevIntegral(l, f; symbolic = true, lambda = lambda, alpha = alpha)
                        summandPlusSym += (-1)^((n-1)*(l+j)) * SymPy.det(XljSym) * MPlusSym[1,j] * integralSym
                    end
                    sumPlusSym += summandPlusSym
                end
                return simplify(1/(2*PI*deltaSym)*sumPlusSym)
            end
            function FMinusSym(f::Union{Function, Number})
                sumMinusSym = 0
                for l = 1:n
                    summandMinusSym = 0
                    for j = 1:n
                        XljSym = get_Xlj(adjointU, l, j; symbolic = true)
                        c = alpha^(l-1)*lambda/2
                        integralSym = get_ChebyshevIntegral(l, f; symbolic = true, lambda = lambda, alpha = alpha)
                        summandMinusSym += (-1)^((n-1)*(l+j)) * SymPy.det(XljSym) * MMinusSym[1,j] * integralSym
                    end
                    sumMinusSym += summandMinusSym
                end
                return simplify((-sympy.E^(-im*lambda))/(2*PI*deltaSym)*sumMinusSym)
            end
            return (FPlusSym, FMinusSym)
            end
    else
        alpha = MathConstants.e^(2pi*im/n)
        (MPlus, MMinus) = get_MPlusMinus(adjointU; symbolic = false)
        function FPlus(lambda::Number, f::Union{Function, Number})
            MPlusLambda, MMinusLambda = MPlus(lambda), MMinus(lambda)
            M = get_M(adjointU; symbolic = false)
            MLambda = convert(Array{Complex}, M(lambda))
            deltaLambda = LinearAlgebra.det(MLambda) # or deltaLambda = (get_delta(adjointU))(lambda)
            sumPlus = 0
            for l = 1:n
                summandPlus = 0
                for j = 1:n
                    Xlj = get_Xlj(adjointU, l, j; symbolic = false)
                    XljLambda = convert(Array{Complex}, Xlj(lambda))
                    integral = get_ChebyshevIntegral(l, f; symbolic = false, lambda = lambda, alpha = alpha)
                    summandPlus += (-1)^((n-1)*(l+j)) * LinearAlgebra.det(XljLambda) * MPlusLambda[1,j] * integral
                end
                sumPlus += summandPlus
            end
            return 1/(2pi*deltaLambda)*sumPlus
        end
        function FMinus(lambda::Number, f::Union{Function, Number})
            MPlusLambda, MMinusLambda = MPlus(lambda), MMinus(lambda)
            M = get_M(adjointU; symbolic = false)
            MLambda = convert(Array{Complex}, M(lambda))
            deltaLambda = LinearAlgebra.det(MLambda) # or deltaLambda = (get_delta(adjointU))(lambda)
            sumMinus = 0
            for l = 1:n
                summandMinus = 0
                for j = 1:n
                    Xlj = get_Xlj(adjointU, l, j)
                    XljLambda = convert(Array{Complex}, Xlj(lambda))
                    integral = get_ChebyshevIntegral(l, f; symbolic = false, lambda = lambda, alpha = alpha)
                    summandMinus += (-1)^((n-1)*(l+j)) * LinearAlgebra.det(XljLambda) * MMinusLambda[1,j] * integral
                end
                sumMinus += summandMinus
            end
            return (-MathConstants.e^(-im*lambda))/(2pi*deltaLambda)*sumMinus
        end
        return (FPlus, FMinus)
    end
end
    

function get_gammaAAngles(a::Number, n::Int; symbolic = false)
    # thetaA = argument(a)
    thetaA = angle(a)
    if symbolic
        thetaStartList = Array{SymPy.Sym}(undef, n) # List of angles that characterize where domain sectors start
        thetaEndList = Array{SymPy.Sym}(undef, n) # List of angles that characterize where domain sectors end
        k = symbols("k")
        counter = 0
        while (2pi*counter + pi/2 - thetaA)/n < 2pi
        # Substituting counter for k
        # while SymPy.N(subs((2PI*k + PI/2 - thetaA)/n, k, counter)) < 2pi
            thetaStart = (2*PI*counter - PI/2 - rationalize(thetaA/pi)*PI)/n
            thetaEnd = (2*PI*counter + PI/2 - rationalize(thetaA/pi)*PI)/n
            counter += 1
            thetaStartList[counter] = thetaStart
            thetaEndList[counter] = thetaEnd
        end
    else
        thetaStartList = Array{Number}(undef, n)
        thetaEndList = Array{Number}(undef, n)
        k = 0
        while (2pi*k + pi/2 - thetaA)/n < 2pi
            thetaStart = (2pi*k - pi/2 - thetaA)/n
            thetaEnd = (2pi*k + pi/2 - thetaA)/n
            k += 1
            thetaStartList[k] = thetaStart
            thetaEndList[k] = thetaEnd
        end
    end
    return (thetaStartList, thetaEndList)
end

function get_gammaAAnglesSplit(a::Number, n::Int; symbolic = false)
    (thetaStartList, thetaEndList) = get_gammaAAngles(a, n; symbolic = symbolic)
    # Split sectors that contain the positive half of the real line (angle = 0)
    zeroIndex = findall(i -> ((is_approxLess(convert(Float64, thetaStartList[i]), 0) && is_approxLess(0, convert(Float64, thetaEndList[i])))), 1:length(thetaStartList))
    if !isempty(zeroIndex)
        index = zeroIndex[1]
        # Insert 0 after thetaStart
        splice!(thetaStartList, (index+1):index, 0)
        # Insert 0 before thetaEnd
        splice!(thetaEndList, index:(index-1), 0)
    end
    # Split sectors that contain the negative half of the real line (angle = pi)
    piIndex = findall(i -> ((is_approxLess( convert(Float64, thetaStartList[i]), pi) && is_approxLess(pi, convert(Float64, thetaEndList[i]) ))), 1:length(thetaStartList))
    if !isempty(piIndex)
        index = piIndex[1]
        if symbolic
            # Insert pi after thetaStart
            splice!(thetaStartList, (index+1):index, pi)
            # Insert pi before thetaEnd
            splice!(thetaEndList, index:(index-1), pi)
        else
            # Use pi*1 instead of pi to avoid "<= not defined for Irrational{:pi}" error in get_gamma()
            splice!(thetaStartList, (index+1):index, pi*1)
            splice!(thetaEndList, index:(index-1), pi*1)
        end
    end
    return (thetaStartList, thetaEndList)
end

function pointOnSector(z::Number, sectorAngles::Tuple{Number, Number})
    (startAngle, endAngle) = sectorAngles
    return is_approx(argument(z), startAngle) || is_approx(argument(z), endAngle) || is_approx(angle(z), startAngle) || is_approx(angle(z), endAngle)
end

function pointInSector(z::Number, sectorAngles::Tuple{Number, Number})
    (startAngle, endAngle) = sectorAngles
    # First check if z is on the sector boundary
    if pointOnSector(z, sectorAngles)
        return false
    else
        # angle(z) would work if it's in the sector with positive real parts and both positive and negative imaginary parts; argument(z) would work if it's in the sector with negative real parts and both positive and negative imaginary parts
        return (angle(z) > startAngle && angle(z) < endAngle) || (argument(z) > startAngle && argument(z) < endAngle) # no need to use is_approxLess because the case of approximatedly equal is already checked in pointOnSector
    end
end

function pointExSector(z::Number, sectorAngles::Tuple{Number, Number})
    return !pointOnSector(z, sectorAngles) && !pointInSector(z, sectorAngles)
end

function get_epsilon(zeroList::Array, a::Number, n::Int)
    (thetaStartList, thetaEndList) = get_gammaAAnglesSplit(a, n; symbolic = false)
    thetaStartEndList = collect(Iterators.flatten([thetaStartList, thetaEndList]))
    truncZeroList = []
    for zero in zeroList
        # If zero is interior to any sector, discard it
        if any(i -> pointInSector(zero, (thetaStartList[i], thetaEndList[i])), 1:n)
        else # If not, append it to truncZeroList
            append!(truncZeroList, zero)
        end
    end
    # List of distance between each zero and each line marking the boundary of some sector
    pointLineDistances = [get_distancePointLine(z, theta) for z in zeroList for theta in thetaStartEndList]
    if length(truncZeroList)>1
        # List of distance between every two zeroes
        pairwiseDistances = [LinearAlgebra.norm(z1-z2) for z1 in zeroList for z2 in truncZeroList]
    else
        pairwiseDistances = []
    end
    distances = collect(Iterators.flatten([pairwiseDistances, pointLineDistances]))
    # Distances of nearly 0 could be instances where the zero is actually on some sector boundary
    distances = filter(x -> !is_approx(x, 0), distances)
    epsilon = minimum(distances)/4
    return epsilon
end

function get_nGonAroundZero(zero::Number, epsilon::Number, n::Int)
    z = zero
    theta = argument(zero)
    deltaAngle = 2pi/n
    vertices = []
    for i = 1:n
        newAngle = pi-deltaAngle*(i-1)
        vertex = z + epsilon*MathConstants.e^(im*(theta+newAngle))
        append!(vertices, vertex)
    end
    # vertices = vcat(vertices, vertices[1])
    return vertices
end

function get_gamma(a::Number, n::Int, zeroList::Array; infty = INFTY, nGon = 8)
    (thetaStartList, thetaEndList) = get_gammaAAnglesSplit(a, n; symbolic = false)
    nSplit = length(thetaStartList)
    gammaAPlus, gammaAMinus, gamma0Plus, gamma0Minus = [], [], [], []
    epsilon = get_epsilon(zeroList, a, n)
    for i in 1:nSplit
        thetaStart = thetaStartList[i]
        thetaEnd = thetaEndList[i]
        # Initialize the boundary of each sector with the ending boundary, the origin, and the starting boundary (start and end boundaries refer to the order in which the boundaries are passed if tracked counterclockwise)
        initialPath = [infty*MathConstants.e^(im*thetaEnd), 0+0*im, infty*MathConstants.e^(im*thetaStart)]
        initialPath = convert(Array{Complex{Float64}}, initialPath)
        if thetaStart >= 0 && thetaStart <= pi && thetaEnd >= 0 && thetaEnd <= pi # if in the upper half plane, push the boundary path to gamma_a+
            push!(gammaAPlus, initialPath) # list of lists
        else # if in the lower half plane, push the boundary path to gamma_a-
            push!(gammaAMinus, initialPath)
        end
    end
    # Sort the zeroList by norm, so that possible zero at the origin comes last. We need to leave the origin in the initial path unchanged until we have finished dealing with all non-origin zeros because we use the origin in the initial path as a reference point to decide where to insert the deformed path
    zeroList = sort(zeroList, lt=(x,y)->!isless(LinearAlgebra.norm(x), LinearAlgebra.norm(y)))
    for zero in zeroList
        # println(zero)
        # If zero is not at the origin
        if !is_approx(zero, 0+0*im)
            # Draw an n-gon around it
            vertices = get_nGonAroundZero(zero, epsilon, nGon)
            # If zero is on the boundary of some sector
            if any(i -> pointOnSector(zero, (thetaStartList[i], thetaEndList[i])), 1:nSplit)
                # Find which sector(s) zero is on
                indices = findall(i -> pointOnSector(zero, (thetaStartList[i], thetaEndList[i])), 1:nSplit)
                # If zero is on the boundary of one sector
                if length(indices) == 1
                    # if vertices[2] is interior to any sector, include vertices on the other half of the n-gon in the contour approximation
                    z0 = vertices[2]
                    if any(i -> pointInSector(z0, (thetaStartList[i], thetaEndList[i])), 1:nSplit)
                        # Find which sector vertices[2] is in
                        index = findall(i -> pointInSector(z0, (thetaStartList[i], thetaEndList[i])), 1:nSplit)[1]
                    else # if vertices[2] is exterior, include vertices on this half of the n-gon in the contour approximation
                        # Find which sector vertices[length(vertices)] is in
                        z1 = vertices[length(vertices)]
                        index = findall(i -> pointInSector(z1, (thetaStartList[i], thetaEndList[i])), 1:nSplit)[1]
                    end
                    thetaStart = thetaStartList[index]
                    thetaEnd = thetaEndList[index]
                    # Find all vertices exterior to or on the boundary of this sector, which would form the nGonPath around the zero
                    nGonPath = vertices[findall(vertex -> !pointInSector(vertex, (thetaStart, thetaEnd)), vertices)]
                    # If this sector is in the upper half plane, deform gamma_a+
                    if thetaStart >= 0 && thetaStart <= pi && thetaEnd >= 0 && thetaEnd <= pi
                        gammaAPlusIndex = findall(path -> (is_approx(argument(zero), argument(path[1])) || is_approx(argument(zero), argument(path[length(path)]))), gammaAPlus)[1]
                        deformedPath = copy(gammaAPlus[gammaAPlusIndex])
                        if any(i -> is_approx(argument(zero), thetaStartList[i]) || is_approx(angle(zero), thetaStartList[i]), 1:nSplit) # if zero is on the starting boundary, insert the n-gon path after 0+0*im
                            splice!(deformedPath, length(deformedPath):(length(deformedPath)-1), nGonPath)
                        else # if zero is on the ending boundary, insert the n-gon path before 0+0*im
                            splice!(deformedPath, 2:1, nGonPath)
                        end
                        deformedPath = convert(Array{Complex{Float64}}, deformedPath)
                        gammaAPlus[gammaAPlusIndex] = deformedPath
                    else # if sector is in the lower half plane, deform gamma_a-
                        # # Find all vertices interior to or on the boundary of this sector, which would form the nGonPath around the zero
                        # nGonPath = vertices[find(vertex -> !pointExSector(vertex, (thetaStart, thetaEnd)), vertices)]
                        gammaAMinusIndex = findall(path -> (is_approx(argument(zero), argument(path[1])) || is_approx(argument(zero), argument(path[length(path)]))), gammaAMinus)[1]
                        deformedPath = copy(gammaAMinus[gammaAMinusIndex])
                        if any(i -> is_approx(argument(zero), thetaStartList[i]) || is_approx(angle(zero), thetaStartList[i]), 1:nSplit) # if zero is on the starting boundary, insert the n-gon path after 0+0*im
                            splice!(deformedPath, length(deformedPath):(length(deformedPath)-1), nGonPath)
                        else # if zero is on the ending boundary, insert the n-gon path before 0+0*im
                            splice!(deformedPath, 2:1, nGonPath) 
                        end
                        deformedPath = convert(Array{Complex{Float64}}, deformedPath)
                        gammaAMinus[gammaAMinusIndex] = deformedPath
                    end
                else # If zero is on the boundary of two sectors, then it must be on the real line, and we need to deform two sectors
                    # Find out which vertices are in the lower half plane
                    nGonPath = vertices[findall(vertex -> !pointInSector(vertex, (0, pi)), vertices)]
                    for index in indices
                        thetaStart = thetaStartList[index]
                        thetaEnd = thetaEndList[index]
                        # If this is the sector in the upper half plane, deform gamma_a+
                        if thetaStart >= 0 && thetaStart <= pi && thetaEnd >= 0 && thetaEnd <= pi
                            gammaAPlusIndex = findall(path -> (is_approx(argument(zero), argument(path[1])) || is_approx(argument(zero), argument(path[length(path)]))), gammaAPlus)[1]
                            deformedPath = copy(gammaAPlus[gammaAPlusIndex])
                            if is_approx(argument(zero), argument(deformedPath[length(deformedPath)])) # if zero is on the starting boundary, insert the n-gon path after 0+0*im
                                splice!(deformedPath, length(deformedPath):(length(deformedPath)-1), nGonPath)
                            else # if zero is on the ending boundary, insert the n-gon path before 0+0*im
                                splice!(deformedPath, 2:1, nGonPath)
                            end
                            deformedPath = convert(Array{Complex{Float64}}, deformedPath)
                            gammaAPlus[gammaAPlusIndex] = deformedPath
                        else # If this is the sector in the lower half plane, deform gamma_a-
                            gammaAMinusIndex = findall(path -> (is_approx(argument(zero), argument(path[1])) || is_approx(argument(zero), argument(path[length(path)]))), gammaAMinus)[1]
                            deformedPath = copy(gammaAMinus[gammaAMinusIndex])
                            if is_approx(argument(zero), argument(deformedPath[length(deformedPath)])) # if zero is on the starting boundary, insert the n-gon path after 0+0*im
                                splice!(deformedPath, length(deformedPath):(length(deformedPath)-1), nGonPath)
                            else # if zero is on the ending boundary, insert the n-gon path before 0+0*im
                                splice!(deformedPath, 2:1, nGonPath)
                            end
                            deformedPath = convert(Array{Complex{Float64}}, deformedPath)
                            gammaAMinus[gammaAMinusIndex] = deformedPath
                        end
                    end
                end
                # Sort each sector's path in the order in which they are integrated over
                gammaAs = [gammaAPlus, gammaAMinus]
                for j = 1:length(gammaAs)
                    gammaA = gammaAs[j]
                    for k = 1:length(gammaA)
                        inOutPath = gammaA[k]
                        originIndex = findall(x->x==0+0*im, inOutPath)[1]
                        inwardPath = inOutPath[1:(originIndex-1)]
                        outwardPath = inOutPath[(originIndex+1):length(inOutPath)]
                        # Sort the inward path and outward path
                        if length(inwardPath) > 0
                            inwardPath = sort(inwardPath, lt=(x,y)->!isless(LinearAlgebra.norm(x), LinearAlgebra.norm(y)))
                        end
                        if length(outwardPath) > 0
                            outwardPath = sort(outwardPath, lt=(x,y)->isless(LinearAlgebra.norm(x), LinearAlgebra.norm(y)))
                        end
                        inOutPath = vcat(inwardPath, 0+0*im, outwardPath)
                        inOutPath = convert(Array{Complex{Float64}}, inOutPath)
                        gammaA[k] = inOutPath
                    end
                    gammaAs[j] = gammaA 
                end
                gammaAPlus, gammaAMinus = gammaAs[1], gammaAs[2]
            # If zero is interior to any sector (after splitting by real line), ignore it
            # If zero is exterior to the sectors, avoid it
            elseif all(i -> pointExSector(zero, (thetaStartList[i], thetaEndList[i])), 1:nSplit)
                nGonPath = vcat(vertices, vertices[1]) # counterclockwise
                nGonPath = convert(Array{Complex{Float64}}, nGonPath)
                # If zero is in the upper half plane, add the n-gon path to gamma_0+
                if argument(zero) >= 0 && argument(zero) <= pi
                    push!(gamma0Plus, nGonPath)
                else # If zero is in the lower half plane, add the n-gon path to gamma_0-
                    push!(gamma0Minus, nGonPath)
                end
            end
        else # If zero is at the origin, we deform all sectors and draw an n-gon around the origin
            # deform each sector in gamma_a+
            for i = 1:length(gammaAPlus)
                deformedPath = gammaAPlus[i]
                # find the index of the zero at origin in the sector boundary path
                index = findall(j -> is_approx(deformedPath[j], 0+0*im), 1:length(deformedPath))
                # If the origin is not in the path, then it has already been bypassed
                if isempty(index)
                else # If not, find its index
                    index = index[1]
                end
                # create a path around zero (origin); the origin will not be the first or the last point in any sector boundary because it was initialized to be in the middle, and only insertions are performed. Moreover, the boundary path has already been sorted into the order in which they will be integrated over, so squarePath defined below has deformedPath[index-1], deformedPath[index+1] in the correct order.
                squarePath = [2*epsilon*MathConstants.e^(im*argument(deformedPath[index-1])), 2*epsilon*MathConstants.e^(im*argument(deformedPath[index+1]))]
                # replace the zero with the deformed path
                deleteat!(deformedPath, index) # delete the origin
                splice!(deformedPath, index:(index-1), squarePath) # insert squarePath into where the origin was at
                deformedPath = convert(Array{Complex{Float64}}, deformedPath)
                gammaAPlus[i] = deformedPath
            end
            # deform each sector in gamma_a-
            for i = 1:length(gammaAMinus)
                deformedPath = gammaAMinus[i]
                index = findall(j -> is_approx(deformedPath[j], 0+0*im), 1:length(deformedPath))
                if isempty(index)
                else
                    index = index[1]
                end
                squarePath = [2*epsilon*MathConstants.e^(im*argument(deformedPath[index-1])), 2*epsilon*MathConstants.e^(im*argument(deformedPath[index+1]))]
                deleteat!(deformedPath, index)
                splice!(deformedPath, index:(index-1), squarePath)
                deformedPath = convert(Array{Complex{Float64}}, deformedPath)
                gammaAMinus[i] = deformedPath
            end
            # Draw an n-gon around the origin and add to gamma_0+
            vertices = get_nGonAroundZero(zero, epsilon, nGon)
            nGonPath = vcat(vertices, vertices[1])
            nGonPath = convert(Array{Complex{Float64}}, nGonPath)
            push!(gamma0Plus, nGonPath)
        end
    end
    return (gammaAPlus, gammaAMinus, gamma0Plus, gamma0Minus)
end


  
 

function plot_contour(gamma::Array; infty = INFTY)
    sectorPathList = Array{Any}(undef,length(gamma),1)
    for i = 1:length(gamma)
        # For each sector path in the gamma contour, plot the points in the path and connect them in the order in which they appear in the path
        sectorPath = gamma[i]
        # labels = map(string, collect(1:1:length(sectorPath)))
        sectorPathList[i] = layer(x = real(sectorPath), y = imag(sectorPath), Geom.line(preserve_order=true))
    end
    coord = Coord.cartesian(xmin=-infty, xmax=infty, ymin=-infty, ymax=infty, fixed=true)
    Gadfly.plot(Guide.xlabel("Re"), Guide.ylabel("Im"), coord, sectorPathList...)
end

function solve_IBVP(L::LinearDifferentialOperator, U::VectorBoundaryForm, a::Number, zeroList::Array, f::Function; FPlusFunc = lambda->get_FPlusMinus(adjointU; symbolic = false)[1](lambda, f), FMinusFunc = lambda->get_FPlusMinus(adjointU; symbolic = false)[2](lambda, f), pDerivMatrix = get_pDerivMatrix(L), infty = INFTY)
    n = length(L.pFunctions)-1
    adjointU = get_adjointU(L, U, pDerivMatrix)
    (gammaAPlus, gammaAMinus, gamma0Plus, gamma0Minus) = get_gamma(a, n, zeroList)
    function q(x,t)
        integrandPlus(lambda) = MathConstants.e^(im*lambda*x)*MathConstants.e^(-a*lambda^n*t) * FPlusFunc(lambda)
        integrandMinus(lambda) = MathConstants.e^(im*lambda*x)*MathConstants.e^(-a*lambda^n*t) * FMinusFunc(lambda)
        tick()
            println("integrandPlus = $(integrandPlus(1+im))")
            println("integrandMinus = $(integrandMinus(1+im))")
        tock()
        # Integrate over individual paths in the Gamma contours
        println("gamma0Plus = $gamma0Plus")
        tick()
            integralGamma0Plus = 0
            for path in gamma0Plus
                println("path = $path")
                if length(path) == 0
                    path = [im,im]
                end
                tick()
                integralGamma0Plus += quadgk(integrandPlus, path...)[1]
                tock()
            end
        tock()
        
        println("int_0_+ = $integralGamma0Plus")
        
        println("gammaAPlus = $gammaAPlus")
        tick()
        
            integralGammaAPlus = 0
            for path in gammaAPlus
                println("path = $path")
                if length(path) == 0
                    path = [im,im]
                end
                tick()
                integralGammaAPlus += quadgk(integrandPlus, path...)[1]
                tock()
            end
        tock()
        
        println("int_a_+ = $integralGammaAPlus")
        
        println("gamma0Minus = $gamma0Minus")
        tick()
            integralGamma0Minus = 0
            for path in gamma0Minus
                println("path = $path")
                if length(path) == 0
                    path = [-im,-im]
                end
                tick()
                integralGamma0Minus += quadgk(integrandMinus, path...)[1]
                tock()
            end
        tock()

        println("int_0_- = $integralGamma0Minus")
        
        println("gammaAMinus = $gammaAMinus")
        tick() 
            integralGammaAMinus = 0
            for path in gammaAMinus
                println("path = $path")
                if length(path) == 0
                    path = [-im,-im]
                end
                tick()
                integralGammaAMinus += quadgk(integrandMinus, path...)[1]
                tock()
            end
        tock()
        println("int_a_- = $integralGammaAMinus")
        return (integralGamma0Plus + integralGammaAPlus + integralGamma0Minus + integralGammaAMinus)
    end
    return q
end