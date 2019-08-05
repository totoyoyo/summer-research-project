##########################################################################################################################################################
#Date created: June 17, 2019
#Name: Sultan Aitzhan
#Description: The relevant code for constructing multipoint adjoint matrices is copypasted here from $\textit{Construct_Multipoint_Adjoint_v1.ipynb}$ and the test functions are updated. ##########################################################################################################################################################

# Importing packages
##########################################################################################################################################################
using SymPy
using Distributions
using ApproxFun
import LinearAlgebra
##########################################################################################################################################################

# Global Constants
TOL = 1e-05
DIGITS = 3
INFTY = 10
##########################################################################################################################################################

# Helper functions

function check_all(array, condition)
    for x in array
        if !condition(x)
            return false
        end
    end
    return true
end

#
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

#
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

# 
function partition(n::Int)
    output = []
    for i = 0:n
        j = n - i
        push!(output, (i,j))
    end
    return output
end

#
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

#
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

# 
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

#
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


# 
function get_polynomial(coeffList::Array)
    polynomial = 0
    n = length(coeffList)-1
    for i in 0:n
        newTerm = t -> coeffList[i+1] * t^(n-i)
        polynomial = add_func(polynomial, newTerm)
    end
    return polynomial
end

# 
function get_polynomialDeriv(coeffList::Array, k::Int)
    if k < 0
        throw(error("Only nonnegative degrees are allowed"))
    elseif k == 0
        newCoeffList = coeffList
    else
        for counter = 1:k
            n = length(coeffList)
            newCoeffList = hcat([0],[(n-i)*coeffList[i] for i in 1:(n-1)]')
            coeffList = newCoeffList
        end
    end
    return get_polynomial(newCoeffList)
end

#
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

#
function prettyPrint(x::Union{Number, SymPy.Sym})
    expr = x
    if isa(expr, SymPy.Sym)
        prettyExpr = expr
        for a in sympy.preorder_traversal(expr)
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

#
function array_hcat_helper(matr1, matr2)
    j = size(matr1[1])[2]
    k = size(matr1[1])[1]
    arr = convert(Array{Complex}, Array{Float64}(undef, size(matr1[1])[1], 2k)) # Avoid InexactError
    t = 0
    for i=1:length(matr1)
        arr[1:k, (t*j+1):(t+1)*j] = matr1[i]
        arr[1:k, ((t+1)*j+1):(t+2)*j] = matr2[i]
        t +=2
    end
    return arr
end

#
function checker_matrix(matr1, matr2, tol) 
    m = size(matr1)[1]
    n = size(matr1)[2]
    for i=1:m
        for j=1:n
            if isapprox(matr1[i,j], matr2[i,j]; atol = tol) == false
                return false
            end
        end
    end
    return true
end
##########################################################################################################################################################

# Structs
struct StructDefinitionError <: Exception
    msg::String
end

#
struct SymLinearDifferentialOperator
    # Entries in the array should be SymPy.Sym or Number. SymPy.Sym seems to be a subtype of Number, i.e., Array{Union{Number,SymPy.Sym}} returns Array{Number}. But specifying symPFunctions as Array{Number,2} gives a MethodError when the entries are Sympy.Sym objects.
    symPFunctions::Array
    interval::Tuple
    t::SymPy.Sym
    SymLinearDifferentialOperator(symPFunctions::Array, interval::Tuple, t::SymPy.Sym) =
    try
        symL = new(symPFunctions, interval, t)
        check_symLinearDifferentialOperator_input(symL)
        return symL
    catch err
        throw(err)
    end
end

function check_symLinearDifferentialOperator_input(symL::SymLinearDifferentialOperator)
    symPFunctions, interval_tuple, t = symL.symPFunctions, symL.interval, symL.t
    for symPFunc in symPFunctions
        if isa(symPFunc, SymPy.Sym)
            if size(free_symbols(symPFunc)) != (1,) && size(free_symbols(symPFunc)) != (0,)
                throw(StructDefinitionError(:"Only one free symbol is allowed in symP_k"))
            end
        elseif !isa(symPFunc, Number)
            throw(StructDefinitionError(:"symP_k should be SymPy.Sym or Number"))
        end
    end
    for x_i in interval_tuple
        if !isa(x_i, Number)
            throw(StructDefinitionError(:"Interval must consist of numbers"))
        end
    end
    for i=1:length(interval_tuple)-1
        if (interval_tuple[i] < interval_tuple[i+1]) == false
            throw(StructDefinitionError(:"Terms in interval must be strictly increasing"))
        end
    end
    return true
end


# 
struct LinearDifferentialOperator
    pFunctions::Array # Array of julia functions or numbers representing constant functions
    interval::Tuple
    symL::SymLinearDifferentialOperator
    LinearDifferentialOperator(pFunctions::Array, interval::Tuple, symL::SymLinearDifferentialOperator) =
    try
        L = new(pFunctions, interval, symL)
        check_linearDifferentialOperator_input(L)
        return L
    catch err
        throw(err)
    end
end

function check_func_sym_equal(func::Union{Function,Number}, symFunc, interval::Tuple, t::SymPy.Sym)
    a = interval[1]
    b = interval[length(interval)]
    for i = 1:1000
        if i == 1
            x = a
        elseif i == 2
            x = b
        else
            x = rand(Uniform(a,b), 1)[1,1]
        end
        funcEvalX = evaluate(func, x)
        if isa(symFunc, SymPy.Sym)
            symFuncEvalX = SymPy.N(subs(symFunc,t,x))
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

function check_linearDifferentialOperator_input(L::LinearDifferentialOperator)
    pFunctions, interval, symL = L.pFunctions, L.interval, L.symL
    symPFunctions, t = symL.symPFunctions, symL.t
    p0 = pFunctions[1]
    if !check_all(pFunctions, pFunc -> (isa(pFunc, Function) || isa(pFunc, Number)))
        throw(StructDefinitionError(:"p_k should be Function or Number"))
    elseif length(pFunctions) != length(symPFunctions)
        throw(StructDefinitionError(:"Number of p_k and symP_k do not match"))
    elseif interval != symL.interval
        throw(StructDefinitionError(:"Intervals of L and symL do not match"))
    elseif !all(i -> check_func_sym_equal(pFunctions[i], symPFunctions[i], interval, t), 1:length(pFunctions))
        warn("symP_k does not agree with p_k on the interval") 
    else
        return true
    end
end

# 
struct VectorMultipointForm
    MM::Array
    NN::Array
    VectorMultipointForm(MM::Array, NN::Array) =
    try
        U = new(MM, NN)
        check_vectorMultipointForm_input(U)
        return U
    catch err
        throw(err)
    end
end

function check_vectorMultipointForm_input(U::VectorMultipointForm)
    # M, N = U.M, U.N
    # Avoid Inexact() error when taking rank()
    M = copy(U.MM)
    N = copy(U.NN)
    checker = Array{Bool}(undef, 1, length(M))
    for i = 1:length(M)
        M_i, N_i = M[i], N[i]
        if !(check_all(M_i, x -> isa(x, Number)) && check_all(N_i, x -> isa(x, Number)))
            throw(StructDefinitionError(:"Entries of M_i, N_i should be Number"))
        elseif size(M_i) != size(N_i)
            throw(StructDefinitionError(:"M_i, N_i dimensions do not match"))
        else
            checker[i] = true
        end
    end
    if LinearAlgebra.rank(array_hcat_helper(M, N)) != size(M[1])[1]
        throw(StructDefinitionError(:"Boundary operators are not linearly independent"))
    end
    for x in checker
        if !(x == true)
            return false
        end
    return true
    end
end

##########################################################################################################################################################
# Main Functions
function get_L(symL::SymLinearDifferentialOperator)
    symPFunctions, interval, t = symL.symPFunctions, symL.interval, symL.t
    if check_all(symPFunctions, x->!isa(x, SymPy.Sym))
        pFunctions = symPFunctions
    else
        pFunctions = sym_to_func.(symPFunctions)
    end
    L = LinearDifferentialOperator(pFunctions, interval, symL)
    return L
end

# 
function get_URank(U::VectorMultipointForm)
    M = U.MM
    N = U.NN
    checker = Array{Int64}(undef, 1, length(M))
    for i=1:length(M) 
        M_i = convert(Array{Complex}, M[i])
        N_i = convert(Array{Complex}, N[i])
        MHcatN = hcat(M_i, N_i)
        checker[i] = LinearAlgebra.rank(MHcatN)
    end
    if all(y->y == checker[1], checker) == true
        return checker[1]
    else
        throw(StructDefinitionError(:"Matrices M_i, N_i are not appropriate."))
    end
end

# Find Uc, a complementary form of U
function get_Uc(U::VectorMultipointForm)
    try
        check_vectorMultipointForm_input(U)
        M, N = copy(U.MM), copy(U.NN)
        P = copy(M)
        Q = copy(N)
        m = length(M)
        n = size(M[1])[2]
        I = complex(Matrix{Float64}(LinearAlgebra.I, 2n*m, 2n*m))
        H_top = array_hcat_helper(M, N)
        
        mat = convert(Array{Complex}, H_top) 
        for k = 1:(2n*m)
            newMat = vcat(mat, I[k:k,:])
            newMat = convert(Array{Complex}, newMat)
            if LinearAlgebra.rank(newMat) == LinearAlgebra.rank(mat) + 1
                mat = newMat
            end
        end
        H_bottom = mat[(size(M[1])[1]+1):2n*m, :]
        j = 0
        for i = 1:length(M)
            P[i] = H_bottom[:, (j*n+1):(j+1)*n]
            Q[i] = H_bottom[:, ((j+1)*n+1):(j+2)*n]
            j += 2
        end
        Uc = VectorMultipointForm(P, Q)
        return Uc
    catch err
        return err
    end
end

# Construct H from M, N, Mc, Nc
function get_H(U::VectorMultipointForm, Uc::VectorMultipointForm)
    M, N = copy(U.MM), copy(U.NN) # need to use copies of U and Uc, not the actual things
    P, Q = copy(Uc.MM), copy(Uc.NN)
    MHcatN = array_hcat_helper(M, N)
    McHcatNc = array_hcat_helper(P, Q)
    H = vcat(MHcatN, McHcatNc)
    return H
end

#
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


# 
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

# 
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

# Construct B_hat. Since all entries of B_hat are evaluated, BHat is a numeric matrix.
function get_BHat(L::LinearDifferentialOperator, B::Array)
    pFunctions, interval = L.pFunctions, L.interval
    k = length(interval) - 1
    BHats_Array = Array{Any}(undef, 1, k)
    n = length(pFunctions)-1
    for i=1:k
        BHat = Array{Float64, 2}(undef, 2n, 2n)
        BHat = convert(Array{Complex}, BHat)
        BEvalA = evaluate.(B, interval[i]) # B at x_{l-1} 
        BEvalB = evaluate.(B, interval[i+1]) # B at x_l
        BHat[1:n,1:n] = -BEvalA
        BHat[(n+1):(2n),(n+1):(2n)] = BEvalB
        BHat[1:n, (n+1):(2n)] = zeros(n, n)
        BHat[(n+1):(2n), 1:n] = zeros(n, n)
        BHats_Array[i] = BHat
    end
    BBHat = complex(Matrix{Float64}(LinearAlgebra.I, 2k*n, 2k*n))
    j = 0
    for i=1:k
        BBHat[(j*n+1):(j+2)*n, (j*n+1):(j+2)*n] = BHats_Array[i]
        j += 2
    end
    return BBHat
end

#
function get_J(BHat, H)
    J = (BHat * inv(H))'
    return J
end

# 
function get_adjoint_Candidate(J, n, k)
    PStar = Array{Any}(undef, 1, k)
    QStar = Array{Any}(undef, 1, k)
    J = J[(n*k+1):2n*k, :]
    j = 0
    for i=1:k
        PStar[i] = J[:, (j*n+1):(j+1)*n]
        QStar[i] = J[:, ((j+1)*n+1):(j+2)*n]
        j += 2
    end
    adjointU = VectorMultipointForm(PStar, QStar)
    return adjointU
end

# 
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


# 
function get_Ux(L::LinearDifferentialOperator, U::VectorMultipointForm, xSym)
    interval = L.interval
    n = length(L.pFunctions)-1 # order of the problem
    k = length(interval)-1 # number of points minus 1
    xi = get_xi(L; symbolic = false, xSym = xSym)
    summand = zeros(n*k, 1)
    M, N = copy(U.MM), copy(U.NN)
    for i=1:k
        xiEvalA = evaluate.(xi, interval[i])
        xiEvalB = evaluate.(xi, interval[i+1])
        Ux = M[i]*xiEvalA + N[i]*xiEvalB
        summand = Ux + summand 
    end
    return summand
end

# 
function check_adjoint(L::LinearDifferentialOperator, U::VectorMultipointForm, adjointU::VectorMultipointForm, B::Array)
    interval = L.interval
    n = length(L.pFunctions) - 1
    k = length(interval) - 1
    M, N = copy(U.MM), copy(U.NN)
    P, Q = (copy(adjointU.MM)), (copy(adjointU.NN))
    checker = Array{Bool}(undef, 1, k)
    left = convert(Array{Complex}, Array{Float64, 2}(undef, n*k, n*k))
    right = convert(Array{Complex}, Array{Float64, 2}(undef, n*k, n*k))
    for i=1:k
        BEvalA = convert(Array{Complex}, evaluate.(B, interval[i]))
        BEvalB = convert(Array{Complex}, evaluate.(B, interval[i+1]))
    
        left += M[i] * inv(BEvalA) * (P[i])'
        right += N[i] * inv(BEvalB) * (Q[i])'
    end
    tol = set_tol(left, right)
    return checker_matrix(left, right, tol) 
end

#
function get_adjointU(L::LinearDifferentialOperator, U::VectorMultipointForm, pDerivMatrix=get_pDerivMatrix(L))
    B = get_B(L; pDerivMatrix = pDerivMatrix)
    BHat_Arr = get_BHat(L, B)
    Uc = get_Uc(U)
    H_Arr = get_H(U, Uc)
    J_Arr = get_J(BHat_Arr, H_Arr)
    adjointU = get_adjoint_Candidate(J_Arr, length(L.pFunctions) -1, length(L.interval) - 1)
    return adjointU
    if check_adjoint(L, U, adjointU, B)
        return adjointU
    else
        throw(error("Adjoint found not valid"))
    end
end
