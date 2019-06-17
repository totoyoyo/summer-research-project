##########################################################################################################################################################
# Date created: June 12, 2019
# Name: Sultan Aitzhan
# Description: Unit tests for construct_adjoint.jl.
##########################################################################################################################################################
# Importing packages and modules
##########################################################################################################################################################
# upload the .jl file with the code that constructs the adjoint
include("/Users/ww/Desktop/SRP/capstone-master/work_in_julia_v1.1.1/Construct_Multipoint_Adjoint_v1.jl")
# using construct_adjoint
##########################################################################################################################################################
# Helper functions
##########################################################################################################################################################
function generate_symPFunctions(n; random = false, constant = false)
    global t = symbols("t")
    if random
        symPFunctions = Array{Number}(undef, 1,n+1)
        for i = 1:(n+1)
            seed = rand(0:1)
            if seed == 0 # constant
                symPFunctionRe = rand(Uniform(1.0,10.0), 1, 1)[1]
                symPFunctionIm = rand(Uniform(1.0,10.0), 1, 1)[1]
                symPFunction = symPFunctionRe + symPFunctionIm*im
            else # variable
                coeffsNo = rand(1:5)
                pFunctionCoeffsRe = rand(Uniform(1.0,10.0), 1, coeffsNo)
                pFunctionCoeffsIm = rand(Uniform(1.0,10.0), 1,  coeffsNo)
                pFunctionCoeffs = pFunctionCoeffsRe + pFunctionCoeffsIm*im
                symPFunction = sum([pFunctionCoeffs[i+1]*t^(length(pFunctionCoeffs)-1-i) for i in 0:(length(pFunctionCoeffs)-1)])
            end
            symPFunctions[i] = symPFunction
        end
    else
        if constant # constant
            symPFunctionsRe = rand(Uniform(1.0,10.0), 1, (n+1))
            symPFunctionsIm = rand(Uniform(1.0,10.0), 1, (n+1))
            symPFunctions = symPFunctionsRe + symPFunctionsIm*im
        else # variable
            symPFunctions = Array{Number}(undef, 1,n+1)
            for i = 1:(n+1)
                # Each p_k is a polynomial function with random degree between 0 to 4 and random coefficients between 0 and 10
                coeffsNo = rand(1:5)
                pFunctionCoeffsRe = rand(Uniform(1.0,10.0), 1, coeffsNo)
                pFunctionCoeffsIm = rand(Uniform(1.0,10.0), 1,  coeffsNo)
                pFunctionCoeffs = pFunctionCoeffsRe + pFunctionCoeffsIm*im
                symPFunction = sum([pFunctionCoeffs[i+1]*t^(length(pFunctionCoeffs)-1-i) for i in 0:(length(pFunctionCoeffs)-1)])
                symPFunctions[i] = symPFunction
            end
        end
    end
    return symPFunctions
end 

function generate_pFunctions(n; random = false, constant = false)
    if random
        pFunctions = Array{Union{Function, Number}}(undef,1,n+1)
        pDerivMatrix = Array{Union{Function, Number}}(undef, n,n)
        for i = 1:(n+1)
            seed = rand(0:1)
            if seed == 0 # constant
                pFunctionRe = rand(Uniform(1.0,10.0), 1, 1)[1]
                pFunctionIm = rand(Uniform(1.0,10.0), 1, 1)[1]
                pFunction = pFunctionRe + pFunctionIm*im
                if i < n+1
                    pDerivMatrix[i,1] = pFunction
                    pDerivMatrix[i:i, 2:n] = zeros(1, n-1)
                end
            else # variable
                coeffsNo = rand(1:5)
                pFunctionCoeffsRe = rand(Uniform(1.0,10.0), 1, coeffsNo)
                pFunctionCoeffsIm = rand(Uniform(1.0,10.0), 1, coeffsNo)
                pFunctionCoeffs = pFunctionCoeffsRe + pFunctionCoeffsIm*im
                pFunction = get_polynomial(pFunctionCoeffs)
                if i < n+1
                    pDerivMatrix[i:i, 1:n] = [get_polynomialDeriv(pFunctionCoeffs, k) for k = 0:(n-1)]
                end
            end
            pFunctions[i] = pFunction
        end
    else
        if constant # constant
            pFunctionsRe = rand(Uniform(1.0,10.0), 1, (n+1))
            pFunctionsIm = rand(Uniform(1.0,10.0), 1, (n+1))
            pFunctions = pFunctionsRe + pFunctionsIm*im
            pDerivMatrix = complex(Matrix{Float64}(LinearAlgebra.I, n, n))
            for i = 1:n
                for j = 1:n
                    if j == 1
                        pDerivMatrix[i,j] = pFunctions[i]
                    else
                        pDerivMatrix[i,j] = 0
                    end
                end
            end
        else # variable
            pFunctions = Array{Union{Function, Number}}(undef,1,n+1)
            pDerivMatrix = Array{Union{Function, Number}}(undef,n,n)
            for i = 1:(n+1)
                # Each p_k is a polynomial function with random degree between 0 to 4 and random coefficients between 0 and 10
                coeffsNo = rand(1:5)
                pFunctionCoeffsRe = rand(Uniform(1.0,10.0), 1, coeffsNo)
                pFunctionCoeffsIm = rand(Uniform(1.0,10.0), 1, coeffsNo)
                pFunctionCoeffs = pFunctionCoeffsRe + pFunctionCoeffsIm*im
                if i < n+1
                    pDerivMatrix[i:i, 1:n] = [get_polynomialDeriv(pFunctionCoeffs, k) for k = 0:(n-1)]
                end
                pFunction = get_polynomial(pFunctionCoeffs)
                pFunctions[i] = pFunction
            end
        end
    end
    return pFunctions, pDerivMatrix
end

function generate_interval(n, a = 0, b = 1)
    if n == 1 || n == 2
        n = 3
    end
    array = Array{Any}(undef, 1, n)
    t = a
    for j=2:(n-1)
        array[j] = rand(Uniform(t, b))
        t = array[j]
    end
    array[1] = a
    array[n] = b
    array = prettyRound.(array; digs = 5)
    interval = ntuple(i -> array[i], n)
    return interval
end

function generate_fake_interval_1(n, a = 0, b = 1)
    if n == 1 || n == 2
        n = 3
    end
    array = Array{Any}(undef, 1, n)
    t = 0
    for j=2:(n-1)
        array[j] = rand()
        t = array[j]
    end
    array[1] = a
    array[n] = "str"
    interval = ntuple(i -> array[i], n)
    return interval
end

function generate_fake_interval_2(n, a = 0, b = 1)
    if n == 1 || n == 2
        n = 4
    end
    array = Array{Any}(undef, 1, n)
    t = 0
    for j=2:(n-1)
        array[n-j+1] = rand(Uniform(t, b))
        t = array[n-j+1]
    end
    array[1] = a
    array[n] = b
    interval = ntuple(i -> array[i], n)
    return interval
end

function generate_pFunctionsAndSymPFunctions(n; random = false, constant = false)
    global t = symbols("t")
    interval = generate_interval(n,)
    if random
        pFunctions = Array{Union{Function, Number}}(undef,1,n+1)
        symPFunctions = Array{Number}(undef, 1,n+1)
        pDerivMatrix = Array{Union{Function, Number}}(undef,n,n)
        for i = 1:(n+1)
            seed = rand(0:1)
            if seed == 0 # constant
                pFunctionRe = rand(Uniform(1.0,10.0), 1, 1)[1]
                pFunctionIm = rand(Uniform(1.0,10.0), 1, 1)[1]
                pFunction = pFunctionRe + pFunctionIm*im
                symPFunction = pFunction
                if i < n+1
                    pDerivMatrix[i,1] = pFunction
                    pDerivMatrix[i:i, 2:n] = zeros(1, n-1)
                end
            else # variable
                coeffsNo = rand(1:5)
                pFunctionCoeffsRe = rand(Uniform(1.0,10.0), 1, coeffsNo)
                pFunctionCoeffsIm = rand(Uniform(1.0,10.0), 1, coeffsNo)
                pFunctionCoeffs = pFunctionCoeffsRe + pFunctionCoeffsIm*im
                if i < n+1
                    pDerivMatrix[i:i, 1:n] = [get_polynomialDeriv(pFunctionCoeffs, k) for k = 0:(n-1)]
                end
                pFunction = get_polynomial(pFunctionCoeffs)
                symPFunction = sum([pFunctionCoeffs[i+1]*t^(length(pFunctionCoeffs)-1-i) for i in 0:(length(pFunctionCoeffs)-1)])
            end
            pFunctions[i] = pFunction
            symPFunctions[i] = symPFunction
        end
    else
        if constant # constant
            pFunctionsRe = rand(Uniform(1.0,10.0), 1, (n+1))
            pFunctionsIm = rand(Uniform(1.0,10.0), 1, (n+1))
            pFunctions = pFunctionsRe + pFunctionsIm*im
            symPFunctions = pFunctions
            pDerivMatrix = Array{Union{Function, Number}}(undef, n, n)
            for i = 1:n
                for j = 1:n
                    if j == 1
                        pDerivMatrix[i,j] = pFunctions[i]
                    else
                        pDerivMatrix[i,j] = 0
                    end
                end
            end
        else # variable
            t = symbols("t")
            pFunctions = Array{Function}(undef, 1, n+1)
            symPFunctions = Array{Number}(undef, 1, n+1)
            pDerivMatrix = Array{Union{Function, Number}}(undef, n,n)
            for i = 1:(n+1)
                # Each p_k is a polynomial function with random degree between 0 to 4 and random coefficients between 0 and 10
                coeffsNo = rand(1:5)
                pFunctionCoeffsRe = rand(Uniform(1.0,10.0), 1, coeffsNo)
                pFunctionCoeffsIm = rand(Uniform(1.0,10.0), 1, coeffsNo)
                pFunctionCoeffs = pFunctionCoeffsRe + pFunctionCoeffsIm*im
                if i < n+1
                    pDerivMatrix[i:i, 1:n] = [get_polynomialDeriv(pFunctionCoeffs, k) for k = 0:(n-1)]
                end
                pFunction = get_polynomial(pFunctionCoeffs)
                pFunctions[i] = pFunction
                symPFunction = sum([pFunctionCoeffs[i+1]*t^(length(pFunctionCoeffs)-1-i) for i in 0:(length(pFunctionCoeffs)-1)])
                symPFunctions[i] = symPFunction
            end
        end
    end
    return pFunctions, symPFunctions, pDerivMatrix, interval
end

function rank_def(n)
    U = rand(Uniform(1.0,10.0), n, n)
    V = rand(Uniform(1.0,10.0), n, n)
    Fu = LinearAlgebra.qr(U)
    Fv = LinearAlgebra.qr(V)
    
    diags = rand(Uniform(1.0,10.0), 1, n-2)
    diag_vals = Array{Number}(undef, 1, n)
    for i=1:(n-2)
        diag_vals[i] = diags[i]
    end
    diag_vals[n], diag_vals[n-1] = 0, 0
    S = zeros(n,n)
    for i=1:n
        S[i,i] = diag_vals[i]
    end
    return Fu.Q*S*Fv.Q
end
               
#############################################################################
# Tests
#############################################################################
# Test the algorithm to generate valid adjoint U+
# Test the algorithm to generate valid adjoint U+
function test_generate_adjoint(n, k)
    global results = [true]
    global t = symbols("t")

    for counter = 1:k
        println("Test $counter")
        println("Testing the algorithm to generate valid adjoint U+: Constant p_k")
        (pFunctions, symPFunctions, pDerivMatrix, interval) = generate_pFunctionsAndSymPFunctions(n; random = false, constant = true)
        symL = SymLinearDifferentialOperator(symPFunctions, interval, t)
        L = LinearDifferentialOperator(pFunctions, interval, symL)
        m = length(interval) - 1
        M = Array{Any}(undef, 1, m)
        N = Array{Any}(undef, 1, m)
        for i=1:m
            MCandRe = rand(Uniform(1.0,10.0), n, n)
            MCandIm = rand(Uniform(1.0,10.0), n, n)
            MCand = MCandRe + MCandIm*im
            NCandRe = rand(Uniform(1.0,10.0), n, n)
            NCandIm = rand(Uniform(1.0,10.0), n, n)
            NCand = NCandRe + NCandIm*im
            M[i] = MCand
            N[i] = NCand
        end
        U = VectorMultiBoundaryForm(M, N)
        println("Testing: order of L = $n")
        passed = false
        try
            adjoint = get_adjointU(L, U, pDerivMatrix)
            passed = true
            append!(results, passed)
        catch err
            println("Failed with $err")
        end
        if passed
            println("Passed!")
        end

        println("Testing the algorithm to generate valid adjoint U+: Variable p_k")
        # Generate variable p_k
        (pFunctions, symPFunctions, pDerivMatrix, interval) = generate_pFunctionsAndSymPFunctions(n; random = false, constant = false)
        symL = SymLinearDifferentialOperator(symPFunctions, interval, t)
        L = LinearDifferentialOperator(pFunctions, interval, symL)
        m = length(interval) - 1
        M = Array{Any}(undef, 1, m)
        N = Array{Any}(undef, 1, m)
        for i=1:m
            MCandRe = rand(Uniform(1.0,10.0), n, n)
            MCandIm = rand(Uniform(1.0,10.0), n, n)
            MCand = MCandRe + MCandIm*im
            NCandRe = rand(Uniform(1.0,10.0), n, n)
            NCandIm = rand(Uniform(1.0,10.0), n, n)
            NCand = NCandRe + NCandIm*im
            M[i] = MCand
            N[i] = NCand
        end
        U = VectorMultiBoundaryForm(M, N)
        println("Testing: order of L = $n")
        try
            adjoint = get_adjointU(L, U, pDerivMatrix)
            passed = true
            append!(results, passed)
        catch err
            println("Failed with $err")
        end
        if passed
            println("Passed!")
        end

        println("Testing the algorithm to generate valid adjoint U+: Constant or variable p_k")
        # Generate p_k
        (pFunctions, symPFunctions, pDerivMatrix, interval) = generate_pFunctionsAndSymPFunctions(n; random = true)
        symL = SymLinearDifferentialOperator(symPFunctions, interval, t)
        L = LinearDifferentialOperator(pFunctions, interval, symL)
        m = length(interval) - 1
        M = Array{Any}(undef, 1, m)
        N = Array{Any}(undef, 1, m)
        for i=1:m
            MCandRe = rand(Uniform(1.0,10.0), n, n)
            MCandIm = rand(Uniform(1.0,10.0), n, n)
            MCand = MCandRe + MCandIm*im
            NCandRe = rand(Uniform(1.0,10.0), n, n)
            NCandIm = rand(Uniform(1.0,10.0), n, n)
            NCand = NCandRe + NCandIm*im
            M[i] = MCand
            N[i] = NCand
        end
        U = VectorMultiBoundaryForm(M, N)
        println("Testing: order of L = $n")
        try
            adjoint = get_adjointU(L, U, pDerivMatrix)
            passed = true
            append!(results, passed)
        catch err
            println("Failed with $err")
        end
        if passed
            println("Passed!")
        end
    end

    return all(results)
end

for n = 1:10
    println(test_generate_adjoint(n, 10))
end

# Test the SymLinearDifferentialOperator definition
function test_symLinearDifferentialOperatorDef(n, k)
    global results = [true]
    global t = symbols("t")

    for counter = 1:k
        println("Test $counter")
        println("Testing definition of SymLinearDifferentialOperator: symP_k are Function")
        symPFunctions = generate_symPFunctions(n; random = false, constant = false)
        interval_1 = generate_interval(n)
        passed = false
        try
            SymLinearDifferentialOperator(symPFunctions, interval_1, t)
            passed = true
        catch err
            println("Failed with $err")
        end
        if passed
            println("Passed!")
        end
        append!(results, passed)

        println("Testing definition of SymLinearDifferentialOperator: symP_k are constant")
        symPFunctions = generate_symPFunctions(n; random = false, constant = true)
        interval_2 = generate_interval(n)
        passed = false
        try
            SymLinearDifferentialOperator(symPFunctions, interval_2, t)
            passed = true
        catch err
            println("Failed with $err")
        end
        if passed
            println("Passed!")
        end
        append!(results, passed)

        println("Testing definition of SymLinearDifferentialOperator: symP_k are SymPy.Sym and Number")
        symPFunctions = generate_symPFunctions(n; random = true)
        interval_3 = generate_interval(n)
        passed = false
        try
            SymLinearDifferentialOperator(symPFunctions, interval_3, t)
            passed = true
        catch err
            println("Failed with $err")
        end
        if passed
            println("Passed!")
        end
        append!(results, passed)

        println("Testing StructDefinitionError: symP_k should be SymPy.Sym or Number")
        symPFunctions = hcat(generate_symPFunctions(n-1; random = true), ["str"])
        interval_4 = generate_interval(n)
        passed = false
        try
            SymLinearDifferentialOperator(symPFunctions, interval_4, t)
        catch err
            if isa(err,StructDefinitionError) && err.msg == "symP_k should be SymPy.Sym or Number"
                passed = true
                println("Passed!")
            else
                println("Failed with $err")
            end
        end
        if !passed
            println("Failed!")
        end
        append!(results, passed)

        println("Testing StructDefinitionError: Only one free symbol is allowed in symP_k")
        interval_5 = generate_interval(n)
        r = symbols("r")
        passed = false
        try
            SymLinearDifferentialOperator([t+1 t+1 r*t+1], interval_5, t)
        catch err
            if isa(err,StructDefinitionError) && err.msg == "Only one free symbol is allowed in symP_k"
                passed = true
                println("Passed!")
            else
                println("Failed with $err")
            end
        end
        if !passed
            println("Failed!")
        end
        append!(results, passed)
            
        println("Testing StructDefinitionError: Interval must consist of numbers")
        symPFunctions = generate_symPFunctions(n; random = false, constant = false)
        interval_6 = generate_fake_interval_1(n)
        passed = false
        try
            SymLinearDifferentialOperator(symPFunctions, interval_6, t)
        catch err
            if isa(err,StructDefinitionError) && err.msg == "Interval must consist of numbers"
                passed = true
                println("Passed!")
            else
                println("Failed with $err")
            end
        end
        if !passed
            println("Failed!")
        end
        append!(results, passed)
    
        println("Testing StructDefinitionError: Terms in interval must be strictly increasing")
        symPFunctions = generate_symPFunctions(n; random = false, constant = false)
        interval_7 = generate_fake_interval_2(n)
        passed = false
        try
            SymLinearDifferentialOperator(symPFunctions, interval_7, t)
        catch err
            if isa(err,StructDefinitionError) && err.msg == "Terms in interval must be strictly increasing"
                passed = true
                println("Passed!")
            else
                println("Failed with $err")
            end
        end
        if !passed
            println("Failed!")
        end
        append!(results, passed)
    end
    return all(results)
end

for n = 1:10
    println(test_symLinearDifferentialOperatorDef(n, 10))
end

# Test the LinearDifferentialOperator definition
function test_LinearDifferentialOperatorDef(n, k)
    global results = [true]
    global t = symbols("t")
    
    for counter = 1:k
        println("Test $counter")

        # Variable p_k
        println("Testing definition of LinearDifferentialOperator: p_k are Function")
        (pFunctions, symPFunctions, pDerivMatrix, interval_1) = generate_pFunctionsAndSymPFunctions(n; random = false, constant = false)
        symL = SymLinearDifferentialOperator(symPFunctions, interval_1, t)
        passed = false
        try
            L = LinearDifferentialOperator(pFunctions, interval_1, symL)
            passed = true
        catch err
            println("Failed with $err")
        end
        if passed
            println("Passed!")
        end
        append!(results, passed)

        # Constant coefficients
        println("Testing definition of LinearDifferentialOperator: p_k are Constants")
        (pFunctions, symPFunctions, pDerivMatrix, interval_2) = generate_pFunctionsAndSymPFunctions(n; random = false, constant = true)
        symL = SymLinearDifferentialOperator(symPFunctions, interval_2, t)
        passed = false
        try
            LinearDifferentialOperator(pFunctions, interval_2, symL)
            passed = true
        catch err
            println("Failed with $err")
        end
        if passed
            println("Passed!")
        end
        append!(results, passed)

        # Mixed coefficients
        println("Testing definition of LinearDifferentialOperator: p_k are mixed")
        # (pFunctions, symPFunctions, pDerivMatrix) = generate_pFunctionsAndSymPFunctions(n; random = true)
        symL = SymLinearDifferentialOperator([1 1 t+1], interval_2, t)
        passed = false
        try
            LinearDifferentialOperator([1 t->1 t->t+1], interval_2, symL)
            passed = true
        catch err
            println("Failed with $err")
        end
        if passed
            println("Passed!")
        end
        append!(results, passed)

        println("Testing StructDefinitionError: p_k should be Function or Number")
        pFunctions = hcat(generate_pFunctions(n-1; random = true)[1], ["str"])
        passed = false
        try
            LinearDifferentialOperator(['s' 1 1], interval_2, symL)
        catch err
            if err.msg == "p_k should be Function or Number" && (isa(err,StructDefinitionError))
                passed = true
                println("Passed!")
            else
                println("Failed with $err")
            end
        end
        if !passed
            println("Failed!")
        end
        append!(results, passed)

        println("Testing StructDefinitionError: Number of p_k and symP_k do not match")
        symL = SymLinearDifferentialOperator([1 1 t+1], interval_2, t)
        passed = false
        try
            LinearDifferentialOperator([1 t->1], interval_2, symL)
        catch err
            if err.msg == "Number of p_k and symP_k do not match" && (isa(err, StructDefinitionError))
                passed = true
                println("Passed!")
            else
                println("Failed with $err")
            end
        end
        if !passed
            println("Failed!")
        end
        append!(results, passed)

        println("Testing StructDefinitionError: Intervals of L and symL do not match")
        symL = SymLinearDifferentialOperator([1 1 t+1], interval_2, t)
        interval_3 = generate_interval(n)
        passed = false
        try
            LinearDifferentialOperator([1 t->1 t->t+1], interval_3, symL)
        catch err
            if err.msg == "Intervals of L and symL do not match" && (isa(err, StructDefinitionError))
                passed = true
                println("Passed!")
            else
                println("Failed with $err")
            end
        end
        if !passed
            println("Failed!")
        end
        append!(results, passed)
    
    end
    return all(results)
end

for n = 1:10
    println(test_LinearDifferentialOperatorDef(n, 10))
end

# Test the VectorBoundaryForm definition
function test_VectorMultiBoundaryFormDef(n, k)
    global results = [true]
    if n == 1
        n = 2
    end
    for counter = 1:k
        println("Test $counter")
        
        println("Testing the definition of VectorBoundaryForm")
        interval_1 = generate_interval(n)
        m = length(interval_1) - 1
        M = Array{Any}(undef, 1, m)
        N = Array{Any}(undef, 1, m)
        for j=1:m
            MRe = rand(Uniform(1.0,10.0), n, n)
            MIm = rand(Uniform(1.0,10.0), n, n)
            M_j = MRe + MIm*im
            M[j] = M_j
            NRe = rand(Uniform(1.0,10.0), n, n)
            NIm = rand(Uniform(1.0,10.0), n, n)
            N_j = NRe + NIm*im
            N[j] = N_j
        end
        passed = false
        try
            VectorMultiBoundaryForm(M, N)
            passed = true
        catch err
            println("Failed with $err")
        end
        if passed
            println("Passed!")
        end
        append!(results, passed)

        println("Testing StructDefinitionError: Entries of M_i, N_i should be Number")
        interval_2 = generate_interval(n)
        m = length(interval_2) - 1
        M = Array{Any}(undef, 1, m)
        N = Array{Any}(undef, 1, m)
        for j=1:m
            MRe = rand(Uniform(1.0,10.0), n, n)
            MIm = rand(Uniform(1.0,10.0), n, n)
            M_j = MRe + MIm*im
            M_j = convert(Array{Any}, M_j)
            M_j[j,j] = "str"
            M[j] = M_j
            NRe = rand(Uniform(1.0,10.0), n, n)
            NIm = rand(Uniform(1.0,10.0), n, n)
            N_j = NRe + NIm*im
            N_j = convert(Array{Any}, N_j)
            N_j[j,j] = "str"
            N[j] = N_j
        end
        passed = false
        try
            VectorMultiBoundaryForm(M, N)
        catch err
            if err.msg == "Entries of M_i, N_i should be Number" && isa(err, StructDefinitionError)
                passed = true
                println("Passed!")
            else
                println("Failed with $err")
            end
        end
        if !passed
            println("Failed!")
        end
        append!(results, passed)

        println("Testing StructDefinitionError: M_i, N_i dimensions do not match")
        interval_3 = generate_interval(n)
        m = length(interval_3) - 1
        M = Array{Any}(undef, 1, m)
        N = Array{Any}(undef, 1, m)
        for j=1:m
            MRe = rand(Uniform(1.0,10.0), n, n-1)
            MIm = rand(Uniform(1.0,10.0), n, n-1)
            M_j = MRe + MIm*im
            M[j] = M_j
            NRe = rand(Uniform(1.0,10.0), n, n)
            NIm = rand(Uniform(1.0,10.0), n, n)
            N_j = NRe + NIm*im
            N[j] = N_j
        end
        passed = false
        try
            VectorMultiBoundaryForm(M, N)
        catch err
            if err.msg == "M_i, N_i dimensions do not match" && isa(err,StructDefinitionError)
                passed = true
                println("Passed!")
            else
                println("Failed with $err")
            end
        end
        if !passed
            println("Failed!")
        end
        append!(results, passed)

        println("Testing StructDefinitionError: M_i, N_i should be square matrices")
        interval_4 = generate_interval(n)
        m = length(interval_4) - 1
        M = Array{Any}(undef, 1, m)
        N = Array{Any}(undef, 1, m)
        for j=1:m
            MRe = rand(Uniform(1.0,10.0), n, n-1)
            MIm = rand(Uniform(1.0,10.0), n, n-1)
            M_j = MRe + MIm*im
            M[j] = M_j
            NRe = rand(Uniform(1.0,10.0), n, n-1)
            NIm = rand(Uniform(1.0,10.0), n, n-1)
            N_j = NRe + NIm*im
            N[j] = N_j
        end
        passed = false
        try
            VectorMultiBoundaryForm(M, N)
        catch err
            if err.msg == "M_i, N_i should be square matrices" && isa(err,StructDefinitionError)
                passed = true
                println("Passed!")
            else
                println("Failed with $err")
            end
        end
        if !passed
            println("Failed!")
        end
        append!(results, passed)

        println("Testing StructDefinitionError: Boundary operators not linearly independent")
        
        interval_5 = generate_interval(n)
        m = length(interval_5) - 1
        M = Array{Any}(undef, 1, m)
        N = Array{Any}(undef, 1, m)
        for j=1:m
            M[j] = rank_def(n)
            N[j] = M[j]
        end
        passed = false
        try
            VectorMultiBoundaryForm(M, N)
        catch err
            if err.msg == "Boundary operators not linearly independent" && isa(err,StructDefinitionError)
                passed = true
                    println("Passed!")
                else
                    println("Failed with $err")
                end
            end
        if !passed
            println("Failed!")
        end
        append!(results, passed)
    end

    return all(results)
end

for n = 1:10
    println(test_VectorMultiBoundaryFormDef(n, 10))
end