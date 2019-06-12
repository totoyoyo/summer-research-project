##########################################################################################################################################################
# Date created: June 12, 2019
# Name: Sultan Aitzhan
# Description: Unit tests for construct_adjoint.jl.
##########################################################################################################################################################
# Importing packages and modules
##########################################################################################################################################################
# upload the .jl file with the code that constructs the adjoint
include("/Users/ww/Desktop/SRP/capstone-master/work_in_julia_v1.1.1/Construct_Adjoint_v1.jl")
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

function generate_pFunctionsAndSymPFunctions(n; random = false, constant = false)
    global t = symbols("t")
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
    return pFunctions, symPFunctions, pDerivMatrix
end
               
#############################################################################
# Tests
#############################################################################
# Test the algorithm to generate valid adjoint U+
function test_generate_adjoint(n, k)
    global results = [true]
    global t = symbols("t")
    global (a,b) = (0,1)

    for counter = 1:k
        println("Test $counter")
        println("Testing the algorithm to generate valid adjoint U+: Constant p_k")
        (pFunctions, symPFunctions, pDerivMatrix) = generate_pFunctionsAndSymPFunctions(n; random = false, constant = true)
        symL = SymLinearDifferentialOperator(symPFunctions, (a,b), t)
        L = LinearDifferentialOperator(pFunctions, (a,b), symL)
        MCandRe = rand(Uniform(1.0,10.0), n, n)
        MCandIm = rand(Uniform(1.0,10.0), n, n)
        MCand = MCandRe + MCandIm*im
        NCandRe = rand(Uniform(1.0,10.0), n, n)
        NCandIm = rand(Uniform(1.0,10.0), n, n)
        NCand = NCandRe + NCandIm*im
        U = VectorBoundaryForm(MCand, NCand)
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
        (pFunctions, symPFunctions, pDerivMatrix) = generate_pFunctionsAndSymPFunctions(n; random = false, constant = false)
        symL = SymLinearDifferentialOperator(symPFunctions, (a,b), t)
        L = LinearDifferentialOperator(pFunctions, (a,b), symL)
        MCandRe = rand(Uniform(1.0,10.0), n, n)
        MCandIm = rand(Uniform(1.0,10.0), n, n)
        MCand = MCandRe + MCandIm*im
        NCandRe = rand(Uniform(1.0,10.0), n, n)
        NCandIm = rand(Uniform(1.0,10.0), n, n)
        NCand = NCandRe + NCandIm*im
        U = VectorBoundaryForm(MCand, NCand)
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
        (pFunctions, symPFunctions, pDerivMatrix) = generate_pFunctionsAndSymPFunctions(n; random = true)
        symL = SymLinearDifferentialOperator(symPFunctions, (a,b), t)
        L = LinearDifferentialOperator(pFunctions, (a,b), symL)
        MCandRe = rand(Uniform(1.0,10.0), n, n)
        MCandIm = rand(Uniform(1.0,10.0), n, n)
        MCand = MCandRe + MCandIm*im
        NCandRe = rand(Uniform(1.0,10.0), n, n)
        NCandIm = rand(Uniform(1.0,10.0), n, n)
        NCand = NCandRe + NCandIm*im
        U = VectorBoundaryForm(MCand, NCand)
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
    global (a,b) = (0,1)

    for counter = 1:k
        println("Test $counter")
        println("Testing definition of SymLinearDifferentialOperator: symP_k are Function")
        symPFunctions = generate_symPFunctions(n; random = false, constant = false)
        passed = false
        try
            SymLinearDifferentialOperator(symPFunctions, (a,b), t)
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
        passed = false
        try
            SymLinearDifferentialOperator(symPFunctions, (a,b), t)
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
        passed = false
        try
            SymLinearDifferentialOperator(symPFunctions, (a,b), t)
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
        passed = false
        try
            SymLinearDifferentialOperator(symPFunctions, (a,b), t)
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
        r = symbols("r")
        passed = false
        try
            SymLinearDifferentialOperator([t+1 t+1 r*t+1], (a,b), t)
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
    global (a,b) = (0,1)
    
    for counter = 1:k
        println("Test $counter")

        # Variable p_k
        println("Testing definition of LinearDifferentialOperator: p_k are Function")
        (pFunctions, symPFunctions, pDerivMatrix) = generate_pFunctionsAndSymPFunctions(n; random = false, constant = false)
        symL = SymLinearDifferentialOperator(symPFunctions, (a,b), t)
        passed = false
        try
            L = LinearDifferentialOperator(pFunctions, (a,b), symL)
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
        (pFunctions, symPFunctions, pDerivMatrix) = generate_pFunctionsAndSymPFunctions(n; random = false, constant = true)
        symL = SymLinearDifferentialOperator(symPFunctions, (a,b), t)
        passed = false
        try
            LinearDifferentialOperator(pFunctions, (a,b), symL)
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
        symL = SymLinearDifferentialOperator([1 1 t+1], (a,b), t)
        passed = false
        try
            LinearDifferentialOperator([1 t->1 t->t+1], (a,b), symL)
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
            LinearDifferentialOperator(['s' 1 1], (a,b), symL)
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
        symL = SymLinearDifferentialOperator([1 1 t+1], (a,b), t)
        passed = false
        try
            LinearDifferentialOperator([1 t->1], (a,b), symL)
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
        symL = SymLinearDifferentialOperator([1 1 t+1], (a,b), t)
        passed = false
        try
            LinearDifferentialOperator([1 t->1 t->t+1], (-b,a), symL)
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
function test_VectorBoundaryFormDef(n, k)
    global results = [true]

    for counter = 1:k
        println("Test $counter")

        println("Testing the definition of VectorBoundaryForm")
        MRe = rand(Uniform(1.0,10.0), n, n)
        MIm = rand(Uniform(1.0,10.0), n, n)
        M = MRe + MIm*im
        NRe = rand(Uniform(1.0,10.0), n, n)
        NIm = rand(Uniform(1.0,10.0), n, n)
        N = NRe + NIm*im
        passed = false
        try
            VectorBoundaryForm(M, N)
            passed = true
        catch err
            println("Failed with $err")
        end
        if passed
            println("Passed!")
        end
        append!(results, passed)

        println("Testing StructDefinitionError: Entries of M, N should be Number")
        M = ['a' 2; 3 4]
        N = M
        passed = false
        try
            VectorBoundaryForm(M, N)
        catch err
            if err.msg == "Entries of M, N should be Number" && isa(err, StructDefinitionError)
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

        println("Testing StructDefinitionError: M, N dimensions do not match")
        M = rand(Uniform(1.0,10.0), n, n-1)
        N = rand(Uniform(1.0,10.0), n, n)
        passed = false
        try
            VectorBoundaryForm(M, N)
        catch err
            if err.msg == "M, N dimensions do not match" && isa(err,StructDefinitionError)
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

        println("Testing StructDefinitionError: M, N should be square matrices")
        M = rand(Uniform(1.0,10.0), n, n-1)
        N = rand(Uniform(1.0,10.0), n, n-1)
        passed = false
        try
            VectorBoundaryForm(M, N)
        catch err
            if err.msg == "M, N should be square matrices" && isa(err,StructDefinitionError)
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
        M = [1 2*im; 2 4*im]
        N = [3 4*im; 6 8*im]
        passed = false
        try
            VectorBoundaryForm(M, N)
        catch err
            if err.msg == "Boundary operators not linearly independent" && isa(err,StructDefinitionError)
                passed = true
                println("Passed!")
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
    println(test_VectorBoundaryFormDef(n, 10))
end