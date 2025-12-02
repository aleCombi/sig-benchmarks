#!/usr/bin/env julia
# ChenSignatures.jl adapter for signature benchmarks

using ChenSignatures
using StaticArrays
using JSON

# -------- Path Generators --------

"""Generate linear path: [t, 2t, 2t, ...]"""
function make_path_linear(d::Int, N::Int)
    ts = range(0.0, stop=1.0, length=N)
    [SVector{d,Float64}(ntuple(i -> (i == 1 ? t : 2t), d)) for t in ts]
end

"""Generate sinusoidal path: [sin(2π·1·t), sin(2π·2·t), ...]"""
function make_path_sin(d::Int, N::Int)
    ts = range(0.0, stop=1.0, length=N)
    ω = 2π
    [SVector{d,Float64}(ntuple(i -> sin(ω * i * t), d)) for t in ts]
end

"""Generate path of specified kind"""
function make_path(d::Int, N::Int, kind::String)
    kind_lower = lowercase(kind)
    if kind_lower == "linear"
        return make_path_linear(d, N)
    elseif kind_lower == "sin"
        return make_path_sin(d, N)
    else
        error("Unknown path_kind: $kind")
    end
end

# -------- Manual Timing Loop --------

"""
Execute manual timing loop with warmup and GC disabled.

This is the Julia equivalent of the Python BenchmarkAdapter.manual_timing_loop.

Returns a tuple of (avg_time_ms, avg_alloc_bytes):
    - avg_time_ms: Average time per iteration in milliseconds
    - avg_alloc_bytes: Average bytes allocated per iteration
"""
function manual_timing_loop(func::Function, repeats::Int; warmup_iterations::Int=3)
    # Warmup phase (untimed)
    for _ in 1:warmup_iterations
        func()
    end

    # Timed phase with GC disabled and allocation tracking
    GC.enable(false)
    local t0, t1, total_alloc_bytes
    try
        t0 = time_ns()

        # Track allocations for all iterations
        total_alloc_bytes = @allocated begin
            for _ in 1:repeats
                func()
            end
        end

        t1 = time_ns()
    finally
        GC.enable(true)
    end

    # Calculate average time in milliseconds
    total_time_ns = t1 - t0
    avg_time_ms = (total_time_ns / repeats) / 1e6

    # Calculate average bytes allocated per iteration
    avg_alloc_bytes = div(total_alloc_bytes, repeats)

    return avg_time_ms, avg_alloc_bytes
end

# -------- Benchmark Operations --------

"""
Prepare signature computation kernel.

Returns a closure that performs only the kernel (no setup).
"""
function run_signature(path, m::Int)
    # Setup phase (untimed): path is already prepared
    tensor_type = ChenSignatures.Tensor{Float64}

    # Return kernel closure
    return () -> signature_path(tensor_type, path, m)
end

"""
Prepare logsignature computation kernel.

Returns a closure that performs only the kernel (no setup).
"""
function run_logsignature(path, m::Int)
    # Setup phase (untimed): path is already prepared
    tensor_type = ChenSignatures.Tensor{Float64}

    # Return kernel closure
    return () -> ChenSignatures.log(signature_path(tensor_type, path, m))
end

# -------- Main Benchmark Runner --------

"""Format benchmark result for JSON output"""
function output_result(N, d, m, path_kind, operation, t_ms, alloc_bytes, library, method, path_type)
    return Dict(
        "N" => N,
        "d" => d,
        "m" => m,
        "path_kind" => path_kind,
        "operation" => operation,
        "language" => "julia",
        "library" => library,
        "method" => method,
        "path_type" => path_type,
        "t_ms" => t_ms,
        "alloc_bytes" => alloc_bytes
    )
end

"""Run the benchmark and output result as JSON"""
function run_benchmark(config::Dict)
    # Extract configuration
    N = config["N"]
    d = config["d"]
    m = config["m"]
    path_kind = config["path_kind"]
    operation = config["operation"]
    repeats = config["repeats"]

    # Generate path
    path = make_path(d, N, path_kind)

    # Select operation and prepare kernel
    if operation == "signature"
        kernel = run_signature(path, m)
        method = "signature_path"
    elseif operation == "logsignature"
        kernel = run_logsignature(path, m)
        method = "log"
    else
        # Operation not supported (sigdiff not available yet)
        error("Unsupported operation: $operation")
    end

    # Run manual timing loop
    t_ms, alloc_bytes = manual_timing_loop(kernel, repeats)

    # Format result
    result = output_result(
        N, d, m, path_kind, operation, t_ms, alloc_bytes,
        "ChenSignatures.jl",
        method,
        "Vector{SVector}"
    )

    # Output as JSON
    println(JSON.json(result))
end

# -------- Entry Point --------

if isempty(ARGS)
    println(stderr, "Usage: run_chen.jl '<json_config>'")
    exit(1)
end

# Parse configuration from command line
config_json = JSON.parse(ARGS[1])
# Convert JSON.Object to Dict
config = Dict{String, Any}(config_json)

# Run benchmark
try
    run_benchmark(config)
catch e
    error_result = Dict(
        "error" => string(e),
        "N" => get(config, "N", nothing),
        "d" => get(config, "d", nothing),
        "m" => get(config, "m", nothing),
        "operation" => get(config, "operation", nothing)
    )
    println(stderr, JSON.json(error_result))
    exit(1)
end
