option("nv-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Enable NVIDIA CUDA support")
option_end()

if not has_config("nv-gpu") then
    return
end

add_requires("cuda")

rule("with_nvidia")
    on_load(function (t)
        t:set("languages", "cxx17")
        t:add("packages", "cuda")

        t:add("cxflags", "-DENABLE_NVIDIA_API")
        t:add("mxflags", "-DENABLE_NVIDIA_API")
        t:add("rules", "cuda")

        t:set("policy", "check.auto_ignore_flags", false)

        t:add("cuflags", "-rdc=true", {force = true})
        t:add("cuflags", "-allow-unsupported-compiler", {force = true})
        t:add("cuflags", "-gencode=arch=compute_90,code=sm_90", {force = true})
        t:add("cuflags", "-gencode=arch=compute_80,code=sm_80", {force = true})
        t:add("cuflags", "-gencode=arch=compute_89,code=sm_89", {force = true})
        t:add("cuflags", "--expt-relaxed-constexpr", {force = true})
        t:add("cuflags", "-Wno-deprecated-gpu-targets", {force = true})

        -- Link cuBLAS
        t:add("links", "cublas", "cublasLt")

        if is_mode("debug") then
            t:add("cuflags", "-G", {force = true})
        else
            t:add("cuflags", "-lineinfo", {force = true})
        end
    end)
rule_end()