option("tianshu")
    set_default(false)
    set_showmenu(true)
    set_description("Enable Tianshu (BI-CUDA) support")
option_end()

if not has_config("tianshu") then
    return
end

-- Tianshu's compiler toolchain is typically clang++ based
-- and provides CUDA 10.2 compatible headers.
add_requires("cuda") -- Use generic cuda requirement for now, might need manual path on Tianshu

rule("with_tianshu")
    on_load(function (t)
        t:set("languages", "cxx17")
        t:add("packages", "cuda")

        t:add("cxflags", "-DENABLE_TIANSHU_API")
        t:add("mxflags", "-DENABLE_TIANSHU_API")
        
        -- Use CUDA compilation rules as Tianshu processes .cu files
        t:add("rules", "cuda")

        t:set("policy", "check.auto_ignore_flags", false)

        t:add("cuflags", "-std=c++17", {force = true})
        t:add("cuflags", "-rdc=true", {force = true})
        t:add("cuflags", "-allow-unsupported-compiler", {force = true})
        t:add("cuflags", "--expt-relaxed-constexpr", {force = true})
        t:add("cuflags", "-Wno-deprecated-gpu-targets", {force = true})

        -- Link cudart (Tianshu provides a compatible cudart)
        t:add("links", "cudart")

        if is_mode("debug") then
            t:add("cuflags", "-G", {force = true})
        else
            t:add("cuflags", "-lineinfo", {force = true})
        end
    end)
rule_end()
