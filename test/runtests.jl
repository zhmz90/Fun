src = "../src"
if !in(src,LOAD_PATH)
    push!(LOAD_PATH, src)
end
using Tel
using Base.Test
