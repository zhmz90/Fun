src = "../src"
if !in(src, LOAD_PATH)
    push!(LOAD_PATH, src)
end
using Tel

#=
ev = readcsv("../data/event_type.csv", ASCIIString)
re = readcsv("../data/resource_type.csv", ASCIIString)
se = readcsv("../data/severity_type.csv", ASCIIString)
reformat(ev)
reformat(re)
reformat(se)
=#

log = readcsv("../data/log_feature.csv", ASCIIString)
Tel.reformat3(log)

