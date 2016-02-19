module Tel

const data_dir = "../data"
const result_dir = "../result"


include("preprocess.jl")
include("model.jl")
include("eval.jl")
include("utils.jl")


end
