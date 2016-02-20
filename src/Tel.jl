module Tel

using JLD
using XGBoost
using Logging
@Logging.configure(level=DEBUG)


const data_dir = "../data"
const result_dir = "../result"

include("preprocess.jl")
include("model.jl")
include("eval.jl")
include("utils.jl")
include("reformat.jl")

end
