@doc """ index array data with the first column
""" ->
function index(dt)
    dict = Dict{ASCIIString,Array{ASCIIString,1}}()
    for i = 1:size(dt,1)
        dict[dt[i,1]] = dt[i,2:end]
    end
    dict
end

@doc """ left join data on the first column, assume have the same ids
         And data have no header part
""" ->
function leftjoindata(d1,d2)
    d1_ind = index(d1)
    d2_ind = index(d2)
    @assert issubset(Set(collect(keys(d1_ind))),Set(collect(keys(d2_ind)))) == true
    nrow = size(d1,1)
    ncol = size(d1,2)-1 + size(d2,2)-1 + 1
    data = Array{ASCIIString,2}(nrow,ncol)
    for (i,id) in enumerate(keys(d1_ind))
        data[i,:] = vcat(id,d1_ind[id], d2_ind[id])'
    end
    data
end

function leftjoindata(d1,d2,d3...)
    #left join
    data = leftjoindata(d1,d2)
    for d in d3
        data = leftjoindata(data,d)
    end
    data
end

@doc """ the last column as lable column, ratio is 6:2:2
         assume data doest not contain header part
""" ->
function splitdata{T<:Any}(dt::Array{T,2})
    labels = Set(dt[:,end])
    num_row,num_col = size(dt)
    tr,val,te = Array{T,2}(0,num_col),Array{T,2}(0,num_col),Array{T,2}(0,num_col)
    
    for lb in labels
        inds = dt[:,end] .== lb
        data = dt[inds,:]
        len = size(data,1)
        trval = Int64(round(len*0.6))
        valte = trval + Int64(round(len*0.2))
        tr  = vcat(tr, data[1:trval,:])
        val = vcat(val,data[trval+1:valte,:])
        te  = vcat(te, data[valte+1:end,:])
    end
    
    tr,val,te
end

@doc """ shuffle dataset
random indexes of row
""" ->
function shuffle(data)
    m,n = size(data)
    inds = randperm(m)
    data[inds,:]
end


function makedata()
    ev = readcsv("../data/event_type.csv", ASCIIString)
    re = readcsv("../data/resource_type.csv", ASCIIString)
    se = readcsv("../data/severity_type.csv", ASCIIString)
    log = readcsv("../data/log_feature.csv", ASCIIString)
    ev = reformat(ev)
    re = reformat(re)
    se = reformat(se)
    log = reformat3(log)
    
    tr = readcsv("../data/train.csv", ASCIIString)
    te = readcsv("../data/test.csv", ASCIIString)
    tr_idloc = reformat(tr[:, [1,2]])
    tr_idlb  = tr[:, [1,3]]
    te_idloc = reformat(te[:, [1,2]])
    
    train = leftjoindata(tr_idloc,ev,re,se,log,tr_idlb)
    test = leftjoindata(te_idloc,ev,re,se,log)

    tr,val,te = map(shuffle,  splitdata(train))
    tr  = tr[:,2:end]
    val = val[:,2:end]
    te  = te[:,2:end]
    JLD.save(joinpath(data_dir,"tr.jld"),"tr", tr)
    JLD.save(joinpath(data_dir,"val.jld"),"val", val)
    JLD.save(joinpath(data_dir,"te.jld"),"te", te)
    JLD.save(joinpath(data_dir,"test.jld"),"test", test)
end
