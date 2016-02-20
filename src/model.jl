function minmeanmedianmax(vec)
    (minimum(vec),round(mean(vec),2),median(vec),maximum(vec))
end
function statcols(data)
    num_col = size(data,2)
    for ind in 1:num_col
        tmp = data[:,ind]
        print("min mean median max for $ind class: ")
        println(minmeanmedianmax(tmp))
    end
end

function model_flow()
    train_path = joinpath(data_dir, "tr.jld")
    val_path   = joinpath(data_dir, "val.jld")
    test_path  = joinpath(data_dir, "te.jld")

    train = JLD.load(train_path, "tr")
    val   = JLD.load(val_path, "val")
    test  = JLD.load(test_path, "te")

    train = map(x->parse(Float64,x), train)
    val = map(x->parse(Float64,x), val)
    test = map(x->parse(Float64,x), test)
    @show size(train)
    @show size(val)
    @show size(test)


    #println("minmeanmedianmax(sum(vcat(train,val,test).== 1.0,2)):")
    #println(minmeanmedianmax(sum(vcat(train,val,test)[:,1:end-1].== 1.0,2)))
    #statcols(train)
    sleep(1)
    info("training model using Gradient Boosting decision tree")
    num_iter = 5000
    bst = model_xgboost(train,val,test,num_iter)
end

function sparse_stat()
    train_path = joinpath(data_dir, "train.jld")
    val_path   = joinpath(data_dir, "val.jld")
    test_path  = joinpath(data_dir, "test.jld")    
    train = JLD.load(train_path, "train")
    val   = JLD.load(val_path, "val")
    test  = JLD.load(test_path, "test")

    data = vcat(train,val,test)
end

function apply_model(bst,vector::Array{Float32,1})
    test = DMatrix(vector')
    label = XGBoost.predict(bst,test)
    label
end
function apply_model(mdl::ASCIIString,vector::Array{Float32,1})
    bst = Booster(model_file = mdl)
    test = DMatrix(vector')
    label = XGBoost.predict(bst,test)
    label
end

@doc """ modeling data with Gradient Boosting decesion tree
""" ->
function model_xgboost(data)
    model_xgboost(data[1],data[2],data[3])
end

@doc """ xgboost for train,val,test dataset without gridtune
         eval with weighted F1-score.
""" ->
function model_xgboost(train, val, test, num_iter::Int64)
    train_X,train_Y = train[:,1:end-1],train[:,end]
    val_X,val_Y = val[:,1:end-1],val[:,end]
    test_X,test_Y = test[:,1:end-1],test[:,end]
    
    num_class = maximum(train_Y)+1
    m_tr,n_tr   = size(train_X)
    m_val,n_val = size(val_X)    
    info("in the model, there are $num_class classes")
    info("train_X size is :$m_tr,$n_tr")
    info("val_X size is : $m_val, $n_tr")
    #numcancersample_data(train,cancer_names())
    #sleep(3)
    
    num_round = num_iter
    
    dtrain = DMatrix(train_X, label = train_Y)
    dval   = DMatrix(val_X,   label = val_Y)
    dtest  = DMatrix(test_X,  label = test_Y)   
    
    watch_list = [(dtrain,"train"), (dval,"val")]
    param      = Dict{ASCIIString,Any}("max_depth"=>7,"eta"=>0.01,"nthread"=>50,
                                       "objective"=>"multi:softprob","silent"=>1,
                                       #"alpha"=>0.7,
                                       
                                       "sub_sample"=>0.9) #,"num_class"=>num_class)
   
    bst = xgboost(dtrain,num_round,watchlist=watch_list,param=param,num_class=num_class,
                  metrics=["mlogloss"], seed=2015)
    
    #XGBoost.save(bst, cancer33_mdl)
    test_preds = XGBoost.predict(bst, test_X)
    
    #info("on test dataset mean-f1-score is $mean_f1")
    yprob = reshape(test_preds, Int64(num_class), size(test_X,1))
    ind_prob = map(ind->indmax(yprob[:,ind]), 1:size(test_X,1))
    test_preds = ind_prob - 1 # for index start different
    
    #test_preds = map(x->convert(Int64,x), test_preds)
    #test_Y = map(x->convert(Int64,x), test_Y)
    #num_class = convert(Int64, num_class)
    #ROC_print(test_Y.+1,test_preds.+1,num_class = num_class, challenge_list=cancer_names())
    #numcancersample_data(train,cancer_names())
    return bst
end


### Model Tuning useful in MLBase
function estfun(max_depth,eta)
    param["max_depth"] = max_depth
    param["eta"] = eta
    
    if CV
        bst = nfold_cv(dtrain,num_round,nfold,param=param,feval=evaluerror_softmax,show_stdv=false)   
    else
        bst = xgboost(dtrain,num_round,watchlist=watchlist,param=param,feval=evaluerror_softmax,seed=0, early_stopping_rounds=80)
    end
    return bst
end


function evalfun(bst)
    test_preds = XGBoost.predict(bst,test_X) 
    mean_f1 = evaluerror_softmax(test_preds,dtest)
    return mean_f1    
end

function gridtune()
    if debug
        r = gridtune(estfun,evalfun,("max_depth",[2,10]),("eta",[0.1]))   
    else
        r = gridtune(estfun,evalfun,("max_depth",[2,4,6,10,30,50,100]),("eta",[0.01,0.05,0.1,0.3,0.5,0.8,1]))
    end

    best_model, best_cfg, best_score = r
    @printf "===============================================================================================\n"
    @printf "softmax_best %10s  is %8f \n" best_score[1] best_score[2]
    @printf "===============================================================================================\n"
    f = open("softmax_best_scores.txt","a")
    @printf f "softmax_best %10s is %8f \n"  best_score[1] best_score[2]
    close(f)


    #save(best_model,"../models/softmax.model")
    
    #bst = Booster(model_file = "../models/softmax.model")
    #p = XGBoost.predict(bst,test[:,1:end-1]) .+ 1
    
    p = XGBoost.predict(best_model,test[:,1:end-1]) .+ 1

    test_Y_gt,train_pre = convert(Array{Int64},test[:,end]),convert(Array{Int64},p)

    ROC_print(test_Y_gt,train_pre,num_class=num_class)
    #   if WIN
    p = XGBoost.predict(best_model,test_I[:,1:end-1]) .+ 1

    test_Y_gt,train_pre = convert(Array{Int64},test_I[:,end]),convert(Array{Int64},p)
    
    ROC_print(test_Y_gt.+1,train_pre,num_class=num_class)
    #  end
    
end

@doc """eval ROC_value of model
"""->
function ROC_print(gt,pred;num_class=2,challenge_list=cancer_list)
    
    C = confusmat(num_class,gt,pred)
    println(C)

    output = Array{Pair{ASCIIString,Float64},1}(num_class)
    
    for i=1:num_class
        gt_tmp = (gt .== i)
        pred_tmp = (pred .== i)
        r_tmp = MLBase.roc(gt_tmp,pred_tmp)
        f1_tmp = MLBase.f1score(r_tmp)
        output[i] = Pair{ASCIIString,Float64}(challenge_list[i],f1_tmp)

    end
    output = sort(output,by=x->x[2],rev=true)
    for i=1:num_class
        @printf "%35s\t %8f\n" output[i][1] output[i][2]
    end
end

@doc """ shuffle dataset
         random indexes of row
""" ->
function shuffle(data::SparseMatrixCSC{Float64,Int64})
    m,n = size(data)
    inds = randperm(m)
    data[inds,:]
end

function split_cancer_data(data::SparseMatrixCSC{Float64,Int64})
    m,n = size(data)
    point1 = round(Int64, 0.6*m)
    point2 = round(Int64, 0.8*m)
    train = data[1:point1,:]
    val   = data[point1+1:point2,:]
    test  = data[point2+1:end,:]

    train,val,test
end

@doc """ split each cancer dataset
""" ->
function split_data(data::SparseMatrixCSC{Float64,Int64})
    num_label = maximum(data[:,end])

    function reducer(data1, data2)
        (vcat(data1[1],data2[1]), vcat(data1[2], data2[2]), vcat(data1[3], data2[3]))
    end
    
    data = @parallel (reducer) for i = 1:num_label
        split_cancer_data(data[data[:,end] .== i,:])
    end

    data
end
