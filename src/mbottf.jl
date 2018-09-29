#!/usr/bin/env julia
include("helper.jl")

len_in = 100
len_out = 20
nbatch = 32  
nepochs = 10
model_dir = mktempdir(abspath(""))
frac_test = .3
@info "model is stored in $model_dir"

using PyCall
using Queryverse, DataFrames
@pyimport pydoc
df = load("postproc3.feather") |> DataFrame
delete!(df, :datevalue)
data = convert(Array{Float64}, df)
len_train = floor(Int64, size(data)[1] * (1-frac_test))  
train_data = data[1:len_train,:]
test_data = data[len_train+1:end,:]
# data = diff(data, 1)

params = Dict("lr_start"=>.0001,
              "lr_decay"=>.99999,
              "len_in"=>len_in,
              "len_out"=>len_out,
              "nepochs"=>nepochs,
              "Nh"=>10, # size of hidden layer
              "Nc"=>12, # size of context
              "D"=>size(train_data)[2])

localpath = "pybot/input.py"
filepath = abspath(joinpath(dirname(@__FILE__),localpath))
specname = "pybot.input" # need pybot inorder to allow relative import
pymodule = pyimport_module(filepath, specname)
functionname = "input_wrapper_function"
pyinput_fn = pymodule[Symbol(functionname)](train_data, len_in, len_out, nbatch, nepochs)

localpath = "pybot/model.py"
filepath = abspath(joinpath(dirname(@__FILE__),localpath))
modulename = "pybot.model" # need pybot inorder to allow relative import
functionname = "model_fn"
modelmod = pyimport_module(filepath, modulename)
pymodel_fn = modelmod[Symbol(functionname)]

@pyimport tensorflow as pytf
pytf.logging[:set_verbosity](pytf.logging[:DEBUG])
config = pytf.estimator[:RunConfig](save_summary_steps=10000,
                                    tf_random_seed=1234,
                                    save_checkpoints_secs=600)
estimator = pytf.estimator[:Estimator](model_fn=pymodel_fn, params=params,
                                       model_dir=model_dir, config=config)

estimator[:train](pyinput_fn)
