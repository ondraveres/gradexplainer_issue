using Pkg
using Flux, MLDataPattern, Mill, JsonGrinder, JSON, Statistics, IterTools, StatsBase, ThreadTools
using JsonGrinder: suggestextractor, ExtractDict
using Mill: reflectinmodel
using CSV, DataFrames
using Random
using Dates
using Plots
using Printf
using JLD2
using ExplainMill #added via ] dev path_to_local_ExplainMill.jl
using Setfield



PATH_TO_REPORTS = "/mnt/data/jsonlearning/Avast_cuckoo/"
PATH_TO_REDUCED_REPORTS = PATH_TO_REPORTS * "public_small_reports/"

df_labels = CSV.read(PATH_TO_REPORTS * "public_labels.csv", DataFrame);

train_samples = df_labels[1:4, :]
test_samples = df_labels[5:9, :]


df_labels = vcat(train_samples, test_samples)


jsons = map(df_labels.sha256) do s
    try
        open(JSON.parse, "$(PATH_TO_REDUCED_REPORTS)$(s).json")
    catch e
        @error "Error when processing sha $s: $e"
    end
end;

sch = JsonGrinder.schema(vcat(jsons, Dict()))

extractor = suggestextractor(sch)
data = map(json -> extractor(json, store_input=true), jsons);


neurons = 32
model = reflectinmodel(sch, extractor,
    k -> Dense(k, neurons, relu),
    all_imputing=true
)
model = @set model.m = Chain(model.m, Dense(neurons, 10))

model(data[2])

stochastic_mask = ExplainMill.explain(GradExplainer(), data[1], model, pruning_method=:Flat_HAdd, rel_tol=0.1)
