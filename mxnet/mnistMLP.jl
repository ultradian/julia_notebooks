# Written by Milton Huang
# MIT license
#
using DataFrames

@time train = readtable("data/train.csv");

# to get the size/shape of the DataFrame
size(train)

# first element is label, rest is 28x28=784 pixel values, so to look at first row
reshape(Array(train[1,2:end]), 28, 28)

using Plots
# I like Plotly for interactivity.  
plotly(legend=false)
# to use it on GitHub, you have to setup Plotly Online and inject their code into the notebook
#  could switch to PyPlot
# pyplot()

# heatmap() plots a 2D array
@time heatmap(reshape(Array(train[1,2:end]), 28, 28), aspect_ratio=:equal)

# to see it upright, use the rotl90() function to rotate left 90 degrees
# there are also rotr90() and rot180() functions
@time heatmap(rotl90(reshape(Array(train[1,2:end]), 28, 28)), aspect_ratio=:equal)

# we can use an array comprehension to plot the first sixteen rows
# the Plots library automatically resizes everything for you and adds the axes
plot([heatmap(rotl90(reshape(Array(train[i,2:end]), 28, 28)), aspect_ratio=:equal) for i=1:16]...)

# note that summary in Julia just gives basic information as it is supposed to work on ANY datatype
#  those who are used to R may be looking for the describe() function which is for DataFrames
summary(train)

# like R, there are head() and tail() functions to get the first or last 6 rows
head(train)

# the describe() function gives basic stats
# you could describe(train) to do the entire DataFrame, but that would take a lot of space here
describe(train[:,406])

# can also specify a column in a DataFrame with a Symbol
# Symbols in Julia have a colon before them and are reserved when created (:pixel404)
# I just just picked 404 because it is near the center for 28x28
describe(train[:pixel404])

# see the distribution of values in that column
histogram(train[:pixel404])

# a point a little more to the edge has a lot more zeros
histogram(train[:pixel397])

# distribution of labels - seems fairly even
histogram(train[:label], ticks=collect(0:9))

# we can plot different subsets to eyeball variation
histogram([train[1:8400,:label] train[8401:16800,:label] train[16801:25200,:label] train[25201:33600,:label] train[33601:end,:label]], ticks=collect(0:9), legend=true, 
label=["first fifth" "second fifth" "third fifth" "fourth fifth" "last fifth"])

# need to split train into training and eval sets
# convert DataFrame to an Array, remembering Julia is column-major like R and Matlab (unlike Python)
X = transpose(Array(train[:,2:end]))
y = Array(train[:,1])

size(X)

N = size(X)[2]

length(y)

extrema(X)

mean(X)

# scale X to range 0-1
X = X./255;

# we need to split the train data into a training set (cv_X) and an eval set (eval_X)
split = 0.8
cv_X = X[:,1:floor(Int,split*N)]
eval_X = X[:,floor(Int,split*N)+1:N]

cv_y = y[1:floor(Int,split*N)]
eval_y = y[floor(Int,split*N)+1:N]

using MXNet

batch_size = 1000

# since this is a training set, we specify shuffle to help randomize the training
train_provider = mx.ArrayDataProvider(cv_X, cv_y, batch_size=batch_size, shuffle=true)

# we don't want to randomize the eval set during training, so shuffle is false
eval_provider = mx.ArrayDataProvider(eval_X, eval_y, batch_size=batch_size, shuffle=false)

data = mx.Variable(:data)
fc1 = mx.FullyConnected(data, name=:fc1, num_hidden=300)
act1 = mx.Activation(fc1, name=:tanh1, act_type=:tanh)
fc2 = mx.FullyConnected(act1, name=:fc2, num_hidden=10)
mlp1 = mx.SoftmaxOutput(fc2, name=:softmax)

# save file - note that the `do` block automatically closes the filestream
open("mlp1graph.dot", "w") do fs
    print(fs, mx.to_graphviz(mlp1))
end

# we can run the `dot` program to convert to png if it is installed on your computer
run(pipeline(`dot -Tpng mlp1graph.dot`, stdout="mlp1graph.png"))

# change context to gpu(number) if you have a gpu and want to use that for processing
model = mx.FeedForward(mlp1, context=mx.cpu())

# we are going to use the basic Stochastic Gradient Descent optimizer with a fixed learning rate and momentum
optimizer = mx.SGD(lr=0.1, momentum=0.9, weight_decay=0.00001)

@time mx.fit(model, optimizer, train_provider, eval_data=eval_provider, n_epoch=1)

# note the input layer has 784*300 = 235,200 trainable weights stored in a NDArray. The output layer has 3000 weights.
model.arg_params

# let's examine the weights in the hidden layer
w2 = copy(model.arg_params[:fc2_weight])

# let's get the extrema mean and standard deviation for each of the 10 groups
[ [extrema(w2[:,i]) for i in 1:10] [mean(w2[:,i]) for i in 1:10] [std(w2[:,i]) for i in 1:10] ]

# plot mean and std
plot([ [mean(w2[:,i]) for i in 1:10] [std(w2[:,i]) for i in 1:10] ], legend=true, label=["mean" "std"])

# plot them as 30x10 arrays
heatmap(reshape(w2[:,1], 10,30), aspect_ratio=:equal)

# switch to PyPlot because Plotly doesn't support clims attribute and I will want to compare later
pyplot()

plot([heatmap(reshape(w2[:,i], 10,30), aspect_ratio=:equal, clims=(-0.6,0.6)) for i=1:10]...)

@time mx.fit(model, optimizer, train_provider, eval_data=eval_provider, n_epoch=9)

# let's examine how the weights in the hidden layer have changed after 10 more epochs of training
w211 = copy(model.arg_params[:fc2_weight])

# extrema mean and standard deviation for each of the 10 groups
[ [extrema(w211[:,i]) for i in 1:10] [mean(w211[:,i]) for i in 1:10] [std(w211[:,i]) for i in 1:10] ]

# plot comparison to previous 
plot([ [mean(w2[:,i]) for i in 1:10] [std(w2[:,i]) for i in 1:10] [mean(w211[:,i]) for i in 1:10] [std(w211[:,i]) for i in 1:10] ], 
label=["mean 1" "std 1" "mean 10" "std 10"], legend=true)

plot([heatmap(reshape(w211[:,i], 10,30), aspect_ratio=:equal, clims=(-0.6,0.6)) for i=1:10]...)

# test on eval set
preds = mx.predict(model, eval_provider)
correct = 0
for i = 1:size(preds)[2]
    if indmax(preds[:,i]) == eval_y[i]+1
        correct += 1
    end
end
correct/size(preds)[2]

@time mx.fit(model, optimizer, train_provider, eval_data=eval_provider, n_epoch=10)

@time test = readtable("data/test.csv");

size(test)

test_X = transpose(Array(test))

plot([heatmap(rotl90(reshape(Array(test_X[1:end,i]), 28, 28)), aspect_ratio=:equal) for i=1:16]...)

extrema(test_X)

mean(test_X)

# scale test_X to range 0-1
test_X = test_X./255;

test_provider = mx.ArrayDataProvider(test_X, batch_size=batch_size, shuffle=false)

# this uses the previously trained model and the DataProvider specified above
@time tpreds = mx.predict(model, test_provider)

# create submission
open("MLP1submission.csv", "w") do f
    write(f, "ImageId,Label\n")
    for i = 1:size(tpreds)[2]
        write(f, string(i),",",string(indmax(tpreds[:,i])-1),"\n")
    end
end

# define new network
mlp2 = @mx.chain mx.Variable(:data) => 
 mx.FullyConnected(name=:fc1, num_hidden=300) =>
mx.Activation(name=:tanh1, act_type=:tanh) =>
 mx.FullyConnected(name=:fc2, num_hidden=100) =>
mx.Activation(name=:tanh2, act_type=:tanh) =>
 mx.FullyConnected(name=:fc3, num_hidden=10) =>
 mx.SoftmaxOutput(name=:softmax)

# save file - note that the `do` block automatically closes the filestream
open("mlp2graph.dot", "w") do fs
    print(fs, mx.to_graphviz(mlp2))
end

# we can run the `dot` program to convert to png if it is installed on your computer
run(pipeline(`dot -Tpng mlp2graph.dot`, stdout="mlp2graph.png"))

# change context to gpu(number) if you have a gpu
model = mx.FeedForward(mlp2, context=mx.cpu())

@time mx.fit(model, optimizer, train_provider, eval_data=eval_provider, n_epoch=20)

# this uses the recently trained model and the same DataProvider on the test set 
@time tpreds = mx.predict(model, test_provider)

# create submission
open("MLP2submission.csv", "w") do f
    write(f, "ImageId,Label\n")
    for i = 1:size(tpreds)[2]
        write(f, string(i),",",string(indmax(tpreds[:,i])-1),"\n")
    end
end
