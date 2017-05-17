require "nn"
require "torch"
require "optim"
require "cunn"
require "cutorch"
require "loadCharDataset.lua"

torch.manualSeed(1)
numClasses = 2
cmd = torch:CmdLine()
cmd:text()
cmd:option("-batchSize","10","Batchsize")
cmd:option("-coefL1", "0", "l1 coefficient")
cmd:option("-coefL2", "0", "l2 coefficient")
cmd:option("-numepochs", "10", "number of epochs")

--parse input params
params = cmd:parse(arg)

model = nn.Sequential()
model:add(nn.SpatialConvolutionMM(1, 96, 8, 8):cuda())
model:add(nn.ReLU())
model:add(nn.SpatialAveragePooling(5, 5, 5, 5 ))
model:add(nn.SpatialConvolutionMM(96, 256, 2, 2))
model:add(nn.ReLU())
model:add(nn.SpatialAveragePooling(2, 2, 2, 2))
model:add(nn.View(2*2*256))

model:add(nn.Linear(2*2*256, numClasses))
model:add(nn.LogSoftMax())

model = model:cuda()

criterion = nn.ClassNLLCriterion()
criterion = criterion:cuda()

traindata = train
testdata = test

parameters, gradParameters = model:getParameters()

local sum = function(t)
	--sum the errors
	s = 0
	for i, v in ipairs(t) do
		s = s + v
	end
	return s
end
for i = 1, params.numepochs do
	print(string.format("Running %d epoch", i))
	print("Starting to shuffle input data")
	local shuffle = torch.randperm(traindata:size())
	local shuff_inputs = torch.Tensor(traindata.data:size()):cuda()
	local shuff_labels = torch.Tensor(traindata.labels:size()):cuda()

	for idx = 1, traindata.data:size(1) do
		shuff_inputs[idx] = traindata.data[shuffle[idx]]
		shuff_labels[idx] = traindata.labels[shuffle[idx]]
	end

	print("Completed shuffling")
	criterionTable={}
	for j = 1, traindata:size(),params.batchSize do
		local end_j = math.min(j + params.batchSize, traindata:size())
		local inputs = shuff_inputs[{{j, end_j}}]
		local labels = shuff_labels[{{j, end_j}}]

		local feval = function(x)
			--get new parameters
			if x ~= parameters then
				parameters:copy(x)
			end
			--reset gradients
			gradParameters:zero()
			--evaluate function for minibatch
			local outputs = model:forward(inputs)
			--print(outputs)
			--print(labels)
			local f = criterion:forward(outputs, labels) + params.coefL1 * torch.norm(parameters,1) + params.coefL2 * torch.norm(parameters, 2)^2/2
			table.insert(criterionTable, f)
			--estimate df/dw
			local df_dw = criterion:backward(outputs, labels)
			model:backward(inputs, df_dw)

			gradParameters:add(torch.sign(parameters):mul(params.coefL1) + parameters:clone():mul(params.coefL2))
			gradParameters:mul(1/inputs:size(1))
			--print(gradParameters)
			--print("Train loss:"..f.." , grad:"..gradParameters:norm())

			return f, gradParameters
		end
		sgdState = {
			learningRate = 0.1,
			momentum = 0.5,
			learningRateDecay = 5e-7,
			dampening = 1.01
		}
		optim.sgd(feval, parameters, sgdState)
		xlua.progress(j, traindata:size())
	end
	print("Error after this epoch: "..sum(criterionTable))
end

--with the trained model predict for testdataset
--output = model:forward(testdata.data)
--get the maximum probability label
classes = {'1','2'}
confusion = optim.ConfusionMatrix(classes)
outputs = model:forward(testdata.data:cuda())
for i=1, testdata:size() do
	confusion:add(outputs[i], testdata.labels[i])
end

print(confusion)

print("Saving log likelihoods...")
nll = {}
nll.outputs = outputs
nll.labels = testdata.labels
torch.save("nll.t7", nll)

model = model:clearState()
floatmodel = model:float()
torch.save("charDetectionModel.t7", model)
