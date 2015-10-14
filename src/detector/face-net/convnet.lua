require 'nn'
require 'image'
require 'underscore'


---------------------------------------------------------------------------
-- Create convNet metatable.  Then we can create convNet objects.

convNet = {}
convNet.__index = convNet
setmetatable(convNet, {
	__call = function (cls, ...)
		return cls.new(...)
	end,
})

function convNet.new(opt)
	local self = setmetatable({},convNet)
	self.net = nn.Sequential()
	for key, value in pairs(opt) do
		self[key] = value
	end
	
	if self.cuda then
		require 'cunn'
	end
	
	
	

	return self
end

---------------------------------------------------------------------------

function convNet:setNet(net)
	self.net = net
end


function convNet:getNet()
	return self.net
end


-- options for transfer functions
local trans_layer = {}
trans_layer['ReLU'] = nn.ReLU
trans_layer['Sigmoid'] = nn.Sigmoid
trans_layer['Tanh'] = nn.Tanh



function convNet:build()  -- first let's get something working, then try to build a flexible net
	
	print('==> BUILDING NET')

	if self.cuda then
		self.net:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor'))
	end
	
	-- convNet structure for 64x64 images
	
	--local normkernel = torch.ones(9)
	
	-- SpatialConfolutionCUDA, SpatialMaxPoolingCUDA
	self.net:add( nn.SpatialConvolution(3, 16, 5, 5) )  -- 3 x 64 x 64 -> 16 x 60 x 60
	self.net:add( trans_layer[self.transfer]() )
	self.net:add( nn.SpatialMaxPooling(2, 2, 2, 2) )  -- 16 x 60 x 60 -> 16 x 30 x 30
	--self.net:add(nn.SpatialSubtractiveNormalization(16, normkernel))
	
	self.net:add( nn.SpatialConvolution(16, 16, 5, 5) )  -- 16 x 30 x 30 -> 16 x 26 x 26
	self.net:add( trans_layer[self.transfer]() )
	self.net:add( nn.SpatialMaxPooling(2, 2, 2, 2) )  -- 16 x 26 x 26 -> 16 x 13 x 13
	--self.net:add(nn.SpatialSubtractiveNormalization(16, normkernel))
	
	self.net:add( nn.SpatialConvolution(16, 16, 13, 13) )  -- 16 x 13 x 13 -> 16 x 1 x 1
	self.net:add( nn.SpatialConvolution(16, 8, 1, 1) )  -- 16 x 1 x 1 -> 8 x 1 x 1
	self.net:add( nn.Dropout(.4) )
	self.net:add( nn.SpatialConvolution(8, 1, 1, 1) )  -- 8 x 1 x 1 -> 1 x 1 x 1
	
	self.net:add( nn. Sigmoid() )  -- prep output for binary cross-entropy criterion (requires output between 0 and 1)
	
	if self.cuda then
		self.net:cuda()
		self.net:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor'))
	end
	
	
--[[  BELOW IS A CONVNET STRUCTURE FOR 128x128 images
	
	self.net:add( nn.SpatialConvolution(3, 32, 5, 5) )  -- 3 x 128 x 128 -> 32 x 124 x 124
	self.net:add( trans_layer[self.transfer]() )
	self.net:add( nn.SpatialMaxPooling(2, 2, 2, 2) )  -- 32 x 124 x 124 -> 32 x 62 x 62
	--self.net:add(nn.SpatialSubtractiveNormalization(dimension, normkernel))
	--self.net:add(nn.SpatialDropout(p))
	
	self.net:add( nn.SpatialConvolution(32, 32, 5, 5) )  -- 32 x 62 x 62 -> 32 x 58 x 58
	self.net:add( trans_layer[self.transfer]() )
	self.net:add( nn.SpatialMaxPooling(2, 2, 2, 2) )  -- 32 x 58 x 58 -> 32 x 29 x 29
	
	self.net:add( nn.SpatialConvolution(32, 16, 4, 4) )  -- 32 x 29 x 29 -> 16 x 26 x 26
	self.net:add( trans_layer[self.transfer]() )
	self.net:add( nn.SpatialMaxPooling(2, 2, 2, 2) )  -- 16 x 26 x 26 -> 16 x 13 x 13
	
	self.net:add( nn.SpatialConvolution(16, 16, 13, 13) )  -- 16 x 13 x 13 -> 16 x 1 x 1
	self.net:add( nn.SpatialConvolution(16, 8, 1, 1) )  -- 16 x 1 x 1 -> 8 x 1 x 1
	self.net:add( nn.Dropout(.4) )
	self.net:add( nn.SpatialConvolution(8, 1, 1, 1) )  -- 8 x 1 x 1 -> 1 x 1 x 1
	
	self.net:add( nn. Sigmoid() )  -- prep output for binary cross-entropy criterion (requires output between 0 and 1)
--]]
	
	self:reset()  -- maybe reset should be called initialize

end

function convNet:reset()
	self.criterion = nn.BCECriterion()
	self.parameters, self.gradParameters = self.net:getParameters()
end


function convNet:getWeights()
	return self.parameters
end

function convNet:setWeights(weights)
	self.parameters:copy(weights)
end

function convNet:zeroGradients()
	self.gradParameters:zero()
end


function convNet:trainStep(input, targets)  -- input is image of type torch.Tensor(channel, height, width); label is 'pos' or 'neg'
	
	self.net:training()  -- activates dropout layers of net
	
	-- if using CUDA, move the batch of training images to GPU
	if self.cuda then input = input:cuda() end
	
	--print('INPUT SIZE:', input:size())
	-- FORWARD PROPAGATE through NET
	local hypothesis = nn.Reshape(1):forward( self.net:forward(input) )  -- batch_size x 1 x 1 x 1 -> batch_size x 1
	--print('OUPUT SIZE:', hypothesis:size())

	
	-- FORWARD PROPAGATE through COST FUNCTION
	local cost = self.criterion:forward( hypothesis , targets )  -- should this have a :float() at the end
	
	-- BACKWARD PROPAGATE through COST FUNCTION
	local dcost_dout = self.criterion:backward( hypothesis , targets )
	
	-- BACKWARD PROPAGATE through NET
	self.net:backward(input, dcost_dout)  -- this automatically upgrades self.gradParameters
	
	-- make sure hypothesis is back on CPU side (IS THIS NECESSARY?  What about cost?)
	hypothesis = hypothesis:float()
	
	return hypothesis, cost, self.gradParameters
end



function convNet:testStep(input)
	
	self.net:evaluate()  -- deactivates dropout layers of net
	
	if self.cuda then input = input:cuda() end

	-- FORWARD PROPAGATE through NET
	return self.net:forward(input):float()
	
end



--[[
function convNet:augmentedTrainStep(input, targets)	
	local hypothesis, cost, self.gradParameters = self:trainStep(input, targets)
	local err = val['err']
	local output = val['output']
	val = self:trainStep(image.hflip(img),label)
	err = err + val['err']
	table.insert(output, val['output'])
	local tmp_img = img
	for i=0,3 do
		tmp_img = image.rotate(tmp_img, math.pi/2)
		val = self:trainStep(tmp_img, label)
		err = err + val['err']
		table.insert(output, val['output'])
		val = self:trainStep(image.hflip(tmp_img), label) 
		err = err + val['err']
		table.insert(output, val['output'])
	end
	return output, err  -- output should be 'pos' or 'neg'
end
--]]


