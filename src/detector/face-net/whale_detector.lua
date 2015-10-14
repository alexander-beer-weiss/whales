require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')

require 'nn'
require 'xlua'  -- progress bar
require 'optim'  -- confusion matrix; gradient decent optimization
require 'paths'  -- read OS directory structure

dofile 'convnet.lua'

local headDir = '../../..'

local imgDir = headDir .. '/imgs/detector_imgs'
local NNsaveDir = headDir .. '/NNsave'

cmd = torch.CmdLine()
cmd:text()
cmd:text('SVHN Training/Optimization')
cmd:text()
cmd:text('Options:')
cmd:option('-preprocessedImgs', imgDir .. '/preprocessed/preprocessed_imgs.dat', 'path to preprocessed images')
cmd:option('-netDatadir', NNsaveDir .. '/detector', 'location of neural net data directory')
cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
cmd:option('-trainOn', 85, 'percent of training data to use for training; the rest is used for cross-validation')
cmd:option('-useWarped', true, 'augment training set with warped images')
cmd:option('-posMultiples', 5, 'number of (augmented) multiples of each positive image to include in each epoch')
cmd:option('-negMultiples', 1, 'number of (augmented) multiples of each negative image to include in each epoch')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
cmd:option('-learningRate', 1e-1, 'learning rate at t=0')  -- could make this variable.  Start big and decay.
cmd:option('-batchSize', 32, 'mini-batch size (1 = pure stochastic)')
cmd:option('-sgdUntil',2,'epoch at which to switch over to different optimization')
cmd:option('-weightDecay', .01, 'weight decay (SGD only)')
cmd:option('-transfer', 'ReLU', 'transfer function: Tanh | ReLU | Sigmoid')
cmd:option('-dropout', '0,0.2,0.2', 'fraction of connections to drop: comma seperated numbers in the range 0 to 1')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:option('-t0', 4, 'start averaging at t0 (ASGD only), in nb of epochs')
cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS')
cmd:option('-maxEpoch', 12, 'maximum number of epochs during training')  -- set to -1 for unlimited epochs
cmd:option('-loadNet', '', 'load from previous opt')
cmd:option('-learningRateScale', 0.5, 'factor to reduce the learning rate if the score is increasing')
cmd:option('-cuda', false, 'use GPU')
cmd:text()
opt = cmd:parse(arg or {})

if opt.cuda then
	require 'cutorch'
	require 'cunn'
	require 'cudnn'
end

print('OPT.CUDA',opt.cuda)

opt.img_multiples = { pos = opt.posMultiples , neg = opt.negMultiples }

local dropout = {}
for match in opt.dropout:gmatch("([%d%.%+%-]+),?") do
  table.insert(dropout,tonumber(match))
end
opt.dropout = dropout

local img_data = torch.load(opt.preprocessedImgs)
-- img_data[1] = positive examples
-- img_data[2] = negative examples
-- img_data[pos_or_neg][x] = {'file name' = table of warped multiples}  (for x = 1,2,...)


-- (opt.trainOn)% of the data is used to train on
-- the rest of the data is set aside for cross-validation
local training_data = { pos = {}, neg = {} }  -- training_data['pos'] = {positive examples}; training_data['neg'] = {negative examples}
local cv_data = { pos = {}, neg = {} }


--image_files, image_targets, image_paths = table_shuffle(image_files, image_targets, image_paths)
local trainFraction = opt.trainOn / 100
if trainFraction < 0 then trainFraction = 0  -- should never be set to zero anyway (except maybe for testing CV code)
elseif trainFraction > 1 then trainFraction = 1 end



-- Seperate a set of cross-validation examples
for pos_or_neg, examples in pairs(img_data) do
	-- examples is a table containing all pos/neg training examples
	-- each examples[idx] is a table with one entry: { filename = {warped_multiples} }
	local shuffle_tensor = torch.randperm(#examples)
	for idx = 1, #examples do
		local shuffled_idx = shuffle_tensor[idx]
		-- randomly inserting single entry tables { filename = {warped_multiples} } into training/cross-validation sets
		if idx <= trainFraction * #examples then
			table.insert(training_data[pos_or_neg], {['filename'] = examples[shuffled_idx].filename, ['img'] = examples[shuffled_idx].img})
		else
			-- for the cross-validation set, only store the unwarped image
			table.insert(cv_data[pos_or_neg], {['filename'] = examples[shuffled_idx].filename, ['img'] = examples[shuffled_idx].img} )
		end
	end
end

-- Set the order in which positive vs. negative training examples are fed through the neural net
-- This is experimental; perhaps training would be more effective with positive/negative examples alternating
local pos_neg_training_order = {}
local num_examples = {pos = opt.img_multiples.pos * #training_data.pos, neg = opt.img_multiples.neg * #training_data.neg}
local pos_neg_training_ratio = num_examples.pos / num_examples.neg
local pos_idx = 1
for neg_idx = 1, num_examples.pos do
	while (neg_idx >= pos_idx * pos_neg_training_ratio) do
		table.insert(pos_neg_training_order, 'neg')
		--print('pos:', pos_idx)
		pos_idx = pos_idx + 1
	end
	table.insert(pos_neg_training_order, 'pos')
	--print('neg:', neg_idx)
end




-- initial a convolutionaal neural network object called myNet
local myNet = convNet(opt)

--[[ ************************** RESTORE THIS FUNCTIONALITY
if opt.loadNet ~= '' then
	print( 'Loading ' .. opt.netDatadir .. '/' .. opt.loadNet .. '.dat')
	myNet.net = torch.load(opt.netDatadir .. '/' .. opt.loadNet .. '.dat')
	myNet:reset()
else
	print('Building a new net to train')
	myNet:build({32,64}, {3,2,2}, {1,1,1}, {2,2,1})
end
--]]


-- build neural net
myNet:build()



dofile 'config_optimizer.lua'  -- optim.sgd, optim.asgd, optim.lbfgs, optim.cg 
dofile 'train.lua'  -- train convnet
dofile 'cross_validate.lua'  -- test convnet with cross-validation data
dofile 'save_nets.lua'

netSaver = save_nets(opt)
netSaver:prepNNdirs()

local best_accuracy = 0  -- best accuracy on cross-validation set
local best_confusion = nil
local best_confusion_filenames = nil
local best_epoch = 1
local prev_training_accuracy = 0
for epoch = 1, opt.maxEpoch do
	local training_accuracy = train(epoch, myNet, training_data, pos_neg_training_order)
	local accuracy, confusion, confusion_filenames = cross_validate(epoch, myNet, cv_data)
	
	if best_accuracy < accuracy then
		best_accuracy = accuracy
		best_confusion = confusion
		best_confusion_filenames = confusion_filenames
		best_epoch = epoch
	end
	
	if training_accuracy < prev_training_accuracy then
		print('Training accuracy went down! Reducing learning rate')
		optimState.learningRate = optimState.learningRate * opt.learningRateScale
	end
	
	prev_training_accuracy = training_accuracy

	netSaver:saveNN(epoch, myNet:getNet())  -- save current state (i.e., parameters) of neural net
end

-- copy net with best accuracy to NN.dat
netSaver:saveBestNet(best_epoch, os.date(), best_confusion, best_confusion_filenames, opt)

-- PRINT CONFUSION MATRIX TO FILE INSTEAD OF ACCURACY!!!



