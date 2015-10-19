
local label_to_number = { pos = 1, neg = 0 }


function train(epoch, netObject, training_data, pos_neg_training_order, num_warps)  -- epoch counts number of times through training data
	-- local vars
	-- local time = sys.clock()
	
	-- do one epoch
	print('')
	print('==> doing epoch on training data:')
	print('==> epoch # ' .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
	
	local shuffle = {}
	shuffle.pos = torch.randperm(#training_data.pos)  -- array of shuffled indices for positive training examples
	shuffle.neg = torch.randperm(#training_data.neg)  -- array of shuffled indices for negative training examples
	
	--local misclassify = {}
	local confusion = optim.ConfusionMatrix({'positive','negative'})  -- can we just make assignment pos='positive' or [1]='positive', [0]='negative

	local num_training_examples = opt.img_multiples.pos * #training_data.pos + opt.img_multiples.neg * #training_data.neg
	
	-- determine the number of channels, the height and the width of training images by looking at an arbitrary example
	local num_channels, img_height, img_width
	    = training_data.pos[1].img:size(1), training_data.pos[1].img:size(2), training_data.pos[1].img:size(3)
	
	-- keep track of the number of correct predictions
	local correct_predictions = 0
	
	-- keep track of contributions to the logloss score
	local logloss = 0
	
	-- indices for positive and negative training examples stored in training_data['pos'] and training_data['neg']
	local target_counter = { pos = 0, neg = 0 };
	
	-- Here we loop over batches using a course-grained index
	-- The index gives the first training example in each batch
	for coarse_training_idx = 1, num_training_examples, opt.batchSize do
		
		-- display progress bar
		xlua.progress(coarse_training_idx, num_training_examples)  -- could instead do a fine grained progress bar in the inner-loop
		
		-- determine the true size of current batch
		-- on the last training batch of the epoch, actual batch size may be smaller than opt.batchSize
		local batch_size = math.min(opt.batchSize, num_training_examples - coarse_training_idx + 1)
		
		
		
		-- now package a batch of training images into contiguous memory as a 4D torch tensor
		local batch_input = torch.Tensor(batch_size, num_channels, img_height, img_width)
		local batch_targets = torch.Tensor(batch_size)

		for fine_training_idx = 1, batch_size do
			local target = pos_neg_training_order[coarse_training_idx + fine_training_idx - 1]  -- target = 'pos' or 'neg'.
			local shuffled_idx = shuffle[target][ target_counter[target] % #training_data[target] + 1 ]  -- this will make opt.img_multiple[target] passes
			
			--local example_idx = shuffled_idx % #training_data[target] + 1  -- +1 because table indexing begins at 1
			--local warp_idx = math.ceil( shuffled_idx / #training_data[target] )  -- ceil, not floor, because table indexing begins at 1
			
			-- copy image into contiguous block of memory
			batch_input[fine_training_idx] = training_data[target][shuffled_idx].img:clone()
			local curr_img = batch_input[fine_training_idx]  -- create a convenience reference to the current chunk of continguous memory
			
			-- randomly flip ~50% of images
			if math.random(0,1) == 0 then
				image.hflip(curr_img, curr_img)  -- flip curr_img and store back in same memory location
			end
			
			-- rotate by 0, 90, 180, 270, ...
			local rotation_angle = (epoch - 1 + math.floor( target_counter[target] / #training_data[target] ) ) * math.pi / 2
			image.rotate(curr_img, curr_img, rotation_angle)
			
			-- after the fourth training epoch, warp curr_img ~50% of the time (hopefully this makes the network more robust)
			if epoch > 4 and math.random(0,1) == 0 then
				local flow = torch.randn(2,img_width,img_height) * 0.3  -- random components from normal distribution with standard deviation 0.3 (small distortions)
				image.warp(curr_img, curr_img, flow)
			end
			
			-- save labels into contiguous array
			batch_targets[fine_training_idx] = label_to_number[target]  -- 'pos' -> 1; 'neg' -> 0
			
			target_counter[target] = target_counter[target] + 1
			
			collectgarbage()
		end
		
		--************************** begin cost-eval closure **************************--
		
		-- create closure to evaluate cost(X) and dcost/dX
		-- the closure will be passed to optimMethod below to optimize training
		local cost_eval = function(X)
			-- get new network weights
			if X ~= netObject:getWeights() then
				netObject:setWeights(X)
			end
			
			-- reset network gradients
			netObject:zeroGradients()
			
			-- FORWARD & BACKWARD PROPAGATE THROUGH NEURAL NETWORK
			local hypothesis, cost, gradients = netObject:trainStep(batch_input, batch_targets)
			
			-- update confusion matrix
			local binary_hypothesis = torch.ceil(-hypothesis + 1.5):reshape(hypothesis:size(1))
			local binary_targets = -batch_targets + 2
			confusion:batchAdd(binary_hypothesis, binary_targets)
			
			-- create binary array to label correct predictions ( 1 = true prediction, 0 = false prediction )
			local batch_correct_predictions = torch.eq(binary_hypothesis, binary_targets)
			
			-- record number of accurate predictions
			correct_predictions = correct_predictions + batch_correct_predictions:sum()
			
			-- recoord logloss contribution
			logloss = logloss + nn.BCECriterion():forward(hypothesis, batch_targets) * ( batch_size / num_training_examples )
			  -- ideally need to regularize elementwise as max(min(hyp,1−10−15),10−15)
			  -- could first take elementwise log and then do max(min(-log(hyp),-log(10−15)),log(10−15)
			
			-- return cost and dcost/dX
			return cost, gradients
			
		end  -- end cost_eval loop
		
		--************************** end cost_eval closure **************************--
		
		
		-- optimize on current mini-batch
		if optimMethod == optim.asgd then
			_, _, average = optimMethod(cost_eval, netObject:getWeights(), optimState)  -- what is this about... (global) averages go unused...
		else
			optimMethod(cost_eval, netObject:getWeights(), optimState)
		end
		
	end  -- end coarse-grained (batch) training loop

	
	print(confusion)  -- print confusion matrix
	--confusion:zero()  -- reset confusion matrix for next batch
	print('LOGLOSS: ' .. logloss)
	
	local accuracy = correct_predictions / num_training_examples
	return accuracy, logloss
	
end
