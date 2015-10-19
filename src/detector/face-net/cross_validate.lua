
local label_to_number = { pos = 1, neg = 0 }


-- CV function
function cross_validate(epoch, netObject, cv_data)  -- epoch counts number of times through training data
	
	-- averaged param use?
	if average then
		cachedparams = parameters:clone()
		parameters:copy(average)
	end
	
	local confusion = optim.ConfusionMatrix({'positive','negative'})
	
	-- test over test data
	print('==> cross validating on CV set:')
	--local time = sys.clock()

	-- determine the total number of CV examples
	local num_cv_examples = #cv_data.pos + #cv_data.neg

	-- determine the number of channels, the height and the width of training images by looking at an arbitrary example
	local num_channels, img_height, img_width
	    = cv_data.pos[1].img:size(1), cv_data.pos[1].img:size(2), cv_data.pos[1].img:size(3)
	
	-- keep track of the number of correct predictions
	local correct_predictions = 0
	
	-- keep track of contributions to the logloss score
	local logloss = 0
	
	-- keep track of filenames for positive/negative images that are guessed correctly/incorrectly
	local confusion_filenames = { true_pos = {}, false_pos = {}, false_neg = {}, true_neg = {} }
	
	local progress_idx = 0
	for _, pos_or_neg in pairs({'pos','neg'}) do
		-- Here we loop over batches using a course-grained index
		-- The index gives the first CV example in each batch
		for coarse_cv_idx = 1, #cv_data[pos_or_neg], opt.batchSize do
			
			-- display progress bar
			xlua.progress(progress_idx, num_cv_examples)  -- could instead do a fine grained progress bar in the inner-loop
			
			-- determine the true size of current batch
			-- actual batch size of the final batch may be smaller than opt.batchSize
			local batch_size = math.min(opt.batchSize, #cv_data[pos_or_neg] - coarse_cv_idx + 1)
			
			-- now package a batch of CV images into contiguous memory as a 4D torch tensor
			local batch_input = torch.Tensor(batch_size, num_channels, img_height, img_width)
			local batch_targets = torch.Tensor(batch_size)
			local batch_img_filenames = {}  -- array of associated filenames for current batch of images
			for fine_cv_idx = 1, batch_size do
				local example_idx = coarse_cv_idx + fine_cv_idx - 1
				
				-- copy image into contiguous block of memory
				batch_input[fine_cv_idx] = cv_data[pos_or_neg][example_idx].img:clone()
				
				-- save labels into contiguous array
				batch_targets[fine_cv_idx] = label_to_number[pos_or_neg]  -- 'pos' -> 1; 'neg' -> 0
				
				-- put filename into lua array
				table.insert(batch_img_filenames, cv_data[pos_or_neg][example_idx].filename)
				
				collectgarbage()
			end
			
			
			-- FORWARD PROPAGATE THROUGH NEURAL NETWORK
			local hypothesis = nn.Reshape(1):forward( netObject:testStep(batch_input) )  -- note that for training, reshape was done by convNet method
			
			-- update confusion matrix
			local binary_hypothesis = torch.ceil(-hypothesis + 1.5):reshape(hypothesis:size(1))  -- positive hypothesis = 1, negative hypothesis = 2
			local binary_targets = -batch_targets + 2  -- positive target label = 1, negative target label = 2
			confusion:batchAdd(binary_hypothesis, binary_targets)
			
			-- create binary array to label correct predictions ( 1 = true prediction, 0 = false prediction )
			local batch_correct_predictions = torch.eq(binary_hypothesis, binary_targets)
			
			-- record number of accurate predictions
			correct_predictions = correct_predictions + batch_correct_predictions:sum()
			
			
			-- recoord logloss contribution
			logloss = logloss + nn.BCECriterion():forward(hypothesis, batch_targets) * ( batch_size / num_cv_examples )
			  -- ideally need to regularize elementwise as max(min(hyp,1−10−15),10−15)
			  -- could first take elementwise log and then do max(min(-log(hyp),-log(10−15)),log(10−15)
			
			
			for fine_cv_idx = 1, batch_size do
				if batch_correct_predictions[fine_cv_idx] == 1 then
					if pos_or_neg == 'pos' then
						table.insert(confusion_filenames.true_pos, batch_img_filenames[fine_cv_idx])
					else
						table.insert(confusion_filenames.true_neg, batch_img_filenames[fine_cv_idx])
					end
				else
					if pos_or_neg == 'pos' then
						table.insert(confusion_filenames.false_neg, batch_img_filenames[fine_cv_idx])
					else
						table.insert(confusion_filenames.false_pos, batch_img_filenames[fine_cv_idx])
					end
				end
			end
			
			progress_idx = progress_idx + batch_size
		end
	end
	
	-- print confusion matrix
	print(confusion)
	--confusion:zero()
	print('LOGLOSS: ' .. logloss)
	
	-- averaged param use?
	if average then
		-- restore parameters
		parameters:copy(cachedparams)
	end
	
	local accuracy = correct_predictions / num_cv_examples
	return accuracy, confusion, confusion_filenames, logloss
	
end
