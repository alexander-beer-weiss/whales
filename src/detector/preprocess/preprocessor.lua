require 'nn'  -- LCN
require 'image'  -- decode jpegs
require 'paths'  -- read OS directory structure


---------------------------------------------------------------------------
-- Create preprocessor metatable.  Then we can create preprocessor objects.

preprocessor = {}
preprocessor.__index = preprocessor
setmetatable(preprocessor, {
  __call = function (cls, ...)
    return cls.new(...)
  end,
})

function preprocessor.new(opt)
  local self = setmetatable({},preprocessor)
  for key, value in pairs(opt) do
    self[key] = value
  end
  return self
end

---------------------------------------------------------------------------



-- load jpg images and preprocess


function preprocessor:loadFromJPG()
	print 'Loading training data from JPG'
	
	local imgs = { pos = {}, neg = {} }  -- hash table with 'pos' and 'neg' keys and table initializers for values

	local cropped_dirs = { pos = self.posDir, neg = self.negDir }
	
	local num_imgs = 0
	local mean_sum = 0
	
	for pos_or_neg, cropped_dir in pairs(cropped_dirs) do
		print('Loading images from: ' .. cropped_dir)
		for file in paths.files(cropped_dir) do
			if string.match(file,'%.jpg$') then  -- skip '.', '..', and any improperly labeled training example files
				
				-- load images
				local img = image.load(cropped_dir .. '/' .. file)
				
				-- perform normalizations
				img = image.scale(img, self.height, self.width)
				
				if self.normStyle == 'independent' then
					img = ( img - torch.mean(img) ) / torch.std(img)
				else
					mean_sum = mean_sum + torch.mean(img)
				end
				
				num_imgs = num_imgs + 1
				
				-- could also try segmenting first; or LCN first, then segmenting, then Z-normalize
				
				table.insert(imgs[pos_or_neg], {['filename'] = file, ['img'] = img})

				-- torch and lua don't always talk nicely
				-- manual garbage collection prevents memory leaks from opened images
				collectgarbage()
			end
		end
	end
	
	if self.normStyle == 'collective' then
	
		local mean = mean_sum / num_imgs
		local variance_sum = 0
	
		print('Normalizing images')
		for pos_or_neg, img_table in pairs(imgs) do
			for _, example in ipairs(img_table) do
				example.img = example.img - mean
				variance_sum = variance_sum + torch.sum( torch.pow(example.img,2) ) / ( self.height * self.width )
			end
		end
	
		local std = math.sqrt( variance_sum / num_imgs )

		for pos_or_neg, img_table in pairs(imgs) do
			for _, example in ipairs(img_table) do
				example.img = example.img / std
			end
		end	
	
		-- return table of positive and negative training examples for whale face detector
		-- each postive/negative example is a pair: [1] file name; [2] table of warped images for that file
		imgs.mean = mean
		imgs.std = std
		
	end
		
	return imgs
end

