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
	
	--local normkernel = image.gaussian1D(9)  -- would we be better with a different LCN (check dp module)?  or sharpening with image.convolution?

	for pos_or_neg, cropped_dir in pairs(cropped_dirs) do
		print('Loading images from: ' .. cropped_dir)
		for file in paths.files(cropped_dir) do
			if string.match(file,'%.jpg$') then  -- skip '.', '..', and any improperly labeled training example files
				
				-- load images
				local img = image.load(cropped_dir .. '/' .. file)
				
				-- perform normalizations
				img = image.scale(img, self.height, self.width)
				img = ( img - torch.mean(img) ) / torch.std(img)
				--img = nn.SpatialContrastiveNormalization(3, normkernel):forward(img)  -- 3 means RGB; should this be before Z-values normalization?
				
				-- could also try segmenting first; or LCN first, then segmenting, then Z-normalize
				
				table.insert(imgs[pos_or_neg], {['filename'] = file, ['img'] = img})

				-- torch and lua don't always talk nicely
				-- manual garbage collection prevents memory leak from opened images
				collectgarbage()
			end
		end
	end

	-- return table of positive and negative training examples for whale face detector
	-- each postive/negative example is a pair: [1] file name; [2] table of warped images for that file
	return imgs
end

