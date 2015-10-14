require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')

require 'image'
require 'paths'
require 'xlua'  -- progress bar

dofile '../face-net/convnet.lua'  -- not sure if this is necessary... is it saved with netObject??

local headDir = '../../..'

--local imgDir = '../../../imgs'
--local train_dir = imgDir .. '/train_imgs'
local imgDir = headDir .. '/fake_imgs/train_imgs'

local NN_dir = headDir .. '/NNsave/detector/bestNets/2015_10_5/14_13'

-- load netObject from appropriate NNsave directory

local hyperParams = torch.load(NN_dir .. '/hyperParams.dat')
local net = torch.load(NN_dir .. '/NN.dat')  -- this loads most recent best net... should really specify which net is wanted.

local netObject = convNet(hyperParams)
netObject:setNet(net)
netObject:reset()  -- this method should really be called initialize!



-- ALSO NEED TO GRAB HYPER-PARAMETER FILE AND READ OFF INFORMATION ABOUT CONVNET STRUCTURE
-- OR MAYBE CAN JUST GRAB THIS FROM netObject (the structure is save in nn.Sequential object; i.e. use getNet() method and then look for conv and pool layers)




--local window_height, window_width = 128, 128
local window_height, window_width = 64, 64  -- this will stay fixed; the training images will scale to accomodate comparison at different scales

math.randomseed(os.time())


function tensor_max(t) -- assuming t is a tensor of type torch.Tensor(1,height,width); returns max value in tensor and associated index in tensor
	
	local max_val_by_col, max_idx_by_col = torch.max(t,2)
	local max_val_by_row, max_idx_by_row = torch.max(t,3)
	
	local max_val, max_x_idx = torch.max(max_val_by_col,3)
	local _, max_y_idx = torch.max(max_val_by_row,2)
	
	local max_idx = torch.Tensor({max_x_idx[1][1][1],max_y_idx[1][1][1]})

	return max_val[1][1][1], max_idx  -- scalar and two component table
end



local all_hypotheses = {}

for _,dir in pairs(paths.dir(imgDir)) do
	if not string.match(dir,'^%.') then  -- skip any directories begining with '.'
		print(dir)
		for file in paths.files(imgDir .. '/' .. dir) do
			local file_label = string.match(file,'_%d+%.jpg')
			if(file_label) then  -- skip '.', '..', and any improperly labeled training example files
				print(file)
				
				local img = image.load(imgDir .. '/' .. dir .. '/' .. file, 3, float)
				
				img_height = img:size(2)
				img_width = img:size(3)
				
				local img_aspect_ratio = img:size(3) / img:size(2)  -- width/height
				
				
				local hypotheses = {}
				
				for inv_scale = 3, 9, 2 do  -- we consider windows of size 1/3, 1/5, 1/7, 1/9
					
					xlua.progress((inv_scale-1)/2, 4)
					
					local preliminary_scaled_height = window_height * inv_scale
					local preliminary_scaled_width = window_width * img_aspect_ratio * inv_scale
					
					-- need to make sure the scaled dimensions work with convolution/pooling structure of neural network
					-- for now assume pattern conv(5x5) -> pool(2x2) -> conv(5x5) -> pool(2x2)
					-- this can be generalized later
					
					-- determine width of of output maps after convolutions/pooling
					-- we must enforce that these are integer valued
					--local integer_post_conv_height = math.floor( ( ( (preliminary_scaled_height - 4) / 2 - 4 ) / 2 - 3 ) / 2 + 0.5 )  -- add .5; take floor
					--local integer_post_conv_width = math.floor( ( ( (preliminary_scaled_width - 4) / 2 - 4 ) / 2 - 3 ) / 2 + 0.5 )
					local integer_post_conv_height = math.floor( ( (preliminary_scaled_height - 4) / 2 - 4 ) / 2 + 0.5 )  -- add .5; take floor
					local integer_post_conv_width = math.floor( ( (preliminary_scaled_width - 4) / 2 - 4 ) / 2 + 0.5 )
					
					-- determine appropriate dimensions for input images by now reversing convolution/pooling process
					--local scaled_height = ( ( integer_post_conv_height * 2 + 3 ) * 2 + 4 ) * 2 + 4
					--local scaled_width = ( ( integer_post_conv_width * 2 + 3 ) * 2 + 4 ) * 2 + 4
					local scaled_height = ( integer_post_conv_height * 2 + 4 ) * 2 + 4
					local scaled_width = ( integer_post_conv_width * 2 + 4 ) * 2 + 4
					
					-- finally, perform scaling of original images
					local scaled_img = image.scale(img, scaled_height, scaled_width)  -- should keep track of scale factor *********
					
					
					-- FORWARD PROPAGATE CROPPED WINDOW THROUGH NEURAL NETWORK
					-- what about batches???  (Maybe no point with big images.)
					local hypothesis = netObject:testStep(scaled_img):clone()
					-- should extract torch.max of this, but for now want to print a map
					-- also, will need to map back the peak signal in hypothesis to pixel address in original image
					
					-- POSSIBLY... THE PICTURE WITH THE LARGEST TOTAL PERCENTAGE OF HIGH INTENSITY HYPOTHESIS PIXELS IS CORRECT!
					-- or try mean-pooling 2x2 or 3x3 with stride 1x1 to get back image with same dimension-1
					
					table.insert(hypotheses, hypothesis)  -- doesn't tell us the scale
					--print('')
					--print('')
					--print('')
					--print('HYP',hypothesis)
					
					
					local _, max_hypothesis_idx = tensor_max(hypothesis)
					print('')
					print('')
					print('')
					--print('MAXHYP',max_hypothesis_idx[1],max_hypothesis_idx[2])
					--max_hypothesis = torch.Tensor( max_hypothesis_idx )
					print('MAXHYP',torch.type(max_hypothesis),max_hypothesis_idx[1],max_hypothesis_idx[2])
					
					
					--local orig_window_idx = ( ( ( (max_hypothesis_idx-.5) + (13-1)/2 ) * 2 + (4-1)/2  ) * 2 + (5-1)/2 ) * 2 + (5-1)/2
					local orig_window_idx = ( ( ( (max_hypothesis_idx-.5) + (13-1)/2 ) * 2 + (5-1)/2 ) * 2 + (5-1)/2
					-- this will give me the scaled window location.  Could unscale...
					
					local cropped_window = image.crop(scaled_img,
					                                  orig_window_idx[1] - window_width / 2,orig_window_idx[2] - window_height / 2,
					                                  orig_window_idx[1] + window_width / 2,orig_window_idx[2] + window_height / 2)
					
					image.save(headDir..'/notebooks/crop'..inv_scale..file_label, cropped_window)
					
				end
				
				
				for i=1,4 do all_hypotheses[#all_hypotheses+1] = hypotheses[i]:clone() end
				
				torch.save(headDir .. '/notebooks/hypotheses.dat', all_hypotheses)  -- this gets overwritten on each file loop				
				
				collectgarbage()  -- torch and lua don't always talk nicely... prevents memory leak from opened images
				
			end
			
		end
	end
end



