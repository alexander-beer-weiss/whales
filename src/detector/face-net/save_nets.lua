
require 'paths'  -- read OS directory structure
require 'optim'

local month =
{
  ['Jan'] = '01',
  ['Feb'] = '02',
  ['Mar'] = '03',
  ['Apr'] = '04',
  ['May'] = '05',
  ['Jun'] = '06',
  ['Jul'] = '07',
  ['Aug'] = '08',
  ['Sep'] = '09',
  ['Oct'] = '10',
  ['Nov'] = '11',
  ['Dec'] = '12'
}


local function date_stamp(date_string)
  local date = {}
  for w in string.gmatch(date_string,'%w+') do
    table.insert(date,w)
  end
  
  return date[7] .. '_' .. month[ date[2] ] .. '_' .. date[3]
end


local function time_stamp(time_string)
  local time = {}
  for w in string.gmatch(time_string,'%w+') do
    table.insert(time,w)
  end
  
  return time[4] .. '_' .. time[5]
end



---------------------------------------------------------------------------
-- Create save_nets metatable.  Then we can create save_nets objects.

save_nets = {}
save_nets.__index = save_nets
setmetatable(save_nets, {
	__call = function (cls, ...)
		return cls.new(...)
	end,
})

function save_nets.new(opt)
	local self = setmetatable({},save_nets)
	for key, value in pairs(opt) do
		self[key] = value
	end
	return self
end

---------------------------------------------------------------------------



function save_nets:prepNNdirs()
	if not paths.dir(self.netDatadir) then
		print('==> creating directory ' .. self.netDatadir)
		paths.mkdir(self.netDatadir)
	elseif paths.dir(self.netDatadir .. '/currNets') then
		print('==> removing directory ' .. self.netDatadir .. '/currNets from previous training session')
		os.execute('rm -r ' .. self.netDatadir .. '/currNets')
	end
  
	print('==> creating directory ' .. self.netDatadir .. '/currNets to store temporary data for current network')
	paths.mkdir(self.netDatadir .. '/currNets')
end



function save_nets:saveNN(NN_id, localNetObj)  -- do we really need localNetObj; does self = localNetObj ???
	torch.save(self.netDatadir .. '/currNets/NN_' .. NN_id .. '.dat', localNetObj)
	
	-- this is probably also saving gradParameters and probably other junk; do we need any of it?
	-- Can we just save parameters (NOTE, THAT WOULD NOT ALLOW US TO DETERMINE THE ARCHITECTURE OF THE NETWORK IF THE CODE WERE TO CHANGE)
	-- Maybe we can just save parameters and modules (what is included in modules?)
end


function save_nets:deleteNN(NN_id)
	os.execute('rm ' .. self.netDatadir .. '/currNets/NN_' .. NN_id .. '.dat')
end


  
function save_nets:saveBestNet(NN_idx, date_string, confusion, logloss, confusion_filenames, hyperParams)
	if not paths.dir(self.netDatadir .. '/bestNets') then
		print('==> creating directory ' .. self.netDatadir .. '/bestNets')
		paths.mkdir(self.netDatadir .. '/bestNets')
	end
	
	local date_dir = date_stamp(date_string)
	if not paths.dir(self.netDatadir .. '/bestNets/' .. date_dir) then
		print('==> creating directory ' .. self.netDatadir .. '/bestNets/' .. date_dir)
		paths.mkdir(self.netDatadir .. '/bestNets/' .. date_dir)
	end
	
	local time_dir = time_stamp(date_string)
	local time_append = ''
	if paths.dir(self.netDatadir .. '/bestNets/' .. date_dir .. '/' .. time_dir) then
		time_append = '_other' -- hack!
	end
	
	print('==> creating directory ' .. self.netDatadir .. '/bestNets/' .. date_dir .. '/' .. time_dir)
	paths.mkdir(self.netDatadir .. '/bestNets/' .. date_dir .. '/' .. time_dir .. time_append)
	
	os.execute('mv ' .. self.netDatadir .. '/currNets/NN_' .. NN_idx .. '.dat '
	            .. self.netDatadir .. '/bestNets/' .. date_dir .. '/' .. time_dir .. time_append .. '/NN.dat')
	
	if confusion then
		local file_stream = io.open(self.netDatadir .. '/bestNets/' .. date_dir .. '/' .. time_dir .. time_append .. '/confusion.txt', 'w')
		file_stream:write(tostring(confusion), '\n', 'LOGLOSS: ' .. logloss, '\n')
		file_stream:close()
	end
	
	if confusion_filenames then
		torch.save(self.netDatadir .. '/bestNets/' .. date_dir .. '/' .. time_dir .. time_append .. '/confusion_filenames.dat', confusion_filenames)
	end
	
	if hyperParams then
		torch.save(self.netDatadir .. '/bestNets/' .. date_dir .. '/' .. time_dir .. time_append .. '/hyperParams.dat', hyperParams)
	end
	
	--os.execute('ln -f -s ' .. self.netDatadir .. '/bestNets/' .. date_dir .. '/' .. time_dir .. time_append .. '/NN.dat' .. ' '
	--            .. self.netDatadir .. '/NN.dat')  -- the symlink won't load using torch.load (osexecute(readlink ...) for OSX or osexecute(readlink -f ...) for Linux)
	
	
	os.execute('rm -r ' .. self.netDatadir .. '/currNets')
	
end
  
-- ******* SHOULD CREATE A VERY_BEST_NET DIRECTORY WITH NET WITH VERY BEST ACCURACY ON CV
  
    