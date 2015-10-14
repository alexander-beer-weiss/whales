
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
	elseif paths.dir(self.netDatadir .. '/newNets') then
		if paths.dir(self.netDatadir .. '/oldNets') then
			print('==> deleting directory ' .. self.netDatadir .. '/oldNets')
			paths.rmall(self.netDatadir .. '/oldNets', 'yes')
		end
		print('==> moving directory ' .. self.netDatadir .. '/newNets'
		.. ' to ' .. self.netDatadir .. '/oldNets')
		os.execute('mv ' .. self.netDatadir .. '/newNets ' .. self.netDatadir .. '/oldNets')
	end
  
	print('==> creating directory ' .. self.netDatadir .. '/newNets')
	paths.mkdir(self.netDatadir .. '/newNets')
end



function save_nets:saveNN(NN_id, localNetObj)  -- do we really need localNetObj; does self = localNetObj ???
	self.net=nn.Sequential()
	torch.save(self.netDatadir .. '/newNets/NN_' .. NN_id .. '.dat', localNetObj)
end


  
function save_nets:saveBestNet(NN_idx, date_string, confusion, confusion_filenames, hyperParams)
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
	
	os.execute('cp ' .. self.netDatadir .. '/newNets/NN_' .. NN_idx .. '.dat '
	            .. self.netDatadir .. '/bestNets/' .. date_dir .. '/' .. time_dir .. time_append .. '/NN.dat')
	
	if confusion then
		local file_stream = io.open(self.netDatadir .. '/bestNets/' .. date_dir .. '/' .. time_dir .. time_append .. '/confusion.txt', 'w')
		file_stream:write(tostring(confusion), '\n')
		file_stream:close()
	end
	
	if confusion_filenames then
		torch.save(self.netDatadir .. '/bestNets/' .. date_dir .. '/' .. time_dir .. time_append .. '/confusion_filenames.dat', confusion_filenames)
	end
	
	if hyperParams then
		torch.save(self.netDatadir .. '/bestNets/' .. date_dir .. '/' .. time_dir .. time_append .. '/hyperParams.dat', hyperParams)
	end
	
	os.execute('ln -f -s ' .. self.netDatadir .. '/bestNets/' .. date_dir .. '/' .. time_dir .. time_append .. '/NN.dat' .. ' '
	            .. self.netDatadir .. '/NN.dat')
	
end
  
-- ******* SHOULD CREATE A VERY_BEST_NET DIRECTORY WITH NET WITH VERY BEST ACCURACY ON CV
  
    