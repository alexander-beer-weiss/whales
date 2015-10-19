require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
dofile 'preprocessor.lua'

cmd = torch.CmdLine()
cmd:text()
cmd:text('SVHN Training/Optimization')
cmd:text()
cmd:text('Options:')
cmd:option('-posDir', '../../../imgs/detector_imgs/cropped/positives', 'location of positive whale face images')
cmd:option('-negDir', '../../../imgs/detector_imgs/cropped/negatives', 'location of negative whale face images')
cmd:option('-preprocessedDir', '../../../imgs/detector_imgs/preprocessed', 'location of preprocessed directory')
cmd:option('-cropSize', 64, 'defines the height and width of cropped images')
cmd:option('-normStyle', 'collective', 'collective | independent : normalize each image based on collective mean/std or independently')
cmd:text()
opt=cmd:parse(arg)

opt.height, opt.width = opt.cropSize, opt.cropSize

-- now with objects
prep = preprocessor(opt)
imgs = prep:loadFromJPG()

torch.save(opt.preprocessedDir .. '/' .. 'preprocessed_imgs.dat', imgs)
