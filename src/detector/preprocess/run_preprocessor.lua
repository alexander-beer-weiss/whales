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
cmd:option('-height', 64, 'rescale height')
cmd:option('-width', 64, 'rescale width')
cmd:text()
opt=cmd:parse(arg)
opt.numWarps = { pos = opt.numPosWarps, neg = opt.numNegWarps }

-- now with objects
prep = preprocessor(opt)
imgs = prep:loadFromJPG()

torch.save(opt.preprocessedDir .. '/' .. 'preprocessed_imgs.dat', imgs)
