require 'cunn'
require 'cudnn'
require 'optim'
require 'nn'
require 'xlua'
require 'image'

---------------------DEFINING OPTIONS------------------
opt = lapp[[
   -s,--save                  (default "logs/")      subdirectory to save logs
   -b,--batchSize             (default 128)          batch size
   -r,--learningRate          (default 1)       learning rate(not much used)
   --activebatchSize           (default 1000)     batch size from which we filter 200 examples
   --alpha                     (default 0.001)   learning rate for the ndf(not much used)
   --gamma                     (default 0.95)      gamma used to find the cumulative reward
   --beta                     (default 0.001)      learning rate for actorcritic(not much used)
   --episodes                  (default 500)       maximum number of episodes
   --maxiter                  (default 3000)      maximum number of iterations in an episode
   --tou                      (default 0.8)       accuracy threshold
   --learningRateDecay        (default 1e-7)      learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.9)         momentum
   --epoch_step               (default 25)          epoch step
   --epochs                   (3000)                no. of epochs
   --spiltval                 (default 50000)       spilt size to train ndf or start episode
    
   --backend                  (default cudnn)            backend
   --type                     (default cuda)          cuda/float/cl
   --startfrom                (default 0)             from which epoch should I start the training
   --eps                      (default 0.05)            epsilon for greedy policy 
   --eta                      (default 0.001)           eta during updating weights
   
   --loadprev                 (default 0)          load previous  epoch
]]


--Algorithm for active learning
--L-initial-train = 30k
-- ->for  i = 1 to model.T :
--     for each iteration you have to train a new model to fit a new labeled set L
--     then select some unlabeled instances(around 1000/2000) to label using NDF.
--     and add it to the current labeled set, L = L U L' (Notice the letter U, it means union thus the increase in the L may not be size(L'))
--     shuffle before sending in the L to train_ndf()
--     For every 10k addition to the training set during training you need to update NDF.


-------------------default tensor type-------------------------------------------------------------

torch.setdefaulttensortype('torch.CudaTensor')
--------------------classes------------------------------------------------------------------------
classes  =  {'0','1','2','3','4','5','6','7','8','9'}

--------------------LOADING THE DATASET------------------------------------------------------------
local mnist = require 'mnist'

local trainset = mnist.traindataset().data -- 60000x32x32
local testset = mnist.testdataset().data -- 10000x32x32
-- add +1 to all the labels so that you can use those as indices
local targtrain = mnist.traindataset().label+1 --60000
local targtest = mnist.testdataset().label+1   --10000 

-------------------defining networks of NDF policy and mnist model and criterions------------------
--for NDF the input has 12 features  10 log probabilities+ normalized iteration+ average historical training accuracy.  

local ndf = nn.Sequential()
ndf:add(nn.Linear(12,6)):add(nn.Tanh()):add(nn.Linear(6,1)):add(nn.Sigmoid())
if filep(opt.save .. 'ndf.net') then
   ndf = torch.load(opt.save .. 'ndf.net')
end

   
local model = nn.Sequential()
ndf:add(nn.Linear(784,500)):add(nn.Tanh()):add(nn.Linear(500,10)):add(nn.LogSoftMax())
if filep(opt.save .. 'model.net') then
   ndf = torch.load(opt.save .. 'model.net')
end
 
-- local modelentropy = nn.Sequential()
-- ndf:add(nn.Linear(784,500)):add(nn.Tanh()):add(nn.Linear(500,10)):add(nn.LogSoftMax())
-- if filep(opt.save .. 'modelentropy.net') then
--    ndf = torch.load(opt.save .. 'modelentropy.net')
-- end

local actorcritic = nn.Sequential()
local parallel = nn.ParallelTable()
parallel:add(nn.Identity())
parallel:add(nn.Linear(opt.batchSize,12,true))
actorcritic:add(parallel)
actorcritic:add(nn.MM())
actorcritic:add(nn.ReLU())
actorcritic:add(nn.Linear(opt.batchSize,1))
actorcritic:add(nn.Sigmoid())





local criterion = nn.ClassNLLCriterion()
local criteriondf = nn.BCECriterion()
-------------------------initializing state features and state-------------------------------------------------
-- local labelcateg = torch.Tensor(opt.batchSize,#classes):zero()
local logprobs = torch.Tensor(opt.activebatchSize,#classes):fill(0.1)
-- local correctprob = torch.Tensor(opt.batchSize,1):fill(0.1)
-- local margin = torch.Tensor(opt.batchSize,1):fill(0)
local normalizediter = torch.Tensor(opt.activebatchSize,1):fill(0)
local trainacc = torch.Tensor(opt.activebatchSize,1):fill(0)
local margin = torch.Tensor(opt.activebatchSize,1):fill(0)
local entropy = torch.Tensor(opt.activebatchSize,1):fill(0)
function getstate()
   return torch.cat(logprobs,normalizediter,trainacc,2)
end
------------------------- weight initialization functions----------------------------------------------
function weightinit(model) 
   for k,v in pairs(model:findModules('nn.Linear')) do
      print({v})
      v.bias:fill(0)
      if k == 2 then
         v.bias:fill(2)
      end
      v.weight:normal(0,0.1)
   end
end
--initialize ndf
weightinit(ndf)
-------------------------confusion matrix and optimstate-------------------------------------------------------------
local confusion = optim.ConfusionMatrix(10)
-- local optimState = {
--   learningRate = opt.learningRate,
--   weightDecay = opt.weightDecay,
--   momentum = opt.momentum,
--   learningRateDecay = opt.learningRateDecay,
-- }

-------------------------testdev function-------------------------------------------------------------
function testdev(trainsetdev,targdev)
   local conf = optim.ConfusionMatrix(10)
   local outputs = model:forward(trainsetdev)
   conf:batchAdd(outputs,targdev)
   conf:updateValids()
   return conf.totalValid 
end
------------------------ trainNdf function--------------------------------------------------------------
--Algorithm for training datafilter
-- L-initial-ndf = 15k 
-- for t = 1 to ndf.episodes: 
--     for  i = 1 to ndf.T :
--         --when should I stop this iteration? when I reached 
--         for each iteration you have to train a new model to fit a new labeled set L
--         then select some unlabeled instances(around 200/1000)(by 200 I mean 200 new instances) to label using NDF.
--         and add it to the current labeled set, L = L U L'


--this function has inputs:  L : the labeled set among which half shall be used for pretraining the model and half shall 
--be used for labeling.

function trainNdf(L)
   pretrainsetind = L:spilt(L:size(1)/2)[1]
   posttrainsetind = L:spilt(L:size(1)/2)[2]

end


---------------------- trainModel function ------------------------------------------------------------------
--this function takes train dataset, validation dataset and outputs the validation accuracy.
function trainModel(L,Lval)


end