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


print(opt)
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
 
local modelentropy = nn.Sequential()
ndf:add(nn.Linear(784,500)):add(nn.Tanh()):add(nn.Linear(500,10)):add(nn.LogSoftMax())
if filep(opt.save .. 'modelentropy.net') then
   ndf = torch.load(opt.save .. 'modelentropy.net')
end

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
local logprobs = torch.Tensor(opt.batchSize,#classes):fill(0.1)
-- local correctprob = torch.Tensor(opt.batchSize,1):fill(0.1)
-- local margin = torch.Tensor(opt.batchSize,1):fill(0)
local normalizediter = torch.Tensor(opt.batchSize,1):fill(0)
local trainacc = torch.Tensor(opt.batchSize,1):fill(0)

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
-------------------------trainfill to indices function------------------------------------------------------
function tftoind(trainfill)
   local total = trainfill:sum()
   local indices = torch.Tensor(total):zero()
   local j = 0
   for i = 1,60000 do 
      if trainfill[i] then
         j++
         indices[j] = i 
      end
   end
   return indices
end
------------------------- trainpolicy function--------------------------------------------------------------
--this takes no. of training examples, model as inputs and run episodes on those training examples
-- these training examples can be derived from a one hot vector which notes what examples had undergone training. when the no. of new examples 
--reaches more than 5000 this function is invoked and thus ndf is trained.
--you may use many reward functions for 
--how does the original training goes.
--You should add examples to the training set which bear value to the training set. So you should select examples which when selected increases
--the generalization error. So in each episode you have many iterations and in each iteration you add 1000 examples to the training set.
--Does the model gets reinitialized after every iteration, if not then you get to choose better examples after every iteration.
local trainfill = torch.Tensor(60000):zero()
local curreward = 0
local prevreward = 0
local filter = torch.Tensor(opt.batchSize):fill(1)
local filtbatch;
local numb = 20000
local numbdev = 2000
trainfill[{{1,numb}}]:fill(1)
trainfill = trainfill:index(1,torch.randperm(60000):long())
-- the model in this function gets initiated every episode so be careful. send in a different temporary model or a cloned one.
-- the the max iter way of ending an episode here is a little naive so thinking of a better way is necessary.
function trainpolicy(initmodel,numb,numbdev) 

   --pick numb from 60k using trainfill and then start the episode 
   
   local trainsett = tftoind(trainfill)
   --initialize ndf
   
   for l = 1, opt.episodes do
      confusion:zero()
      trainsetdevin = trainsett:long():split(numb-numbdev)[2]
      local trainsetdev = trainset:index(1,trainsetdevin)
      local targdev = trainset:index(1,trainsetdevin)
      trainsett = trainset:index(1,trainsett:long():split(numb-numbdev)[1])
      local model = initmodel:clone()
      local stop = false
      local T = 0
      while ~stop do
         T = T+1
         if T > opt.maxiter then
            stop = true
         end
         local shuffle = torch.randperm(trainsett:size(1)):long()
         local batchindx = trainsett:index(1,shuffle):split(opt.batchSize)[1]
         collectgarbage()
         local batch = trainset:index(1,batchindx)
         batch = torch.reshape(batch,batch:size(1),batch:size(2)*batch:size(3))
         local targets = targtrain:index(1,batchindx)

         --forward pass every instance in the batch and get state of that instance 
         local states;
         local prevstates = getstate()
         local prevfilter = filter
         confusion:updateValids()
         for i = 1,batch:size(1) do
            local output = model:forward(batch[i])
            logprobs[i] = torch.log(output)
            -- labelcateg[i]:zero()
            -- labelcateg[i][targets[i]] = 1
            -- correctprob[i][1] = logprobs[i][targets[i]]
            -- val,indx = torch.max(output)
            -- margin[i][1] = output[targets[i]]-val
            normalizediter[i][1] = T/opt.maxiter

            trainacc[i][1] = confusion.totalValid
            states = getstate()
         end

         --find out filtered batch
         ndfprobs = ndf:forward(states)
         filter = torch.ge(ndfprobs,0.5)
         local num = torch.sum(filter)
         local indx = torch.CudaLongTensor(num)
         local j = 0;
         for i = 1,opt.batchSize do
            if filter[i] == 1 then
               j = j + 1
               indx[j] = i
            end
         end

         batch = batch:index(1,indx)
         targets = targets:index(1,indx)
         collectgarbage()
         if filtbatch then
            filtbatch = torch.cat(filtbatch,batch,1)
         else
            filtbatch = batch
         end
         --if filtered batch has more than M instances then train that network with that batch
         if filtbatch:size(1)>=opt.batchSize then
            filtbatch = filtbatch:split(opt.batchSize)
            batch = filtbatch[1]
            filtbatch = filtbatch[2]
            local parametersm,gradParametersm = model:getParameters()
            local feval = function(x)
               if x ~= parametersm then parametersm:copy(x) end
               gradParametersm:zero()
               local outputs = model:forward(batch)
               f = criterion:forward(outputs, targets)
               local df_do = criterion:backward(outputs, targets)
               model:backward(inputs, df_do)
               confusion:batchAdd(outputs, targets)
               return f,gradParametersm
            end
            optim.adam(feval,parametersm)
            --find the reward 
            local tmp = torch.randperm(trainsetdev:size(1)):split(1000)[1]
            curreward =  testdev(trainsetdev:index(1,tmp),targdev:index(1,tmp))
            
            
            --update the ndf and actor critic
            --first you need to find Q
            local Q = actorcritic:forward{states,filter}
            local parametersn,gradParametersn = ndf:getParameters()
            local ndfeval = function(x)
               if x ~= parametersn then parametersn:copy(x) end
               gradParametersn:zero()
               local outputs = ndf:forward(states)
               f = criteriondf:forward(outputs, torch.Tensor(opt.batchSize):fill(1))
               local df_do = criteriondf:backward(outputs, torch.Tensor(opt.batchSize):fill(1))
               ndf:backward(states, Q*df_do)
               return f,gradParametersn
            end
            optim.adam(ndfeval,parametersn)
            --so you found the Q now substitute it in the function and multiply it with the gradients
            --after updating the ndf you need to update actor critic.
            local q = prevreward + opt.gamma*actorcritic:forward(states,filter)-actorcritic:forward(prevstates,prevfilter)
            local parametersa,gradParametersa = actorcritic:getParameters()
            local actoreval = function(x)
               if x ~= parametersa then parametersa:copy(x) end
               gradParametersa:zero()
               local outputs = ndf:forward{prevstates,prevfilter}
               actorcritic:backward({prevstates,prevfilter}, q)
               return f,gradParametersa
            end
            optim.adam(actoreval,parametersa)
         end
         
         prevreward = curreward

      end
   end
end

