require 'cunn'
require 'cudnn'
require 'optim'
require 'nn'
require 'xlua'
require 'image'
require 'paths'
cudnn.benchmark=true

---------------------DEFINING OPTIONS------------------
opt = lapp[[
   -s,--save                  (default "logs/")      subdirectory to save logs
   -b,--batchSize             (default 128)          batch size
   -r,--learningRate          (default 1)       learning rate(not much used)
   --activebatchSize           (default 200)     batch size from which we filter 200 examples
   --trainval                   (default 5000)   validation set size to measure generalization in train function
   --ndfval                     (default 2000)   validation set size to measure generalization in ndf      
   --ndfinterval                (default 10000)  after 10000 additions to labeled dataset ndf starts retraining.
   --alpha                     (default 0.001)   learning rate for the ndf(not much used)
   --gamma                     (default 0.95)      gamma used to find the cumulative reward
   --beta                     (default 0.001)      learning rate for actorcritic(not much used)
   --episodes                  (default 500)       maximum number of episodes
   --maxiter                  (default 1500)      maximum number of iterations in an episode
   --tou                      (default 0.8)       accuracy threshold
   --learningRateDecay        (default 1e-7)      learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.9)         momentum
   --epoch_step               (default 25)          epoch step
   --epochs                   (default 3000)                no. of epochs
   --splitval                 (default 50000)       split size to train ndf or start episode
    
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

-- torch.setdefaulttensortype('torch.CudaTensor')
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
ndf:add(nn.Linear(15,8)):add(nn.Tanh()):add(nn.Linear(8,1)):add(nn.Sigmoid())
if paths.filep(opt.save .. 'ndf.net') then
   ndf = torch.load(opt.save .. 'ndf.net')
end

   
-- local model = nn.Sequential()
-- ndf:add(nn.Linear(784,500)):add(nn.Tanh()):add(nn.Linear(500,10)):add(nn.LogSoftMax())
-- if filep(opt.save .. 'model.net') then
--    ndf = torch.load(opt.save .. 'model.net')
-- end
 
-- local modelentropy = nn.Sequential()
-- ndf:add(nn.Linear(784,500)):add(nn.Tanh()):add(nn.Linear(500,10)):add(nn.LogSoftMax())
-- if filep(opt.save .. 'modelentropy.net') then
--    ndf = torch.load(opt.save .. 'modelentropy.net')
-- end

-- local actorcritic = nn.Sequential()
-- local parallel = nn.ParallelTable()
-- parallel:add(nn.Identity())
-- parallel:add(nn.Linear(opt.batchSize,12,true))
-- actorcritic:add(parallel)
-- actorcritic:add(nn.MM())
-- actorcritic:add(nn.ReLU())
-- actorcritic:add(nn.Linear(opt.batchSize,1))
-- actorcritic:add(nn.Sigmoid())





local criterion = nn.ClassNLLCriterion()
local criteriondf = nn.BCECriterion()
criterion = criterion:cuda()
criteriondf = criteriondf:cuda()
-------------------------initializing state features and state-------------------------------------------------
-- local labelcateg = torch.Tensor(opt.batchSize,#classes):zero()
local logprobs = torch.Tensor(opt.activebatchSize,#classes):fill(0.1)
-- local correctprob = torch.Tensor(opt.batchSize,1):fill(0.1)
-- local margin = torch.Tensor(opt.batchSize,1):fill(0)
local normalizediter = torch.Tensor(opt.activebatchSize,1):fill(0)
local trainacc = torch.Tensor(opt.activebatchSize,1):fill(0)
local margin = torch.Tensor(opt.activebatchSize,1):fill(0)
local entropy = torch.Tensor(opt.activebatchSize,1):fill(0)
local valacc = torch.Tensor(opt.activebatchSize,1):fill(0)
function getstate()
   return torch.cat({logprobs,normalizediter,trainacc,margin,entropy,valacc},2)
end
------------------------- weight initialization functions----------------------------------------------
function weightinit(model) 
   for k,v in pairs(model:findModules('nn.Linear')) do
      -- print({v})
      v.bias:fill(0)
      if k == 2 then
         v.bias:fill(2)
      end
      v.weight:normal(0,0.1)
   end
end
function weightinitm(model) 
   for k,v in pairs(model:findModules('nn.Linear')) do
      
      v.bias:fill(0)
      v.weight:normal(0,0.1)
   end
end
--initialize ndf
weightinit(ndf)
--converting the ndf to cudnn
ndf = ndf:cuda()
cudnn.convert(ndf, cudnn)

-------------------------confusion matrix and optimstate-------------------------------------------------------------
local confusion = optim.ConfusionMatrix(10)
-- local optimState = {
--   learningRate = opt.learningRate,
--   weightDecay = opt.weightDecay,
--   momentum = opt.momentum,
--   learningRateDecay = opt.learningRateDecay,
-- }

-------------------------testdev function-------------------------------------------------------------
function testdev(model,trainsetdev,targdev,batchSize)
   local conf = optim.ConfusionMatrix(10)
   local outputs = model:forward(torch.reshape(trainsetdev,trainsetdev:size(1),784):cuda())
   conf:batchAdd(outputs,targdev)
   conf:updateValids()
   return conf.totalValid 
end
------------------------ trainNdf function--------------------------------------------------------------
--Algorithm for training datafilter
-- L-initial-ndf = 15k 
-- for t = 1 to ndf.episodes: 
--     shuffle L and then partition pretrain, val and posttrain dataset.
--     for  i = 1 to ndf.T :
--         select some unlabeled instances(around 200) to label using NDF.
--         if the filtered batch is less than 200 then don't update the model until 200 instances arrive. 
--         and add it to the current labeled set, L = L U L'
--         for each iteration you have to train a new model to fit a new labeled set L
--         get the reward and update the ndf.



--this function has inputs: L : the labeled set among which half shall be used for pretraining the model and half shall 
--be used for labeling.

function trainNdfReinforce(L)
   print('training ndf...')
   local model = nil
   local baseline = 0
   local reward = 0
   for l = 1, opt.episodes do
      local filtind = nil
      local rnd = torch.randperm(L:size(1)):long()
      L = L:index(1,rnd)
      pretrainsetind = L:split(L:size(1)/2)[1]
      posttrainsetind = L:split(L:size(1)/2)[2]
      pretrainsetind = pretrainsetind:split(pretrainsetind:size(1)-opt.ndfval)
      prevalset = pretrainsetind[2]
      pretrainsetind = pretrainsetind[1]
      unlabeledpool = posttrainsetind
      local modifiedtrain = pretrainsetind
      local stop = false
      model,reward,confusion = trainModel(modifiedtrain,prevalset)
      baseline = 0.8*baseline + 0.2*reward
      local unlabeledstates = nil
      local T = 0
      while not stop do
         T = T+1
         if modifiedtrain:size(1)/(L:size(1)-opt.ndfval) >= 0.98 then
            stop = true
         end
         local activebatch = unlabeledpool:split(unlabeledpool:size(1)-opt.activebatchSize)
         unlabeledpool = activebatch[1]
         activebatch = activebatch[2]
         --find the states needed for activebatch
         --update state variables
         if activebatch:size(1) <opt.activebatchSize then
            break
         end
         logprobs = torch.log(model:forward(torch.reshape(trainset:index(1,activebatch),opt.activebatchSize,784):cuda()))
         logprobs  = logprobs:double()
         normalizediter:fill(T/opt.maxiter)
         trainacc:fill(confusion.totalValid)
         local expprobs = logprobs:exp()
         local expprobs,ind = torch.sort(expprobs,2,true)
         
         -- print({expprobs[{{},{1}}]-expprobs[{{},{2}}]})
         -- t = expprobs
         -- print(t,{t})
         -- print({margin[{{},{1}}]})
         margin[{{},{1}}] = (expprobs[{{},{1}}]-expprobs[{{},{2}}])
         entropy[{{},{1}}] = torch.sum(-torch.cmul(logprobs,logprobs:exp()),2)
         valacc[{{},{1}}]:fill(reward)
         states = getstate()
         -- print({states})
         ndfoutputs = ndf:forward(states:cuda())
         if not unlabeledstates then
            unlabeledstates = states
         else
            unlabeledstates = torch.cat(unlabeledstates,states,1)
         end
         --filter the batch.
         local filter = torch.ge(ndfoutputs,0.5)
         local oppfilter = torch.lt(ndfoutputs,0.5)
         local oppnum = torch.sum(oppfilter)
         local num = torch.sum(filter)
         if num>0 then
            local indx = torch.CudaLongTensor(num)
            local oppindx = torch.CudaLongTensor(oppnum)
            local j = 0;
            for i = 1,opt.activebatchSize do
               if filter[i] == 1 then
                  j = j + 1
                  indx[j] = i
               end
            end
            local j = 0;
            for i = 1,opt.activebatchSize do
               if oppfilter[i] == 1 then
                  j = j + 1
                  oppindx[j] = i
               end
            end
            collectgarbage()
            if filtind then
               filtind = torch.cat(filtind,indx)
            else
               filtind = indx
            end
         
         --if filtered batch has length less than opt.activebatchSize then do not train else add it to the training set and start training.
            if filtind:size(1)>=opt.activebatchSize then
               local batch = filtind:split(opt.activebatchSize)[1]
               filtind = filtind:split(opt.activebatchSize)[2]
               modifiedtrain = torch.cat(modifiedtrain,batch)
               unlabeledpool = torch.cat(unlabeledpool,oppindx)
               confusion:zero()
               model,reward,confusion = trainModel(modifiedtrain,prevalset)
            end
         end
      end
      --now train the ndf by using unlabeled states.
      unlabeledstates = unlabeledstates:split(opt.activebatchSize)
      for t = 1,(#unlabeledstates) do
         local parametersn,gradParametersn = ndf:getParameters()
         local ndfeval =  function(x) 
            if x ~= parametersn then parametersn:copy(x) end
            gradParametersn:zero()
            print(t,unlabeledstates)
            local outputs = ndf:forward(unlabeledstates[t]:cuda())
            local targets = torch.ge(outputs,0.5):cuda()
            print({outputs},{targets})
            f = criteriondf:forward(outputs,targets)
            df_do = criteriondf:backward(outputs,targets)
            ndf:backward(unlabeledstates[t]:cuda(),(reward-baseline)*df_do)
            return f,gradParametersn
         end
         optim.adam(ndfeval,parametersn)
      end
   end
end


---------------------- trainModel function ------------------------------------------------------------------
--this function takes train dataset, validation dataset and outputs the validation accuracy.
function trainModel(L,Lval)
   print('training model...')
   local model = nn.Sequential()
   model:add(nn.Linear(784,500)):add(nn.Tanh()):add(nn.Linear(500,10)):add(nn.LogSoftMax())
   weightinitm(model)

   -- converting the model to cudnn
   cuda = model:cuda()
   cudnn.convert(model, cudnn)

   local parameters,gradParameters = model:getParameters()
   local epochs = math.floor(L:size(1)/trainset:size(1)*500)
   local batchSize = 128
   local reward = 0
   for ep = 1,epochs do
      -- print({L})
      xlua.progress(ep,epochs)
      local shuffle = torch.randperm(L:size(1)):long()
      batches = L:index(1,shuffle):long():split(batchSize)
      for i = 1 , (#batches)-1 do
         local batch = trainset:index(1,batches[i])
         local targets = targtrain:index(1,batches[i])
         local feval = function(x)
            if x ~= parameters then parameters:copy(x) end
            gradParameters:zero()
            -- print(torch.reshape(batch,batchSize,784))
            -- print({batch})
            batch = torch.reshape(batch,batchSize,784):cuda()
            local outputs = model:forward(batch)
            -- print({outputs},{targets})
            -- print({outputs},{targets:cuda()})
            f = criterion:forward(outputs, targets)
            local df_do = criterion:backward(outputs, targets:cuda())
            -- print({batch},{df_do})
            model:backward(batch, df_do)
            confusion:batchAdd(outputs, targets)
            return f,gradParameters
         end
         optim.adam(feval,parameters)
      end
      -- if ep%5 == 0 then
      --    reward = testdev(trainset:index(1,Lval),targtrain:index(1,Lval))
      -- end
   end
   reward = testdev(model,trainset:index(1,Lval),targtrain:index(1,Lval),batchSize)
   return model,reward,confusion
end

------------------------------------------------final train function-------------------------------------------------
--Algorithm for active learning
--train the ndf first
--L-initial-train = 30k
-- ->for  i = 1 to model.T :
--     then select some unlabeled instances(around 200) to label using NDF.
--     and add it to the current labeled set, L = L U L' (Notice the letter U, it means union thus the increase in the L may not be size(L'))
--     for each iteration you have to train a new model to fit a new labeled set L
--     For every 10k addition to the training set during training you need to update NDF.(shuffle before sending in the L to train_ndf())

function train() 
   local filtind = nil
   local rnd = torch.randperm(trainset:size(1)):long()
   trainset = trainset:index(1,rnd)
   pretrainsetind = rnd:split(trainset:size(1)/2)[1]
   posttrainsetind = rnd:split(trainset:size(1)/2)[2]
   pretrainsetind = pretrainsetind:split(pretrainsetind:size(1)-opt.trainval)
   prevalset = pretrainsetind[2]
   pretrainsetind = pretrainsetind[1]
   unlabeledpool = posttrainsetind
   unlabeledpool = unlabeledpool:split(opt.activebatchSize)
   local modifiedtrain = pretrainsetind
   model,reward,confusion = trainModel(modifiedtrain,prevalset)
   trainNdfReinforce(modifiedtrain)
   local unlabeledstates = nil
   local T = 0
   while not stop do
      T = T+1
      if modifiedtrain:size(1)/(trainset:size()-opt.trainval) >= 0.98 then
         stop = true
      end
      local activebatch = unlabeledpool:split(unlabeledpool:size(1)-opt.activebatchSize)
      unlabeledpool = activebatch[1]
      activebatch = activebatch[2]
      --find the states needed for activebatch
      --update state variables
      logprobs = torch.log(model:forward(activebatch))
      normalizediter:fill(T/opt.maxiter)
      trainacc:fill(confusion.totalValid)
      local expprobs = logprobs:exp()
      local expprobs,ind = torch.sort(2,expprobs)
      margin[{{},{1}}] = expprobs[{{},{-1}}]-expprobs[{{},{-2}}]
      entropy[{{},{1}}] = torch.sum(-torch.cmul(logprobs,logprobs:exp()),2)
      valacc[{{},{1}}]:fill(reward)
      states = getstate()
      ndfoutputs = ndf:forward(states)
      if not unlabeledstates then
         unlabeledstates = states
      else
         unlabeledstates = torch.cat(unlabeledstates,states)
      end
      --filter the batch.
      local filter = torch.ge(ndfoutputs,0.5)
      local oppfilter = torch.lt(ndfoutputs,0.5)
      local oppnum = torch.sum(oppfilter)
      local num = torch.sum(filter)
      local indx = torch.CudaLongTensor(num)
      local oppindx = torch.CudaLongTensor(oppnum)
      local j = 0;
      for i = 1,opt.activebatchSize do
         if filter[i] == 1 then
            j = j + 1
            indx[j] = i
         end
      end
      local j = 0;
      for i = 1,opt.activebatchSize do
         if oppfilter[i] == 1 then
            j = j + 1
            oppindx[j] = i
         end
      end
      collectgarbage()
      if filtind then
         filtind = torch.cat(filtind,indx)
      else
         filtind = indx
      end
         

      --if filtered batch has length less than opt.activebatchSize then do not train else add it to the training set and start training.
      if filtind:size(1)>=opt.activebatchSize then
         local batch = filtind:split(opt.activebatchSize)[1]
         filtind = filtind:split(opt.activebatchSize)[2]
         modifiedtrain = torch.cat(modifiedtrain,batch)
         unlabeledpool = torch.cat(unlabeledpool,oppindx)
         confusion:zero()
         model,reward,confusion = trainModel(modifiedtrain,prevalset)
      end
      -- for every 10K addition to training set update ndf
      if (modifiedtrain:size(1)-pretrainsetind:size(1))%opt.ndfinterval then
         trainNdfReinforce(modifiedtrain)
      end
   end
end

-- run it up
train()

