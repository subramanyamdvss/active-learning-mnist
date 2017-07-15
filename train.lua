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


--Algorithm for active learning
--L-initial-train = 30k
-- ->for  i = 1 to model.T :
--     for each iteration you have to train a new model to fit a new labeled set L
--     then select some unlabeled instances(around 1000/2000) to label using NDF.
--     and add it to the current labeled set, L = L U L'

--Algorithm for training datafilter
-- L-initial-ndf = 15k 
-- for t = 1 to ndf.episodes: 
--     for  i = 1 to ndf.T :
--         for each iteration you have to train a new model to fit a new labeled set L
--         then select some unlabeled instances(around 1000/2000) to label using NDF.
--         and add it to the current labeled set, L = L U L'


--draw up some partitions of the dataset
