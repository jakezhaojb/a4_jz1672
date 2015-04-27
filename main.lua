--
----  Implementation of query_sentence and auxiliary functions
----  By Jake Zhao
----
--
ok,cunn = pcall(require, 'fbcunn')
if not ok then
    ok,cunn = pcall(require,'cunn')
    if ok then
        print("warning: fbcunn not found. Falling back to cunn") 
        LookupTable = nn.LookupTable
    else
        print("Could not find cunn or fbcunn. Either is required")
        os.exit()
    end
else
    deviceParams = cutorch.getDeviceProperties(1)
    cudaComputeCapability = deviceParams.major + deviceParams.minor/10
    LookupTable = nn.LookupTable
end
require('nngraph')
require('base')
ptb = require('data')

-- Some cmd parser, not needed in executation.
cmd = torch.CmdLine()
cmd:option('--submit', true, 'Testing mode.')
cmd:option('--devid', 1, 'device id')
opt = cmd:parse(arg)

-- Best model obtained
local params = {batch_size=20,
                seq_length=50,
                layers=3,
                decay=2,
                rnn_size=600,
                dropout=0.1,
                init_weight=0.1,
                lr=1,
                vocab_size=50,
                max_epoch=4,
                max_max_epoch=10,
                max_grad_norm=5}

function transfer_data(x)
  return x:cuda()
end

local modelname = 'clstm'
for k, v in pairs(params) do
   modelname = modelname .. '-' .. k .. '=' .. v
end
--print(modelname)

model = {}

function reset_state(state)
  state.pos = 1
  if model ~= nil and model.start_s ~= nil then
    for d = 1, 2 * params.layers do
      model.start_s[d]:zero()
    end
  end
end

function submit_test()
  reset_state(state_valid)
  g_disable_dropout(model.rnns)
  g_replace_table(model.s[0], model.start_s)
  while true do
     -- Load one character
     line = io.read("*line")
     if line == nil then
        break
     end
     -- Conversion to CudaTensor
     local x = transfer_data(map(line, params.batch_size))
     -- Forwarding
     _, model.s[1], pred = unpack(model.rnns[1]:forward({x[1], x[1], model.s[0]}))
     for i = 1, pred:size(2) do
        io.write(pred[1][i] .. ' ')
     end
     io.write('\n')
     io.flush()
     g_replace_table(model.s[0], model.s[1])
  end
  g_enable_dropout(model.rnns)
end

--g_init_gpu({opt.devid})
-- Loading data, for vocabulary building
state_train = {data=transfer_data(ptb.traindataset(params.batch_size))}
state_valid =  {data=transfer_data(ptb.validdataset(params.batch_size))}
--print("Network parameters:")
--print(params)
local states = {state_train, state_valid}
for _, state in pairs(states) do
 reset_state(state)
end
step = 0
epoch = 0
total_cases = 0
----
---- This part is the core part of submission
----
if opt.submit then
   print("Starting testing.")
   io.flush()
   io.write("OK GO\n")
   io.flush()
   local tmp = torch.load('/scratch/jz1672/lstm/models/' .. 'model' .. '.t7b', 'binary')
   model = tmp.model
   submit_test()
   os.exit(1)
end
----
---- Submission of testing done.
----
