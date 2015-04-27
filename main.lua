--
----  Copyright (c) 2014, Facebook, Inc.
----  All rights reserved.
----
----  This source code is licensed under the Apache 2 license found in the
----  LICENSE file in the root directory of this source tree. 
----
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
print(modelname)

model = {}

function reset_state(state)
  state.pos = 1
  if model ~= nil and model.start_s ~= nil then
    for d = 1, 2 * params.layers do
      model.start_s[d]:zero()
    end
  end
end

function reset_ds()
  for d = 1, #model.ds do
    model.ds[d]:zero()
  end
end

function fp(state)
  g_replace_table(model.s[0], model.start_s)
  if state.pos + params.seq_length > state.data:size(1) then
    reset_state(state)
  end
  for i = 1, params.seq_length do
    local x = state.data[state.pos]
    local y = state.data[state.pos + 1]
    local s = model.s[i - 1]
    model.err[i], model.s[i] = unpack(model.rnns[i]:forward({x, y, s}))
    state.pos = state.pos + 1
  end
  g_replace_table(model.start_s, model.s[params.seq_length])
  return model.err:mean()
end

function bp(state)
  paramdx:zero()
  reset_ds()
  for i = params.seq_length, 1, -1 do
    state.pos = state.pos - 1
    local x = state.data[state.pos]
    local y = state.data[state.pos + 1]
    local s = model.s[i - 1]
    local derr = transfer_data(torch.ones(1))
    local tmp = model.rnns[i]:backward({x, y, s},
                                       {derr, model.ds, 0})[3]
    g_replace_table(model.ds, tmp)
    cutorch.synchronize()
  end
  state.pos = state.pos + params.seq_length
  model.norm_dw = paramdx:norm()
  if model.norm_dw > params.max_grad_norm then
    local shrink_factor = params.max_grad_norm / model.norm_dw
    paramdx:mul(shrink_factor)
  end
  paramx:add(paramdx:mul(-params.lr))
end

function run_valid()
  reset_state(state_valid)
  g_disable_dropout(model.rnns)
  local len = (state_valid.data:size(1) - 1) / (params.seq_length)
  local perp = 0
  for i = 1, len do
    perp = perp + fp(state_valid)
  end
  print("Validation set perplexity : " .. g_f3(5.6*torch.exp(perp / len)))
  g_enable_dropout(model.rnns)
end

function run_test()
  reset_state(state_test)
  g_disable_dropout(model.rnns)
  local perp = 0
  local len = state_test.data:size(1)
  g_replace_table(model.s[0], model.start_s)
  for i = 1, (len - 1) do
    local x = state_test.data[i]
    local y = state_test.data[i + 1]
    local s = model.s[i - 1]
    perp_tmp, model.s[1] = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
    perp = perp + perp_tmp[1]
    g_replace_table(model.s[0], model.s[1])
  end
  print("Test set perplexity : " .. g_f3(5.6*torch.exp(perp / (len - 1))))
  g_enable_dropout(model.rnns)
end

function submit_test()
  reset_state(state_valid)
  g_disable_dropout(model.rnns)
  g_replace_table(model.s[0], model.start_s)
  while true do
     line = io.read("*line")
     if line == nil then
        break
     end
     local x = transfer_data(map(line, params.batch_size))
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

g_init_gpu({opt.devid})
state_train = {data=transfer_data(ptb.traindataset(params.batch_size))}
state_valid =  {data=transfer_data(ptb.validdataset(params.batch_size))}
print("Network parameters:")
print(params)
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
   print("OK GO")
   io.flush()
   local tmp = torch.load('/scratch/jakez/lstm/models/' .. modelname .. '.t7b', 'binary')
   model = tmp.model
   submit_test()
   os.exit(1)
end
----
---- Submission of testing done.
----
