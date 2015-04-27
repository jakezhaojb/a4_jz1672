--[[
Q1
--]]

require 'nn'
require 'nngraph'

-- TODO would it be garbage collected?
local x1 = nn.Identity()()
local x2 = nn.Identity()()
local x3 = nn.Linear(5,4)()

local x23 = nn.CMulTable()({x2, x3}):annotate{name = 'CMul'}
local x123 = nn.CAddTable()({x1, x23}):annotate{name = 'CAdd'}

model = nn.gModule({x1,x2,x3}, {x123})

graph.dot(model.fg, 'model', 'model')

local data1 = torch.rand(2,4)
local data2 = torch.rand(2,4)
local data3 = torch.rand(2,5)

print(model:forward({data1, data2, data3}))
