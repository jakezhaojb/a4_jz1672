--
----  Implementation of query_sentence and auxiliary functions
----  By Jake Zhao
----
--
function query_sentence()
   while true do
     print("Query: len word1 word2 etc")
     -- Reading from stdin
     line = io.read('*line')
     -- Parse it and pre-process it
     if line == nil then error('None input') end
     line = stringx.split(line)
     if tonumber(line[1]) == nil then
        error('First should be digit')
     end
     for i = 2, #line do
        -- Map is a LookupTable conversion
        if vocab_map[line[i]] == nil then
           error("No word '" .. line[i] .. "' in vocab.") 
        end
     end
     local number = tonumber(line[1])
     local tmp = {}
     for i = 2, #line do
        tmp[i-1] = line[i]
     end
     line = tmp
     local len = #line
     local num_pred = number - len  -- The number to be predicted
     assert(num_pred > 0)
     -- Set the state
     g_replace_table(model.s[0], model.start_s)
     for i = 1, num_pred do
        local query = map(line, params.batch_size)
        query = transfer_data(query)
        -- Initialize a table storing query-states
        state_query = {}
        state_query.pos = 1
        state_query.data = query
        reset_state(state_query)
        g_disable_dropout(model.rnns)
        -- Predict the next
        local x = query[len+i-2]
        local y = query[len+i-1]
        local s = model.s[i - 1]
        perp_tmp, model.s[1], pred = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
        -- Fetch the maximum and output
        local _, imax = pred:max(2)
        imax = imax:squeeze(2)
        -- Inverse map to entry to a word in LookUpTable
        local new_word = inverse_map(imax[1])
        -- Update model state
        g_replace_table(model.s[0], model.s[1])
        table.insert(line, new_word)
     end
     -- Print out
     print_table(line)
     g_enable_dropout(model.rnns)
   end
end


function map(line, batch_size)
   local data
   if type(line) == 'table' then
      -- If the line has been parsed to table
      data = line
   elseif type(line) == 'string' then
      -- If the line has NOT been parsed to table
      data = stringx.replace(line, '\n', '<eos>')
      data = stringx.split(data)
   else
      error('No supported input')
   end
   local x = torch.zeros(#data)
   for i = 1, #data do
      x[i] = vocab_map[data[i]]
   end
   -- RepeatTensor for batch_size match-up
   x = x:reshape(x:nElement(), 1)
   x = torch.repeatTensor(x, 1, batch_size)
   return x
end


function inverse_map(data)
 for k, v in pairs(vocab_map) do
    if data == v then
       return k
    end
 end
 return '' -- No found
end

