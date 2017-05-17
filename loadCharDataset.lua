train =  torch.load("train.t7")
test =  torch.load("test.t7")
setmetatable(train, 
    {__index = function(t, i) 
                    return {t.data[i], t.labels[1][i]} 
                end}
);
setmetatable(test, 
    {__index = function(t, i) 
                    return {t.data[i], t.labels[1][i]} 
                end}
);
