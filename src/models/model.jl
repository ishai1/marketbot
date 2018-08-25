using Flux: glorot_uniform, zeros, chunk
Nin = 15
Nh = 10 # size of hidden layer

# A recurrent model which takes a token and returns a context-dependent
# annotation.

Nh = 30 # size of hidden layer

# A recurrent model which takes a token and returns a context-dependent
# annotation.

forward  = LSTM(Nin, Nh÷2)
backward = LSTM(Nin, Nh÷2)
encode(tokens) = vcat.(forward.(tokens), flip(backward, tokens))

Ws = param(glorot_uniform(1, Nh))
Wt = param(glorot_uniform(1, Nh))
b = param(zeros(1))
align(s,t) = Ws*s .+ Wt*t .+ b

# A recurrent model which takes a sequence of annotations, attends, and returns a predicted output token.

recur   = LSTM(Nh+Nin, Nh)
toalpha = Dense(Nh, Nin)
function asoftmax(xs)
    xs = [exp.(x) for x in xs]
    s = sum(xs)
    return [x ./ s for x in xs]
end
function decode1(tokens, phone)
    weights = asoftmax([align(recur.state[2], t) for t in tokens])
    context = sum(map((a, b) -> a .* b, weights, tokens))
    y = recur(vcat(phone, context))
    return softmax(toalpha(y))
end

decode(tokens, phones) = [decode1(tokens, phone) for phone in phones]

# The full model

state = (forward, backward, recur, toalpha) # Dense doesn't hold a state... right?

function model(x, y)
    ŷ = decode(encode(x), y)
    reset!(state)
    return ŷ
end

loss(x, yo, y) = sum(crossentropy.(model(x, yo), y))
