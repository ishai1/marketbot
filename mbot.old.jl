using Logging
LogLevel(Logging.Info)
using DataFrames, Queryverse
df = load("postproc.feather") |> DataFrame

len_seq_in = 10
len_seq_out = 2
nbatch = 48
nchannels = size(df)[2]
end_index = size(df)[1] - (len_seq_in + len_seq_in)
start_indices = 1:end_index

using Flux
"""
    Returns a batch generator where each batch is a pair of arrays with shape T(_in-1)|(_out) x B x D.
    Applies first difference over time axis on original observations.
"""
function get_generator()
    ix_batches = Iterators.partition(shuffle(start_indices), nbatch)
    # single entry in an input / output batch with shape: width_in/out,batch size, channels
    range_in = 0:(len_seq_in-1)
    range_out = (len_seq_in):(len_seq_in+len_seq_out-1)
    function get_observations(ix)
      obs = convert(Array{Float32}, df[ix .+ ([range_in..., range_out...]),:])
      diff(obs, dims=1)
    end
    function get_batch(b)
      """
        Return time major batches
      """
      batch = Flux.stack(get_observations.(b), 2)
      batch[1 .+ range_in[1:end-1],:,:], batch[range_out,:,:]
    end
    Base.Generator(get_batch, ix_batches)
end

using TensorFlow
const tf = TensorFlow
sess = Session(Graph())
summary = tf.summary

D = size(df)[2] # number of features in input
xpl = tf.placeholder(Float32, shape=[len_seq_in-1, nbatch, D], name="xpl")
ypl = tf.placeholder(Float32, shape=[len_seq_out, nbatch, D], name="ypl")

learning_rate = .001
n_epochs= 10
Nh = 10 # size of hidden layer
Nc = 12 # size of context
Nfwd = Nh÷2
Nbwd = Nh÷2

forward  = nn.rnn_cell.LSTMCell(Nfwd)
backward = nn.rnn_cell.LSTMCell(Nbwd)

x = tf.unstack(xpl, axis=1)
fout, fstate = nn.rnn(forward, x, scope="fwRNN")
bout, bstate = nn.rnn(backward, reverse(x, 1), scope="bwRNN")
reverse!(bout)
enc = tf.stack([tf.concat([f,b], 2) for (f,b) in zip(fout, bout)])

using Distributions
variable_scope("align"; initializer=Normal(0, .001)) do
  global Wat = get_variable("Wat", [Nh*2, Nc], Float32)
  global Was = get_variable("Was", [1, 1, Nh, Nc], Float32)
  global B = get_variable("B", [Nc], Float32, initializer=tf.ones(Nc))
  global vt = get_variable("vt", [Nc], Float32)
end

score(e,t) = dropdims(tf.tanh(tf.concat([e,t], 2)*Wat + B)*expand_dims(vt, 2))
align(E,t) = tf.stack(map(e->score(e,t), tf.unstack(E, axis=1)), axis=1)

variable_scope("to_recur_state"; initializer=Normal(0, .001)) do
  global Wts = get_variable("W", [Nh, Nh*2], Float32)
  global Bts = get_variable("B", [Nh*2], Float32)
end

variable_scope("from_recur_state"; initializer=Normal(0, .001)) do
  global Wtso = get_variable("Wo", [Nh, D], Float32)
  global Btso = get_variable("Bo", [D], Float32)
end

state_init = tf.concat([fstate.c, fstate.h], 2)*Wts + Bts
# range corresponding to c and h
crange = 1:Nh
hrange = 1+Nh:2*Nh
cₜ = state_init[:, crange]
hₜ = state_init[:, hrange]
recur = variable_scope("recur") do
  nn.rnn_cell.LSTMCell(Nh)
end

preds = []
for i = 1:len_seq_out
  global hₜ
  global cₜ
  α = expand_dims(align(enc, hₜ), 3)
  yi = tf.reduce_sum(α .* enc; axis=1)
  si = tf.nn.rnn_cell.LSTMStateTuple(cₜ, hₜ)
  yo, so = variable_scope("recuriter"; reuse=i>1) do
    recur(yi, si)
  end
  cₜ, hₜ = so.c, so.h
  push!(preds, yo*Wtso + Btso)
end
ŷ = tf.stack(preds)
loss_MSE = reduce_mean((ypl.-ŷ).^2)
loss_summary = summary.scalar("MSE loss", loss_MSE)
optimizer=train.AdamOptimizer(learning_rate)
# Gradient decent with gradient clipping
# gvs = train.compute_gradients(optimizer, loss_MSE)
# capped_gvs = [(clip_by_norm(grad, 5.), var) for (grad, var) in gvs]
# opt_step = train.apply_gradients(optimizer,capped_gvs)
opt_step = train.minimize(optimizer, loss_MSE)

run(sess, global_variables_initializer())

using ProgressMeter
basic_train_loss = Float64[]

merged_summary_op = summary.merge_all()
# Create a summary writer
summary_dir = mktempdir()
@info "summary directory: $(summary_dir)"
summary_writer = summary.FileWriter(summary_dir)

step = 0
@showprogress for epoch in 1:n_epochs
  epoch_loss = Float64[]
  batchgen = get_generator()
  for i in batchgen
    @debug "$((size(i[1]),size(i[2]),size(xpl),size(ypl)))"
    if size(i[1])[2] == nbatch
      loss_o, summaries, _ = run(sess, (loss_MSE, merged_summary_op, opt_step), Dict(xpl=>i[1], ypl=>i[2]))
      step = step+1
      write(summary_writer, summaries, step)
    end
  end
  push!(basic_train_loss, mean(epoch_loss))
end

using Plots
backend(pyplot())
plot(basic_train_loss, label="training loss")
