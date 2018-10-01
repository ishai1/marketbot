function get_generator()
  ix_batches = Iterators.partition(shuffle(start_indices), nbatch)
  # single entry in an input / output batch with shape: width_in/out, channels, batch size
  get_input(ix) = reshape(convert(Array{Float32}, df[ix:ix+(len_seq_in-1),:]),
                          (len_seq_in, 1, size(df)[2]))
  get_output(ix) = reshape(convert(Array{Float32}, df[ix+(len_seq_in):ix+(len_seq_in+len_seq_out-1),:]),
                           (len_seq_out, 1, size(df)[2]))
  get_batch(b) = [cat(get_input.(b)..., dims=2), cat(get_output.(b)..., dims=2)]
  Base.Generator(get_batch, ix_batches)
end


step = 0
@showprogress for epoch in 1:n_epochs
  epoch_loss = Float64[]
  batchgen = get_generator()
  for i in batchgen
    # @debug "$((size(i[1]),size(i[2]),size(xpl),size(ypl)))"
    loss_o, summaries, _ = run(sess, (loss_MSE, merged_summary_op, opt_step), Dict(xpl=>i[1], ypl=>i[2]))
    step = step+1
    write(summary_writer, summaries, step)
  end
  push!(basic_train_loss, mean(epoch_loss))
end

using Plots
backend(pyplot())
plot(basic_train_loss, label="training loss")
