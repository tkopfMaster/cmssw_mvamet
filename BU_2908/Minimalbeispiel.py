
    queue_train = tf.RandomShuffleQueue(capacity=data_train.shape[0], min_after_dequeue=0,
                                        dtypes=[tf.float32, tf.float32, tf.float32], shapes=[tf.TensorShape(data_train.shape[1]),tf.TensorShape(labels_train.shape[1]),tf.TensorShape(weights_train.shape[1])])

    queue_val = tf.RandomShuffleQueue(capacity=data_val.shape[0], min_after_dequeue=0,
                                      dtypes=[tf.float32, tf.float32, tf.float32], shapes=[tf.TensorShape(data_val.shape[1]),tf.TensorShape(labels_val.shape[1]),tf.TensorShape(weights_val.shape[1])])


    batch_size = 40000
    qtrain = tf.FIFOQueue(capacity=100000, dtypes=[tf.float32, tf.float32, tf.float32], shapes=[tf.TensorShape(data_val.shape[1]),tf.TensorShape(labels_val.shape[1]),tf.TensorShape(weights_val.shape[1])])
    enqueue_op_train = qtrain.enqueue_many([data_train, labels_train, weights_train])
    qval = tf.FIFOQueue(capacity=100000,dtypes=[tf.float32, tf.float32, tf.float32], shapes=[tf.TensorShape(data_val.shape[1]),tf.TensorShape(labels_val.shape[1]),tf.TensorShape(weights_val.shape[1])])
    enqueue_op_val = qval.enqueue_many([data_val, labels_val, weights_val])


    sess = tf.Session()
    queue_runner_train = tf.train.QueueRunner(qtrain, [enqueue_op_train])
    tf.train.add_queue_runner(queue_runner_train)
    queue_runner_val = tf.train.QueueRunner(qval, [enqueue_op_val])
    tf.train.add_queue_runner(queue_runner_val)
    coordinator = tf.train.Coordinator()
    threads_train = queue_runner_train.create_threads(sess, coord=coordinator, start=True)
    threads_val = queue_runner_val.create_threads(sess, coord=coordinator, start=True)


    # ## Define the neural network architecture (how many )
    batch_train = qtrain.dequeue_many(batch_size)
    batch_val = qval.dequeue_many(batch_size)
    print('wichtig',data_train.shape[0], data_train.shape[1])


    logits_train, f_train= NNmodel(batch_train[0], reuse=False)
    logits_val, f_val= NNmodel(batch_val[0], reuse=True)

    # ## Add training operations to the graph
    print('len len(batch_train[1])', batch_train[1])
    print('len logits_train', logits_train.shape)

    print("loss fct", loss_fct)
    if (loss_fct=="mean_squared_error"):
        loss_train = tf.reduce_mean(
            tf.losses.mean_squared_error(labels=batch_train[1], predictions=logits_train))
        loss_val = tf.reduce_mean(
            tf.losses.mean_squared_error(labels=batch_val[1], predictions=logits_val))
        minimize_loss = tf.train.AdamOptimizer().minimize(loss_train)
    elif (loss_fct=="Response"):
        print("Loss Function Response: ", loss_fct)
        loss_train = costResponse(batch_train[1], logits_train, batch_train[2])
        loss_val = costResponse(batch_val[1], logits_val, batch_val[2])
        minimize_loss = tf.train.AdamOptimizer().minimize(loss_train)
    elif (loss_fct=="Resolution_para"):
        print("Loss Function Resolution_para: ", loss_fct)
        loss_train = costResolution_para(batch_train[1], logits_train, batch_train[2])
        loss_val = costResolution_para(batch_val[1], logits_val, batch_val[2])
        minimize_loss = tf.train.AdamOptimizer().minimize(loss_train)
    elif (loss_fct=="Resolution_perp"):
        print("Loss Function Resolution_perp: ", loss_fct)
        loss_train = costResolution_perp(batch_train[1], logits_train, batch_train[2])
        loss_val = costResolution_perp(batch_val[1], logits_val, batch_val[2])
        minimize_loss = tf.train.AdamOptimizer().minimize(loss_train)
    elif (loss_fct=="Angle_Response"):
        print("Loss Function Angle_Response: ", loss_fct)
        loss_train = costExpected(batch_train[1], logits_train, batch_train[2])
        loss_val = costExpected(batch_val[1], logits_val, batch_val[2])
        minimize_loss = tf.train.AdamOptimizer().minimize(loss_train)
    elif (loss_fct=="Angle_relResponse"):
        print("Loss Function Angle_Response rel: ", loss_fct)
        loss_train = costExpectedRel(batch_train[1], logits_train, batch_train[2])
        loss_val = costExpectedRel(batch_val[1], logits_val, batch_val[2])
        minimize_loss = tf.train.AdamOptimizer().minimize(loss_train)
    elif (loss_fct=="Resolutions"):
        print("Loss Function Angle_Response: ", loss_fct)
        loss_train = costResolutions(batch_train[1], logits_train, batch_train[2])
        loss_val = costResolutions(batch_val[1], logits_val, batch_val[2])
        minimize_loss = tf.train.AdamOptimizer().minimize(loss_train)
    else:
        factor_response, factor_res_para, factor_res_perp, factor_mse = 1,1,1,1
        loss_response = costResponse(batch_train[1], logits_train, batch_train[2])
        loss_res_para = costResolution_para(batch_train[1], logits_train, batch_train[2])
        loss_res_perp = costResolution_perp(batch_train[1], logits_train, batch_train[2])
        loss_MSE = costMSE(batch_train[1], logits_train, batch_train[2])
        loss_final = factor_response * loss_response + factor_res_para * loss_res_para + factor_res_perp * loss_res_perp + factor_mse * loss_MSE
        train_op = tf.optimizer.Adam().minimize(loss_final)





    # ## Run the training
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    losses_train = []
    losses_val = []
    loss_response, loss_resolution_para, loss_resolution_perp, loss_mse = [], [], [], []

    summary_train = tf.summary.scalar("loss_train", loss_train)
    summary_val = tf.summary.scalar("loss_val", loss_val)
    writer = tf.summary.FileWriter("./logs/{}".format(
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")), sess.graph)
    saver = tf.train.Saver()
    saveStep = 1000
    print("training started")
    for i_step in range(12000):
        start_loop = time.time()

        summary_, loss_, _ = sess.run([summary_train, loss_train, minimize_loss])
        #summary_, loss_, _ = sess.run([summary_train, loss_train, minimize_loss], feed_dict={x: data_train, y: labels_train, w: weights_train})
        #summary_, loss_, loss_response_, loss_resolution_para_, loss_resolution_perp_, loss_mse_, _ = sess.run([summary_train, loss_train, minimize_loss])
        losses_train.append(loss_)
        #loss_response.append(loss_response_)
        #loss_resolution_para.append(loss_resolution_para_)
        #loss_resolution_perp.append(loss_resolution_perp_)
        #loss_mse.append(loss_mse_)
        writer.add_summary(summary_, i_step)

        summary_, loss_ = sess.run([summary_val, loss_val], feed_dict={x: data_val, y: labels_val, w: weights_val})
        losses_val.append(loss_)
        writer.add_summary(summary_, i_step)
        end_loop = time.time()
        if i_step % saveStep == 0:
            saver.save(sess, "%sNNmodel"%outputDir, global_step=i_step)
            print('gradient step No ', i_step)
            print("gradient step time {0} seconds".format(end_loop-start_loop))

    coordinator.request_stop()
    coordinator.join(threads_train)
    coordinator.join(threads_val)
