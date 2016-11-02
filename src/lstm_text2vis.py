import sys, os, getopt
import random
import time
import numpy as np
import tensorflow as tf
from batcher_lstmbased import Batcher
from evaluation_measure import evaluation
from helpers import *
from paths import PATH
from flags import define_commom_flags


def main(argv=None):

    err_exit(argv[1:], "Error in parameters %s (--help for documentation)." % argv[1:])

    Path = PATH(FLAGS.fc_layer, FLAGS.debug)

    do_training = FLAGS.train
    do_test = FLAGS.test
    run_name = FLAGS.run
    if not run_name:
        run_name=time.strftime("%d_%b_%Y")+'::'+time.strftime("%H:%M:%Sh")
    if FLAGS.debug:
        run_name+='-debug'

    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # set stdout to unbuffered

    #---------------------------------------------------------------------
    # Inits
    #---------------------------------------------------------------------

    buckets_def = [15, 20, 40] if not FLAGS.debug else [15]
    batches = Batcher(Path.tr_captions_file, Path.tr_visual_embeddings_file, buckets_def=buckets_def, batch_size=FLAGS.batch_size, lemma=FLAGS.lemma, dowide=FLAGS.dowide)

    vocabulary_size=batches.vocabulary_size()
    print('Vocabulary size %d' % vocabulary_size)

    graph = tf.Graph()
    with graph.as_default():
      # Placeholders
      train_wide1hot = tf.placeholder(tf.float32, shape=[None, vocabulary_size], name='wide_1hot') if FLAGS.dowide else None

      # Model parameters
      embeddings = tf.Variable(tf.random_uniform([vocabulary_size, FLAGS.embedding_size], -1.0, 1.0), name='word_embeddings')
      state2text_weight, state2text_bias = projection_weights(FLAGS.num_nodes, vocabulary_size, 'output-text')
      state2vis_weight, state2vis_bias = projection_weights(FLAGS.num_nodes*FLAGS.lstm_stacked_layers, FLAGS.visual_dim, 'hidden-visual')
      if FLAGS.dowide:
        wide_weight, wide_bias = projection_weights(vocabulary_size, FLAGS.visual_dim, 'wide-projection')
      else:
        wide_weight, wide_bias = None,None

      # Bucket-independent computations
      wide_prediction = (tf.matmul(train_wide1hot, wide_weight) + wide_bias) if FLAGS.dowide else None

      # ----------------------------------------------------------------------------------------------
      # Generate a bucket-specific unrolled net
      def bucket_net(bucket_size, stacked_lstm):
          # ----------------------------------------------------------------------------------------------
          # Placeholders
          train_tokens = list()
          train_labels = list()
          for i in range(bucket_size-1):
            train_tokens.append(tf.placeholder(tf.int64, shape=[None], name='x_'+str(i)))
            train_labels.append(tf.placeholder(tf.float32, shape=[None, vocabulary_size], name='x_'+str(i+1)))

          embedding_inputs = list()
          for i in range(len(train_tokens)):
              embedding_inputs.append(tf.nn.embedding_lookup(embeddings, train_tokens[i]))

          visual_outputs = tf.placeholder(tf.float32, shape=[None, FLAGS.visual_dim], name='visual-embedding')

          # ----------------------------------------------------------------------------------------------
          # Unrolled LSTM loop.
          outputs, final_state = tf.nn.rnn(stacked_lstm, embedding_inputs, dtype=tf.float32)
          final_state = tf.concat(1,[_state.c for _state in final_state]) if FLAGS.lstm_stacked_layers > 1 else final_state.c
          tf.get_variable_scope().reuse_variables()
          logits = tf.matmul(tf.concat(0, outputs), state2text_weight) + state2text_bias
          deep_prediction = tf.matmul(final_state, state2vis_weight) + state2vis_bias
          visual_prediction = tf.nn.relu(wide_prediction+deep_prediction) if FLAGS.dowide else tf.nn.relu(deep_prediction)

          # ----------------------------------------------------------------------------------------------
          # Losses
          lstm_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf.concat(0, train_labels)))
          visual_loss = tf.reduce_mean(tf.square(visual_prediction - visual_outputs))
          loss = lstm_loss +  visual_loss

          # ----------------------------------------------------------------------------------------------
          # Tensorboard data: loss summaries
          if FLAGS.boarddata:
              lstm_loss_summary = tf.scalar_summary('loss/lstm_loss', lstm_loss)
              visual_loss_summary = tf.scalar_summary('loss/visual_loss', visual_loss)
              loss_summary = tf.scalar_summary('loss/loss', loss)
              summaries = tf.merge_summary([loss_summary, lstm_loss_summary, visual_loss_summary])
          else: summaries = None

          #----------------------------------------------------------------------------------------------
          # Optimizer.
          def optimizer(someloss):
              global_step = tf.Variable(0)
              optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
              gradients, v = zip(*optimizer.compute_gradients(someloss))
              gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
              optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)
              return optimizer

          return Model(bucket_size, train_tokens, train_labels, visual_outputs, visual_prediction, loss, lstm_loss, visual_loss,
                   optimizer(loss), optimizer(lstm_loss), optimizer(visual_loss), summaries)

      # ----------------------------------------------------------------------------------------------
      # Defines the computation graph
      print('Creating bucket-specific unrolled nets:')
      models = dict()
      lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=FLAGS.num_nodes, use_peepholes=True, state_is_tuple=True)
      if FLAGS.lstm_stacked_layers > 1:
          lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * FLAGS.lstm_stacked_layers)
      for bucket_i in buckets_def:
          with tf.name_scope('bucket-net-'+str(bucket_i)):
            models[bucket_i] = bucket_net(bucket_i, lstm_cell)
            print('\tcreated model for bucket %d' % bucket_i)

      # ----------------------------------------------------------------------------------------------
      # Add ops to save and restore all the variables.
      saver = tf.train.Saver(max_to_keep=1) #defaults are: max_to_keep=5, keep_checkpoint_every_n_hours=10000.0

    #---------------------------------------------------------------------
    # Model Params
    #---------------------------------------------------------------------

    def get_model_params(batches_):
        img_labels, caps_pos, caps, wide, viss, bucket_size = batches_.next()
        model = models[bucket_size]
        params = dict()
        if FLAGS.dowide: params[train_wide1hot] = wide
        for i in range(bucket_size - 1):
            params[model.train_tokens[i]] = caps[:, i]
            params[model.train_labels[i]] = batches_.from_batchlabel2batch_onehot(caps[:, i + 1])
            params[model.visual_outputs] = viss
        return model, params, img_labels, caps_pos

    # ---------------------------------------------------------------------
    # Graph run
    # ---------------------------------------------------------------------

    with tf.Session(graph=graph) as session:
      tf.initialize_all_variables().run()

      # Train the net
      if do_training:
          #interactive mode: allows the user to save & quit or quit w/o saving
          interactive = InteractiveOptions()
          tensorboard = TensorboardData(FLAGS.boarddata)

          val_batches = Batcher(Path.val_caption_file, Path.val_visual_embeddings_file,
                                buckets_def=buckets_def, batch_size=FLAGS.validation_batch,
                                lemma = FLAGS.lemma,
                                dowide = FLAGS.dowide,
                                word2id=batches.get_word2id(), id2word=batches.get_id2word())

          tensorboard.open(Path.summaries_dir, run_name, session.graph)

          best_val, val_lv = None, None
          tr_losses = model_losses('tr')
          val_losses = model_losses('val')
          last_improve = 0
          for step in range(1, FLAGS.num_steps):
            tr_model,tr_feed_dict,img_labels,caps_pos = get_model_params(batches)
            optimizer = get_optimizer(op_loss=tr_model.optimizer_loss, op_vis=tr_model.optimizer_visual_loss, op_tex=tr_model.optimizer_lstm_loss,
                                      stochastic_loss=FLAGS.stochastic_loss, visual_prob=FLAGS.stochastic_visual_prob)

            _,tr_l,tr_ll,tr_lv = session.run([optimizer, tr_model.loss, tr_model.lstm_loss, tr_model.visual_loss], feed_dict=tr_feed_dict)
            tr_losses.accum(tr_l,tr_ll,tr_lv, len(img_labels))
            tensorboard.add_train_summary(tr_model.summaries.eval(feed_dict=tr_feed_dict), step)

            if step % FLAGS.summary_frequency == 0:
              print('[Step %d][Bucket=%d] %s' % (step, tr_model.bucket_size, tr_losses))
              tr_losses.clear()

            if step % FLAGS.validation_frequency == 0:
                val_losses.clear()
                samples_proc = 0
                val_epoch = val_batches.epoch
                while True:
                    val_model, val_feed_dict,img_ids,cap_ids = get_model_params(val_batches)
                    val_l, val_ll, val_lv = session.run([val_model.loss, val_model.lstm_loss, val_model.visual_loss], feed_dict=val_feed_dict)
                    val_losses.accum(val_l,val_ll,val_lv,len(img_ids))
                    samples_proc+=len(img_ids)
                    if val_batches.epoch != val_epoch: break
                print('\t[Step %d] %s' % (step, val_losses))
                _,_,valid_loss = val_losses.mean_flush()
                tensorboard.add_valid_summary(val_model.summaries.eval(feed_dict=val_feed_dict), step)

                if not best_val or valid_loss < best_val:
                    best_val = valid_loss
                    last_improve = step
                    if step > 2500:
                        savemodel(session, step, saver, Path.checkpoint_dir, run_name)

            if interactive.status != interactive.Continue:
                if interactive.status in [interactive.SaveQuit, interactive.Save, interactive.SaveGoTest]:
                    savemodel(session, step, saver, Path.checkpoint_dir, run_name, posfix='_usersaved')
                if interactive.status in [interactive.SaveQuit, interactive.Quit]:
                    do_test = False
                    break
                if interactive.status in [interactive.GoTest, interactive.SaveGoTest]:
                    do_test = True
                    break
                interactive.status = interactive.Continue

            # early-stop condition
            if step - last_improve >= 10000:
                print ("Early stop at step %d" % step)
                break

          tensorboard.close()

      # Test the net
      if do_test:
          restore_checkpoint(saver, session, Path.checkpoint_dir, checkpoint_path=FLAGS.checkpoint)

          print('Starts evaluation...')
          test_batches = Batcher(Path.test_caption_file, Path.test_visual_embeddings_file,
                                 buckets_def=buckets_def, batch_size=FLAGS.batch_size,
                                 lemma=FLAGS.lemma,
                                 dowide=FLAGS.dowide,
                                 word2id=batches.get_word2id(), id2word=batches.get_id2word())

          print('Getting predictions...')
          test_img_ids, test_cap_id = [], []
          buckets_processed, samples_processed  = 0, 0
          test_losses = model_losses('test')
          predictions = []
          while test_batches.epoch == 0:
              test_model, test_feed_dict, img_ids, cap_ids = get_model_params(test_batches)
              batch_predictions, te_l, te_ll, te_lv = session.run([test_model.visual_prediction,
                                                                   test_model.loss, test_model.lstm_loss, test_model.visual_loss],
                                                                  feed_dict=test_feed_dict)
              test_losses.accum(te_l, te_ll, te_lv, len(img_ids))
              predictions.append(batch_predictions)
              test_img_ids += img_ids
              test_cap_id += cap_ids
              buckets_processed += 1
              samples_processed += len(img_ids)
              if buckets_processed % 100 == 0:
                  print('Processed %d examples' % (samples_processed))
          predictions = np.concatenate((predictions), axis=0)

          predictions_file = Path.predictions_dir + '/' + run_name + '.txt'
          visual_ids, visual_vectors = test_batches._visual.get_all_vectors()

          evaluation(test_batches, visual_ids, visual_vectors, predictions, test_img_ids, test_cap_id, predictions_file,
                     method=FLAGS.retrieval,
                     mean_file=Path.mean_file, eigen_file=Path.eigen_file,
                     test_loss=str(test_losses), save_predictions=FLAGS.save_predictions)

#-------------------------------------
if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS

    define_commom_flags(flags, num_steps=50001, summary_frequency=10)

    # net settings
    flags.DEFINE_integer('num_nodes', 512, 'Number of nodes of the LSTM internal representation (default 512).')
    flags.DEFINE_integer('lstm_stacked_layers', 2, 'Number of stacked layers for the LSTM cell (default 2).')
    flags.DEFINE_boolean('stochastic_loss', False, 'Determines if the Stochastic-loss heuristic is activated (default False).')
    flags.DEFINE_float('stochastic_visual_prob', 0.5, 'If stochastic_loss is active, determines the probability of optimizing the visual loss (default 0.5).')
    flags.DEFINE_boolean('lemma', True, 'Lemmatizes the captions before processing them.')
    flags.DEFINE_integer('embedding_size', 100, 'Dimensionality of the word embeddings space (default 100).')

    tf.app.run()

