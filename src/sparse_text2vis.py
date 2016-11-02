import sys, getopt, os
import numpy as np
import tensorflow as tf
from batcher_sparse import Batcher
from helpers import *
from paths import PATH
from evaluation_measure import evaluation
from flags import define_commom_flags
from mscoco_captions_reader import MSCocoCaptions


def main(argv=None):

    err_exit(argv[1:], "Error in parameters %s (--help for documentation)." % argv[1:])
    Path = PATH(FLAGS.fc_layer, debug_mode=FLAGS.debug, use_ngrams=FLAGS.ngrams)
    run_name = FLAGS.run+('-N' if FLAGS.ngrams else '-U')
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # set stdout to unbuffered

    # Default parameters
    l2factor = 0.00000001

    vocabulary_size = 25000 if FLAGS.ngrams else 10000
    # The training file (captions and visual embeddings) are used as training data, whereas the validation file (captions and visual embeddings) is split into validation and test
    batches = Batcher(captions_file = Path.tr_captions_file,
                    visual_file = Path.tr_visual_embeddings_file,
                    batch_size=FLAGS.batch_size,
                    max_vocabulary_size=vocabulary_size)

    input_size = batches.vocabulary_size()
    print("Input size: %d" % input_size )

    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # GRAPH
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

    graph = tf.Graph()
    with graph.as_default():
        # Input/Output data.
        # -------------------------------------------------------
        caption_input = tf.placeholder(tf.float32, shape=[None, input_size])
        caption_output = tf.placeholder(tf.float32, shape=[None, input_size])
        visual_embedding_output = tf.placeholder(tf.float32, shape=[None, FLAGS.visual_dim])

        global_step = tf.placeholder(tf.float32)  # training iteration

        # Model parameters
        # -------------------------------------------------------
        # caption-embedding
        cap2vec_weights, cap2vec_biases = projection_weights(input_size, FLAGS.hidden_size, 'cap2hid')

        # embedding-caption
        vec2cap_weights, vec2cap_biases = projection_weights(FLAGS.hidden_size, input_size, 'hid2cap')

        # embedding-visual
        vec2vis_weights, vec2vis_biases = projection_weights(FLAGS.hidden_size, FLAGS.visual_dim, 'hid2vis')

        # NNet
        # -------------------------------------------------------
        hidden_layer = tf.nn.relu(tf.matmul(caption_input, cap2vec_weights) + cap2vec_biases)
        caption_reconstruc = tf.nn.relu(tf.matmul(hidden_layer, vec2cap_weights) + vec2cap_biases)
        visual_prediction = tf.nn.relu(tf.matmul(hidden_layer, vec2vis_weights) + vec2vis_biases)

        # Losses functions
        # -------------------------------------------------------
        l2loss = l2factor * (tf.nn.l2_loss(vec2vis_weights) + tf.nn.l2_loss(vec2vis_biases))
        visual_loss = tf.reduce_mean(tf.square(visual_prediction - visual_embedding_output)) + l2loss
        caption_loss = tf.reduce_mean(tf.square(caption_output - caption_reconstruc))
        loss = visual_loss + caption_loss

        # Optimizers
        # -------------------------------------------------------
        visual_optimizer = tf.train.AdamOptimizer().minimize(visual_loss)
        caption_optimizer = tf.train.AdamOptimizer().minimize(caption_loss)
        full_optimizer = tf.train.AdamOptimizer().minimize(loss)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver(max_to_keep=1)  # defaults are: max_to_keep=5, keep_checkpoint_every_n_hours=10000.0

        # Tensorboard data
        auto_loss_summary = tf.scalar_summary('loss/auto_loss', caption_loss)
        visual_loss_summary = tf.scalar_summary('loss/visual_loss', visual_loss)
        loss_summary = tf.scalar_summary('loss/loss', loss)
        summaries = tf.merge_summary([loss_summary, auto_loss_summary, visual_loss_summary])

        print("Graph built!")

    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # RUN
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()

        do_training = FLAGS.train
        do_test = FLAGS.test

        # Train the net
        if do_training:
            # interactive mode: allows the user to save & quit or quit w/o saving
            interactive = InteractiveOptions()
            tensorboard = TensorboardData(FLAGS.boarddata)

            val_batches = Batcher(captions_file=Path.val_caption_file, visual_file=Path.val_visual_embeddings_file,
                                  batch_size=FLAGS.batch_size,
                                  word2id = batches.get_word2id(), id2word = batches.get_id2word())

            tensorboard.open(Path.summaries_dir, run_name, session.graph)

            best_val, val_lv = None, None
            tr_losses = model_losses('tr')
            val_losses = model_losses('val')
            last_improve = 0
            for step in range(1, FLAGS.num_steps):
                img_labels, caps_pos, wide_in, wide_out, visual_embeddings = batches.next()
                optimizer = get_optimizer(op_loss=full_optimizer, op_vis=visual_optimizer, op_tex=caption_optimizer,
                                          stochastic_loss=FLAGS.stochastic_loss,
                                          visual_prob=FLAGS.v_loss_prob)
                tr_feed_dict = {caption_input:wide_in, caption_output:wide_out, visual_embedding_output:visual_embeddings}
                _, tr_l, tr_ll, tr_lv = session.run([optimizer, loss, caption_loss, visual_loss], feed_dict=tr_feed_dict)
                tr_losses.accum(tr_l, tr_ll, tr_lv, len(img_labels))
                tensorboard.add_train_summary(summaries.eval(feed_dict=tr_feed_dict), step)

                if step % FLAGS.summary_frequency == 0:
                    print('[Step %d] %s' % (step, tr_losses))
                    tr_losses.clear()

                if step % FLAGS.validation_frequency == 0:
                    val_losses.clear()
                    samples_proc = 0
                    val_epoch = val_batches.epoch
                    while True:
                        img_labels, caps_pos, wide_in, wide_out, visual_embeddings = val_batches.next()
                        val_feed_dict = {caption_input:wide_in, caption_output:wide_out, visual_embedding_output:visual_embeddings}
                        val_l, val_ll, val_lv = session.run([loss, caption_loss, visual_loss], feed_dict=val_feed_dict)
                        val_losses.accum(val_l, val_ll, val_lv, len(img_labels))
                        samples_proc += len(img_labels)
                        if val_batches.epoch != val_epoch: break
                    print('\t[Step %d] %s' % (step, val_losses))
                    _, _, valid_loss = val_losses.mean_flush()
                    tensorboard.add_valid_summary(summaries.eval(feed_dict=val_feed_dict), step)

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
                if step - last_improve >= 20000:
                    print ("Early stop at step %d" % step)
                    break

            tensorboard.close()

            # Test the net
        if do_test:
            restore_checkpoint(saver, session, Path.checkpoint_dir, checkpoint_path=FLAGS.checkpoint)

            print('Starts evaluation...')
            test_batches = Batcher(captions_file=Path.test_caption_file, visual_file=Path.test_visual_embeddings_file,
                                  batch_size=FLAGS.batch_size,
                                  word2id = batches.get_word2id(), id2word = batches.get_id2word())

            print('Getting predictions...')
            test_img_ids, test_cap_id = [], []
            samples_processed = 0
            batch_processed = 0
            test_losses = model_losses('test')
            predictions = []
            while test_batches.epoch == 0:
                img_ids, cap_ids, wide_in, wide_out, visual_embeddings = test_batches.next()
                test_feed_dict = {caption_input: wide_in, caption_output: wide_out, visual_embedding_output: visual_embeddings}
                batch_predictions, te_l, te_ll, te_lv = session.run([visual_prediction, loss, caption_loss, visual_loss], feed_dict=test_feed_dict)
                test_losses.accum(te_l, te_ll, te_lv, len(img_ids))
                predictions.append(batch_predictions)
                test_img_ids += img_ids
                test_cap_id += cap_ids
                batch_processed += 1
                samples_processed += len(img_ids)
                if batch_processed % 100 == 0:
                    print('Processed %d examples' % (samples_processed))
            predictions = np.concatenate((predictions), axis=0)

            predictions_file = Path.predictions_dir + '/' + run_name + '.txt'
            visual_ids, visual_vectors = test_batches._visual.get_all_vectors()

            ref_test_captions = test_batches._captions
            if FLAGS.ngrams:
                #the reference captions should be taken without ngrams, otherwise the ROUGE has a different bias
                uni_captions_file = Path.test_caption_file.replace('.ngrams','')
                ref_test_captions = MSCocoCaptions(uni_captions_file)

            evaluation(ref_test_captions, visual_ids, visual_vectors, predictions, test_img_ids, test_cap_id, predictions_file,
                   method=FLAGS.retrieval,
                   mean_file=Path.mean_file, eigen_file=Path.eigen_file,
                   test_loss=str(test_losses), save_predictions=FLAGS.save_predictions)

#-------------------------------------
if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS

    flags = define_commom_flags(flags, num_steps=100001, summary_frequency=100)

    flags.DEFINE_boolean('ngrams', True, 'If True uses the (previously extracted) n-grams file (default True).')
    flags.DEFINE_boolean('stochastic_loss', True, 'Activates the stochastic loss heuristic (default True).')
    flags.DEFINE_float('v_loss_prob', 0.5, 'Visual loss probability (default=0.5).')
    flags.DEFINE_integer('hidden_size', 1024, 'Hidden size (default 1024).')

    tf.app.run()



