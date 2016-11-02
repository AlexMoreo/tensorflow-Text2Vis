import numpy as np
import tensorflow as tf
from batcher_word2visualvec import Batcher
from evaluation_measure import evaluation
from helpers import *
import gc
from paths import PATH
from flags import define_commom_flags
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def load_wordembeddings(sentences_file, we_dim):
    saved_model = sentences_file + ".model"
    if not os.path.exists(saved_model):
        print('Pre-trained model for %s not found. Generating word embeddings...' % sentences_file)
        sentences = gensim.models.word2vec.LineSentence(sentences_file)
        model = gensim.models.Word2Vec(sentences, size=we_dim, sg=1, workers=8)  # default: window=5, min_count=5, iter=5
        model.save(saved_model)
        print('Done! file saved in %s' % saved_model)
    return gensim.models.Word2Vec.load(saved_model)

class PATH_WE(PATH):
    def __init__(self, fclayer, debug_mode=False, lemma=False):
        super(PATH_WE, self).__init__(fclayer, debug_mode)
        self.wordembeddings_file = '../wordembeddings/YFCC100Muser_tags'+('_lemma'if lemma else '')+'.bz2'
        notexist_exit(self.wordembeddings_file)

def main(argv=None):

    err_exit(argv[1:], "Error in parameters %s (--help for documentation)." % argv[1:])
    Path = PATH_WE(FLAGS.fc_layer, debug_mode=FLAGS.debug, lemma=FLAGS.lemma)
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # set stdout to unbuffered

    #---------------------------------------------------------------------
    # Inits
    #---------------------------------------------------------------------
    do_training = FLAGS.train
    do_test = FLAGS.test
    run_name = 'word2visualvec'+FLAGS.run

    we_dim = FLAGS.embedding_size
    wordembeddings = load_wordembeddings(Path.wordembeddings_file, we_dim)

    batches = Batcher(captions_file=Path.tr_captions_file,
                     visual_file=Path.tr_visual_embeddings_file,
                     we_dim=we_dim,
                     batch_size = FLAGS.batch_size,
                     lemmatize=FLAGS.lemma,
                     model=wordembeddings)

    hidden_sizes = [1000, 2000, 3000] if FLAGS.large else [1000, 2000]

    graph = tf.Graph()
    with graph.as_default():
      # Placeholders
      output = tf.placeholder(tf.float32, shape=[None, FLAGS.visual_dim], name='visual-embedding')
      input = tf.placeholder(tf.float32, shape=[None, we_dim], name='pooled_word_embeddings')
      keep_p = tf.placeholder(tf.float32)

      def add_layer(layer, hidden_size, drop, name):
          weight, bias = projection_weights(layer.get_shape().as_list()[1], hidden_size, name=name)
          activation = tf.nn.relu(tf.matmul(layer,weight)+bias)
          if drop:
              return tf.nn.dropout(activation, keep_prob=keep_p)
          else:
              return activation

      current_layer = input
      for i,hidden_dim in enumerate(hidden_sizes):
          current_layer = add_layer(current_layer, hidden_dim, drop=True, name='layer'+str(i))
      out_layer = add_layer(current_layer, FLAGS.visual_dim, drop=False, name='output_layer')


      # losses
      loss = tf.reduce_mean(tf.square(out_layer - output))

      # Optimizer.
      optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001, decay=0.9, epsilon=1e-6).minimize(loss)

      # Add ops to save and restore all the variables.
      saver = tf.train.Saver(max_to_keep=1) #defaults are: max_to_keep=5, keep_checkpoint_every_n_hours=10000.0

      # Tensorboard data
      summaries = tf.merge_summary([tf.scalar_summary('loss/loss', loss)]) if FLAGS.boarddata else None


    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()

        # Train the net
        if do_training:
            # interactive mode: allows the user to save & quit or quit w/o saving
            interactive = InteractiveOptions()
            tensorboard = TensorboardData(FLAGS.boarddata)

            val_batches = Batcher(captions_file=Path.val_caption_file, visual_file=Path.val_visual_embeddings_file,
                                  we_dim=we_dim,
                                  batch_size=FLAGS.batch_size,
                                  lemmatize=FLAGS.lemma,
                                  model=wordembeddings)

            tensorboard.open(Path.summaries_dir, run_name, session.graph)

            train_loss = 0.0
            valid_loss = 0.0
            best_val = None
            last_improve = 0
            for step in range(1, FLAGS.num_steps):
                _, _, embedding_pool, vis_embeddings = batches.next()
                feed_dict = {input: embedding_pool, output: vis_embeddings, keep_p: FLAGS.drop_keep_p}
                _,tr_l = session.run([optimizer, loss], feed_dict=feed_dict)
                train_loss += tr_l
                tensorboard.add_train_summary(summaries.eval(feed_dict=feed_dict), step)

                if step % FLAGS.summary_frequency == 0:
                    print('[epoch=%d][Step %d] %0.5f' % (batches.epoch, step, (train_loss / FLAGS.summary_frequency)))
                    train_loss = 0.0

                if step % FLAGS.validation_frequency == 0:
                    valid_loss = 0.0
                    samples_proc = 0
                    val_epoch = val_batches.epoch
                    while True:
                        _, _, pooled_embeddings, visual_embeddings = val_batches.next()
                        samples = len(pooled_embeddings)
                        feed_dict = {input: pooled_embeddings, output: visual_embeddings, keep_p: 1.0}
                        val_l = loss.eval(feed_dict=feed_dict)
                        valid_loss += val_l*samples
                        samples_proc += samples
                        if val_batches.epoch != val_epoch: break
                    valid_loss /= samples_proc
                    print('\t[epoch=%d][Step %d] %0.5f' % (val_batches.epoch, step, valid_loss))
                    tensorboard.add_valid_summary(summaries.eval(feed_dict=feed_dict), step)

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
                if step - last_improve >= 30000:
                    print ("Early stop at step %d" % step)
                    break

            tensorboard.close()

        # Test the net
        if do_test:
          restore_checkpoint(saver, session, Path.checkpoint_dir, checkpoint_path=FLAGS.checkpoint)

          print('Starts evaluation...')
          test_batches = Batcher(captions_file=Path.test_caption_file, visual_file=Path.test_visual_embeddings_file,
                                 we_dim=we_dim,
                                 batch_size=FLAGS.batch_size,
                                 lemmatize=FLAGS.lemma,
                                 model=wordembeddings)

          print('Getting predictions...')
          test_img_ids,test_cap_id,predictions = [],[],[]
          test_loss = 0.0
          tests_processed = 0
          batch_processed = 0
          while True:
              img_labels, caps_pos, pooled_embeddings, visual_embeddings = test_batches.next()
              samples = len(img_labels)
              feed_dict = {input: pooled_embeddings, output: visual_embeddings, keep_p: 1.0}
              batch_predictions, test_l = session.run([out_layer, loss], feed_dict=feed_dict)
              test_loss += (test_l*samples)
              predictions.append(batch_predictions)
              test_img_ids += img_labels
              test_cap_id += caps_pos
              tests_processed += samples
              batch_processed += 1
              if batch_processed % 100 == 0:
                  print('Processed %d test examples' % (tests_processed))
              if test_batches.epoch != 0: break
          predictions = np.concatenate((predictions), axis=0)
          test_loss /= tests_processed
          wordembeddings = None
          gc.collect()

          predictions_file = Path.predictions_dir + '/' + run_name + '.txt'
          visual_ids, visual_vectors = test_batches._visual.get_all_vectors()

          evaluation(test_batches, visual_ids, visual_vectors, predictions, test_img_ids, test_cap_id, predictions_file,
                   method=FLAGS.retrieval,
                   mean_file=Path.mean_file, eigen_file=Path.eigen_file,
                   test_loss=str(test_loss), save_predictions=FLAGS.save_predictions)



#-------------------------------------
if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS

    flags = define_commom_flags(flags, num_steps=500001, summary_frequency=100)

    # net settings
    flags.DEFINE_boolean('lemma', True, 'Determines whether to use the word embeddings trained after lemmatizing.')
    flags.DEFINE_boolean('large', True, 'If true adds an additional layer to the net (default True).')
    flags.DEFINE_integer('embedding_size', 500, 'Dimensionality of the word embeddings space (default 500).')
    flags.DEFINE_float('drop_keep_p', 0.85, 'Keep probability for dropout (default 0.85).')



    tf.app.run()



