import os, sys
import tensorflow as tf
import signal
import math, random
import shutil

#--------------------------------------------------------------
# Model helpers
#--------------------------------------------------------------
"""
This class abstracts the model losses into a single instance, to ease its cummulation and average.
"""
class model_losses:
    def __init__(self, name=''):
        self.clear()
        self.__name__ = name

    def accum(self, l, llstm, lvisual, items=1):
        self.l+=l*items
        self.llstm+=llstm*items
        self.lvisual+=lvisual*items
        self.items+=items

    def mean(self):
        l = self.l / self.items
        llstm = self.llstm / self.items
        lvisual = self.lvisual / self.items
        return l, llstm, lvisual

    def mean_flush(self):
        means = self.mean()
        self.clear()
        return means

    def clear(self):
        self.l = self.llstm = self.lvisual = self.items = 0

    def __str__(self):
        return self.__name__+ ('[L=%.04f Ll=%.04f Lv%.04f]' % self.mean())

class Model:
    def __init__(self, bucket_size,
                 train_tokens, train_labels, visual_outputs,
                 visual_prediction,
                 loss, lstm_loss, visual_loss,
                 optimizer_loss, optimizer_lstm_loss, optimizer_visual_loss,
                 summaries):
        self.bucket_size = bucket_size
        self.train_tokens = train_tokens
        self.train_labels = train_labels
        self.visual_outputs = visual_outputs
        self.visual_prediction = visual_prediction
        self.loss = loss
        self.lstm_loss = lstm_loss
        self.visual_loss = visual_loss
        self.optimizer_loss = optimizer_loss
        self.optimizer_lstm_loss = optimizer_lstm_loss
        self.optimizer_visual_loss = optimizer_visual_loss
        self.summaries = summaries

def variable_summaries(var, scope_name, name):
    """Attach summaries to a Tensor."""
    embeddings_summaries = []
    with tf.name_scope(scope_name):
        mean = tf.reduce_mean(var)
        embeddings_summaries.append(tf.scalar_summary(scope_name + '/mean', mean))
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        embeddings_summaries.append(tf.scalar_summary(scope_name + '/sttdev', stddev))
        embeddings_summaries.append(tf.scalar_summary(scope_name + '/max', tf.reduce_max(var)))
        embeddings_summaries.append(tf.scalar_summary(scope_name + '/min', tf.reduce_min(var)))
        embeddings_summaries.append(tf.histogram_summary(scope_name + '/' + name, var))
    return embeddings_summaries

def savemodel(session, step, saver, checkpoint_dir, run_name, posfix=""):
    sys.stdout.write('Saving model...')
    sys.stdout.flush()
    save_path = saver.save(session, checkpoint_dir + '/' + run_name + posfix, global_step=step + 1)
    print('[Done]')
    return save_path

def projection_weights(orig_size, target_size, name):
    weight = tf.Variable(tf.truncated_normal([orig_size, target_size], stddev=1.0 / math.sqrt(target_size)), name=name + '/weight')
    bias = tf.Variable(tf.zeros([target_size]), name=name + '/bias')
    return weight, bias


#--------------------------------------------------------------
# Run helpers
#--------------------------------------------------------------
def err_exit(exit_condition=True, err_msg=""):
    if exit_condition:
        if not err_msg:
            err_msg = (sys.argv[0]+ " Error")
        print(err_msg)
        sys.exit()

def notexist_exit(path):
    if isinstance(path, list):
        [notexist_exit(p) for p in path]
    elif not os.path.exists(path):
        print("Error. Path <%s> does not exist or is not accessible." %path)
        sys.exit()

def create_if_not_exists(dir):
    if not os.path.exists(dir): os.makedirs(dir)
    return dir

def restore_checkpoint(saver, session, checkpoint_dir, checkpoint_path=None):
    if not checkpoint_path:
        print('Restoring last checkpoint in %s' % checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        err_exit(not (ckpt and ckpt.model_checkpoint_path),
                 'Error: checkpoint directory %s not found or accessible.' % ckpt.model_checkpoint_path)
        saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print('Restoring checkpoint %s' % os.path.join(checkpoint_dir,checkpoint_path))
        saver.restore(session, os.path.join(checkpoint_dir,checkpoint_path))

class TensorboardData:
    def __init__(self, generate_tensorboard_data):
        self.generate_tensorboard_data=generate_tensorboard_data

    def open(self, summaries_dir, run_name, graph):
        if self.generate_tensorboard_data:
            train_path = summaries_dir + '/train_' + run_name
            valid_path = summaries_dir + '/valid_' + run_name
            if os.path.exists(train_path): shutil.rmtree(train_path)
            if os.path.exists(valid_path): shutil.rmtree(valid_path)
            self.train_writer = tf.train.SummaryWriter(summaries_dir + '/train_' + run_name, graph, flush_secs=30)
            self.valid_writer = tf.train.SummaryWriter(summaries_dir + '/valid_' + run_name, graph, flush_secs=120)

    def add_train_summary(self, summary, step):
        if self.generate_tensorboard_data:
            self.train_writer.add_summary(summary, step)

    def add_valid_summary(self, summary, step):
        if self.generate_tensorboard_data:
            self.valid_writer.add_summary(summary, step)

    def close(self):
        if self.generate_tensorboard_data:
            self.train_writer.close()
            self.valid_writer.close()


def get_optimizer(op_loss, op_vis, op_tex, stochastic_loss = False, visual_prob=0.5):
    if not stochastic_loss: return op_loss
    return op_vis if random.random() < visual_prob else op_tex

class InteractiveOptions:
    SaveQuit, Quit, Save, Continue, SaveGoTest, GoTest = range(6)
    _available_signals = set({SaveQuit, Quit, Save, Continue, SaveGoTest, GoTest})

    def __init__(self):
        self.status = self.Continue
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signal, frame):
        self.status = self.get_option_keyboard()

    def get_option_keyboard(self):
        print('Process interrupted by user:')
        print('\t' + str(self.SaveQuit) + '. Save model and quit.')
        print('\t' + str(self.Quit) + '. Quit witout saving.')
        print('\t' + str(self.Save) + '. Save model (posfix "_usersaved").')
        print('\t' + str(self.Continue) + '. Continue.')
        print('\t' + str(self.SaveGoTest) + '. Save and Test the model.')
        print('\t' + str(self.GoTest) + '. Test the last saved model.')
        option = int(raw_input('Choose option [1-5]:'))
        if option not in self._available_signals:
            print("Wrong option.")
            option = self.get_option_keyboard()
        return option





