from helpers import err_exit


def define_commom_flags(flags, num_steps=50001, summary_frequency=10):
    flags.DEFINE_boolean('debug', False, 'Activates the debug mode.')

    # training settings
    flags.DEFINE_integer('num_steps', num_steps, 'Maximum number of training steps (default '+str(num_steps)+').')
    flags.DEFINE_integer('batch_size', 64, 'Batch size (default 64).')
    flags.DEFINE_integer('summary_frequency', summary_frequency, 'Determines the number of steps after which to show the summaries (default '+str(summary_frequency)+').')
    flags.DEFINE_integer('validation_frequency', 1000, 'Number of steps after which to validate next batch (default 1000).')
    flags.DEFINE_integer('validation_batch', 64, 'Batch size for the validation set (default 64).')

    # net settings
    flags.DEFINE_integer('fc_layer', 6, 'fc feature layer from the AlexNet. "6" for fc6 (default), "7" for fc7.')
    flags.DEFINE_integer('visual_dim', 4096, 'Dimensionality of the visual embeddings space (default 4096).')
    flags.DEFINE_boolean('train', True, 'Set the model to be trained (default True).')
    flags.DEFINE_boolean('test', True, 'Set the model to be tested (after training, if eventually activated; Default True).')
    flags.DEFINE_boolean('boarddata', True, 'Set to False to desactivate the Tensorboard data generation (default True).')
    flags.DEFINE_string('run', '', 'Specifies a name for the run (defaults to the date and time when it is run).')
    flags.DEFINE_boolean('save_predictions', True, 'Set to True to save all predictions on a file.')
    flags.DEFINE_string('checkpoint', None, 'Path for a custom checkpoint file to restore the net parameters.')
    flags.DEFINE_string('retrieval', 'pca', "Chooses the retrieval algorithm ['pca'(default), 'cosine']")

    FLAGS = flags.FLAGS
    err_exit(FLAGS.fc_layer not in [6, 7], "Error, parameter 'fc_layer' should be either '6' or '7'.")
    if not FLAGS.checkpoint is None:
        err_exit(FLAGS.train != False or FLAGS.test == False, "Error, a checkpoint for testing was especified but the run is not"
                                                              " set for testing. Use --notrain --test when specifying a checkpoint.")

    return flags