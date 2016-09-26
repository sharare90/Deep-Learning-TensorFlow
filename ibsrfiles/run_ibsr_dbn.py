import os
from random import shuffle

import numpy as np
import tensorflow as tf

from command_line import config

from yadlt.models.rbm_models import dbn
from yadlt.utils import utilities

from ibsrfiles import settings
from ibsrfiles.read_data import get_file
from ibsrfiles.utils import windowing

# #################### #
#   Flags definition   #
# #################### #
flags = tf.app.flags
FLAGS = flags.FLAGS

# Global configuration
flags.DEFINE_string('dataset', 'custom', 'Which dataset to use. ["mnist", "cifar10", "custom"]')
flags.DEFINE_string('train_dataset', '', 'Path to train set .npy file.')
flags.DEFINE_string('train_labels', '', 'Path to train labels .npy file.')
flags.DEFINE_string('valid_dataset', '', 'Path to valid set .npy file.')
flags.DEFINE_string('valid_labels', '', 'Path to valid labels .npy file.')
flags.DEFINE_string('test_dataset', '', 'Path to test set .npy file.')
flags.DEFINE_string('test_labels', '', 'Path to test labels .npy file.')
flags.DEFINE_string('cifar_dir', '', 'Path to the cifar 10 dataset directory.')
flags.DEFINE_string('model_name', 'dbn', 'Name of the model.')
flags.DEFINE_string('save_predictions', '', 'Path to a .npy file to save predictions of the model.')
flags.DEFINE_string('save_layers_output_test', '',
                    'Path to a .npy file to save test set output from all the layers of the model.')
flags.DEFINE_string('save_layers_output_train', '',
                    'Path to a .npy file to save train set output from all the layers of the model.')
flags.DEFINE_boolean('do_pretrain', False, 'Whether or not pretrain the network.')
flags.DEFINE_boolean('restore_previous_model', False, 'If true, restore previous model corresponding to model name.')
flags.DEFINE_integer('seed', -1, 'Seed for the random generators (>= 0). Useful for testing hyperparameters.')
flags.DEFINE_integer('verbose', 1, 'Level of verbosity. 0 - silent, 1 - print accuracy.')
flags.DEFINE_string('main_dir', 'dbn/', 'Directory to store data relative to the algorithm.')
flags.DEFINE_float('momentum', 0.5, 'Momentum parameter.')

# RBMs layers specific parameters
flags.DEFINE_string('rbm_layers', '250,', 'Comma-separated values for the layers in the sdae.')
flags.DEFINE_boolean('rbm_gauss_visible', False, 'Whether to use Gaussian units for the visible layer.')
flags.DEFINE_float('rbm_stddev', 0.1, 'Standard deviation for Gaussian visible units.')
flags.DEFINE_string('rbm_learning_rate', '0.001,', 'Initial learning rate.')
flags.DEFINE_string('rbm_num_epochs', '100,', 'Number of epochs.')
flags.DEFINE_string('rbm_batch_size', '8000,', 'Size of each mini-batch.')
flags.DEFINE_string('rbm_gibbs_k', '1,', 'Gibbs sampling steps.')

# Supervised fine tuning parameters
flags.DEFINE_string('finetune_act_func', 'sigmoid', 'Activation function.')
flags.DEFINE_float('finetune_learning_rate', 0.5, 'Learning rate.')
flags.DEFINE_float('finetune_momentum', 0.7, 'Momentum parameter.')
flags.DEFINE_integer('finetune_num_epochs', 100, 'Number of epochs.')
flags.DEFINE_integer('finetune_batch_size', 8000, 'Size of each mini-batch.')
flags.DEFINE_string('finetune_opt', 'gradient_descent', '["gradient_descent", "ada_grad", "momentum", "adam"]')
flags.DEFINE_string('finetune_loss_func', 'softmax_cross_entropy',
                    'Loss function. ["mean_squared", "softmax_cross_entropy"]')
flags.DEFINE_float('finetune_dropout', 1, 'Dropout parameter.')

# Conversion of Autoencoder layers parameters from string to their specific type
rbm_layers = utilities.flag_to_list(FLAGS.rbm_layers, 'int')
rbm_learning_rate = utilities.flag_to_list(FLAGS.rbm_learning_rate, 'float')
rbm_num_epochs = utilities.flag_to_list(FLAGS.rbm_num_epochs, 'int')
rbm_batch_size = utilities.flag_to_list(FLAGS.rbm_batch_size, 'int')
rbm_gibbs_k = utilities.flag_to_list(FLAGS.rbm_gibbs_k, 'int')

# Parameters validation
assert FLAGS.dataset in ['mnist', 'cifar10', 'custom']
assert FLAGS.finetune_act_func in ['sigmoid', 'tanh', 'relu']
assert FLAGS.finetune_loss_func in ['mean_squared', 'softmax_cross_entropy']
assert len(rbm_layers) > 0

if __name__ == '__main__':

    utilities.random_seed_np_tf(FLAGS.seed)
    models_dir = os.path.join(config.models_dir, FLAGS.main_dir)
    data_dir = os.path.join(config.data_dir, FLAGS.main_dir)
    summary_dir = os.path.join(config.summary_dir, FLAGS.main_dir)

    # Create the object
    finetune_act_func = utilities.str2actfunc(FLAGS.finetune_act_func)

    srbm = dbn.DeepBeliefNetwork(
        models_dir=models_dir, data_dir=data_dir, summary_dir=summary_dir,
        model_name=FLAGS.model_name, do_pretrain=FLAGS.do_pretrain,
        rbm_layers=rbm_layers, dataset=FLAGS.dataset, main_dir=FLAGS.main_dir,
        finetune_act_func=finetune_act_func, rbm_learning_rate=rbm_learning_rate,
        verbose=FLAGS.verbose, rbm_num_epochs=rbm_num_epochs, rbm_gibbs_k=rbm_gibbs_k,
        rbm_gauss_visible=FLAGS.rbm_gauss_visible, rbm_stddev=FLAGS.rbm_stddev,
        momentum=FLAGS.momentum, rbm_batch_size=rbm_batch_size, finetune_learning_rate=FLAGS.finetune_learning_rate,
        finetune_num_epochs=FLAGS.finetune_num_epochs, finetune_batch_size=FLAGS.finetune_batch_size,
        finetune_opt=FLAGS.finetune_opt, finetune_loss_func=FLAGS.finetune_loss_func,
        finetune_dropout=FLAGS.finetune_dropout)

    number_of_slices_for_each_image = (settings.HEIGHT - (settings.WINDOW_HEIGHT - 1)) * (settings.WIDTH - (
        settings.WINDOW_WIDTH - 1))

    counter_0 = 0
    counter_128 = 0
    counter_192 = 0
    counter_254 = 0
    trX0 = []
    trY0 = []
    trX128 = []
    trY128 = []
    trX192 = []
    trY192 = []
    trX254 = []
    trY254 = []
    number_of_pictures = 7000
    for i in xrange(100):
        print 'Finetune: %d pictures loaded.' % (i + 1)
        image, label = get_file(i, column_format=False)
        imgs, lbls = windowing(image, label, column_format=True, preprocess=True)
        for j in xrange(len(imgs)):
            if lbls[j] == 0:
                if counter_0 < number_of_pictures:
                    trX0.append(imgs[j])
                    trY0.append(lbls[j])
                    counter_0 += 1
            elif lbls[j] == 1:
                if counter_128 < number_of_pictures:
                    trX128.append(imgs[j])
                    trY128.append(lbls[j])
                    counter_128 += 1
            elif lbls[j] == 2:
                if counter_192 < number_of_pictures:
                    trX192.append(imgs[j])
                    trY192.append(lbls[j])
                    counter_192 += 1
            elif lbls[j] == 3:
                if counter_254 < number_of_pictures:
                    trX254.append(imgs[j])
                    trY254.append(lbls[j])
                    counter_254 += 1

    trX0 = np.multiply(trX0, 1)
    trX0 = trX0.reshape(number_of_pictures,
                        settings.WINDOW_WIDTH * settings.WINDOW_HEIGHT)
    trY0 = np.multiply(trY0, 1)
    trY0 = trY0.reshape(number_of_pictures, 1)
    trY0 = np.eye(settings.NUMBER_OF_CLASSES)[[trY0]].reshape(trY0.shape[0], settings.NUMBER_OF_CLASSES)

    trX128 = np.multiply(trX128, 1)
    trX128 = trX128.reshape(number_of_pictures,
                            settings.WINDOW_WIDTH * settings.WINDOW_HEIGHT)
    trY128 = np.multiply(trY128, 1)
    trY128 = trY128.reshape(number_of_pictures, 1)
    trY128 = np.eye(settings.NUMBER_OF_CLASSES)[[trY128]].reshape(trY128.shape[0], settings.NUMBER_OF_CLASSES)

    trX192 = np.multiply(trX192, 1)
    trX192 = trX192.reshape(number_of_pictures,
                            settings.WINDOW_WIDTH * settings.WINDOW_HEIGHT)
    trY192 = np.multiply(trY192, 1)
    trY192 = trY192.reshape(number_of_pictures, 1)
    trY192 = np.eye(settings.NUMBER_OF_CLASSES)[[trY192]].reshape(trY192.shape[0], settings.NUMBER_OF_CLASSES)

    trX254 = np.multiply(trX254, 1)
    trX254 = trX254.reshape(number_of_pictures,
                            settings.WINDOW_WIDTH * settings.WINDOW_HEIGHT)
    trY254 = np.multiply(trY254, 1)
    trY254 = trY254.reshape(number_of_pictures, 1)
    trY254 = np.eye(settings.NUMBER_OF_CLASSES)[[trY254]].reshape(trY254.shape[0], settings.NUMBER_OF_CLASSES)

    TRX = np.concatenate([trX0, trX128, trX192, trX254])
    TRY = np.concatenate([trY0, trY128, trY192, trY254])

    # Fit the model (unsupervised pre-training)
    if FLAGS.do_pretrain:
        srbm.pretrain(TRX, TRX)

    # fine-tuning
    print('Start deep belief net finetuning...')
    srbm.fit(TRX, TRY, TRX, TRY, restore_previous_model=FLAGS.restore_previous_model)

    print('Test set accuracy for 0: {}'.format(srbm.compute_accuracy(trX0, trY0)))
    print('Test set accuracy for 128: {}'.format(srbm.compute_accuracy(trX128, trY128)))
    print('Test set accuracy for 192: {}'.format(srbm.compute_accuracy(trX192, trY192)))
    print('Test set accuracy for 254: {}'.format(srbm.compute_accuracy(trX254, trY254)))

    dcs = srbm.compute_dice_coefficient(TRX, TRY, [0, 1, 2, 3])
    for i in range(4):
        print 'Dice Coefficients for label %d: %0.3f.' % (i, dcs[i])

    for i in range(4):
        print '%0.3f' % dcs[i],

    # print('Test set accuracy: {}'.format(srbm.compute_accuracy(teX, teY)))

    # TEST_IMAGE_NUMBER = 20
    # image, label = get_file(TEST_IMAGE_NUMBER, False)
    #
    # label[np.where(label == 1)] = 128
    # label[np.where(label == 2)] = 192
    # label[np.where(label == 3)] = 254
    #
    # colored_label = np.zeros([256, 256, 3])
    # # import pdb
    # # pdb.set_trace()
    # # guess = srbm.predict(image.reshape(1, settings.WINDOW_HEIGHT * settings.WINDOW_WIDTH))
    # # if guess != 0:
    # #     print guess
    # #     quit()
    # for i in xrange(100, 140):
    #     for j in xrange(100, 140):
    #         sample = image[i - settings.WINDOW_HEIGHT / 2: i + settings.WINDOW_HEIGHT / 2 + 1,
    #                  j - settings.WINDOW_WIDTH / 2: j + settings.WINDOW_WIDTH / 2 + 1]
    #         # import ipdb
    #         # ipdb.set_trace()
    #         guess = srbm.predict(sample.reshape(1, settings.WINDOW_HEIGHT * settings.WINDOW_WIDTH))
    #         if guess != 0:
    #             print guess
    #             quit()
    #             if guess == 1:
    #                 colored_label[i, j, 0] = 255
    #             elif guess == 2:
    #                 colored_label[i, j, 1] = 255
    #             if guess == 3:
    #                 colored_label[i, j, 2] = 255
    #
    # import matplotlib.pyplot as plt
    #
    # plt.imshow(colored_label)
    # # plt.imshow(image)
    # plt.show()

    # # Save the predictions of the model
    # if FLAGS.save_predictions:
    #     print('Saving the predictions for the test set...')
    #     np.save(FLAGS.save_predictions, srbm.predict(teX))
    #
    #
    # def save_layers_output(which_set):
    #     if which_set == 'train':
    #         trout = srbm.get_layers_output(trX)
    #         for i, o in enumerate(trout):
    #             np.save(FLAGS.save_layers_output_train + '-layer-' + str(i + 1) + '-train', o)
    #
    #     elif which_set == 'test':
    #         teout = srbm.get_layers_output(teX)
    #         for i, o in enumerate(teout):
    #             np.save(FLAGS.save_layers_output_test + '-layer-' + str(i + 1) + '-test', o)
    #
    #
    # # Save output from each layer of the model
    # if FLAGS.save_layers_output_test:
    #     print('Saving the output of each layer for the test set')
    #     save_layers_output('test')
    #
    # # Save output from each layer of the model
    # if FLAGS.save_layers_output_train:
    #     print('Saving the output of each layer for the train set')
    #     save_layers_output('train')
