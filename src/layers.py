"""
Definition of the layers necessary for the ESIM model.

Inspired from the code on:
https://github.com/yuhsinliu1993/Quora_QuestionPairs_DL
"""

import keras.backend as K
from keras.models import Sequential
from keras.layers import *


class EmbeddingLayer(object):
    """
    Layer to transform words represented by indices to word embeddings.
    """

    def __init__(self, voc_size, output_dim, embedding_weights=None,
                 max_length=100, trainable=True, mask_zero=False):
        self.voc_size = voc_size
        self.output_dim = output_dim
        self.max_length = max_length

        if embedding_weights is not None:
            self.model = Embedding(voc_size, output_dim,
                                   weights=[embedding_weights],
                                   input_length=max_length,
                                   trainable=trainable, mask_zero=mask_zero,
                                   name='embedding')
        else:
            # If no pretrained embedding weights are passed to the initialiser,
            # the model is set to be trainable by default.
            self.model = Embedding(voc_size, output_dim,
                                   input_length=max_length, trainable=True,
                                   mask_zero=mask_zero, name='embedding')

    def __call__(self, input):
        return self.model(input)


class EncodingLayer(object):
    """
    Layer to encode variable length sentences with a BiLSTM.
    """

    def __init__(self, hidden_units, max_length=100, dropout=0.5,
                 activation='tanh', sequences=True):
        self.layer = Bidirectional(LSTM(hidden_units, activation=activation,
                                        return_sequences=sequences,
                                        dropout=dropout,
                                        recurrent_dropout=dropout),
                                   merge_mode='concat')
# hidden_units是cell输出的维度，在MLP中，输出维度与神经元数量是相等的，但是在LSTM中神经元数量与输出维度不想等，1个cell里有4个gate（激活函数），所以参数数量也就是神经元个数远远大于输出维度， 一般好似M级的。
# activation - 输入门的输入经过tahn变换
# recurrent_activation - 其他3个门经过sigmoid变换
# dropout - activation那一层之后的dropout
# recurrent_dropout - recurrent_activation之后的dropout
# return_sequences - 如果return_sequences=True：返回形如（samples，timesteps，output_dim）的3D张量否则，每个时间戳的output都返回。
# return_sequences=False：返回形如（samples，output_dim）的2D张量, 就是只返回最后一个时间戳的output。
# 注意多层LSTM： 前面层都要return_sequences=True, 因为后面LSTM层需要这些output作为输入，只有最后一层return_sequences=False.因为最后一层只取一个输出值。


    def __call__(self, input):
        return self.layer(input)


class LocalInferenceLayer(object):
    """
    Layer to compute local inference between two encoded sentences a and b.
    """

    def __call__(self, inputs):
        a = inputs[0]
        b = inputs[1]

        attention = Lambda(self._attention,
                           self._attention_output_shape)(inputs)

        align_a = Lambda(self._soft_alignment,
                         self._soft_alignment_output_shape)([attention, a])
        align_b = Lambda(self._soft_alignment,
                         self._soft_alignment_output_shape)([attention, b])

        # Enhancement of the local inference information obtained with the
        # attention mecanism and soft alignments.
        sub_a_align = Lambda(lambda x: x[0]-x[1])([a, align_a])
        sub_b_align = Lambda(lambda x: x[0]-x[1])([b, align_b])

        mul_a_align = Lambda(lambda x: x[0]*x[1])([a, align_a])
        mul_b_align = Lambda(lambda x: x[0]*x[1])([b, align_b])

        m_a = concatenate([a, align_a, sub_a_align, mul_a_align])
        m_b = concatenate([b, align_b, sub_b_align, mul_b_align])

        return m_a, m_b

    def _attention(self, inputs):
        """
        Compute the attention between elements of two sentences with the dot
        product.

        Args:
            inputs: A list containing two elements, one for the first sentence
                    and one for the second, both encoded by a BiLSTM.

        Returns:
            A tensor containing the dot product (attention weights between the
            elements of the two sentences).
        """
        attn_weights = K.batch_dot(x=inputs[0],
                                   y=K.permute_dimensions(inputs[1],
                                                          pattern=(0, 2, 1)))
        return K.permute_dimensions(attn_weights, (0, 2, 1))

    def _attention_output_shape(self, inputs):
        input_shape = inputs[0]
        embedding_size = input_shape[1]
        return (input_shape[0], embedding_size, embedding_size)

    def _soft_alignment(self, inputs):
        """
        Compute the soft alignment between the elements of two sentences.

        Args:
            inputs: A list of two elements, the first is a tensor of attention
                    weights, the second is the encoded sentence on which to
                    compute the alignments.

        Returns:
            A tensor containing the alignments.
        """
        attention = inputs[0]
        sentence = inputs[1]

        # Subtract the max. from the attention weights to avoid overflows.
        exp = K.exp(attention - K.max(attention, axis=-1, keepdims=True))
        exp_sum = K.sum(exp, axis=-1, keepdims=True)
        softmax = exp / exp_sum

        return K.batch_dot(softmax, sentence)

    def _soft_alignment_output_shape(self, inputs):
        attention_shape = inputs[0]
        sentence_shape = inputs[1]
        return (attention_shape[0], attention_shape[1], sentence_shape[2])


class InferenceCompositionLayer(object):
    """
    Layer to compose the local inference information.
    """

    def __init__(self, hidden_units, max_length=100, dropout=0.5,
                 activation='tanh', sequences=True):
        self.hidden_units = hidden_units
        self.max_length = max_length
        self.dropout = dropout
        self.activation = activation
        self.sequences = sequences

    def __call__(self, input):
        composition = Bidirectional(LSTM(self.hidden_units,
                                         activation=self.activation,
                                         return_sequences=self.sequences,
                                         recurrent_dropout=self.dropout,
                                         dropout=self.dropout))(input)
        reduction = TimeDistributed(Dense(self.hidden_units,
                                          kernel_initializer='he_normal',
                                          activation='relu'))(composition)

        return Dropout(self.dropout)(reduction)


class PoolingLayer(object):
    """
    Pooling layer to convert the vectors obtained in the previous layers to
    fixed-length vectors.
    """

    def __call__(self, inputs):
        a = inputs[0]
        b = inputs[1]

        a_avg = GlobalAveragePooling1D()(a)
        a_max = GlobalMaxPooling1D()(a)

        b_avg = GlobalAveragePooling1D()(b)
        b_max = GlobalMaxPooling1D()(b)

        return concatenate([a_avg, a_max, b_avg, b_max])


class MLPLayer(object):
    """
    Multi-layer perceptron for classification.
    """

    def __init__(self, hidden_units, n_classes, dropout=0.5,
                 activations=['tanh', 'softmax']):
        self.model = Sequential()
        self.model.add(Dense(hidden_units, kernel_initializer='he_normal',
                             activation=activations[0],
                             input_shape=(4*hidden_units,)))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(n_classes, kernel_initializer='zero',
                             activation=activations[1]))

    def __call__(self, input):
        return self.model(input)
