#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import math

# # Scaled Dot-Product attention function

def dot_product_attn(q, k, v, mask = None):
    
    lenQ = q.get_shape()[-1]
    
    energies = tf.multiply(1/lenQ**0.5, tf.matmul(q, k, transpose_b = True))
    
    if not mask is None:
        mask = (1. - mask) * -1e9
        energies = tf.add(energies, mask)
    
    alphas = tf.nn.softmax(energies, axis = -1)
    
    context = tf.matmul(alphas, v)
    
    return context


# # Multihead Projection

class MultiHeadProjection(tf.keras.layers.Layer):
    
    def __init__(self, projected_dim, heads = 8, **kwargs):
        super().__init__(**kwargs)
        self.h = heads
        self.projected_dim = projected_dim
        
    def build(self, input_shape):
         
        assert(len(input_shape) == 3), 'Expected input of rank 3: (m, Tx, d_model)'
        
        self.m, self.k, self.model_dim = input_shape
        
        self.W = self.add_weight(
                name = 'W',
                shape = (self.h, self.model_dim, self.projected_dim), 
                initializer = 'glorot_normal', 
                trainable = True)
        
        self.b = self.add_weight(
                name = 'b',
                shape = (self.h, 1, self.projected_dim), initializer = 'Zeros', 
                trainable = True)
        
    def call(self, X):
        
        X = tf.expand_dims(X, 1) # adds a head layer
        
        output = tf.add(tf.matmul(X, self.W), self.b)
        
        return output



class AttentionLayer(tf.keras.layers.Layer):
   
    def __init__(self, projected_dim, heads = 8, **kwargs):
        super().__init__(**kwargs)
        self.h = heads
        self.projected_dim = projected_dim
        
    def build(self, input_shape):
        
        for input_ in input_shape:
            assert(len(input_) == 3), 'Expected input shape of (m, Tx, d)'
        
        (self.projQ, self.projK, self.projV) = (MultiHeadProjection(self.projected_dim, self.h) 
                                       for input_ in input_shape)
        
        output_d = input_shape[-1][-1]
        
        self.reshaper = tf.keras.layers.Reshape(target_shape = (-1, self.projected_dim * self.h))
        
        self.dense = tf.keras.layers.Dense(output_d)
        
    def call(self, X, mask = None):
        '''
        Arguments
        X: list of (Q, K, V)
        mask: for softmax layer
        '''
        
        (Q,K,V) = X
        
        Q, K, V = self.projQ(Q), self.projK(K), self.projV(V)
        
        #print(Q.get_shape(), K.get_shape(), V.get_shape())
        
        attention = dot_product_attn(Q, K, V, mask = mask)
        
        #print(attention.get_shape())
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        
        flattened = self.reshaper(attention)
        
        #print(flattened.get_shape())
        
        output = self.dense(flattened)
        
        return output


# # Fully Connected Layer

class FCNNLayer(tf.keras.layers.Layer):
    
    def __init__(self, d_model, dff, **kwargs):
        super().__init__(**kwargs)
        self.d_model, self.dff = d_model, dff
        
    def build(self, input_shape):
        self.dense1 = tf.keras.layers.Dense(self.dff, activation = 'relu')
        self.dense2 = tf.keras.layers.Dense(self.d_model, activation = 'linear')
        
    def call(self, X):
        return self.dense2(self.dense1(X))


# #  Encoder Layer

class TransformerEncoder(tf.keras.layers.Layer):
    
    def __init__(self, dff = 2048, heads = 8, dropout = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.h = heads
        self.dropout = dropout
        self.dff = dff
        
    def build(self, input_shape):
        assert(len(input_shape) == 3), 'Expected input shape of (m, Tx, d)'
        
        (self.m, self.k, self.d_model) = input_shape
        
        self.projected_dim = self.d_model//self.h
        
        self.attn = AttentionLayer(self.projected_dim, self.h)
        self.drop1 = tf.keras.layers.Dropout(self.dropout)
        self.norm1 = tf.keras.layers.LayerNormalization()
        
        self.fcnn = FCNNLayer(self.d_model, self.dff)
        self.drop2 = tf.keras.layers.Dropout(self.dropout)
        self.norm2 = tf.keras.layers.LayerNormalization()
                
    def call(self, X, training = True, mask = None):
        
        attn_output = self.drop1(self.attn([X,X,X], mask = mask), training = training)
        
        X = self.norm1(attn_output + X)
        
        fcnn_output = self.drop2(self.fcnn(X), training = training)
        
        X = self.norm2(fcnn_output + X)
        
        return X  

# # DecoderLayer

class TransformerDecoder(tf.keras.layers.Layer):
    
    def __init__(self, dff = 2048, heads = 8, dropout = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.h = heads
        self.dropout = dropout
        self.dff = dff
        
    def build(self, input_shape):
        assert(len(input_shape) == 3), 'Expected input shape of (m, Tx, d)'
        
        (self.m, self.k, self.d_model) = input_shape
        
        self.projected_dim = self.d_model//self.h
        
        self.intr_attn = AttentionLayer(self.projected_dim, self.h)
        self.drop1 = tf.keras.layers.Dropout(self.dropout)
        self.norm1 = tf.keras.layers.LayerNormalization()
        
        self.enc_dec_attn = AttentionLayer(self.projected_dim, self.h)
        self.drop2 = tf.keras.layers.Dropout(self.dropout)
        self.norm2 = tf.keras.layers.LayerNormalization()
        
        self.fcnn = FCNNLayer(self.d_model, self.dff)
        self.drop3 = tf.keras.layers.Dropout(self.dropout)
        self.norm3 = tf.keras.layers.LayerNormalization()
                
    def call(self, X, encoder_output, lookahead_mask = None, encoder_padding_mask = None, training = True):
        
        # attention mechanism 1
        attn_output = self.drop1(self.intr_attn([X,X,X], mask = lookahead_mask), training = training)
        X = self.norm1(attn_output + X)
                              
        # attention mechanism 2
        attn_output = self.drop2(self.enc_dec_attn([X,encoder_output, encoder_output], mask = encoder_padding_mask), training = training)
        X = self.norm2(attn_output + X)
                                 
        # fcnn
        fcnn_output = self.drop3(self.fcnn(X), training = training)
        X = self.norm3(fcnn_output + X)               
                
        return X  


# # Position Encoding Layer

class PositionalEmbedding(tf.keras.layers.Layer):
    
    def __init__(self, max_seq_len = 1000, **kwargs):
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len
    
    def build(self, input_shape):
        
        (self.m, self.k, self.d_model) = input_shape
        self.k = self.max_seq_len
        
        pos = np.arange(self.k).reshape(-1,1)
        
        i = 1 / np.power(10000, 2 * np.arange(self.d_model) / self.d_model)
        
        embeddings = pos * i
        
        evens = np.arange(0, self.d_model, 2)
        
        odds = evens + 1
        
        embeddings[:, evens] = np.sin(embeddings[:, evens])
        
        embeddings[:, odds] = np.cos(embeddings[:, odds])
        
        self.embeddings = tf.convert_to_tensor(np.expand_dims(embeddings, 0), dtype = 'float32')
        
    def call(self, X):

        Tx = X.get_shape()[1]
        
        X = X + self.embeddings[:, :Tx, :]
        
        return tf.multiply(X, self.d_model**0.5)

# # Encoder Stack

class EncoderStack(tf.keras.layers.Layer):
    
    def __init__(self, num_classes, d_model = 512, num_layers = 6, num_heads = 8, dropout = 0.1, dff = 2048, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.dff = dff
        
    def build(self, input_shape):

        self.embedding = tf.keras.layers.Embedding(self.num_classes, self.d_model, mask_zero = True)
        
        self.positional_embedding = PositionalEmbedding()

        self.embedding_dropout = tf.keras.layers.Dropout(self.dropout)
        
        self.encoders = [
            TransformerEncoder(dff = self.dff, heads = self.num_heads, dropout = self.dropout) 
            for i in range(self.num_layers)
        ]
        
    def call(self, seqs, training = True):
        
        X = self.embedding(seqs)
        
        #expand the mask from the embedding layer from (m, Tx) to (m, 1, 1, Tx) for multihead softmax
        encoder_mask = tf.dtypes.cast(self.embedding.compute_mask(seqs), 'float32')[:, tf.newaxis, tf.newaxis, :]
        
        X = self.positional_embedding(X)

        X = self.embedding_dropout(X, training = training)
        
        X = X * self.d_model**0.5
        
        #call(self, X, encoder_output, lookahead_mask = None, encoder_padding_mask = None, training = True)
        for encoder in self.encoders:
            X = encoder(X, mask = encoder_mask, training = training)
            
        return X, encoder_mask


# # Decoder Stack

class DecoderStack(tf.keras.layers.Layer):
    
    
    def __init__(self, num_classes, d_model = 512, num_layers = 6, num_heads = 8, dropout = 0.1, dff = 2048, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.dff = dff
        
    def build(self, input_shape):
        
        assert(len(input_shape) == 3), 'Expected input with len 3 in the form of (decoder_input, encoder_output, encoder_mask)'
        #assert(input_shape[0][1] == input_shape[1][1]), 'Expected encoder output and decoder input to have same time dimension'
        
        #(_, k) = input_shape[0]
        
        self.embedding = tf.keras.layers.Embedding(self.num_classes, self.d_model, mask_zero = True)
        
        self.embedding_dropout = tf.keras.layers.Dropout(self.dropout)
        
        self.positional_embedding = PositionalEmbedding()
        
        self.decoders = [
            TransformerDecoder(dff = self.dff, heads = self.num_heads, dropout = self.dropout) 
            for i in range(self.num_layers)
        ]
            
    def call(self, inputs, training = True):
        
        (seqs, encoder_output, encoder_mask) = inputs

        (_, k) = seqs.shape

        num_ones = 0.5 * (k**2 + k)

        trailing_mask = tfp.math.fill_triangular(tf.ones(num_ones), upper = False)
        
        loss_mask = self.embedding.compute_mask(seqs)
        #expand the mask from the embedding layer from (m, Tx) to (m, 1, 1, Tx) for multihead softmax
        decoder_padding_mask = tf.dtypes.cast(loss_mask, 'float32')[:, tf.newaxis, tf.newaxis, :]
        #then add trailing mask to it
        decoder_mask = tf.multiply(decoder_padding_mask, trailing_mask)
        
        #print(decoder_mask)
        X = self.embedding(seqs)
        
        X = self.positional_embedding(X)

        X = self.embedding_dropout(X, training = training)

        X = X * self.d_model**0.5
        
        #call(self, X, encoder_output, lookahead_mask = None, encoder_padding_mask = None, training = True)
        for decoder in self.decoders:
            X = decoder(X, encoder_output, lookahead_mask = decoder_mask, 
                        encoder_padding_mask = encoder_mask, training = training)
            
        X = tf.matmul(X, tf.transpose(self.embedding.embeddings))
            
        return X, loss_mask
        

# # Transformer Model

class TransformerModel(tf.keras.Model):

    def __init__(self, num_encoder_classes, num_decoder_classes, d_model = 512, num_layers = 6, num_heads = 8, dropout = 0.1, dff = 2048):
        super().__init__()

        self.encoder_stack = EncoderStack(num_encoder_classes, d_model, num_layers, num_heads, dropout, dff)

        self.decoder_stack = DecoderStack(num_decoder_classes, d_model, num_layers, num_heads, dropout, dff)

    def call(self, X, Y, training):

            enc_output, encoder_mask = self.encoder_stack(X, training = training)
        
            logits, mask = self.decoder_stack((Y, enc_output, encoder_mask), training = training)

            return logits, mask
        

class TransformerLoss():

    def __init__(self):
        
        self.loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(
                        from_logits=True, reduction='none')

    def __call__(self, labels, logits, loss_mask):

        losses = self.loss_obj(labels, logits)

        mean_loss = tf.reduce_mean(tf.boolean_mask(losses, loss_mask))

        return mean_loss 


# # Optimizer

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def TransformerOptimizer(d_model):
    
    learning_rate = CustomSchedule(d_model)

    return tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)


class Transformer():

    def __init__(self, num_encoder_classes, num_decoder_classes, d_model = 512, num_layers = 6, num_heads = 8, dropout = 0.1, dff = 2048):

        self.model = TransformerModel(num_encoder_classes, num_decoder_classes, d_model, num_layers, num_heads, dropout, dff)

        self.loss = TransformerLoss()

        self.opt = TransformerOptimizer(d_model)


    def train_step(self, X,Y, train = True):

        decoder_input = Y[:,:-1] # don't include end token pushed into decoder
        decoder_target = Y[:,1:] # don't include start token in decoder output label

        with tf.GradientTape() as tape:

            predictions, mask = self.model(X, decoder_input, training = train)

            loss = self.loss(decoder_target, predictions, mask)

        gradients = tape.gradient(loss, self.model.trainable_variables)    
        self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss.numpy()


    def fit_on_tfds(self, dataset, exp_loss_beta = 0.9, train = True):

        batch_losses = []
        epoch_loss = 0
        update_print = ''

        for (batch_num, (batch_x, batch_y)) in enumerate(dataset):

            batch_loss = self.train_step(batch_x, batch_y, train = train)
        
            if train:
                epoch_loss = (exp_loss_beta * epoch_loss + (1 - exp_loss_beta) * batch_loss)/(1 - exp_loss_beta**(batch_num + 1))
                update_print = '\tBatch ' + str(batch_num + 1) + ', Loss: {0:.5f}'.format(epoch_loss)
                print(update_print, end='\r')

            batch_losses.append(batch_loss)

        print(update_print)

        if train:
            return epoch_loss
        else:
            return np.mean(batch_losses)



    def train_in_batches(self, X, Y, batch_size, exp_loss_beta = 0.9, EMA = True):

        m = X.shape[0]
        
        num_batches = np.ceil(m / batch_size)

        losses = []

        epoch_loss = 0
        
        for (batch_num, step) in enumerate(range(0, m, batch_size)):

                batch_x = X[step : np.min(step + batch_size, m)]
                batch_y = Y[step : np.min(step + batch_size, m)]

                batch_loss = self.train_step(batch_x, batch_y)

                if EMA:
                    epoch_loss = (exp_loss_beta * epoch_loss + (1 - exp_loss_beta) * batch_loss)/(1 - exp_loss_beta**(batch_num + 1))
                    print('\tBatch ', batch_num + 1, '/', num_batches, ', Loss: ', epoch_loss, end = '\r')
                
                losses.append(batch_loss)

        return np.mean(losses)


    def train(self, X, Y, epochs = 100, batch_size = 32, validation_split = 0.1):

        if validation_split > 0:
            m = X.shape[0]
            validation_m = (validation_split * m)//1
            m -= validation_m

            trainX, trainY, validationX, validationY = X[:m], Y[:m], X[m:], Y[m:]
        else:
            trainX, trainY = X,Y

        epoch_losses = []

        try:

            for epoch in range(epochs):

                print('-- EPOCH ', epoch + 1, ' --')

                epoch_loss = self.train_in_batches(trainX, trainY, batch_size, exp_loss_beta = 0.9, EMA = True)

                print('\tEpoch ', epoch + 1, ' Loss: ', epoch_loss)

                if validation_split > 0:

                    validation_loss = self.train_in_batches(validationX, validationY, batch_size, EMA = False)

                    print('\tValidation Loss: ', validation_loss)
                
                epoch_losses.append(epoch_loss)

        except KeyboardInterrupt:
            pass

        return epoch_losses


    def infer(self, X, Y):

        assert(len(X.shape) == 2 and len(Y.shape == 2)), 'X and Y must both be rank 2 matrices: (m, Tx) where each entry is lookup index for embedding layer'
        assert(X.shape[0] == 0), 'Batch size must be 1 for inference'

        infer_Tx = np.count_nonzero(Y)

        predictions = self.model((X,Y), training = False)[0,infer_Tx] # returns rank 1 array of num_classes

        probabilities = tf.nn.softmax(predictions, axis = -1)

        return probabilities

print('Loading Transformer Module')