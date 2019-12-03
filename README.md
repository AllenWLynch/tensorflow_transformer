# SOTA Seq2Seq: Transformer

Architecture as described in <a href="https://arxiv.org/abs/1706.03762"><i>Vaswani et al. 2017</i></a>, implemented using Tensorflow 2.0.

Author: Allen Lynch, 2019

## Usage

<i>tf_transformer.py</i> contains the transformer library. To instantiate a transformer model, complete with implemented loss computation (mask-modified categorical crossentropy), learning rate scheduling, training loop, and inference mode, import the **Transformer** class and specify the network size.

## Validation

Transformer was confirmed to converge on data provided by TensorFlow core tutorial: <a href= "https://www.tensorflow.org/tutorials/text/transformer">https://www.tensorflow.org/tutorials/text/transformer</a>.
