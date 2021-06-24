import pandas as pd
# import numpy as np
import re
# import time
import tensorflow_datasets as tfds
# import tensorflow as tf
from models.Modelling import *
from config.FilePathConfig import *

# Load Model Module
class InsideOut:
    def __init__(self, bot_type, num_layers):
        train_data = pd.read_csv(TRAIN_DATA[bot_type], index_col=0)
        checkpoint_path = CHECKPOINT[bot_type]

        # Hyper-parameters
        self.D_MODEL = 256
        self.NUM_LAYERS = num_layers
        self.NUM_HEADS = 8
        self.DFF = 512
        self.DROPOUT = 0.1

        self.questions = []
        for sentence in train_data['Q']:
            sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
            sentence = sentence.strip()
            self.questions.append(sentence)

        self.answers = []
        for sentence in train_data['A']:
            sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
            sentence = sentence.strip()
            self.answers.append(sentence)

        self.tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            self.questions + self.answers, target_vocab_size=2 ** 13)

        self.START_TOKEN, self.END_TOKEN = [self.tokenizer.vocab_size], [self.tokenizer.vocab_size + 1]
        self.VOCAB_SIZE = self.tokenizer.vocab_size + 2
        self.MAX_LENGTH = 40

        tf.keras.backend.clear_session()

        self.model = transformer(
            vocab_size = self.VOCAB_SIZE,
            num_layers = self.NUM_LAYERS,
            dff = self.DFF,
            d_model = self.D_MODEL,
            num_heads = self.NUM_HEADS,
            dropout = self.DROPOUT)

        self.model.load_weights(checkpoint_path)

    def preprocess_sentence(self, sentence):
        sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
        sentence = sentence.strip()
        return sentence

    def evaluate(self, sentence):
        sentence = self.preprocess_sentence(sentence)

        sentence = tf.expand_dims(
            self.START_TOKEN + self.tokenizer.encode(sentence) + self.END_TOKEN, axis=0)

        output = tf.expand_dims(self.START_TOKEN, 0)

        for i in range(self.MAX_LENGTH):
            predictions = self.model(inputs=[sentence, output], training=False)
            predictions = predictions[:, -1:, :]
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            if tf.equal(predicted_id, self.END_TOKEN[0]):
                break
            output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0)

    def predict(self, sentence):
        prediction = self.evaluate(sentence)

        predicted_sentence = self.tokenizer.decode(
            [i for i in prediction if i < self.tokenizer.vocab_size])

        return predicted_sentence



