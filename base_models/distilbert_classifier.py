#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 00:07:54 2025

@author: dev
"""


from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
import tensorflow as tf


def init_model(num_classes=2):
    """
    Initializes a DistilBERT text classifier.
    Returns the model, tokenizer, and type.

    Hyperparameters can be edited as needed.
    """
    # Load tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        'distilbert-base-uncased')

    # Load DistilBERT model
    model = TFDistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=num_classes
    )

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model, tokenizer, 'tensorflow'
