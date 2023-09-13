# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 15:05:59 2020

@author: Tom
"""

import tensorflow as tf
import numpy as np

class NegativeLogLikelihood():
    def __init__(self, followup, event):
        
        self.followup = tf.constant(followup, dtype='float32')
        self.event = tf.constant(event, dtype='float32')
    
    def loss(self, ids_batch, y_pred):
        """ Courtesy of https://nbviewer.jupyter.org/github/sebp/survival-cnn-estimator/blob/master/tutorial_tf2.ipynb
        Normalize risk scores to avoid exp underflowing.

        Note that only risk scores relative to each other matter.
        If minimum risk score is negative, we shift scores so minimum
        is at zero.
        """
        
        ids_batch = tf.squeeze(ids_batch)
        y_pred = tf.squeeze(y_pred)

        y_pred_min = tf.math.reduce_min(y_pred)
        c = tf.zeros_like(y_pred_min)
        norm = tf.where(y_pred_min < 0, -y_pred_min, c)
        y_pred_normed = y_pred + norm
        
        ids_batch = tf.cast(ids_batch, 'int32')
        followup_batch = tf.gather(self.followup, ids_batch)
        events_batch = tf.cast(tf.gather(np.array(self.event), ids_batch), 'float32')
        ids_sorted = tf.argsort(followup_batch, direction='DESCENDING')
        y_pred_ordered = tf.gather(y_pred_normed, ids_sorted)
        events = tf.gather(events_batch, ids_sorted)

        #Also, for numerical stability, subtract the maximum value before taking the exponential
        amax = tf.reduce_max(y_pred_ordered)
        y_pred_shift = y_pred_ordered - amax

        hazard_ratio = tf.math.exp(y_pred_shift)
        log_risk = tf.math.log(tf.math.cumsum(hazard_ratio, axis=0)) + amax
        uncensored_likelihood = log_risk - y_pred_ordered
        censored_likelihood = tf.math.multiply(events, tf.squeeze(uncensored_likelihood))
        
        m = tf.math.reduce_mean(censored_likelihood)
        
        return m