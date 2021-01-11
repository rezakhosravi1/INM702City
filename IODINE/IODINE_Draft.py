# %% 
import tensorflow as tf 
import tensorflow_probability as tfp
import numpy as np
import math
# %%
tfk = tf.keras
tfkl = tfk.layers
tfpl = tfp.layers
tfd = tfp.distributions
# %%
class Sampling(tfkl.Layer):

    def call(self, inputs):
        z_mean, z_log_var = tf.split(inputs, num_or_size_splits=2, axis=-1)
        epsilon = tfk.backend.random_normal(shape=tf.shape(z_mean))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        return z_mean, z_log_var, z

# TO HELP PREVENT NAN IN LOSS
@tf.function
def my_exponential_linear_unit(input):
    return (tfk.backend.elu(input) + tfk.backend.epsilon())

batch_size = 32
T = 5
K = 4 
latent_dim = 64
kernel_size = (5,5)
IMAGE_SIZE = [35, 35]

x = tf.linspace(-1.0,1.0, IMAGE_SIZE[1])
y = tf.linspace(-1.0,1.0, IMAGE_SIZE[0])
xx_, yy_ = tf.meshgrid(x, y)
indices = tf.range(len(tf.reshape(xx_, [-1])), dtype=tf.int32)
shuffled_indices = tf.random.shuffle(indices, seed=101)
xx = tf.gather(tf.reshape(xx_, [-1]), shuffled_indices )
xx = tf.reshape(xx, tf.shape(xx_))
yy = tf.gather(tf.reshape(yy_, [-1]), shuffled_indices )
yy = tf.reshape(yy, tf.shape(yy_))
xx = xx[tf.newaxis, tf.newaxis, :, :, tf.newaxis]
yy = yy[tf.newaxis, tf.newaxis, :, :, tf.newaxis]

xx_coor = tfk.backend.tile(xx, (batch_size,K,1,1,1))
yy_coor = tfk.backend.tile(yy, (batch_size,K,1,1,1))

x_coordinate = tf.constant(xx_coor)
y_coordinate = tf.constant(yy_coor)

# BUILD THE SPATIAL BROADCASTING LAYER 
class SpatialBroadCast(tfkl.Layer):

    def call(self, inputs, x_coordinate, y_coordinate):
        z_newaxis = inputs[:, :, tf.newaxis, tf.newaxis, :]
        z_tile = tfk.backend.tile(z_newaxis, (1,1,IMAGE_SIZE[0],IMAGE_SIZE[1],1))
        z_spatial_broadcast = tf.concat([z_tile, 
                                         x_coordinate,
                                         y_coordinate],
                                        axis=-1)
        return z_spatial_broadcast


# DECODER 
latent_inputs = tfk.Input(shape=(K,latent_dim,),
                          name='z_samling')
z_spatial_broadcast = SpatialBroadCast()(latent_inputs, x_coordinate, y_coordinate)
decoder_output = tfkl.Conv2D(filters=32, kernel_size=kernel_size, padding='same', 
                activation=my_exponential_linear_unit, name='decoder_conv_1')(z_spatial_broadcast)
decoder_output = tfkl.Conv2D(filters=32, kernel_size=kernel_size, padding='same', 
                activation=my_exponential_linear_unit, name='decoder_conv_2')(decoder_output)
decoder_output = tfkl.Conv2D(filters=32, kernel_size=kernel_size, padding='same', 
                activation=my_exponential_linear_unit, name='decoder_conv_3')(decoder_output)
decoder_output = tfkl.Conv2D(filters=32, kernel_size=kernel_size, padding='same', 
                activation=my_exponential_linear_unit, name='decoder_conv_4')(decoder_output)
rgb_mask = tfkl.Conv2D(filters=4, kernel_size=kernel_size, padding='same', 
                activation=tfk.activations.linear, name='decoder_conv_5')(decoder_output)
generator = tfk.Model(inputs=latent_inputs,
                      outputs=rgb_mask,
                      name='generator')

# %%
# # DEFINE THE SLOTS AND RECONSTRUCTION MODEL
class Split(tfkl.Layer):

    def call(self, inputs):
        return tf.split(inputs, num_or_size_splits=4, axis=-1)

class MyMerge(tfkl.Layer):

    def call(self, inputs):
        return tf.transpose(inputs, [1,0,2])

class Expand(tfkl.Layer):

    def call(self, inputs):
        if len(tf.shape(inputs)) == 4:
            inputs = inputs[:, tf.newaxis, :, :, :]
            return tfk.backend.tile(inputs, (1,4,1,1,1))
        else:
            return inputs
# ENCODER
original_image = tfk.Input(shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3),
                   name='original_image')
likelihood = tfk.Input(shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3),
                   name='likelihood')
mus_input = tfk.Input(shape=(K,IMAGE_SIZE[0],IMAGE_SIZE[1],3),
                   name='mus')
mus_gradient_input = tfk.Input(shape=(K,IMAGE_SIZE[0],IMAGE_SIZE[1],3),
                   name='mus_gradient')
masks_input = tfk.Input(shape=(K,IMAGE_SIZE[0],IMAGE_SIZE[1],1),
                   name='masks')
masks_gradient_input = tfk.Input(shape=(K,IMAGE_SIZE[0],IMAGE_SIZE[1],1),
                   name='masks_gradient')
masks_logits_input = tfk.Input(shape=(K,IMAGE_SIZE[0],IMAGE_SIZE[1],1),
                   name='masks_logits')
x_coordinate_input = tfk.Input(shape=(K,IMAGE_SIZE[0],IMAGE_SIZE[1],1),
                   name='x_coordinate')
y_coordinate_input = tfk.Input(shape=(K,IMAGE_SIZE[0],IMAGE_SIZE[1],1),
                   name='y_coordinate')
# THE POSTERIOR AND ITS GRADIENT AS INPUTS FOR THE TOP OF THE MODEL
posterior_input = tfk.Input(shape=(K,latent_dim),
                   name='posterior')
posterior_gradient_input = tfk.Input(shape=(K,latent_dim),
                   name='posterior_gradient')
# CONCATENATE INPUTS
# FIRST EXPAND ORIGINAL IMAGE AND LIKELIHOOD
original_image_expanded = Expand(name='original_image_expanded')(original_image)
likelihood_expanded = Expand(name='likelihood_expanded')(likelihood)
encoder_inputs = tfkl.concatenate([original_image_expanded,
                                   likelihood_expanded,
                                   mus_input,
                                   mus_gradient_input,
                                   masks_input,
                                   masks_gradient_input,
                                   masks_logits_input,
                                   x_coordinate_input,
                                   y_coordinate_input])
encoder_output = tfkl.Conv2D(filters=32, kernel_size=kernel_size, 
                activation=my_exponential_linear_unit, name='encoder_conv_1')(encoder_inputs)
encoder_output = tfkl.Conv2D(filters=32, kernel_size=kernel_size, 
                activation=my_exponential_linear_unit, name='encoder_conv_2')(encoder_output)
encoder_output = tfkl.Conv2D(filters=32, kernel_size=kernel_size, 
                activation=my_exponential_linear_unit, name='encoder_conv_3')(encoder_output)
encoder_output_ave = tfkl.AveragePooling3D()(encoder_output)
encoder_output_flattened = tfkl.Flatten()(encoder_output)
encoder_output_dropout = tfkl.Dropout(0.5)(encoder_output_flattened)
encoder_output_slot_1, encoder_output_slot_2, \
encoder_output_slot_3, encoder_output_slot_4 = Split(name='split')(encoder_output_dropout)
encoder_output_slot_1 = tfkl.Dense(units=128, activation=my_exponential_linear_unit, 
                           name='encoder_fc_1_slot_1')(encoder_output_slot_1)
encoder_output_slot_2 = tfkl.Dense(units=128, activation=my_exponential_linear_unit, 
                           name='encoder_fc_1_slot_2')(encoder_output_slot_2)
encoder_output_slot_3 = tfkl.Dense(units=128, activation=my_exponential_linear_unit, 
                           name='encoder_fc_1_slot_3')(encoder_output_slot_3)
encoder_output_slot_4 = tfkl.Dense(units=128, activation=my_exponential_linear_unit, 
                           name='encoder_fc_1_slot_4')(encoder_output_slot_4)
# posterior_flattened = tfkl.Flatten()(posterior_input)
encoder_output_slot_1_cat = tfkl.concatenate([encoder_output_slot_1,
                                          posterior_input[:,0,:],
                                          posterior_gradient_input[:,0,:]])
encoder_output_slot_2_cat = tfkl.concatenate([encoder_output_slot_2,
                                          posterior_input[:,1,:],
                                          posterior_gradient_input[:,1,:]])
encoder_output_slot_3_cat = tfkl.concatenate([encoder_output_slot_3,
                                          posterior_input[:,2,:],
                                          posterior_gradient_input[:,2,:]])
encoder_output_slot_4_cat = tfkl.concatenate([encoder_output_slot_4,
                                          posterior_input[:,3,:],
                                          posterior_gradient_input[:,3,:]])
encoder_output_slot_1_final = tfkl.Dense(units=latent_dim + latent_dim, activation=tfk.activations.linear,
                              name='encoder_fc_2_slot_1')(encoder_output_slot_1)
encoder_output_slot_2_final = tfkl.Dense(units=latent_dim + latent_dim, activation=tfk.activations.linear,
                              name='encoder_fc_2_slot_2')(encoder_output_slot_2)
encoder_output_slot_3_final = tfkl.Dense(units=latent_dim + latent_dim, activation=tfk.activations.linear,
                              name='encoder_fc_2_slot_3')(encoder_output_slot_3)
encoder_output_slot_4_final = tfkl.Dense(units=latent_dim + latent_dim, activation=tfk.activations.linear,
                              name='encoder_fc_2_slot_4')(encoder_output_slot_4)
encoder_output = MyMerge()([encoder_output_slot_1_final, encoder_output_slot_2_final,
                            encoder_output_slot_3_final, encoder_output_slot_4_final])
z_mean, z_log_var, z = Sampling()(encoder_output)
refinement = tfk.Model(inputs=[original_image,
                               likelihood,
                               mus_input,
                               mus_gradient_input,
                               masks_input,
                               masks_gradient_input,
                               masks_logits_input,
                               x_coordinate_input,
                               y_coordinate_input,
                               posterior_input,
                               posterior_gradient_input],
                      outputs=[z_mean,
                               z_log_var,
                               z])

# %%
@tf.function
def log_normal_pdf(z, z_mean, z_log_var):
    return tf.reduce_sum(
        -0.5 * ((z - z_mean) ** 2. * tf.exp(-z_log_var) + z_log_var + tf.math.log(2.* math.pi)),
        axis=[1,2])

# DEFINE THE METRIC(S) WE WISH TO TRACK AND WRITE THE SUMMARY
loss_tracker = tfk.metrics.Mean(name='loss')

class VAE(tfk.Model):

    def __init__(self, generator, refinement, image_size, latent_dim, iterations, num_slots):
        super().__init__()
        self.generator = generator
        self.refinement = refinement
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.iterations = iterations
        self.num_slots = num_slots
        # NORMALIZE THROUGH ALL CHANNELS AND SPATIAL DIMS AND POSTERIOR GRADIENT
        self.layer_normalization_likelihood = tfkl.LayerNormalization(axis=[1,2,3])
        self.layer_normalization_posterior_gradient = tfkl.LayerNormalization(axis=-1)
        self.layer_normalization_mus_gradient = tfkl.LayerNormalization(axis=[2,3,4])
        self.layer_normalization_masks_gradient = tfkl.LayerNormalization(axis=-1)

    def compile(self, g_optimizer, ref_optimizer, loss_fn):
        super().compile()
        self.g_optimizer = g_optimizer
        self.ref_optimizer = ref_optimizer
        self.loss_fn = loss_fn
    
    
    @tf.function
    def train_step(self, data):

        original_image = tf.cast(data['image'], tf.float32)/255.0 + tfk.backend.epsilon()
        batch_size = tf.shape(original_image)[0]

        posterior_mean = tf.zeros(shape=(batch_size, self.num_slots, self.latent_dim))
        posterior_log_var = tf.zeros(shape=(batch_size, self.num_slots, self.latent_dim))
        _,_,posterior = Sampling()(tfkl.concatenate([posterior_mean,
                                                      posterior_log_var],
                                                      axis=-1))
        
        rgb_mask = self.generator(posterior)
        mus, masks_logits = tf.split(rgb_mask, num_or_size_splits=[3,1], axis=-1)
        masks = tf.nn.softmax(masks_logits, axis=1)
        masks_tiled = tfk.backend.tile(masks, (1,1,1,1,3))
        epsilon = tfk.backend.random_normal(shape=tf.shape(mus))
        likelihood = tf.reduce_sum(masks_tiled * (mus + tf.exp(0.5 * 0) * epsilon), axis=1)

        with tf.GradientTape(persistent=True) as tape:
            
            tape.watch([masks_logits,
                        masks,
                        mus,
                        likelihood,
                        posterior,
                        self.generator.trainable_variables])
            
            rgb_mask = self.generator(posterior)
            mus, masks_logits = tf.split(rgb_mask, num_or_size_splits=[3,1], axis=-1)
            masks = tf.nn.softmax(masks_logits, axis=1)
            masks_tiled = tfk.backend.tile(masks, (1,1,1,1,3))
            epsilon = tfk.backend.random_normal(shape=tf.shape(mus))
            likelihood = tf.reduce_sum(masks_tiled * (mus + tf.exp(0.5 * 0) * epsilon), axis=1)
            
            log_likelihood_loss =  -tf.reduce_sum(self.loss_fn(labels=original_image, logits=likelihood), axis=[1,2,3])
            log_true_prior_density_distribution_loss = log_normal_pdf(posterior, 0.0, 0.0)
            log_posterior_conditional_density_distribution_loss = log_normal_pdf(posterior,
                                                                             posterior_mean,
                                                                             posterior_log_var)
            # MINIMIZING THE NEGATIVE ELBO IS LIKE MAXIMIZING THE ELBO
            # WE USE MONTE CARLO ESTIMATION OF ELBO
            loss = (1.0/ tf.cast(batch_size, tf.float32)) * (1.0 / tf.cast(self.iterations, dtype=tf.float32)) *  \
                   -tf.reduce_mean(log_likelihood_loss + \
                                   log_true_prior_density_distribution_loss - \
                                   log_posterior_conditional_density_distribution_loss)
            
        generator_grads = tape.gradient(loss, self.generator.trainable_variables)
        generator_grads_clipped = [tf.clip_by_norm(gg, 5.0) for gg in generator_grads]
        self.g_optimizer.apply_gradients(zip(generator_grads_clipped, self.generator.trainable_variables))

        likelihood = self.layer_normalization_likelihood(likelihood)
        posterior_gradient = tape.gradient(loss, posterior)
        posterior_gradient_clipped_layer_normalized = self.layer_normalization_posterior_gradient(posterior_gradient)
        mus_gradient = tape.gradient(loss, mus)
        mus_gradient_layer_normalized = self.layer_normalization_mus_gradient(mus_gradient)
        masks_gradient = tape.gradient(loss, masks)
        masks_gradient_layer_normalized = self.layer_normalization_masks_gradient(masks_gradient)

        for i in tf.range(2, self.iterations + 1, 1):
            with tf.GradientTape(persistent=True) as tape:
                tape.watch([masks_logits,
                            masks,
                            mus,
                            likelihood,
                            posterior,
                            self.generator.trainable_variables,
                            self.refinement.trainable_variables])

                z_mean, z_log_var, z = self.refinement([original_image,
                                                        likelihood,
                                                        mus,
                                                        masks,
                                                        masks_logits,
                                                        x_coordinate,
                                                        y_coordinate,
                                                        posterior,
                                                        posterior_gradient_clipped_layer_normalized])

                posterior_mean += z_mean
                posterior_log_var += z_log_var
                _,_,posterior = Sampling()(tfkl.concatenate([posterior_mean,
                                                             posterior_log_var],
                                                             axis=-1))

                masks_logits, masks, mus, likelihood = self.generator(posterior)

                
                log_likelihood_loss =  -tf.reduce_sum(self.loss_fn(labels=original_image, logits=likelihood), axis=[1,2,3])
                log_true_prior_density_distribution_loss = log_normal_pdf(posterior, 0.0, 0.0)
                log_posterior_conditional_density_distribution_loss = log_normal_pdf(posterior,
                                                                                posterior_mean,
                                                                                posterior_log_var)
                # MINIMIZING THE NEGATIVE ELBO IS LIKE MAXIMIZING THE ELBO
                # WE USE MONTE CARLO ESTIMATION OF ELBO
                loss += (1.0 / tf.cast(batch_size, tf.float32)) * (tf.cast(i, dtype=tf.float32)/tf.cast(self.iterations, dtype=tf.float32)) *\
                       -tf.reduce_mean(log_likelihood_loss + \
                                       log_true_prior_density_distribution_loss - \
                                       log_posterior_conditional_density_distribution_loss)
                
            generator_grads = tape.gradient(loss, self.generator.trainable_variables)
            generator_grads_clipped = [tf.clip_by_norm(gg, 5.0) for gg in generator_grads]
            self.g_optimizer.apply_gradients(zip(generator_grads_clipped, self.generator.trainable_variables))
            refinement_grads = tape.gradient(loss, self.refinement.trainable_variables)
            refinement_grads_clipped = [tf.clip_by_norm(rg , 5.0) for rg in refinement_grads]
            self.ref_optimizer.apply_gradients(zip(refinement_grads_clipped, self.refinement.trainable_variables))
            
            likelihood = self.layer_normalization_likelihood(likelihood)
            posterior_gradient = tape.gradient(loss, posterior)
            posterior_gradient_clipped_layer_normalized = self.layer_normalization_posterior_gradient(posterior_gradient)
            mus_gradient = tape.gradient(loss, mus)
            mus_gradient_layer_normalized = self.layer_normalization_mus_gradient(mus_gradient)
            masks_gradient = tape.gradient(loss, masks)
            masks_gradient_layer_normalized = self.layer_normalization_masks_gradient(masks_gradient)
        
        # COMPUTE OUR OWN METRICS
        loss_tracker.update_state(loss)
        return {'loss': loss_tracker.result()}

    @property
    def metrics(self,):
        # TO RESET METRICS AT THE START OF EACH EPOCH
        return [loss_tracker,]

# %%

checkpoint_cb = tfk.callbacks.ModelCheckpoint(
    'IODINE_model_on_tetrominoes.h5', monitor='loss',
    mode='min', save_best_only=True, verbose=2
)

early_stopping_cb = tfk.callbacks.EarlyStopping(
    monitor='loss', min_delta=0.001, patience=50, 
    mode='auto', restore_best_weights=True
)


vae = VAE(generator, refinement, image_size=IMAGE_SIZE, latent_dim=latent_dim, iterations=T, num_slots=K)
vae.compile(g_optimizer=tfk.optimizers.Adam(learning_rate=0.0003),
            ref_optimizer=tfk.optimizers.Adam(learning_rate=0.0003),
            loss_fn=tf.nn.sigmoid_cross_entropy_with_logits)
# %%
# LOADING THE DATASET
# THIS IS A DATASET OF TETRIS-LIKE SHAPE (AKA TETROMINOES)
# EACH 35x35 IMAGE CONTAINS THREE TETROMINOES, SAMPLED FROM
# 17 UNIQUE SHAOES/ORIENTATIONS, EACH TETROMINO HAS ONE OF 
# SIX POSSIBLE COLORS (RED, GREEN, BLUE, YELLOW, MAGENTA, CYAN).
# PROVIDE X AND Y POSITION, SHAPE AND COLOR (INTEGER-CODED) AS 
# GROUND-TRUTH FEATURES. DATAPOINTS ALSO INCLUDE A VISIBILITY VECTOR.

COMPRESSION_TYPE = tf.io.TFRecordOptions.get_compression_type_string('GZIP')
IMAGE_SIZE = [35, 35]
# THE MAXIMUM NUMBER OF FOREGROUND AND BACKGROUND ENTITIES IN THE PROVIDED
# DATASET. THIS CORRESPONDS TO THE NUMBER OF SEGMENTATION MASKS RETURNED
# PER SCENE
MAX_NUM_ENTITIES = 4
BYTE_FEATURES = ['mask', 'image']

# CREATE A DICTIONARY MAPPING FEATURES NAMES TO TF.EXAMPLE COMPATIBLE
# SHAPE AND DATA TYPE DESCRIPTORS.
features = {
    'image': tf.io.FixedLenFeature(IMAGE_SIZE+[3], tf.string),
    'mask': tf.io.FixedLenFeature([MAX_NUM_ENTITIES]+IMAGE_SIZE+[1], tf.string),
    'x': tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    'y': tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    'shape': tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    'color': tf.io.FixedLenFeature([MAX_NUM_ENTITIES, 3], tf.float32),
    'visibility': tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
           }
def decode(example_proto):
    # PARS THE INPUT TF.EXAMPLE PROTO USING THE 
    # FEATURE DESCRIPTION DICT ABOVE
    single_example = tf.io.parse_single_example(example_proto, features)
    for k in BYTE_FEATURES:
        single_example[k] = tf.squeeze(tf.io.decode_raw(single_example[k], tf.uint8),
                                       axis=-1)
    return single_example

def dataset(tfrecords_path, read_buffer_size=None, map_parallel_calls=None):
    '''
        Read, decompress, and parse the TFRecords file.
        Args:
            tfrecords_path: str. Path to the dataset file.
            read_buffer_size: int. Number of bytes in the read buffer.
            map_parallel_calls: int. Number of elements decoded 
            asynchronously in parallel.
        Retruns:
            An unbatched tf.data.TFRecordDataset.
    '''
    raw_dataset = tf.data.TFRecordDataset(
        tfrecords_path, compression_type=COMPRESSION_TYPE,
        buffer_size=read_buffer_size)
    return raw_dataset.map(decode, num_parallel_calls=map_parallel_calls)


path = './data/tetrominoes_tetrominoes_train.tfrecords'

dataset = dataset(path)
batched_dataset = dataset.shuffle(1000).batch(batch_size)

for data in batched_dataset.take(1):
    print(data.keys())

# %%
vae.fit(batched_dataset,
        callbacks=[early_stopping_cb,
                   checkpoint_cb],
        epochs=10, 
        verbose=2)

vae.save_weights('tetrominoes.h5')
# %%
# from typing import Any
# from dataclasses import dataclass
# @dataclass
# class EvaluationResults:
#     __slots__ = ['masks_logits',
#                  'mask',
#                  'mus',
#                  'likelihood']

#     masks_logits: Any
#     mask: Any
#     mus: Any
#     likelihood: Any
# %%
# er = EvaluationResults()
# vae.evaluate(batched_dataset_take_10, er)
# %%
# from sklearn.metrics.cluster import adjusted_rand_score
# %%

# checkpoint = tf.train.Checkpoint(vae)
# checkpoint.restore('IODINE_model_on_tetrominoes.h5')
# %%
# generator.load_weights('tetrominoes.h5', by_name=True)
# %%
