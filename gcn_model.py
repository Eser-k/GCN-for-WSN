import tensorflow as tf
from tensorflow.keras import layers, Model, Input

@tf.keras.utils.register_keras_serializable()
def spread_out_fn(logits):
    epsilon = 1e-8
    mean = tf.reduce_mean(logits, axis=[1, 2], keepdims=True)
    std = tf.math.reduce_std(logits, axis=[1, 2], keepdims=True) + epsilon
    alpha = 1.2  
    return (logits - mean) / std * alpha

@tf.keras.utils.register_keras_serializable()
def normalize_adjacency(A):
    N = A.shape[-1]
    I = tf.eye(N, batch_shape=A.shape[:-2])
    A_tilde = A + I
    row_sum = tf.reduce_sum(A_tilde, axis=-1)
    D_inv_sqrt = tf.linalg.diag(1.0 / tf.sqrt(row_sum))
    A_hat = tf.matmul(tf.matmul(D_inv_sqrt, A_tilde), D_inv_sqrt)
    return A_hat

@tf.keras.utils.register_keras_serializable()
class GraphConvolution(layers.Layer):
    def __init__(self, output_dim, activation=None, kernel_initializer=None, bias_initializer=None, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.activation = activation
        self.kernel_initializer = kernel_initializer if kernel_initializer is not None else tf.keras.initializers.HeNormal()
        self.bias_initializer = bias_initializer if bias_initializer is not None else tf.keras.initializers.HeNormal()
    
    def build(self, input_shape):
        feature_dim = input_shape[0][-1]
        self.weight = self.add_weight(
            shape=(feature_dim, self.output_dim),
            initializer=self.kernel_initializer,
            trainable=True,
            name='weight'
        )
        self.bias = self.add_weight(
            shape=(self.output_dim,),
            initializer=self.bias_initializer,
            trainable=True,
            name='bias'
        )
        super(GraphConvolution, self).build(input_shape)
    
    def call(self, inputs):
        X, A_hat = inputs
        AX = tf.matmul(A_hat, X)
        AXW = tf.matmul(AX, self.weight)
        Z = AXW + self.bias
        if self.activation is not None:
            Z = tf.keras.activations.get(self.activation)(Z)
        return Z

    def get_config(self):
        config = super(GraphConvolution, self).get_config()
        config.update({
            "output_dim": self.output_dim,
            "activation": tf.keras.activations.serialize(tf.keras.activations.get(self.activation)),
            "kernel_initializer": tf.keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": tf.keras.initializers.serialize(self.bias_initializer)
        })
        return config

def build_gcn_model(num_nodes, feature_dim, hidden_units=16):
    X_in = Input(shape=(num_nodes, feature_dim), name='X_in')
    A_in = Input(shape=(num_nodes, num_nodes), name='A_in')
    
    A_hat = layers.Lambda(normalize_adjacency,
                          output_shape=lambda input_shape: (input_shape[0], input_shape[1], input_shape[2]),
                          name='Normalize')(A_in)
    
    gc1 = GraphConvolution(hidden_units, activation=tf.tanh, name='GCN_1')([X_in, A_hat])
    gc2 = GraphConvolution(hidden_units, activation=tf.tanh, name='GCN_2')([gc1, A_hat])
    gc3 = GraphConvolution(1, activation=None, name='GCN_3')([gc2, A_hat])
    
    scaled = layers.Lambda(spread_out_fn, name='SpreadOut')(gc3)
    sigmoid_out = layers.Activation('sigmoid', name='SigmoidOut')(scaled)
    output_reshaped = layers.Reshape((num_nodes,), name='Reshape')(sigmoid_out)
    
    model = Model(inputs=[X_in, A_in], outputs=output_reshaped, name='GCN_ClusterHead')
    return model

def compute_reward(consumed_energy, repeated_penalty, proximity_penalty, ch_ratio_penalty):
    
    alpha = 1.0  # Gewichtung für den Energieverbrauch 
    beta  = 1.5  # Gewichtung für Wiederholungsstrafe
    gamma = 1.5  # Gewichtung für Proximitätsstrafe
    delta = 1.5  # Gewichtung für den CH-Anteil

    reward_energy = -alpha * consumed_energy
    total_reward = reward_energy - beta * repeated_penalty - gamma * proximity_penalty - delta * ch_ratio_penalty
    return total_reward


def rl_train_step(model, X, A, consumed_energy, optimizer,
                  repeated_penalty, proximity_penalty, ch_ratio_penalty):
    # Gesamtreward aus allen Komponenten berechnen
    total_reward = compute_reward(consumed_energy, repeated_penalty, proximity_penalty, ch_ratio_penalty)
    
    with tf.GradientTape() as tape:
        probs = model([X, A])
        actions = tf.cast(probs >= 0.7, tf.float32)
        log_probs = actions * tf.math.log(probs + 1e-8) + (1 - actions) * tf.math.log(1 - probs + 1e-8)
        loss = -tf.reduce_mean(log_probs) * total_reward
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Beispiel zum Testen des Modells
def test_gcn_architecture():
    num_nodes = 100
    feature_dim = 6
    hidden_units = 256

    model = build_gcn_model(num_nodes, feature_dim, hidden_units)
    model.summary()
    
    X_dummy = tf.random.uniform(shape=(1, num_nodes, feature_dim), minval=0, maxval=100, dtype=tf.float32)
    A_dummy = tf.cast(tf.random.uniform(shape=(1, num_nodes, num_nodes), minval=0, maxval=1) < 0.4, tf.float32)
    
    outputs = model([X_dummy, A_dummy])
    print("Final Output values:\n", outputs.numpy())

if __name__ == '__main__':
    test_gcn_architecture()