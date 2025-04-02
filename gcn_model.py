import tensorflow as tf
from tensorflow.keras import layers, Model, Input

#############################################
# 1) Hilfsfunktion zur Normalisierung von A
#############################################
def normalize_adjacency(A):
    """
    Normalisiert die Adjazenzmatrix A nach dem Ansatz von Kipf & Welling (GCN).
    A_tilde = A + I
    D_tilde(i,i) = Summe über Zeile i von A_tilde
    A_hat = D_tilde^(-1/2) * A_tilde * D_tilde^(-1/2)   (wobei D_tilde^(-1/2) = 1/sqrt(D_tilde))

    Parameter:
    ----------
    A : tf.Tensor
        Unnormierte Adjazenzmatrix der Form (N, N) oder (batch_size, N, N)

    Returns:
    --------
    A_hat : tf.Tensor
        Normalisierte Adjazenzmatrix (gleiches Format wie A)
    """
    # Self-Loops hinzufügen: I hat die gleiche Größe wie A
    I = tf.eye(A.shape[-1], batch_shape=A.shape[:-2])
    A_tilde = A + I

    # Berechne die Zeilensummen (Grad) für A_tilde
    row_sum = tf.reduce_sum(A_tilde, axis=-1)  # Erwartete Form: (batch_size, N) oder (N,)
    D_inv_sqrt = tf.linalg.diag(1.0 / tf.sqrt(row_sum))
    
    # Berechne A_hat = D^(-1/2) * A_tilde * D^(-1/2)
    A_hat = tf.matmul(tf.matmul(D_inv_sqrt, A_tilde), D_inv_sqrt)
    return A_hat

#############################################
# 2) Custom Layer: Graph Convolution
#############################################
class GraphConvolution(layers.Layer):
    """
    Führt eine Graph Convolution Operation durch:
       Z = A_hat * X * W + b
    wobei A_hat die normalisierte Adjazenzmatrix, X die Knoteneingaben und 
    W, b die trainierbaren Parameter sind.
    """
    def __init__(self, output_dim, activation=None, **kwargs):
        """
        Initialisiert die GraphConvolution-Schicht.

        Parameter:
        - output_dim: Integer, der die Dimension des Ausgabe-Feature-Vektors 
                      für jeden Knoten angibt.
        - activation: Optionale Aktivierungsfunktion (z. B. 'relu', 'sigmoid').
                      Falls None, wird keine Aktivierung angewendet.
        - **kwargs: Zusätzliche benannte Argumente, die an tf.keras.layers.Layer
                    weitergereicht werden.
        """
        super(GraphConvolution, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.activation = tf.keras.activations.get(activation)
    
    def build(self, input_shape):
        """
        Initialisiert die trainierbaren Parameter der GraphConvolution-Schicht.

        Parameter:
        - input_shape: Eine Liste, die die Formen der Eingaben [X, A] enthält.
          * X hat die Form (batch_size, N, F) (wobei N: Anzahl Knoten, F: Feature-Dimension)
          * A hat die Form (batch_size, N, N) (Adjazenzmatrix)
        """
        feature_dim = input_shape[0][-1]
        self.weight = self.add_weight(
            shape=(feature_dim, self.output_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='weight'
        )
        self.bias = self.add_weight(
            shape=(self.output_dim,),
            initializer='zeros',
            trainable=True,
            name='bias'
        )

    def call(self, inputs):
        """
        Führt den Vorwärtsdurchlauf der GraphConvolution-Schicht durch.

        Eingabe:
        - inputs: Eine Liste [X, A_hat], wobei:
            X:     Feature-Matrix der Knoten, Form (batch_size, N, F)
            A_hat: Normalisierte Adjazenzmatrix, Form (batch_size, N, N)
        Ablauf:
        1. Aggregation: Multipliziert A_hat mit X, um die Features der Nachbarn zu aggregieren.
        2. Transformation: Multipliziert das Ergebnis mit der Gewichtsmatrix (self.weight).
        3. Bias hinzufügen: Addiert den Bias (self.bias).
        4. Aktivierung: Wendet die Aktivierungsfunktion an (falls definiert).
        Rückgabe:
        - Z: Tensor der Form (batch_size, N, output_dim)
        """
        X, A_hat = inputs
        AX = tf.matmul(A_hat, X)  # (batch_size, N, F)
        AXW = tf.matmul(AX, self.weight)  # (batch_size, N, output_dim)
        Z = AXW + self.bias
        if self.activation is not None:
            Z = self.activation(Z)
        return Z

#############################################
# 3) GCN-Modell zusammenbauen
#############################################
def build_gcn_model(num_nodes, feature_dim, hidden_units=16):
    """
    Erstellt ein GCN-Modell mit zwei GCN-Schichten und einem Fully Connected Block,
    der aus zwei Dense-Schichten besteht, und einer abschließenden Output-Schicht zur 
    Sigmoid-Ausgabe.

    Parameter:
    ----------
    num_nodes : int
        Anzahl der Knoten im Netzwerk (z. B. 100).
    feature_dim : int
        Anzahl der Merkmale pro Knoten.
    hidden_units : int
        Dimension des Ausgabevektors jeder GCN-Schicht (Hyperparameter).

    Returns:
    --------
    model : tf.keras.Model
        Ein Keras-Modell, das pro Knoten eine Sigmoid-Wahrscheinlichkeit ausgibt.
    """
    # Eingaben definieren: X_in hat Form (batch_size, N, F) und A_in hat Form (batch_size, N, N)
    X_in = Input(shape=(num_nodes, feature_dim), name='X_in')
    A_in = Input(shape=(num_nodes, num_nodes), name='A_in')
    
    # Normalisiere die Adjazenzmatrix A_in.
    A_hat = layers.Lambda(
        lambda A: normalize_adjacency(A),
        output_shape=lambda input_shape: input_shape,
        name='Normalize'
    )(A_in)
    
    # Erste GCN-Schicht mit ReLU-Aktivierung.
    gc1 = GraphConvolution(hidden_units, activation='relu')([X_in, A_hat])
    
    # Zweite GCN-Schicht.
    gc2 = GraphConvolution(hidden_units)([gc1, A_hat])
    
    # Angenommen gc2 hat Form (batch_size, num_nodes, 16).

    # 1) Erste Dense-Schicht (16 Neuronen)
    fc1 = layers.TimeDistributed(
    layers.Dense(16, activation='relu'),
    name='FullyConnected_1'
    )(gc2)  # Ergebnis: (batch_size, num_nodes, 16)

    # 2) Zweite Dense-Schicht (16 Neuronen)
    fc2 = layers.TimeDistributed(
    layers.Dense(16, activation='relu'),
    name='FullyConnected_2'
    )(fc1)  # Ergebnis: (batch_size, num_nodes, 16)

    # 3) Letzte Dense-Schicht (1 Neuron, Sigmoid)
    fc3 = layers.TimeDistributed(
    layers.Dense(1, activation='sigmoid'),
    name='FullyConnected_3'
    )(fc2)  # Ergebnis: (batch_size, num_nodes, 1)

    # Reshape auf (batch_size, num_nodes)
    outputs = layers.Reshape((num_nodes,))(fc3)
    
    # Erstelle das Modell.
    model = Model(inputs=[X_in, A_in], outputs=outputs, name='GCN_ClusterHead')
    return model

#############################################
# 4) Beispiel für die Verwendung
#############################################
if __name__ == '__main__':
    # Beispiel: 100 Knoten, 6 Features
    num_nodes = 100
    feature_dim = 6
    model = build_gcn_model(num_nodes, feature_dim, hidden_units=16)
    
    # Modellübersicht anzeigen
    model.summary()
    
    # Dummy-Daten erstellen mit batch_size = 1 (also ein einzelner Graph)
    X_dummy = tf.random.normal(shape=(1, num_nodes, feature_dim))
    A_dummy = tf.random.uniform(shape=(1, num_nodes, num_nodes),
                                minval=0, maxval=2, dtype=tf.int32)
    A_dummy = tf.cast(A_dummy, tf.float32)
    
    # Vorhersage abrufen
    y_pred = model([X_dummy, A_dummy])
    print("Shape der Ausgabe:", y_pred.shape)  # Erwartet: (1, 100)
    print("Ausgabe-Werte:", y_pred)