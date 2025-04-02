function y_pred = runGCNModel(X, A)
% runGCNModel ruft das TensorFlow GCN-Modell in Python auf und liefert 
% die Vorhersagen für jeden Knoten zurück.
%
% Eingaben:
%   X - Feature-Matrix der Knoten, Größe [num_nodes x feature_dim]
%   A - Adjazenzmatrix des Graphen, Größe [num_nodes x num_nodes]
%
% Ausgabe:
%   y_pred - Vorhersagevektor der Größe [1 x num_nodes] (Wahrscheinlichkeiten, z.B. ob ein
%            Knoten als Clusterhead gewählt wird)
%
% Voraussetzung: Das Python-Modul "gcn_model.py" mit der Funktion 
% build_gcn_model() befindet sich im Python-Pfad und TensorFlow ist in der 
% verwendeten Python-Umgebung installiert.
%
% Beispiel:
%   X = randn(100, 6);
%   A = randi([0,1], 100, 100);
%   y_pred = runGCNModel(X, A);

    % Überprüfen, ob die Python-Umgebung geladen ist.
    pe = pyenv;
    if ~strcmp(pe.Status, 'Loaded')
        error('Python environment is not loaded. Bitte konfiguriere pyenv.');
    end

    % Importiere das Python-Modul "gcn_model"
    gcn_module = py.importlib.import_module('gcn_model');
    
    % Bestimme die Anzahl der Knoten und die Feature-Dimension
    [num_nodes, feature_dim] = size(X);
    
    % Setze den Hyperparameter hidden_units (hier 16)
    hidden_units = int32(16);
    
    % Baue das GCN-Modell in Python: Es wird ein Modell für num_nodes, feature_dim
    % und hidden_units erstellt. Die Funktion erwartet integer Werte.
    model = gcn_module.build_gcn_model(int32(num_nodes), int32(feature_dim), hidden_units);
    
    % Füge eine Batch-Dimension hinzu (batch_size = 1)
    X_batch = reshape(X, [1, num_nodes, feature_dim]);
    A_batch = reshape(A, [1, num_nodes, num_nodes]);
    
    % Konvertiere MATLAB-Arrays in numpy-Arrays
    X_py = py.numpy.array(X_batch);
    % A wird als float32 benötigt – konvertiere deshalb zu single
    A_py = py.numpy.array(single(A_batch));
    
    % Erstelle eine Python-Liste der Eingaben
    inputs = py.list({X_py, A_py});
    
    % Rufe das Modell auf und erhalte die Vorhersage
    y_py = model(inputs);
    
    % Konvertiere die Ausgabe (ein numpy-Array) in einen MATLAB double-Vektor.
    % Diese Methode flacht den Tensor (der die Form (1, num_nodes) hat) zu einem Vektor.
    y_pred_flat = double(py.array.array('d', py.numpy.nditer(y_py)));
    
    % Reshape zurück auf (1, num_nodes)
    y_pred = reshape(y_pred_flat, [1, num_nodes]);
    
end