function y_pred = runGCNModel(X, A, model)    
    
    [num_nodes, feature_dim] = size(X);
    
    X_batch = reshape(X, [1, num_nodes, feature_dim]);
    
    % Konvertiere MATLAB-Arrays in numpy-Arrays
    X_py = py.numpy.array(X_batch);
    
    % Erstelle eine Python-Liste der Eingaben
    inputs = py.list({X_py, A});
    
    % Rufe das Modell auf und erhalte die Vorhersage
    y_py = model(inputs);
    
    % Konvertiere die Ausgabe (ein numpy-Array) in einen MATLAB double-Vektor.
    % Diese Methode flacht den Tensor (der die Form (1, num_nodes) hat) zu einem Vektor.
    y_pred_flat = double(py.array.array('d', py.numpy.nditer(y_py)));
    
    % Reshape zurück auf (1, num_nodes)
    y_pred = reshape(y_pred_flat, [1, num_nodes]);
    
end