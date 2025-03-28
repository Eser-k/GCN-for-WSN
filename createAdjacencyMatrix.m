function A = createAdjacencyMatrix(Sensors, Model)
% createAdjacencyMatrix erstellt die Adjazenzmatrix für das Sensornetzwerk.
%
% INPUT:
%   Sensors - Array der Sensorstrukturen (Index 1 bis Model.n: normale Sensoren)
%   Model   - Struktur mit Parametern, u.a.:
%             Model.n  : Anzahl der Sensoren (ohne Sink)
%             Model.RR : Funkreichweite, die als Schwellenwert für Nachbarschaft dient
%
% OUTPUT:
%   A       - Adjazenzmatrix der Größe [Model.n x Model.n] 
%             (A(i,j)=1, wenn Sensor i und Sensor j verbunden sind, sonst 0)

    n = Model.n;          % Anzahl der Sensoren (ohne Sink)
    RR = Model.RR;        % Funkreichweite als Nachbarschaftsschwelle
    A = zeros(n, n);      % Preallocation der Adjazenzmatrix

    % Für jeden Sensor-Paar (i,j) die euklidische Distanz berechnen
    for i = 1:n
        for j = 1:n
            if i == j
                % Selbstverbindungen (Self-Loops) können hier 0 gelassen werden.
                A(i,j) = 0;
            else
                distance = sqrt((Sensors(i).xd - Sensors(j).xd)^2 + ...
                                (Sensors(i).yd - Sensors(j).yd)^2);
                if distance <= RR
                    A(i,j) = 1;
                else
                    A(i,j) = 0;
                end
            end
        end
    end
end