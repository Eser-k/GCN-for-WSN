function X = createFeatureMatrix(Sensors, Model)
% createFeatureMatrix erstellt die Feature-Matrix für das GCN.
%
% INPUT:
%   Sensors - Array der Sensorstrukturen (Index 1 bis Model.n: Sensoren,
%             Index Model.n+1: Sink, wird hier nicht berücksichtigt)
%   Model   - Struktur mit Parametern, u.a.:
%             Model.n  : Anzahl der Sensoren (ohne Sink)
%             Model.RR : Funkreichweite, die als Schwellenwert für Nachbarn dient
%
% OUTPUT:
%   X       - Feature-Matrix der Größe [Model.n x 6] mit den Merkmalen:
%             [E, xd, yd, Knotengrad, dis2sink, avgNeighborDistance]

    n = Model.n;        % Anzahl Sensoren (ohne Sink)
    RR = Model.RR;      % Funkreichweite als Schwellenwert
    X = zeros(n, 6);    % Preallocation: n Sensoren, 6 Features

    for i = 1:n
        % Feature 1: Restenergie
        X(i,1) = Sensors(i).E;
        % Feature 2 & 3: x- und y-Koordinate
        X(i,2) = Sensors(i).xd;
        X(i,3) = Sensors(i).yd;
        % Feature 4: Distanz zur Basisstation 
        X(i,5) = Sensors(i).dis2sink;
        
        % Initialisierung für Knotengrad und durchschnittliche Nachbardistanz
        neighborCount = 0;
        totalNeighborDistance = 0;
        
        % Für alle anderen Sensoren prüfen, ob sie innerhalb der Funkreichweite liegen
        for j = 1:n
            if i ~= j
                d = sqrt((Sensors(i).xd - Sensors(j).xd)^2 + (Sensors(i).yd - Sensors(j).yd)^2);
                if d <= RR
                    neighborCount = neighborCount + 1;
                    totalNeighborDistance = totalNeighborDistance + d;
                end
            end
        end
        
        % Feature 5: Knotengrad (Anzahl der Nachbarn)
        X(i,4) = neighborCount;
        % Feature 6: Durchschnittliche Distanz zu den Nachbarn
        if neighborCount > 0
            X(i,6) = totalNeighborDistance / neighborCount;
        else
            X(i,6) = RR; % Falls keine Nachbarn vorhanden sind, setze als Platzhalter den Funkreichweitenwert
        end
    end
end