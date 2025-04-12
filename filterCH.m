function [CH, Sensors] = filterCH(Sensors, Model, r, probs)
    CH = [];
    countCHs = 0;
    n = Model.n;
    threshold = 0.7; 

    for i = 1:n
        if Sensors(i).E > 0
            if probs(i) >= threshold
                countCHs = countCHs + 1;
                CH(countCHs).id = i; 
                Sensors(i).type = 'C'; 
            end
        end
    end
end