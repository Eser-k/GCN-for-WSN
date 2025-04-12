function d = distances(X)
    [n, ~] = size(X);
    numPairs = n*(n-1)/2;
    d = zeros(numPairs, 1);
    idx = 1;
    for i = 1:n-1
        for j = i+1:n
            diff = X(i,:) - X(j,:);
            d(idx) = sqrt(sum(diff.^2));
            idx = idx + 1;
        end
    end
end