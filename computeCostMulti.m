function J = computeCostMulti(X, y, theta)

m = length(y);

J = 0;


J = (1/(2*m))*sum(power((X*theta - y),2));



end
