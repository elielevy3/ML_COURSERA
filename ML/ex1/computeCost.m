function J = computeCost(X, Y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(Y); % number of training examples

% Let's calculate a vector of ( h(x⁰), ... , h(x^m) )
H = X*theta;
% error for every elements of the dataset
Error = (H - Y).^2;
% Cost function value
J = sum(Error) / (2*m);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.





% =========================================================================

end
