function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% sig => m*1
sig = arrayfun(@(x) 1 / (1+exp(-x)), X*theta);
% J => 1*1
% J = (1/(2*m)) * sum((sig - y).^2);

J = (1/m)*sum(-y.*arrayfun(@(x) log(x), sig) -  (1.-y).*arrayfun(@(x) log(1-x), sig));
grad = (1/m)*(sig - y)'*X;


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%








% =============================================================

end
