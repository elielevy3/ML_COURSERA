function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
% sig => m*1
sig = arrayfun(@(x) 1 / (1+exp(-x)), X*theta);


% non regularisée
J = (1/m)*sum(-y.*arrayfun(@(x) log(x), sig) - (1.-y).*arrayfun(@(x) log(1-x), sig));

% on ne rajoute pas le faceur regularisateur pour theta1
theta(1) = 0;

% cost
J = (1/m)*sum(-y.*arrayfun(@(x) log(x), sig) - (1.-y).*arrayfun(@(x) log(1-x), sig)) + (lambda/(2*m))*sum(theta.^2);

% grad
grad = (1/m)*(sig - y)'*X + ((lambda/m)*theta)';

end
