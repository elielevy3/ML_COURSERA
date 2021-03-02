function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

theta_1_temp = 0;
theta_2_temp = 0;


for iter = 1:num_iters
    
    theta_1_temp = theta(1) - alpha*(1/m)*sum((X*theta - y));
    theta_2_temp = theta(2) - alpha*(1/m)*sum((X*theta - y)'*X(:, 2));
    
    theta(1) = theta_1_temp;
    theta(2) = theta_2_temp;
    
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    
end

% J_history
% theta

end
