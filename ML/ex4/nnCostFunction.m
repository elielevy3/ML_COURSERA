function [J ,grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)


%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% we add the bias
X = [ones(m, 1), X];
Z2 = X*Theta1';

A2 = arrayfun(@(x) sigmoid(x), Z2);
A2 = [ones(size(X, 1), 1), A2];

Z3 = A2*Theta2';
A3 = arrayfun(@(x) sigmoid(x), Z3);

% we turn y to a m;num_label matrix where new_y(i, j) = 1 means that for
% the ith example belongs to the jth class among K
new_y = zeros(m, num_labels);
for i=1:m
    new_y(i, y(i)) = 1;
end
logA3 = arrayfun(@(x) log(x), A3);
log1MinusA3 = arrayfun(@(x) log(1-x), A3);



for i = 1:m
    
    % we initialize A1 as the ith row of input we are working on
    a1 = X(i, :);
     
   % Now we need to calculate every a^l for l=2,...,L
    
    z2 = a1*Theta1';
    a2 = arrayfun(@(x) sigmoid(x), z2);
    a2 = [1, a2];
    
    z3 = a2*Theta2';
    a3 = arrayfun(@(x) sigmoid(x), z3);

    % Now we compute the cost error for every node in every layer 
    d3 = a3 - new_y(i, :);
    d2 = (((Theta2'*d3')').*a2).*(ones(1, 5) - a2);
    %d2 = test.*a2.*test2;
    %d2= (Theta2'*d3').*a2.*((ones(1, 5) - a2));
   
    % we remove the first element which is linked to the bias term
    d2(1) = [];
    Theta1_grad = Theta1_grad + d2'*(a1);
    Theta2_grad = Theta2_grad + d3'*a2;
    
end


Theta1(:, 1) = zeros(size(Theta1, 1), 1);
Theta2(:, 1) = zeros(size(Theta2, 1), 1);

J = (-1/m)*( sum(new_y.*logA3, "all") + sum((ones(m, 4)-new_y).*log1MinusA3, "all") );
J = J + (lambda/(2*m))*(sum(Theta1.^2, "all") + sum(Theta2.^2, "all"));


Theta1_grad = (Theta1_grad + lambda*Theta1)*(1/m);
Theta2_grad = (Theta2_grad + lambda*Theta2)*(1/m);


%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
