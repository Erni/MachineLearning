function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
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
recY = zeros(m,num_labels);
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% first recode the labels of y as vectors
for i = 1:m
  recY(i,y(i)) = 1;
end

cost = 0;

for i = 1:m
  for k = 1:num_labels
    a1 = X(i,:);
    a1 = [1 a1];
    z2 = Theta1 * a1';
    a2 = sigmoid(z2);
    a2 = [1 ; a2];
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);
    Hx = a3;
    
    cost = cost + ((-recY(i,k) * log(Hx(k))) - (1 - recY(i,k)) * log(1 - Hx(k)));
  end
end

J = (1/m) * cost;

% Regularized cost function
costreg = (sum(sum(Theta1(:,2:end) .* Theta1(:,2:end))) + sum(sum(Theta2(:,2:end) .* Theta2(:,2:end))));

costreg = (lambda / (2 * m)) * costreg;

J = J + costreg;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
a1 = X; % 5000x400
a1 = [ones(m,1) a1]; % 5000x401
z2 = (Theta1 * a1')'; % 5000x25
a2 = sigmoid(z2); % 5000x25
a2 = [ones(m,1) a2]; % 5000x26
z3 = Theta2 * a2'; % 10x5000
a3 = sigmoid(z3)'; % 5000x10
d3 = a3 - recY; % 5000x10

r2 = d3 * Theta2(:,2:end); % 5000x25
d2 = r2 .* sigmoidGradient(z2); % 5000x25

t1 = Theta1 * lambda;
t1(:,1) = 0;
Theta1_grad = (d2' * [ones(m,1) X] + t1) / m;

t2 = Theta2 * lambda;
t2(:,1) = 0;
Theta2_grad = (d3' * a2 + t2) / m;

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
