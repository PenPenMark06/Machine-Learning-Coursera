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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Part 1 satrts here

% As hint states, we need to convert y into binary representation
binary_y = zeros(m, num_labels);

for i = 1:m,
    if(y(i) == 0)
        binary_y(i, 10) = 1;
    else
        binary_y(i, y(i)) = 1;    
    endif
end

% First I need to add hidden unit to input X as instructed
X = [ones(m, 1) X];
z2 = Theta1 * X';
% At this point, a2 has 25 x 5000
a2 = sigmoid(z2);

% Again, add hidden unit manually
a2 = [ones(m, 1) a2']; % 26 x 5000
z3 = Theta2 * a2'; 

% a3 has 10 x 5000
a3 = sigmoid(z3);

inside = sum(binary_y' .* log(a3) + (1 - binary_y)' .* log(1 - a3), 2);
J = -1 * (1/m) * sum(inside);

% Now regularize J by computing complicated theta part
constant = (lambda/(2*m));

num_cols = size(Theta1, 2);
num_cols2 = size(Theta2, 2);

first_theta = sum(Theta1(:, 2:num_cols).^2, 2);
second_theta = sum(Theta2(:, 2:num_cols2).^2, 2);

regular_theta = constant * sum(first_theta + second_theta);

J = J + regular_theta;

% Part2 start here. Backpropagation

Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));

for x = 1:m,
    current_x = X(x, :);
    current_x = current_x'; % current_x is a vector now with 400 x 1

    z2 = Theta1 * current_x;
    a2 = sigmoid(z2);

    a2 = [1 ; a2];

    z3 = Theta2 * a2;
    a3 = sigmoid(z3);

    % Compute deltas for Layer 3 and Layer 2
    delta3 = a3 - binary_y(x, :)';
    delta2 = (Theta2' *  delta3) .* [1; sigmoidGradient(z2)];
    % Note that there's no delta1

    Delta1 += delta2(2:end) * current_x';
    Delta2 += delta3 * a2';

end

Theta1_grad = (1/m) .* Delta1;
Theta2_grad = (1/m) .* Delta2;


% Part3 starts here

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda/m) * Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda/m) * Theta2(:, 2:end);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
