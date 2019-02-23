function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% Get the matrix that has h(x)'s. Same as ex1
hypothesis = sigmoid(X * theta);

% Get J just as formula given during lecture
J = (1/m) * sum((-1 * y).* log(hypothesis) - (1 - y).* log(1 - hypothesis));

% Assign grad as formula given by lecture as well
for i = 1:size(X, 2)
    % Last part is .* so that each row of hypothesis is multiplied to corresponding x(i)
    grad(i) = (1/m) * sum((hypothesis - y) .* X(:, i)); 
                                                        
end






% =============================================================

end
