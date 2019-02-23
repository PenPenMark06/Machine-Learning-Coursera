function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

% Since it is similar to normal costFunction, get left_half of the formula, and gradients
[left_half, grad] = costFunction(theta, X, y);

% We only want to penalize theta2 and above, so get the size of theta
n = size(theta);
% From what we got by cost function, add the missing parts from formula
J = left_half + (lambda/(2 *m)) * sum(theta(2:n, 1).^2);

% Similarly for gradient, add missing parts
for i = 2:size(theta, 1)
    if (i == 1)
        grad(i) = grad(i);
    else
        grad(i) = grad(i) + (lambda/m) * theta(i)
    end   
end

% =============================================================

end