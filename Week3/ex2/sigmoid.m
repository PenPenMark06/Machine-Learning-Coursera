function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

% Key here is that operations other than (+/-) that applies to both Matrix and sclar, we need it to be 
% Applied to all elements, so we denote by .

denom = 1 + e.^(z * -1); % For each element from Z, powered by e (. denotes e^ to all elements)
g = 1./denom; % 1./ denotes we divide 1 by each element from z which was modified by line above



% =============================================================

end
