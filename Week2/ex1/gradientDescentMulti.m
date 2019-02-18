function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    prediction = X * theta; % This is the h(x) for all of them
    error = (prediction - y); % Get the h(x) - y part
    
    % For each colum (feature), I'll times x's to error
    % And get the sum. Store it in temp
    for i = 1:size(X, 2)
        temp = (1/m) * sum(error.* X(:, i));
        if i == 1
            sigma = [temp]; % If temp was the first item, create matric sigma
        else    
            sigma = [sigma; temp]; % Else, append them to next row
        end    
    end

    % Update theta as defined in the lecture videos
    theta = theta - (alpha * sigma);




    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
