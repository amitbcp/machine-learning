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


h = sigmoid(X*theta);
% J = (1/m)*sum(-y .* log(h) - (1 - y) .* log(1-h));
size(theta);
shift_theta = theta(2:size(theta));
size(shift_theta);
theta_reg = [0;shift_theta];  
% filling 0 in the first row so that the vector equation 
%can be normalized and we don't have to 
%update it again and again separately
size(theta_reg);

J = (1/m)*(-y'* log(h) - (1 - y)'*log(1-h))+(lambda/(2*m))*theta_reg'*theta_reg;

% grad_zero = (1/m)*X(:,1)'*(h-y);
% grad_rest = (1/m)*(shift_x'*(h - y)+lambda*shift_theta);
% grad      = cat(1, grad_zero, grad_rest);

grad = (1/m)*(X'*(h-y)+(lambda)*theta_reg);


% =============================================================

end
