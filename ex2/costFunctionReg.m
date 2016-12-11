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


A = 0;
B = 0;
C = 0;
D = 0;
for i = 1:m 
    A = A - y(i,1)*log(sigmoid(X(i,:)*theta));
    B = B - (1-y(i,1))*log(1-sigmoid(X(i,:)*theta));
end

for j = 2:size(theta, 1)  
    C = C + theta(j)^2;
end

D = (lambda*C)/2;

J = ((A+B+D)/m);

C = 0;
for i = 1:m
    C = C + (sigmoid(X(i,:)*theta)-y(i,1))*X(i,1);
end
    
grad(1) = C/m ;

%Grad
for j = 2:size(theta, 1) 
    C = 0;
    for i = 1:m
        C = C + (sigmoid(X(i,:)*theta)-y(i,1))*X(i,j);
    end
    
    grad(j) = (C + lambda*theta(j))/m ;
end



% =============================================================

end
