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


a_1 = [ones(m, 1) X]; % 5000 by 401
z_2 = a_1 * Theta1'; % 5000 by 401 * 401 by 25
a_2 = sigmoid(z_2); % 5000 by 25

a_2 = [ones(m,1) a_2]; %5000 by 26

z_3 = a_2 * Theta2'; % 5000 by 26 * 26 by 10
a_3 = sigmoid(z_3); % 5000 by 10
h = a_3;
h=h';

% yk = zeros(num_labels,m);
% for i = 1:m
%         if( y(i) == 10 ) 
%             yk(i,10) = 1;
%         end
%         
%     for j = 1:9
%         if( y(i) == j)
%             yk(i,10-j) = 1;
%         end
%     end
% end

yk = zeros(num_labels,m);
for i = 1:m
    yk(y(i),i) = 1;
end

cost = sum( (-yk) .* log(h) - (1 - yk) .* log(1 - h) ) ;

J = (1/m)*sum(cost); %sum(cost)까지는 5000에 근사한 값 가지는 것 아닌가..?


%Jreg = lambda / (2 * m) * (sum(sum(Theta1(:, 2:end) .^2)) + sum(sum(Theta2(:, 2:end) .^2)));
% -------------------------------------------------------------
%Theta1_t = sum(Theta1(:,2:end).^2);
Left = sum(sum(Theta1(:,2:end).^2)); % 좌측 끝열 Bias Term 제거

%Theta2_t = sum(Theta2(:,2:end).^2);
Right = sum(sum(Theta2(:,2:end).^2)); % 좌측 끝열 Bias Term 제거

Reg_Term = (lambda/(2*m)) * (Left + Right);

J = J + Reg_Term;

% =========================================================================
%grad = (1/m).*(X'*(h-y)) ;

delta3 = a_3 - yk';% 5000 by 10  여기가 왜 yk일까.....
delta2 = ( delta3 * Theta2(:, 2:end) ) .* sigmoidGradient(z_2);  % 5000 by 10 * 10 by 25  =  5000 by 25

D1 = delta2' * a_1;  % 25 by 5000 * 5000 by 401 (same dimensions as Theta1)
D2 = delta3' * a_2;  % 10 by 5000 * 5000 by 26

Theta1_grad = (1/m)*D1 ; %Backprop 과정을 사용하는 것이 SGD 25 by 401
Theta2_grad = (1/m)*D2 ; %Cost Fcn의 Theta에 대한 GD를 구함. 10 by 26

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m)*Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m)*Theta2(:,2:end);

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
