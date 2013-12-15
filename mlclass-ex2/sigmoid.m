function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
num_rows = size(z,1);
num_cols = size(z,2);

for i = 1:num_rows
  for j = 1:num_cols
    g(i,j) = 1 /( 1 + e^(-z(i,j)));
  end
end

% =============================================================

end
