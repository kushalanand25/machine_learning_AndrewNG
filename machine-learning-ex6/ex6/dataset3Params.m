function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

c = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
u = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sz_c = size(c,1);
sz_u = size(u,1);

all_m = zeros(sz_c, sz_u);

for i = 1:sz_c
  for j = 1:sz_u
    model = svmTrain(X, y, c(i), @(x1, x2) gaussianKernel(x1, x2, u(j)));
    predictions = svmPredict(model, Xval);
    all_m(i,j) = mean(double(predictions ~= yval));
  endfor
endfor

[m,idx] = min(all_m(:));
[rw,cl] = ind2sub(size(all_m),idx);
C = c(rw);
sigma = u(cl);

% =========================================================================

end
