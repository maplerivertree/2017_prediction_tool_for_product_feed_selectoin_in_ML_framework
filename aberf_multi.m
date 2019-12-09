

%% ================ Part 1: Feature Normalization ================

%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = load('abrfdata2.txt');
X = data(:, 1:5);
y = data(:, 6);
m = length(y);

% Print out some data points
fprintf('Let us load the First 10 rows from \n');
fprintf('the dummy data for AbeBooks Rare Feed: \n');
fprintf(' x = [%.0f %.0f %.0f %.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

fprintf('Pausing, Press Enter to Continue.\n');
pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];


%% ================ Part 2: Gradient Descent ================

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.01;
num_iters = 800;

% Init Theta and Run Gradient Descent 
theta = zeros(6, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J - AbeBooks Rare Feed');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Predict the GMS of a listing_id with 1) no association, 2) 3 star seller ratings, 3) 1 wants match, 4) 10 bdp views, and 5) 1 basket adds.
d = [0 3 1 10 1];
d = (d - mu) ./ sigma;
d = [ones(1, 1) d];
price = d * theta; 




% ============================================================
fprintf('Predict the GMS of a listing_id with \n');
fprintf('1) no association, 2) 3 star seller ratings,\n');
fprintf(' 3) 1 wants match, 4) 10 bdp views, and 5) 1 basket adds.\n');
fprintf(['...(using Gradient Descent):\n $%f\n'], price);

fprintf('Program paused. Press enter to continue.\n');
pause;





%% ================ Part 3: Normal Equations ================
fprintf('................................\n');
fprintf('Solving with normal equations...\n');


%% Load Data
data = load('abrfdata2.txt');
X = data(:, 1:5);
y = data(:, 6);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');


% Predict the GMS of a listing_id with 1) no association, 2) 3 star seller ratings, 3) 1 wants match, 4) 10 bdp views, and 5) 1 basket adds.

d = [1 0 3 1 10 1];
price = d * theta; 


% ============================================================
fprintf('Predict the GMS of a listing_id with \n');
fprintf('1) no association, 2) 3 star seller ratings,\n');
fprintf(' 3) 1 wants match, 4) 10 bdp views, and 5) 1 basket adds.\n');
fprintf(['...(using Normal Equation):\n $%f\n'], price);


