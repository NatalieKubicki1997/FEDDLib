clc;   % Clear the command window
clear; % Clear all variables
close all; % Close all figures

% As simulation was done in dimensionless units we do not multiply with
% dimensionless viscosity function

% Parameters for viscosity model
eta0   = 230.63; % Initial viscosity
etaInf = 3.37;   % Infinite shear viscosity
lambda = -300;   % 
a = 2;           % Parameter for the model
n = 0.45;        % Shear-thinning index

% Define the viscosity function (eta) as a function of shear rate (gamma)
eta = @(gamma) etaInf + (eta0 - etaInf) * (1 + (lambda * gamma).^a).^((n - 1) / a);

% Define the ranges for shear rate (gamma), angular frequency (w), and volume fraction (phi)
gamma = logspace(-2, 3, 200); % Shear rate from 10^(-2) to 10^3
w = logspace(0.55, 3.36,200); % (logarithmic scale)
phi = linspace(5, 85, 200);   % Volume fraction (linear scale)

% Prepend 0 to the volume fraction and first w to maintain dimensions
phi = [0 phi];  % Add 0 to the beginning of phi
w   = [w(1) w]; % Add the first element of w to the beginning

% Plot the raw data points
plot(phi, w,'*')
hold on

% Interpolate the data to create a smoother curve
interp_points = 100; % Number of points for interpolation
ydata_interp = linspace(min(phi), max(phi), interp_points); % Interpolation range for phi
ydata_v_interp = interp1(phi, w, ydata_interp, 'linear');   % Linear interpolation for w

% Plot the interpolated data points
plot(ydata_interp , ydata_v_interp, '*' );
hold on;

% Fit the data to an exponential model using Levenberg-Marquardt algorithm
[exp_lm, gof_lm] = fit(phi', w', "exp2", Algorithm="Levenberg-Marquardt");

% Plot the fitted curve on the original data
plot(exp_lm, phi, w)
legend(["data", "predicted value"]) % Legend for the plot
xlabel("phi")  % X-axis label for volume fraction
ylabel("w")    % Y-axis label for angular frequency

% Define the coefficients for the fitted exponential function
a_2 = 1.18;
b = -2.741;
c = 2.368;
d = 0.08088;

% Define the exponential function for plotting
exp_plot = @(phi) a_2 * exp(b * phi) + c * exp(d * phi);

% Create a finer range for phi for smooth plotting
phi_i = linspace(0,90,100);

% Plot the original data and the exponential fit
figure
plot(phi, w,'*' ,phi_i, exp_plot(phi_i))

% Initialize an empty matrix for ETA (viscosity values)
ETA = [];

% Loop through each value of w to compute ETA (viscosity matrix)
for i = 1:length(w)
    if i == 1
        eta0 = etaInf;  % Set initial eta0 to etaInf for the first case
        eta = etaInf + (eta0 - etaInf) * (1 + (lambda * gamma).^a).^((n - 1) / a);
    else
        eta0 = w(i);    % Update eta0 based on the current w value
        eta = etaInf + (eta0 - etaInf) * (1 + (lambda * gamma).^a).^((n - 1) / a);
    end
    ETA = [ETA; eta];  % Append computed eta values to the matrix
end

% Create a surface plot for the viscosity matrix ETA
figure;
[GAMMA, PHI] = meshgrid(gamma, phi); % Create meshgrid for gamma and phi
surf(GAMMA, PHI, ETA, 'EdgeColor', 'none'); % Plot the surface without edges
set(gca, 'zscale', 'log'); % Set z-axis to logarithmic scale
set(gca, 'xscale', 'log'); % Set x-axis to logarithmic scale
set(gca, 'yscale', 'log'); % Set y-axis to logarithmic scale
view(45, 45); % Set the viewing angle for the 3D plot
xlabel('Shear Rate $(\dot{\gamma})$', 'Interpreter', 'latex', 'FontSize', 12); % X-axis label
ylabel('Density $(\phi)$', 'Interpreter', 'latex', 'FontSize', 12); % Y-axis label
zlabel('Viscosity $(\eta)$', 'Interpreter', 'latex', 'FontSize', 12); % Z-axis label
hold on;

% Define specific phi values for extracting eta values at different shear rates
phi_values = [20, 40, 60]; % Selected phi values
eta_gamma = zeros(length(phi_values), length(gamma)); % Initialize a matrix to store eta values

% Loop through phi values and extract corresponding eta values
for k = 1:length(phi_values)
    [~, idx] = min(abs(phi - phi_values(k))); % Find the closest index for each phi value
    eta_gamma(k, :) = ETA(idx, :); % Store the corresponding eta values
end

% Plot the trajectories for each selected phi value
colors = ['y', 'r', 'g']; % Colors for the trajectories
for k = 1:length(phi_values)
    plot3(gamma, phi_values(k) * ones(size(gamma)), eta_gamma(k, :), ...
        'LineWidth', 20, 'Color', colors(k), 'DisplayName', sprintf('$(\\phi) \\approx %d$', phi_values(k)));
end

% Display the legend for the trajectories
legend('show', 'Interpreter', 'latex', 'Location', 'best');
hold on;

% Define specific gamma values for extracting eta values at different phi values
gamma_values = [0.052 , 52.0]; % Selected gamma values
eta_phi = zeros(length(phi), length(gamma_values)); % Initialize a matrix to store eta values

% Loop through gamma values and extract corresponding eta values for each phi
for k = 1:length(gamma_values)
    [~, idx] = min(abs(gamma - gamma_values(k))); % Find the closest index for each gamma value
    eta_phi(:, k) = ETA(:, idx); % Store the corresponding eta values
end

% Plot the trajectories for each selected gamma value
colors = ['b', 'white']; % Colors for the trajectories
for k = 1:length(gamma_values)
    plot3(gamma_values(k) * ones(size(phi)), phi, eta_phi(:, k), ...
        'LineWidth', 20, 'Color', colors(k), 'DisplayName', sprintf('$(\\gamma) \\approx %d$', gamma_values(k)));
end

% Display the legend for the new trajectories
legend('show', 'Interpreter', 'latex', 'Location', 'best');
hold on;

