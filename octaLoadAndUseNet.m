close all
clear
clc

% +++++++++++++++ Load training data (starting solution) +++++++++++++++++

Nth = 100;
Nxi = 100; % I probably removed one sigma!

dd = load('outp_TEST.csv');

th   = dd(:,1);
xi   = dd(:,2);
eigg1 = dd(:,3);
eigg2 = dd(:,4);

% The net is trained on scaled and centered values!
th_min = min(th);
th_max = max(th);
xi_min = min(xi);
xi_max = max(xi);
eigg1_min = min(eigg1);
eigg1_max = max(eigg1);
eigg2_min = min(eigg2);
eigg2_max = max(eigg2);

% OLD SCALING 0 to 1 %%% th   = (th   - min(th))/(max(th) - min(th));
% OLD SCALING 0 to 1 %%% xi   = (xi   - min(xi))/(max(xi) - min(xi));
% OLD SCALING 0 to 1 %%% eigg = (eigg - min(eigg))/(max(eigg) - min(eigg));

% Scaling from -1 to 1
th    = 2*(th    - min(th))/(max(th) - min(th)) - 1;
xi    = 2*(xi    - min(xi))/(max(xi) - min(xi)) - 1;
eigg1 = 2*(eigg1 - min(eigg1))/(max(eigg1) - min(eigg1)) - 1;
eigg2 = 2*(eigg2 - min(eigg2))/(max(eigg2) - min(eigg2)) - 1;

TH    = reshape(th,    Nth, Nxi);
XI    = reshape(xi,    Nth, Nxi);
EIGG1 = reshape(eigg1, Nth, Nxi);
EIGG2 = reshape(eigg2, Nth, Nxi);

% ++++++++++++++++ Create data to test the net +++++++++++++++++++++++++++
% Data is scaled to unit range by the net.

% Ntest_th = 100;
% Ntest_xi = 103;
% 
% th_test = linspace(0,1,Ntest_th);
% xi_test = linspace(0,1,Ntest_xi);
% 
% [TH_TEST, XI_TEST] = meshgrid(xi_test, th_test);

TH_TEST = TH;
XI_TEST = XI;

% +++++++++++++++ LOAD NEURAL NET +++++++++++++++++++
model = load('saved_keras_model.h5');

% Say something on the structure of the net
% activations = {'tanh', 'tanh', 'tanh', 'tanh', 'linear'};
activations = {'tanh', 'tanh', 'tanh', 'linear'};

NNetPredic1 = zeros(size(TH_TEST));
NNetPredic2 = zeros(size(TH_TEST));
 
for ii = 1:numel(NNetPredic1) % Access elements one by one
  [NNetPredic1(ii), NNetPredic2(ii)] = apply_saved_net(model, activations, [TH_TEST(ii); XI_TEST(ii)]);
end

% Compute error
% Rescale the values
NNetPredic1_SCALEBACK = 10.^((NNetPredic1 + 1)/2*(eigg1_max - eigg1_min) + eigg1_min);
EIGG1_SCALEBACK       = 10.^((EIGG1+1)/2*(eigg1_max - eigg1_min) + eigg1_min);

NNetPredic2_SCALEBACK = 10.^((NNetPredic2 + 1)/2*(eigg2_max - eigg2_min) + eigg2_min);
EIGG2_SCALEBACK       = 10.^((EIGG2+1)/2*(eigg2_max - eigg2_min) + eigg2_min);

% SCALING 0 to 1 %%% NNetPredic_SCALEBACK = 10.^(NNetPredic*(eigg_max - eigg_min) + eigg_min);
% SCALING 0 to 1 %%% EIGG_SCALEBACK       = 10.^(EIGG*(eigg_max - eigg_min) + eigg_min);

Err1 = (NNetPredic1_SCALEBACK - EIGG1_SCALEBACK)./(EIGG1_SCALEBACK + 1.0e-15);
Err2 = (NNetPredic2_SCALEBACK - EIGG2_SCALEBACK)./(EIGG2_SCALEBACK + 1.0e-15);

% Err = (NNetPredic - EIGG)./(EIGG + 0.0001);

% NONONONONONONON TO BE COMPUTED ON RESCALED DATA!!!!!!!
% NONONONONONONON TO BE COMPUTED ON RESCALED DATA!!!!!!!

% +++++++++++++++ PLOT +++++++++++++++++++++++++++++
figure
surf(TH_TEST,XI_TEST,NNetPredic1)
xlabel('theta')
ylabel('xi = log10(sigma)')
zlabel('Predicted max WS')

hold on
steppp = 5;
plot3(th(1:steppp:end), xi(1:steppp:end), eigg1(1:steppp:end), 'or', 'linewidth',2)
% zlim([-0.1,1])

%%%%%%%%%

figure
surf(TH_TEST,XI_TEST,NNetPredic2)
xlabel('theta')
ylabel('xi = log10(sigma)')
zlabel('Predicted min WS')

hold on
steppp = 5;
plot3(th(1:steppp:end), xi(1:steppp:end), eigg2(1:steppp:end), 'or', 'linewidth',2)
% zlim([-0.1,1])


%
figure
surf(TH_TEST, XI_TEST, Err1*100)
xlabel('theta')
ylabel('xi')
% caxis([0,1])
title('Error [\%] in max WS (real scale) dimensionless eig)')

%
figure
surf(TH_TEST, XI_TEST, Err2*100)
xlabel('theta')
ylabel('xi')
% caxis([0,1])
title('Error [\%] in min WS (real scale) dimensionless eig)')


fprintf('+++++++++++++++++++++++++++++++++++++++++++\n')
fprintf('++++++++ PRESS RETURN TO EXIT +++++++++++++\n')
fprintf('+++++++++++++++++++++++++++++++++++++++++++\n')
pause()

% 
% 
% figure
% surf(Q(1:5:end, 1:5:end), SIG(1:5:end, 1:5:end), Err(1:5:end, 1:5:end))
% xlabel('q')
% ylabel('sig')
% 
% 
% figure
% surf(Q(1:5:end, 1:5:end), SIG(1:5:end, 1:5:end), EIGG(1:5:end, 1:5:end))
% title('Training data')
% xlabel('q')
% ylabel('sig')
% 
% 
