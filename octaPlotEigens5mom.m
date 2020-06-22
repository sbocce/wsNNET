close all
clear
clc

% LOad data

dd = load('outp.csv');

th_vect  = dd(:,1);
sig_vect = dd(:,2);
maxEig   = dd(:,3);
minEig   = dd(:,4);

Nq = 300;
Nsig = 100;

TH  = reshape(th_vect,  Nq, Nsig);
SIG = reshape(sig_vect, Nq, Nsig);

MAXEIG = reshape(maxEig, Nq, Nsig);
MINEIG = reshape(minEig, Nq, Nsig);

figure
surf(TH, SIG, MAXEIG)
% surf(Q, SIG, log(log(MAXEIG + 1)))
% surf(Q, SIG, MAXEIG - (Q.*(erf(Q*20)/2+0.5)./SIG + 1))
% surf(Q, SIG, MAXEIG./(Q./SIG + 1) )  
% surf(Q, SIG, MAXEIG - (Q./SIG + 1).*(erf(Q*10)/2 + 0.5) )  
% surf(Q, SIG, MAXEIG - (Q./SIG + 1).*(Q >= 0)) 
% surf(Q, SIG, MAXEIG - (Q./SIG + 1))
hold on
% plot3(Q, SIG, Q./SIG + 1, 'xr', 'linewidth', 2)
xlabel('TH')
ylabel('SIG')
zlabel('maxxx')

return

% PROCESSED DATA
MAXEIGPROC = MAXEIG - (Q.*(erf(Q*20)/2+0.5)./SIG + 1);

fid = fopen('scaled.csv','w')

for ii = 1:numel(Q)
  if Q(ii) ~= 1.0e-5 % Skip very first point
    fprintf(fid, '%.15e, %.15e, %.15e\n', Q(ii), SIG(ii), MAXEIGPROC(ii))
  end
end

fclose(fid)
