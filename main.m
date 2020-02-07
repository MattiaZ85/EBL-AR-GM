% Code written by Mattia Zorzi 2019
Install_mex
% algorithm parameters  
e = 10^-3; % hyperprior for Lambda and Gamma
tol=10^-3; % threshold for the reweightingò-like scheme
verb='v'; % no verbosity

% load data
n = 6; % dimension of the process
p = 2;  % order of the AR model
N = 500; % length of the data
y= randn(500,n);% data (i.e. a model with 0 latent variables and S diagonal)              
[Omega,A,R,th,S,Rn,iter,Sp,gamma] = identS(y,p,e,tol,verb);
[Omega,h,S,L,Sp,Lp,gamma,Q] = identSL(y,p,e,e,tol,verb);

   
 