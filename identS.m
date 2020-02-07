function [Omega,A,R,th,S,Rn,iter,Sp,gamma] = identS(X,p,e,tol,verb)
%identS     computes the sparse grafical model using the re-weigthed scheme
%
% Inputs:  X data (row is the time, columns are the components)
%          p order of the AR model
%          e epsilon parameter (>0) in the re-weigthed scheme
%          tol tolerance for stopping criterium of the re-weighted
%          iterations
%          verb 'v'=verbouse modality, ' '=no verbose modality
%
% Outputs: Omega sparsity patters (0 = no edge, 1 = edge)
%          A=[A0 A1 ... Ap] matrix coefficients of the AR model
%        
%             sum_{k=0}^n A_k y(t-k) =e(t),   
%
%             with e(t) WGN with covaraince matrix I_n
%          R inverse of the inverse of PSD (th vector of frequencies)
%          Rn partial coherence function (th vector of frequencies)
%          S=[S0 S1 ... Sn] coefficients of the partial coherence function
%
%          S = S0+0.5sum_{k=1}^n S_k z^{-k}+S_k' z^{k}  
%          iter number of iterations for the reweighted
%          Sp estimates found in the re-weighted iterations
%          gamma regularization values in the re-weighted iterations
% 
% In this case the reweighting is with GML  
%


%% variables
N = size(X,1);
n = size(X,2);

%% weigths of gammma
switch p
    case 0
        dia = 1;
        ofd = 1;
    case 1
        dia = 2;
        ofd = 3;
    case 2 
        dia =3;
        ofd =5;
end

%% init gammas
T_C=symm(sTop(X,p));
L=chol(T_C);
Li=L^-1;
B=Li*Li'*[eye(n); zeros(n*p,n)];
B=B*sqrtm(B(1:n,1:n))^-1;
Sinit = get_D_short(B*B', n, p);
W = weigth(Sinit);
gamma(:,:,1)=2/(N-p)*1./(W+e);
gamma(:,:,1) = dia*diag(diag(gamma(:,:,1)))+ofd*(gamma(:,:,1)-diag(diag(gamma(:,:,1))));

gamma(:,:,1) = symm(gamma(:,:,1));

%% thresholding in the final inverse
thr = 10^-4; 

%% optimize
k=1;
d=1;
while d>=tol
    [Omega,A,R,th,Sp{k},Rn] = sparseGMW2(X,gamma(:,:,k),n,p,thr,verb);
    
    W = weigth(Sp{k});
    gamma(:,:,k+1)=2/(N-p)*1./(W+e);
    gamma(:,:,k+1)=2/(N-p)*1./(W+e);
    gamma(:,:,k+1) = dia*diag(diag(gamma(:,:,k+1)))+ofd*(gamma(:,:,k+1)-diag(diag(gamma(:,:,k+1))));

    gamma(:,:,k+1) = symm(gamma(:,:,k+1));
    if k>1
        d = distance(Sp{k-1},Sp{k});
    end
    k=k+1;
end    
iter = k-1;  % totoal iterations
gamma=gamma(:,:,1:end-1);
S=Sp{end};
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   FUNCTIONS USED IN MAIN CODE   %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Omega,A,R,th,S,Rn,fit,reg] = sparseGMW2(X,gamma,n,p,thr,verb)
%SPARSEGMW computes the sparse grafical model
%
% Inputs:  X data (row is the time, columns are the components)
%          gamma (positive entrywise and symmetric) regularization
%          parameter maitrx
%          n dimension of the process
%          p order of the AR model
%          thr (<1) thresholding in the partial coherence
%          verb 'v'=verbouse modality, ' '=no verbose modality
%
% Outputs: Omega sparsity patters (0 = no edge, 1 = edge)
%          A=[A0 A1 ... Ap] matrix coefficients of the AR model
%        
%             sum_{k=0}^n A_k y(t-k) =e(t),   
%
%          e(t) WGN with covaraince matrix I_n
%          R partial coherence function (th is the vector of frequencies)
%          S=[S0 S1 ... Sn] coefficients of the partial coherence function
%
%          S = S0+0.5sum_{k=1}^n S_k z^{-k}+S_k' z^{k}   
% 


%% parameter in the armijo condition
alpha = 0.1;

%% stopping tolerance for the duality gap
tol=10^-4;

%% maximum number of iterations
max_iter=1000;

%% sample Toeplitz covariance
C=symm(sTop(X,p));

%% init
Z = zeros(n,n,p+1);
obj = phi(C,Z);
t = 0.2;
dgap = tol+1;
it=0;

%% optimization with repsect to Z
while dgap >= tol
    it=it+1;
    if it>max_iter
        break
    end 
    gradZ = grad_phi(C,Z);
    t = min(4*t,0.5);
    flagp = false;
    while not(flagp)
        t = t/2;
        Zn = proj_C1W(Z-t*gradZ,gamma);
        flagp = check_positive_definiteness(C,Zn);
        if t<10^-6
            break
        end
    end
    t = 2*t;
    flaga = false;
    while not(flaga)
        t = t/2;
        Zn = proj_C1W(Z-t*gradZ,gamma);
        flaga = check_armijo(obj,C,Z,Zn,gradZ,alpha,t);
        if t<10^-6
            break
        end        
    end
    Z = Zn;
    obj = phi(C,Z);
    dgap = dual_gap(C, Z, gamma);
    if verb=='v'
        disp(['iteration#' num2str(it) '    obj dual:' num2str(obj) '   duality gap: ' num2str(dgap) ])
    end
end
    
%% primal solution
[dgap,X] = dual_gap(C, Z, gamma);

% coeffieicients of the inverse of the PSD
S = get_D_short(X, n, p);
S(:,:,1)=symm(S(:,:,1));


% inverse of the PSD 
th = linspace(0,pi,1000);
R = zeros(n,n,1000);
Rn = zeros(n,n,1000);
for k=1:size(th,2)
    SS=S(:,:,1);
    for j=2:size(S,3)
        SS = SS+0.5*(S(:,:,j)*exp(-sqrt(-1)*th(k)*(j-1))+S(:,:,j)'*exp(sqrt(-1)*th(k)*(j-1))); 
    end
    R(:,:,k) = SS;
    Rn(:,:,k) = diag(diag(SS))^-0.5*SS*diag(diag(SS))^-0.5;
end

%% sparsity patter with thresholding
Omega = zeros(n,n);
for k=1:n
    for j=1:n
        Omega(j,k) = max(abs(R(j,k,:)));
    end
end
Omega = symm(Omega);
Omega = Omega>thr;


%% coefficients of the AR model 
try
    A=bauer(S,20*p);
catch
    A=[];
    %warning('Spectral factorization aborted')
end


%% computes the fit and regularization term

[fit , reg] = fit_vs_reg(C, Z, gamma);



end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function r = phi(C, Z)
% Function to evaluate phi(C+T(Z)), defined in the paper
    T_Z = get_T(Z);
    n = size(Z,1);
    V = C + T_Z;
    V00 = V(1:n,1:n);
    V1p0 = V(n+1:end,1:n);
    V1p1p = V(n+1:end,n+1:end);
    V_temp = V00 - V1p0'*(V1p1p\V1p0);
    V_temp = symm(V_temp);
    r = -log(det(V_temp));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function grad_Z = grad_phi(C, Z)
% Function to compute the gradient of phi
    n = size(Z, 1);
    p = size(Z, 3) - 1;
    T_Z=get_T(Z);
    V = C + T_Z;
   
%     L=chol(symm(V),'lower');
%     L00 = L(1:n,1:n);
%     L1p0 = L(n+1:end,1:n);
%     L1p1p = L(n+1:end,n+1:end);
%     Ltmp=[eye(n); -L1p1p^-1*L1p0]*L00^-1;
%     grad_Z =- get_D_short(Ltmp*Ltmp', n, p);
    
    Y = sparse([zeros(n*p,n) eye(n*p)]);
    T1 = speye(size(V))/V;
    T2Y = ((Y*V)*Y')\Y;     
    grad_Z = get_D_short(-T1 + Y'*T2Y, n, p);

    % grad_Z-grad_ZZ
% pause
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function h_BM = get_hW(D,gamma)
% Function to compute the weighted "infinity norm".
% D is in a 3d format.
    gamma = symm(gamma);
    Dabs = abs(D);
    Dabs_max_matrix = max(Dabs, [], 3);
    Dabs_max_matrix_offdiag = Dabs_max_matrix - diag(diag(Dabs_max_matrix)); % DO NOT Remove the diagonal
    DU = gamma.*(triu(Dabs_max_matrix_offdiag)+diag(diag(Dabs_max_matrix))); % Upper triangular part plus diagonal
    DL = gamma.*tril(Dabs_max_matrix_offdiag); % Lower triangular part
    DLt = DL'; % Transpose
    h_BM = sum(max([DLt(:), DU(:)], [], 2));
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Z_projected = proj_C1W(Z, gamma) %Z is the U of the equations p.2699
% Function to project onto the C1 constraint

    n = size(Z, 1);
    p = size(Z, 3) - 1;
    X = zeros(1, 2*(p+1)); % A row vector
    
    Z_projected = zeros(size(Z));
    
    Z(:,:,1) = symm(Z(:,:,1)); % Ensure that the first block is symmetric
    
    for i = 1 : n
        
        for j = i : n % For each element we perform LASSO
            % Reformulation using vectors
            if i==j
                gamma(i,j) = 2*gamma(i,j);
            end
            X(1 : p+1) = Z(i,j,:);
            X(p+2 : end) = Z(j,i,:);
            absX = abs(X);
            Z_temp = projectSortC(absX', gamma(i,j)); % Using the mex file from EWOUT VAN DEN BERG, MARY SCHMIDT, MICHAEL P. FRIEDLANDER, AND YEVIN MURPHY
            Z_temp = (sign(X').*Z_temp)';
            Z_projected(i,j,:) = Z_temp(1 : p+1);
            Z_projected(j,i,:) = Z_temp(p+2 : end);

        end
        
    end
    
    Z_projected(:,:,1) = symm(Z_projected(:, :, 1)); % Ensure that the first block is symmetric
        
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [val,X] = dual_gap(C, Z, gamma)
% computes the duality gap
    n = size(Z, 1);
    p = size(Z, 3) - 1;
    T_Z=get_T(Z);
    V = C + T_Z;

    W = get_W(C, Z);
    X = V\(sparse([W, zeros(n,n*p); zeros(n*p,n), zeros(n*p,n*p) ])/V);
    X = symm(X); % numerical accuracy

   val=-trace(X*T_Z)+get_hW(get_D_short(X, n ,p),gamma);

end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function C=sTop(X,p)
% Computes the sample Toeplitz covariance matrix C of order p
% from the given data X (rows is time, columns are the component)
%
% Note that C has dimension n(p+1) x n(p+1) s.t.
%  
%       C=[C0  C1 .... Cn
%          C1' C0 .... : 
%          :           :
%          Cn'          ]
%
% where C_k=E[x(t+k)x(t)^T]
%




n=size(X,2);
N=size(X,1);
C_line=zeros(n,n,2*p+1);
for k=p+1:2*p+1
    for t=1:N-k+p+1
        C_line(:,:,k)=C_line(:,:,k)+N^-1*X(t+k-p-1,:)'*X(t,:);
    end
end
for k=1:p
    C_line(:,:,k)=C_line(:,:,-k+2*p+2)';
end
    
    

C = zeros(n*(p+1));
for i=1:p+1
    for j=1:p+1
        C(((i-1)*n+1):i*n,((j-1)*n+1):j*n)=C_line(:,:,(j-i)+p+1);
    end
end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function T = get_T(Z)
% Function to compute T(Z), outputs a block Toeplitz matrix
    n = size(Z, 1);
    p = size(Z, 3) - 1;
    T = zeros(n*(p+1));
    for i=1:p+1
        for j=1:p+1
            if i<j
                T(((i-1)*n+1):i*n,((j-1)*n+1):j*n)=Z(:,:,(j-i)+1);
            else
                T(((i-1)*n+1):i*n,((j-1)*n+1):j*n)=(Z(:,:,(i-j)+1))';
            end
        end
    end
    T = symm(T); % It should be symmetric
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function D = get_D(X, n, p)
% Function to compute the adjoint of T with output in a matrix of size n x (p+1)    
    D = zeros(n,n*(p+1));
    %D0
    for l=0:p
        D(:,1:n)=D(:,1:n)+X(((l)*n+1):((l+1)*n),((l)*n+1):((l+1)*n));
    end
    
    %Dj for j in [1,p]
    for m=1:p
        for l=0:p-m
            D(:,(m*n+1:(m+1)*n)) = D(:,(m*n+1:(m+1)*n)) + 2*X(((l)*n+1):((l+1)*n),((l+m)*n+1):((l+m+1)*n));
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function D = get_D_short(X, n, p)
% Function to compute the adjoint of T with output in 3-d array
% p is the AR order
% n is the dimension of the process
    D = zeros(n,n,(p+1));
    %D0
    for l=0:p
        D(:,:,1)=D(:,:,1)+X(((l)*n+1):((l+1)*n),((l)*n+1):((l+1)*n));
    end
    %Dj for j in [1,p]
    for m=1:p
        for l=0:p-m
            D(:,:,m+1) = D(:,:,m+1) + 2*X(((l)*n+1):((l+1)*n),((l+m)*n+1):((l+m+1)*n));
        end
    end    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function boolean1 = check_positive_definiteness(C,Z)
% Checking one of the conditions for the backtracking step    
    
    T_Z = get_T(Z);
    
    
    n = size(Z, 1);
    
    V = C + T_Z;
    V = symm(V);
    
    % First part, positive semidefiniteness of V
    cond11 = min(eig(V)) >= -1e-8; % BM: earlier it was 0
    
    % Second part, positive semidefiniteness of the Schur complement
    V00 = V(1:n, 1:n);
    V1p0 = V(n+1 : end, 1:n);
    V1p1p = V(n+1 : end, n+1 : end);
    
    V_temp = V00 - V1p0'*(V1p1p\V1p0);
    V_temp = symm(V_temp); % BM
    cond12 = min(eig(V_temp))>= -1e-8; % BM: earlier it was 0
    
    % Synthesis
    boolean1 = cond12 && cond11;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function boolean1 = check_armijo(obj_old,C,Z,Zn,gradZ,alpha,t)
% Checking che armijo conditions
    
    p=size(Z,3)-1;
    val=0;
    for k=1:p+1
        val=val+trace(gradZ(:,:,k)'*(Zn(:,:,k)-Z(:,:,k)));
    end
    
    if  phi(C,Zn)<= obj_old+alpha*t*val
        boolean1=true;
    else 
        boolean1=false;
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function W = get_W(C, Z)
% Function to compute W
    T_Z = get_T(Z);
    n = size(Z,1);
    V = C + T_Z;
    
    V00 = V(1:n,1:n);
    V1p0 = V(n+1:end,1:n);
    V1p1p = V(n+1:end,n+1:end);
    W = V00 -V1p0'*(V1p1p\V1p0);
    W = symm(W);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function D = symm(D)
% Function to compute the symmetric part of a matrix
    D = 0.5*(D + D');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [A, SS] = bauer(P,Kb)
% Computes the factorization P(z)=A(z)^T*A(z^-1) starting from P
% where A(z^-1)=A0+A1z^-1+...+Anz^-n
% Input  P=[P0 ... Pn] coefficizents of P(z) (3d structure)
%        Kb integer parameter to choose sufficiently large
% Output A=[A0 ... An] coefficients of A(z^-1) (3d structure)


% from 3d to 2d
n1=size(P,1);
p1=size(P,3);
Pt=zeros(n1,n1*p1);
for k=1:p1
    Pt(:,(k-1)*n1+1:k*n1)=P(:,:,k);
end
P=Pt;
% parameters of the polynomial
[m g]=size(P);
n=g/m;
if n>1 % dynamic case
    % band toeplitz matrix
    Stmp=[P zeros(m,m*(Kb-n))];
    SS=[];
    for j=1:Kb
        SS=[SS; Stmp];
        if j<n
            J=(2*j-1)*m+1:(2*j)*m;
            Stmp=[Stmp(:,J)' Stmp];
        else
            Stmp=[zeros(m,m) Stmp];
        end
        Stmp=Stmp(:,1:end-m);
    end
    SS=symm(SS);
    % check positivity
    if min(eig(SS))<=0
        ME = MException('VerifyOutput:OutOfBounds', ...
             'Positivity not guaranteed');
        throw(ME);
    end
    % cholesky decomposition
    for j=1:Kb
        J=(j-1)*m+1:j*m;
        if j==1
            D(J,J)=SS(J,J);
        else
          SUM=zeros(m,m);
          for k=1:j-1
             K=(k-1)*m+1:k*m;
             SUM=SUM+L(J,K)*D(K,K)*L(J,K)';
          end  
          D(J,J)=SS(J,J)-SUM;
        end
        D(J,J)=symm(D(J,J));
        DD(J,J)=chol(D(J,J))';
        L(J,J)=eye(m);
        for i=j+1:Kb
            I=(i-1)*m+1:i*m;
            SUM=zeros(m,m);
            for k=1:j-1
                K=(k-1)*m+1:k*m;
                SUM=SUM+L(I,K)*D(K,K)*L(J,K)';
            end
            L(I,J)=(SS(I,J)-SUM)*D(J,J)^-1;
        end
    end
    L=L*DD;
    % constrution of A
    j=Kb;
    for k=1:n
        J=(k-1)*m+1:k*m;
        K=(j-1)*m+1:j*m;
        A(:,J)=L((Kb-1)*m+1:Kb*m,K)';
        j=j-1;
    end

    % 3d tranformation
    Af=zeros(m,m,n);
    for k=1:n
        Af(:,:,k)=A(:,(k-1)*m+1:k*m);
    end
    A=Af;
else
    A=chol(P); % static case
    SS=P;
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [fit,reg] = fit_vs_reg(C, Z, gamma)
% computes the fit and the regularization part at the optimum 

    n = size(Z, 1);
    p = size(Z, 3) - 1;
    T_Z=get_T(Z);
    V = C + T_Z;

    W = get_W(C, Z);
    X = V\(sparse([W, zeros(n,n*p); zeros(n*p,n), zeros(n*p,n*p) ])/V);
    X = symm(X); % numerical accuracy

   fit = -log(det(X(1:n,1:n)))+trace(X*C);
   reg = get_hW(get_D_short(X, n ,p),gamma);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function d = distance(X,Y)
% computes the distance between two 3d matrices
p=size(X,3)-1;
n=size(X,1);
d=0;
for k=1:p+1
    d=d+trace((X(:,:,k)-Y(:,:,k))*(X(:,:,k)-Y(:,:,k))');
end
d= d/(n^2*(p+1));    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
function W = weigth(D)
% Computes the re-weight  

    n=size(D,1);
    Dabs = abs(D);
    Dabs_max_matrix = max(Dabs, [], 3);
    Dabs_max_matrix_offdiag = Dabs_max_matrix - diag(diag(Dabs_max_matrix)); % Remove the diagonal
    DU = (triu(Dabs_max_matrix_offdiag)+ diag(diag(Dabs_max_matrix))); % Upper triangular part
    DL = tril(Dabs_max_matrix_offdiag); % Lower triangular part
    DLt = DL'; % Transpose
    W=reshape(max([DLt(:), DU(:)], [], 2),n,n);
    W=W+W'-diag(diag(W));
    
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
