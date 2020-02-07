function [Omega,h,S,L,Sp,Lp,gamma,Q] = identSL(X,p,eS,eL,tol,verb)
%identSL computes the S+L grafical model using the re-weigthed scheme
%
% Inputs:  X data (row is the time, columns are the components)
%          p order of the AR model
%          eS epsilon parameter (>0) in the re-weigthed scheme in S
%          eL epsilon parameter (>0) in the re-weigthed scheme in L
%          tol tolerance for stopping criterium of the re-weighted
%          verb 'v'=verbouse modality, ' '=no verbose modality
%
% Outputs: Omega sparsity patters (0 = no edge, 1 = edge)
%          A=[A0 A1 ... Ap] matrix coefficients of the AR model
%        
%             sum_{k=0}^n A_k y(t-k) =e(t),   
%
%             with e(t) WGN with covaraince matrix I_n
%          S=[S0 S1 ... Sn] coefficients of the partial coherence function
%
%          S = S0+0.5sum_{k=1}^n S_k z^{-k}+S_k' z^{k}  
%          L=[L0 L1 ... Ln] coefficients of the partial coherence function
%
%          L = L0+0.5sum_{k=1}^n L_k z^{-k}+L_k' z^{k}  
%          Sp estimates found in the re-weighted iterations for S
%          Lp estimates found in the re-weighted iterations for L
%          gamma regularization values in the re-weighted iterations
%          Q regularization values in the re-weighted iterations
% 
%
% The prior only on L0


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
Delta = get_D_short(B*B', n, p);
X1 = 1.1*Delta;
X2 = 0.1*Delta;
W = weigth(X1);
gamma(:,:,1)=2/(N-p)*1./(W+eS);
gamma(:,:,1) = dia*diag(diag(gamma(:,:,1)))+ofd*(gamma(:,:,1)-diag(diag(gamma(:,:,1))));
gamma(:,:,1) = symm(gamma(:,:,1));
Q(:,:,1)=1/(N-p)*(X2(:,:,1)+eL*eye(n))^-1*(n+1);
Q(:,:,1)=symm(Q(:,:,1));


%% optimize
k=1;
d=1;
max_iter2=0;
while (d>=tol) & (max_iter2<=50)
    
    %% posterior
    [Omega,h(k),Sp{k},Lp{k},cobj,primal_residual,dual_residual,dgap,rho]=slGMW(X,gamma(:,:,k),Q(:,:,k),n,p,verb);
    %% update weigths
    W = weigth(Sp{k});
    gamma(:,:,k+1)=2/(N-p)*1./(W+eS);
    gamma(:,:,k+1) = dia*diag(diag(gamma(:,:,k+1)))+ofd*(gamma(:,:,k+1)-diag(diag(gamma(:,:,k+1))));
    gamma(:,:,k+1) = symm(gamma(:,:,k+1));
    Q(:,:,k+1)=2/(N-p)*(Lp{k}(:,:,1)+eL*eye(n))^-1*(n+1)/2;
    Q(:,:,k+1)=symm(Q(:,:,k+1));
    if k>1
        d = distance(Sp{k-1}-Lp{k-1},Sp{k}-Lp{k});
    end
    if verb=='v'
    fprintf('[%0.4d] Distance = %3.3e, Latent v. =%3.0d, primal resid. = %3.3e, dual resid. =%3.3e, rho = %3.3e\n', k, d, h(k), primal_residual, dual_residual, rho);        
    end
    k=k+1;
    max_iter2=max_iter2+1;
end    

 
gamma=gamma(:,:,1:end-1);
Q=Q(:,:,1:end-1);
S=Sp{end};
L=Lp{end};
h=h(end);
end
    
  






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   FUNCTIONS USED IN MAIN CODE   %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



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


function [Omega,h,S,L,obj,primal_residual,dual_residual,dgap,rho] = slGMW(X,gamma,Q,n,p,verb)

%slGMW computes the sparse + low rank grafical model
%
% Inputs:  X data (row is the time, columns are the components)
%          gamma (positive entrywise and symmetric) regularization
%          parameter maitrx for the sparse part
%          Q (positive definite and symmetric) regularization
%          parameter maitrx for the low rank part (dim n x n)
%          n dimension of the process
%          p order of the AR model
%          verb 'v'=verbouse modality, ' '=no verbose modality
%
% Outputs: Omega sparsity patters (0 = no edge, 1 = edge)
%          h number of latent variables
%          S=[S0 S1 ... Sn] coefficients of the partial coherence function
%                      S = S0+0.5*sum_{k=-n}^n S_k z^{-k}+S_k'z^k
%          S=[L0 L1 ... Ln] coefficients of the partial coherence function
%                      L = L0+0.5*sum_{k=-n}^n L_k z^{-k}+L_kz^k
% obj      value of the objective function at the optimal solution



%% parameter in the armijo condition
alpha = 0.1;

%% stopping tolerance for the duality gap
eps_pri=10^-5;
eps_dual=10^-5;

%% maximum number of iterations
max_iter = 6000; % prima 2000

%% rho for ADMM
rho_max = 1e4;
rho =1;

%% sample Toeplitz covariance
C=symm(sTop(X,p));

%% init
Z = zeros(n,n,p+1);
Y = eye(n*(p+1));
M = zeros(n*(p+1));
obj = cost(C,Z,Y,M,Q,rho); 
t = 0.2;

%% ADMM algorithm 
for iter=1:max_iter
    
    %% First update: variable Z    
    % Gradient step in Z
    gradZ = grad(C,Z,Y,M,Q,rho);
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
        flaga = check_armijo(obj,C,Z,Y,M,Q,Zn,gradZ,alpha,rho,t);
        if t<10^-6
            break
        end        
    end
    
    %% Second update: variable Y
    Yn = proj_C2((M/rho) + kron(eye(p+1),Q) + get_T(Zn)); 
    
    %% Third update for the dual varible M
    Mn = M - rho*(Yn - kron(eye(p+1),Q) - get_T(Zn)); 

    %% Standard stopping criteria based on primal and dual residual error
    primal_residual = norm(Yn - get_T(Zn) - kron(eye(p+1),Q),'fro');
    dual_residual = rho*norm(get_D(Y-Yn, n , p), 'fro');
    
    %% Update rho
%     rhon = min(rho_max, 1.0125*rho); % Boyd's suggestion
rhon=rho;
    if dual_residual<10^-3 % 10^-2  
         rhon = min(rho_max, 1.1*rho);  
    end
    if dual_residual>0.01 % prima uno
        rhon=rho/2;
    end
        
    %             % Second choice from Stephen Boyd's book
    %             if primal_residual > mu*dual_residual,
    %                 rho = rho_inc*rho; % Increase rho
    %             elseif dual_residual > mu*primal_residual
    %                 rho = rho/rho_dec; % Decrease rho
    %             end

            
   %% Update of the variables
   Z = Zn;
   Y = Yn;
   M = Mn;
   rho = rhon;
   obj = cost(C,Z,Y,M,Q,rho);
   
   %% Print info
   if verb=='v'
        fprintf('[%0.4d] Augmented Lagrangian = %7.3e, primal resid. = %3.3e, dual resid. =%3.3e, rho = %3.3e\n', iter, obj, primal_residual, dual_residual, rho);        
    end        

   %% Stopping criteria
   if primal_residual < eps_pri && ...
            dual_residual < eps_dual && ...
            rho == rho_max
        if verb=='v'
            fprintf('\nPrimal and dual residual sufficiently decreased. \n');
        end
        break
   end 
   
end
    
%% primal solution
[Omega, h, S, L, cost_p, cost_d] = compute_costs_and_primal_variables(C,Z,gamma,Q,verb);
dgap=cost_p-cost_d;
obj = cost_p;

%% Print last info
if verb=='v'
        fprintf('Duality gap = %7.3e\n', dgap);        
end        


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
    r = -log(det(V_temp))-n;
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


end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function val = cost(C,Z,Y,M,Q,rho)
% Compute the augmented Lagrangian
%   Z are 3D objects; M C, Y, Q are matrices, Q of dimension n x n
    p = size(Z,3)-1;
    P=kron(eye(p+1),Q);
    T_Z = get_T(Z);
    
    val = phi(C,Z) + trace(M'*T_Z) + rho/2*norm(Y - T_Z - P,'fro')^2;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function grad_Z = grad(C,Z,Y,M,Q,rho)
% Compute the gradient of the augmented Lagrangian
%   Z are 3D objects; M, C, Y, Q are matrices, Q of dimension n x n
    T_Z= get_T(Z);
    n = size(Z, 1);
    p = size(Z, 3) - 1;
    P=kron(eye(p+1),Q);
    grad_Z = grad_phi(C, Z) +  get_D_short(M + rho*(T_Z + P - Y), n, p);
    
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

function Y_truncated = proj_C2(Y)
%  Function to project onto the cone of positive definite matrices
    
    Y = symm(Y);
    [V, D] = eig(Y); %D contains eigenvalues; Columns of V contain corresponding eigenvectors
    d = diag(D);
    w1 = ones(size(V,1),1);
    w1(d<=0)=0;
    w = logical(w1);
    V1 = V(:,w);
    d1 = d(w);
    Y_truncated = symm((V1*diag(d1))*V1');
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function boolean1 = check_armijo(obj_old,C,Z,Y,M,Q,Zn,gradZ,alpha,rho,t)
% Checking che armijo conditions of the augmented Lagrangian wrt Z
    
    p=size(Z,3)-1;
    val=0;
    for k=1:p+1
        val=val+trace(gradZ(:,:,k)'*(Zn(:,:,k)-Z(:,:,k)));
    end
    
    if  cost(C,Zn,Y,M,Q,rho) <= obj_old+alpha*t*val
        boolean1=true;
    else 
        boolean1=false;
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

function [Omega, h, S_short, L_short, cost_p, cost_d] = compute_costs_and_primal_variables(C,Z,gamma,Q,verb)
% Compute the primal variables and the primal and dual costs    
% 
% Output:
%
% Omega   sparsity pattern 
% h       number of latent vriables
% S       Sparse matrix 3D array
% L       low rank matrix
% cost_p  cost primal
% cost_d  cost dual

    
    n = size(Z,1);
    p = size(Z,3)-1;
    
    T_Z = get_T(Z);
    W = get_W(C, Z);
    
    %% Compute Delta=S-L
    
    Delta = (C + T_Z)\(sparse([W, zeros(n,n*p); zeros(n*p,n), zeros(n*p,n*p) ])/(C + T_Z));
    Delta = symm(Delta);
    my_X = get_D_short(Delta, n, p); % ?????
        
    %% Restriction to low rank dimension
    [U D] = eig(kron(eye(p+1),Q) + T_Z);
    d = diag(D);
    w1 = zeros(size(U,1),1);
    w1(abs(d) <= 1e-6) = 1; % before 10^-6
    w = logical(w1);
    V = U(:,w);
    h = size(V, 2);  
    
    %% Sparsity pattern Omega
    Omega = ones(n,n);
    Zabs = abs(Z);
    Zabs_matrix = sum(Zabs, 3);
    Zabs_matrix = Zabs_matrix + Zabs_matrix'; 
    Omega(Zabs_matrix  - gamma < -1e-8) = 0;
    [I J] = find(Omega == 0); % Collect the indices of the sparsity pattern
   
    %% Computing the low-rank and sparse parts
    H_dummy = ones(h,h);
    [IH, JH] = find(triu(H_dummy) == 1); % Indices of the upper triangular part of a symmetric matrix
    IJH = sub2ind(size(H_dummy), IH, JH);
    
    
    if h > 0
        my_V = get_short_V(V,n,p,h);
        
        rhs = zeros((p)*length(I) + length(I)/2, 1);
        
        %         lhs = zeros((p)*length(I) + length(I)/2, h^2); % BM: If taking the full size
        lhs = zeros((p)*length(I) + length(I)/2, h*(h+1)/2); % BM: If taking the half size
        
        count = 0;
        Hzeros = zeros(h, h);
        
        for z = 0 : p % covering all the D operations
            
            for ii = 1 : length(I) % same as the length of J
                k = I(ii); % ith index
                l = J(ii); % jth index
                
                %computing left hand side term
                sumkl = Hzeros;
                for y = 0 : (p-z)
                    
                    left_vector = my_V(k, :, y + 1);
                    right_vector = my_V(l, :, y + 1 + z);
                    
                    if z==0
                        sumkl = sumkl  +  left_vector'*right_vector;
                    else
                        sumkl = sumkl  +  2*left_vector'*right_vector;
                    end
                    
                end
                
                sumkl = (sumkl + sumkl')/2; %Taking symmetric part of first term
                % in the scalar product in order
                % to have symmetric H
                sumklvec_full = sumkl(:);
                
                sumklvec = sumklvec_full(IJH); % BM: Half size
                
                if z == 0
                    if k < l
                        count = count + 1;
                        rhs(count, :) =  -my_X(k,l,z+1);
                        lhs(count, :) = sumklvec';
                    end
                else
                    count = count + 1;
                    rhs(count, :) = -my_X(k,l,z+1);
                    lhs(count, :) = sumklvec';
                    
                end
                
            end
            
            
        end
        
        if sum(Omega(:) == 0) > 0 % It means that there are some active equations to be solved
            
            X_sol = lhs\sparse(rhs); % This is indeed the most computationally expensive step
            
            % If taking only half the size
            H = zeros(h, h);
            for jj = 1 : length(IH);
                if IH(jj) == JH(jj)
                    H(IH(jj), JH(jj)) = X_sol(jj);
                else
                    H(IH(jj), JH(jj)) = X_sol(jj)/2; % BM: Half because of the adjoint operation; manually it should be clear.
                    H(JH(jj), IH(jj)) = X_sol(jj)/2;
                end
            end
            
            H  = symm(H); % Numerical puropose.
            
            [U_H, D_H] = eig(H); % This is a small eigenvalue decomposition, should be okay.
            d_H = diag(D_H);
            
            
            if min(d_H) > 0
                if verb=='v'
                    if norm(lhs*X_sol - rhs, 'fro')/norm(rhs, 'fro') <1e-6
                        fprintf('Accurate solution to linear system for computing low-rank part. \n');
                    else
                        fprintf('Computed low-rank part is Positive Definite but inacurate solution to linear system. \n');
                    end
                end
            else
                if verb =='v'
                    fprintf('No accurate solution to linear system but creating a proper low-rank candidate. \n');
                end
                w1 = ones(h,1);
                w1(d_H <= 0) = 0;
                w = logical(w1);
                U_H1 = U_H(:,w);
                d_H1 = d_H(w);
                H = U_H1*diag(d_H1)*U_H1';
                H  = symm(H);
            end
        else
            % Omega is all ones, i.e., there are no equations to be solved
            % equivalently, H can be not well defined.
            H = eye(h);
            
        end
        
        % construction of L and S
        L = (V*H)*V';
        L = symm(L); % Low-rank part
        S = Delta + L; % Sparse part
        
    else % This corresponds to the case h == 0.
        H = 0;
        L = zeros(n*(p+1),n*(p+1)); % Low rank part
        S = Delta; % Sparse part
    end
    
    %% Projecting the sparse part onto the sparsity pattern
    
    S_short = get_D_short(S, n ,p); % Get the 3-d array
    for jj = 1:p+1 % small loop, affordable
        S_short(:, :, jj) = Omega.*S_short(:, :, jj); % Project back the sparsity pattern
    end
    
    %% Final low rank part
    L_short = get_D_short(L, n ,p); % Get the 3-d array
    
    %% Primal and dual cost evaluation
    
    term1 = -log(det(get_first_block(Delta,n)));
    term2 = (C(:)'*Delta(:));
    term3 = get_hW(S_short,gamma); 
    term4 = trace(kron(eye(p+1),Q)*L); 
    cost_p = term1 + term2 + term3 + term4;
    cost_d = log(det(W)) + n;
    
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


function B = get_first_block(A,n_var)
% Function to extract the first block of a matix of size n(p+1) x n(p+1)
    n = n_var;
    B = A(1:n,1:n);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function S = get_short_V(V, n, p, n_latent)
% From 3D to a block row matrix with diemsnions n , l, p+1
    l = n_latent; % number of latent variables.
    S=zeros(n,l,(p+1));
    for i=1:p+1
        S(:,:,i)=V((i-1)*n+1:i*n,:);
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
