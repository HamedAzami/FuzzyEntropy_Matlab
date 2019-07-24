function entr = FuzEn_MFs(ts, m, mf, rn, local,tau)
% This function calculates fuzzy entropy (FuzEn) of a univariate
% signal ts, using different fuzzy membership functions (MFs)%       
%       Inputs:
%               ts     --- time-series  - a vector of size 1 x N (the number of sample points)
%               m      --- embedding dimension
%               mf     --- membership function, chosen from the follwing
%                          'Triangular', 'Trapezoidal', 'Z_shaped', 'Bell_shaped',
%                          'Gaussian', 'Constant_Gaussian', 'Exponential'
%               rn      --- threshold r and order n (scalar or vector based upon mf)
%                          scalar: threshold
%                          vector: [threshold r, order n]
%               local  --- local similarity (1) or global similarity (0)
%               tau    --- time delay
% 
%  Ref.:
%  [1]  	H. Azami, P. Li, S. Arnold, J. Escudero, and A. Humeau-Heurtier, "Fuzzy Entropy Metrics for the Analysis 
%       of Biomedical  Signals: Assessment and Comparison", IEEE ACCESS, 2019.
% 
% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% If you use the code, please make sure that you cite Reference [1]
% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%
% Authors:      Peng Li and Hamed Azami
%  Emails: pli9@bwh.harvard.edu and hmd.azami@gmail.com
%
%
% For Cr=0.1, the threshold r and order n should be selected as follows: 
%
% 'Triangular'------      rn=0.3
% 'Trapezoidal'------     rn=0.1286
% 'Z_shaped'------      rn=0.1309
% 'Bell_shaped'------    rn=[0.1414 2] % means r=0.1414 and n=2
% 'Bell_shaped'------     rn=[0.1732 3] % means r=0.1732 and n=3
% 'Gaussian'------         rn=0.1253
% 'Constant_Gaussian'------ rn=0.0903
% Exponential------        rn=[0.0077 3] % means r=0.0077 and n=3
% Exponential------       rn=[0.0018 4] % means r=0.0018 and n=4
%
%
% Example x=rand(1,1000);FuzEn_MFs(x, 2, 'Bell_shaped', [0.1414*std(x) 2], 0,1)


if nargin == 5, tau = 1; end
if nargin == 4, local = 0; tau=1; end
if nargin == 3, rn=0.2*std(ts);local = 0; tau=1; end

% parse inputs
narginchk(6, 6);
N     = length(ts);

% normalization
%ts = zscore(ts(:));

% reconstruction
indm = hankel(1:N-m*tau, N-m*tau:N-tau);    % indexing elements for dim-m
indm = indm(:, 1:tau:end);
ym   = ts(indm);

inda = hankel(1:N-m*tau, N-m*tau:N);        % for dim-m+1
inda = inda(:, 1:tau:end);
ya   = ts(inda);

if local
    ym = ym - mean(ym, 2)*ones(1, m);
    ya = ya - mean(ya, 2)*ones(1, m+1);
end

% inter-vector distance
% if N < 1e4
    cheb = pdist(ym, 'chebychev'); % inf-norm
    cm   = feval(mf, cheb, rn);
    
    cheb = pdist(ya, 'chebychev');
    ca   = feval(mf, cheb, rn);
% else % allocating space takes longer than loop, still debugging
%     for k = N-m*tau:-1:1
%         ymrow     = ym(k, :);
%         xmrowmt   = ones(N-m*tau, 1)*ymrow;
%         dmtemp    = max(abs(ym - xmrowmt), [], 2)';
%         cm(k)     = sum(feval(mf, dmtemp, cr));
%         
%         yarow     = ya(k, :);
%         xarowmt   = ones(N-m*tau, 1)*yarow;
%         datemp    = max(abs(ya - xarowmt), [], 2)';
%         ca(k)     = sum(feval(mf, datemp, cr));
%     end
% end


% output
entr = -log(sum(ca) / sum(cm));
end

% membership functions
function c = Triangular(dist, rn)
c = zeros(size(dist));
c(dist <= rn) = 1 - dist(dist <= rn) ./ rn;
end

function c = Trapezoidal(dist, rn)
c = zeros(size(dist));
c(dist <= rn) = 1;
c(dist <= 2*rn & dist > rn) = 2 - dist(dist <= 2*rn & dist > rn) ./ rn;
end

function c = Z_shaped(dist, rn)
c  = zeros(size(dist));
r1 = dist <= rn;
r2 = dist >  rn     & dist <= 1.5*rn;
r3 = dist >  1.5*rn & dist <= 2*rn;
c(r1) = 1;
c(r2) = 1 - 2.*((dist(r2)-rn)./rn).^2;
c(r3) = 2.*((dist(r3)-2*rn)./rn).^2;
end

function c = Bell_shaped(dist, rn)
c = 1 ./ (1 + abs(dist ./ rn(1)).^(2*rn(2)));
end

function c = Gaussian(dist, rn)
c = exp(-(dist./(sqrt(2)*rn)).^2);
end

function c = Constant_Gaussian(dist, rn)
c = ones(size(dist));
c(dist>rn) = exp(-log(2).*((dist(dist>rn) - rn)./rn).^2);
end

function c = Exponential(dist, rn)
c = exp(-dist.^rn(2) ./ rn(1));
end