function [lambda, H,obj, gradient]=wasserstein_NMF_coefficient_step(X,K,D,gamma,rho,H,options,isHist)
%% Wasserstein projection on the non-negative span of D
% This function solves 
%
%               min_{lambda} W_gamma(X,D*lambda) + rho*E(lambda)
%
% Inputs:
%       - X, the input data (n-by-m matrix)
%       - K, exp(-M/gamma) where M is the ground metric (n-by-t matrix). If
%       K is symmetric, consider using options.Kmultiplication='symmetric'.
%       If M is a squared Euclidean distance on a regular grid, consider
%       using options.Kmultiplication='convolution' (see optional inputs 
%       for more information)
%       - D, the dictionary (t-by-k matrix)
%       - gamma, the regularization parameter of Wasserstein (scalar)
%       - rho, the regularization parameter for lambda (scalar)
% 
% Optional inputs:
%       - H, the initial dual variables (t-by-m matrix)
%       - options, an option structure:
%           * options.verbose: if true the function displays information at
%           each iteraion. Default false
%           * options.t0: the initial step size for gradient updates.
%           Default 1
%           * options.alpha: parameter for the backtracking line search.
%           Default 0
%           * options.beta: parameter for the backtracking line search.
%           Default .8
%           * options.dual_descent_stop: stopping criterion. Default 1e-5
%           * options.bigMatrices: if true some operations are made so that
%           less memory is used. It may affect efficiency. Default false.
%           * options.weights: weights on the wasserstein distance (1-by-m
%           matrix). Defaults to empty which is equivalent to ones(1,m).
%           * options.Kmultiplication: A string which defines how to
%           multiply by K or K':
%               + 'symmetric': K=K' so only multiply by K
%               + 'asymmetric': K~=K', if options.bigMatrices is false then
%               K' is precomputed
%               + 'convolutional': the ground metric is squared Euclidean 
%               distance on a regular grid, so multiplication is by
%               exp(-gamma*M) is equivalent to a convolution with a
%               gaussian kernel, which can be computed dimensions one at a
%               time. K should then be a structure:
%                   o K.grid_dimensions: the dimensions of the original
%                   data
%                   o K.kernelSize: the size of the gaussian kernel
%
% Outputs:
%       - lambda: the minimizer (k-by-m matrix)
%       - H: the dual optimal variable (t-by-m matrix)
%       - obj: the optimum (scalar)
%       - gradient: the gradient of the dual on H (t-by-m matrix)
%
% This code is (c) Antoine Rolet 2016.
%
% This Source Code Form is subject to the terms of the Mozilla Public 
% License, v. 2.0. If a copy of the MPL was not distributed with this file,
% You can obtain one at http://mozilla.org/MPL/2.0/.
% 
% This Source Code is distributed on an "AS IS" basis, WITHOUT WARRANTY OF
% ANY KIND, either express or implied. See the License for the specific
% language governing rights and limitations under the License.
%
% The Initial Developers of the Original Code is Antoine Rolet.

options = checkOptionsWasserteinProjection(options);

n=size(X,1);
m=size(X,2);
if strcmp(options.Kmultiplication,'symmetric')
    if ~((size(K,2)==n)&&(sum(sum(abs(K-K')))/sum(sum(K))<1e-3))
        options.Kmultiplication='asymmetric';
    end
end
if strcmp(options.Kmultiplication,'asymmetric')
    n=size(K,2);
end

if ~exist('isHist','var')||~isHist
    if ~isHistogram(X)
        error('Input vectors are not histograms. Check that they are non-negative, sum to one and do not contain NaN of inf')
    end
end



[multiplyK, multiplyKt]=buildMultipliers(K,gamma,options,size(X));

% Precompute the entropy of X
if ~isfield(options,'pX')||isempty(options.pX)
    pX=matrixEntropy(X);
else
    pX=options.pX;
end


% Function that compute the entropy part of the objective and its gradient
    function [obj, grad]=entropyObj(H)
        expD=exp(-D'*H/rho);
        if any(isinf(expD(:)))
            expD(isinf(expD))=max(expD(~isinf(expD)));
        end
        sumE=sum(expD,1);
        obj=rho*sum(log(sumE));
        grad=-bsxfun(@rdivide,D*expD,sumE);
    end

% Function handler that compute the Wasserstein part of the objective and its gradient
if isempty(options.weights)
       wassersteinObj=@(H)computeWassersteinLegendre(X,H,gamma,pX,multiplyK,multiplyKt);
else
       wassersteinObj=@(H)computeWassersteinLegendreWeighted(X,H,gamma,pX,multiplyK,multiplyKt,options.weights);
end


% Function that computes the objective and its gradient
    function [obj, grad, gradNorm]=computeObj(H)
        % Compute the Wasserstein part of the gradient and objective
       [obj, grad]=wassersteinObj(H);        
        
        % Compute the entropy part of the gradient
        [objE, gradE]=entropyObj(H);

        obj=obj+objE;
        grad=grad+gradE;
        
        % Comput the norm
        gradNorm=gather(norm(grad,'fro'));
        grad=grad/gradNorm;

    end

% Initialize
if ~exist('H','var') || isempty(H)
    H=options.createZeroArray([n,m]);
end

% Launch the solver to get the dual optimizer
[H, obj, gradient]=accelerated_gradient(H,@(G)computeObj(G),@(G)G,options.dual_descent_stop,options.t0,options.alpha,options.beta,options.verbose);
obj=-obj;


% Recover lambda from the dual variable
lambda=exp(-D'*H/rho);
lambda=bsxfun(@rdivide,lambda,sum(lambda));

end
