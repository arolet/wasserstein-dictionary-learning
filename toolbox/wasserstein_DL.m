function [D, lambda, objectives, HD, Hlambda]=wasserstein_DL(X,p,M,gamma,rhoL,rhoD,options,initialValues)
%% Dictionary learning with regularized wasserstein cost.
% This function solves
%
%       min_{lambda,D} W_gamma(X,D*lambda) + rhoL*E(lambda) + rhoD*E(D)
%
% where E is the sum of entropy on the columns. If rho_1, rho_2 > 0, 
% this performs NMF. Otherwise D or lambda can have negative values, but
% not D*lambda.
%
% Inputs:
%       - X, the input data (n-by-m matrix)
%       - p, the number of dictionary elements to learn (scalar)
%       - M, the ground metric (n-by-t matrix). If M is symmetric, consider
%       using options.Kmultiplication='symmetric'. If M is a squared 
%       Euclidean distance on a regular grid, consider using 
%       options.Kmultiplication='convolution' (see optional inputs for more
%       information)
%       - gamma, the regularization parameter of Wasserstein (scalar)
% 
% Optional inputs:
%       - rhoL, the regularization parameter for lambda (scalar). Default 0
%       - rhoD, the regularization parameter for D (scalar). Default 0
%       - options, an option structure:
%           * options.verbose: verbose level, 0 (default) for no output, 1
%           for an output at each outer loop iteration, 2 for additional
%           output at each inner loop iteration.
%           * options.t0: the initial step size for gradient updates.
%           Default 1
%           * options.alpha: parameter for the backtracking line search.
%           Default 0 (backtrackinh stops when the objective decreases).
%           * options.beta: parameter for the backtracking line search.
%           Default .8
%           * options.stop: outer stopping criterion. Default 1e-3
%           * options.D_step_stop: inner stopping criterion for the D step.
%           Default 1e-5
%           * options.lambda_step_stop: inner stopping criterion for the D 
%           step. Default 1e-5
%           * options.bigMatrices: if true some operations are made so that
%           less memory is used. It may affect efficiency. Default false.
%           * options.weights: weights on the wasserstein distance (1-by-m
%           matrix). Defaults to empty which is equivalent to ones(1,m).
%           * options.Kmultiplication: A string which defines how to
%           multiply by K=exp(-gamma*M) or K':
%               + 'symmetric': K=K' so only multiply by K
%               + 'asymmetric': K~=K', if options.bigMatrices is false then
%               K' is precomputed
%               + 'convolutional': the ground metric is squared Euclidean 
%               distance on a regular grid, so multiplication is by
%               exp(-gamma*M) is equivalent to a convolution with a
%               gaussian kernel, which can be computed dimensions one at a
%               time. M should then be a structure:
%                   o M.grid_dimensions: the dimensions of the original
%                   data
%                   o M.kernelSize: the size of the gaussian kernel
%           * options.GPU: the GPU which should be used or 0 (default) for
%           no GPU.
%       - initialValues, a structure containing starting points or
%       initialization method:
%           * initialValues.D: the initial dictionary (t-by-p matrix)
%           * initialValues.isInitLambda: if initialValues.D is present and
%           this is true, then initialValues.D is the initial coefficients 
%           (p-by-m matrix)
%           * initialValues.method: if initialValues.D, this is a string
%           which indicates which initialization method to use. Can be:
%               + 'random' (default): initializes D randomly
%               + 'euclidean' : sets lambda as the result of euclidean NMF
%           
%
% Outputs:
%       - lambda: the optimal coefficitients (p-by-m matrix)
%       - D: the optimal dictionary (t-by-p matrix)
%       - objectives: the sequence of objective values
%       - HD: the dual optimal variable for the D step (t-by-m matrix).
%       - Hlambda: the dual optimal variable (t-by-m matrix)
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






[options]=checkOptions(options);

verbose=options.verbose;
if verbose
    options.verbose=verbose-1;
end

if ~isHistogram(X)
    error('Input vectors are not histograms. Check that they are non-negative, sum to one and do not contain NaN of inf')
end

% Load GPU device if applicable
if options.GPU
    gpuDevice(options.GPU);
end

% Create optimizers
if ~exist('rhoL','var') || isempty(rhoL)
    rhoL=0;
end
if ~exist('rhoD','var') || isempty(rhoD)
    rhoD=0;
end
[optimizeLambda, optimizeD,options, sizeD]=createHandlers(options,X,M,p,gamma,rhoL,rhoD);

% Create or assign starting values
if ~exist('initialValues','var') || isempty(initialValues)
    initialValues=struct();
end
[D, HD, Hlambda]=initialValue(options,optimizeD,X,p,sizeD,initialValues);

clear initialValues;
biglast=0;
bigobj=-1;
objectives=[];
niter=0;


% Loop that optimizes alternatively with D or lambda fixed until
% convergence
while abs(bigobj-biglast)>options.stop*(1+abs(bigobj))
    niter=niter+1;
    
    if verbose
        display('Optimize with respect to lambda');
    end
    clear lambda;
    if options.GPU
        Hlambda=gpuArray(Hlambda);
    end
    
    % Optimize with respect to lambda, D is fixed
    [lambda, Hlambda, objL]=optimizeLambda(D,Hlambda);
    
    if options.bigMatrices          % Free GPU memory if needed
        Hlambda=gather(Hlambda);
    end
    
    % Store objective values
    biglast=bigobj;
    tmpObj=objL(end);
    if rhoD>0
        tmpObj=tmpObj-rhoD*matrixEntropy(D);
    end
    objectives=[objectives, tmpObj];
    
    
    if verbose
        display('Optimize with respect to D');
    end
    if options.GPU
        HD=gpuArray(HD);
    end
    
    clear D;
    % Optimize with respect to D, lambda is fixed
    [D, HD,objD]=optimizeD(lambda,HD);
    
    if options.bigMatrices          % Free GPU memory if needed
        HD=gather(HD);
    end
    
    
    % Store objective values
    tmpObj=objD(end);
    if rhoL>0
        tmpObj=tmpObj-rhoL*matrixEntropy(lambda);
    else
        sumD=sum(abs(D));
        sumD(sumD==0)=1;
        D=bsxfun(@rdivide,D,sumD);
    end
    objectives=[objectives tmpObj];
    
    if verbose
        display(['Outer iteration : ', num2str(niter), ' Objective : ', num2str(objectives(end)), ' gamma=', num2str(gamma)]);
        options.plothandler(objL,objD,objectives,D);
    end
        
    bigobj=objectives(end);
end

if options.GPU
    D=gather(D);
    lambda=gather(lambda);
end

end


%
%
%
%
function options=checkOptions(options)

if ~isfield(options,'stop') || isempty(options.stop)
    options.stop=1e-3;
end

if ~isfield(options,'D_step_stop')|| isempty(options.D_step_stop)
    options.D_step_stop=1e-5;
end

if ~isfield(options,'lambda_step_stop')|| isempty(options.lambda_step_stop)
    options.lambda_step_stop=1e-5;
end

if ~isfield(options,'alpha')|| isempty(options.alpha)
    options.alpha=[];
end
if ~isfield(options,'beta')|| isempty(options.beta)
    options.beta=[];
end

if ~isfield(options,'t0')|| isempty(options.t0)
    options.t0=[];
end

if ~isfield(options,'verbose')|| isempty(options.verbose)
    options.verbose=0;
end
if ~isfield(options,'GPU')|| isempty(options.GPU)
    options.GPU=0;
end

if ~isfield(options,'bigMatrices')|| isempty(options.bigMatrices)
    options.bigMatrices=0;
end

if options.GPU
    options.createZeroArray=@(siz)zeros(siz,'gpuArray');
else
    options.createZeroArray=@(siz)zeros(siz);
end

if ~isfield(options,'plothandler')|| isempty(options.plothandler)
    options.plothandler=@(obj1,obj2,obj3,D)1;
end

end


%
%
%
%
function [optimizeLambda, optimizeD, options, sizeD]=createHandlers(options,X,M,p,gamma,rhoL,rhoD)
% Create the solvers for sub-problems on either D or lambda




if options.GPU
    X=gpuArray(X);
end


if ~isfield(options,'Kmultiplication') || (~strcmp(options.Kmultiplication,'convolution')&&~strcmp(options.Kmultiplication,'graphLapalacian'))
    if options.GPU
        K=exp(-gpuArray(M)/gamma);
    else
        K=exp(-M/gamma);
    end
    K(K<1e-200)=1e-200;
    options.convolution=false;
    if (size(K,1)==size(K,2))&&(sum(sum(abs(K-K')))/sum(sum(K))<1e-5)
        options.Kmultiplication='symmetric';
        sizeD=size(X);
    else
        options.Kmultiplication='asymmetric';
        sizeD=[size(K,2), size(X,2)];
    end
elseif strcmp(options.Kmultiplication,'convolution')
    if prod(M.grid_dimensions)~=size(X,1)
        error('When using a convolution to compute the gradient, M.grid_dimensions should be the grid dimensions of the original data the product of its components should be size(X,1)');
    end
    if numel(M.kernelSize)==1
        M.kernelSize=M.kernelSize*ones(1,numel(M.grid_dimensions));
    end
    if any(M.kernelSize>M.grid_dimensions)
        error('Kernel size sould be smaller than dimension');
    end
    K=M;
    sizeD=size(X);
elseif strcmp(options.Kmultiplication,'graphLapalacian')
    K=M;
    sizeD=size(X);
end


% Precompute the entropy of X
options.pX=matrixEntropy(X);

optionsD=options;
options.dual_descent_stop=options.lambda_step_stop;
optionsD.dual_descent_stop=options.D_step_stop;

% Create solvers depending on options.algorithms
if rhoL
    optimizeLambda=@(D,Hlambda)wasserstein_NMF_coefficient_step(X,K,D,gamma,rhoL,Hlambda,options,1);
else
    optimizeLambda=@(D,Hlambda)wasserstein_DL_coefficient_step(X,K,D,gamma,Hlambda,options,1);
end
if rhoD
    optimizeDcomplete=@(lambda,HD)wasserstein_NMF_dictionary_step(X,K,lambda,gamma,rhoD,HD,optionsD,1);
else
    optimizeDcomplete=@(lambda,HD)wasserstein_DL_dictionary_step(X,K,lambda,gamma,HD,optionsD,1);
end
optimizeD=@(lambda,HD)optimizeDWithUnused(lambda,HD,optimizeDcomplete,p);

end



%
%
%
%
function [D, HD, Hlambda]=initialValue(options,optimizeD,X,p,sizeD,initialValues)


if ~isfield(initialValues,'HD')
    HD=options.createZeroArray(sizeD);
end
if ~isfield(initialValues,'Hlambda')
    Hlambda=options.createZeroArray(sizeD);
end
if ~isfield(initialValues,'D') || isempty(initialValues.D)   % No initial value provided
    if ~isfield(initialValues,'method')||strcmp(initialValues.method,'random')
        D=randomInitializationD(options,sizeD,p);
    elseif strcmp(initialValues.method,'fromData')
        if sizeD(1)~=size(X,1)
            error('Initialization from data is only possible when the dictionary has the same dimension as the input')
        end
        D=randomInitializationFromData(X,options,sizeD,p);
    elseif strcmp(initialValues.method,'kmeans')
        [~, D]=fkmeans(X',p);
        dist=pairwiseDistance(D',X);
        lambda=exp(-dist);
        lambda=bsxfun(@rdivide,lambda,sum(lambda));
        D=initializeFromLambda(lambda,p,optimizeD,X,HD,options);
    elseif strcmp(initialValues.method,'euclidean')
        [~, lambda]=nnmf(X,p);
        lambda=bsxfun(@rdivide,lambda,sum(lambda));
        D=initializeFromLambda(lambda,p,optimizeD,X,HD,options);
    else
        error('initialValues.method should be either \"kmeans\", \"euclidean\" or \"random\"');
    end
else                        % Initial value provided
    if options.GPU
        D=gpuArray(initialValues.D);
    else
        D=initialValues.D;
    end
    
    
    if isfield(initialValues,'isInitLambda') && initialValues.isInitLambda  % Initial value provided is lambda
        D=initializeFromLambda(D,p,optimizeD,X,sizeD,HD,options);
    end
end



end

function [D, HD]=initializeFromLambda(lambda,p,optimizeD,X,HD,options)

if options.GPU
    lambda=gpuArray(lambda);
end
[D, HD, obj, Qind]=optimizeD(lambda,HD);

hasUnused=sum(Qind)<p;
if hasUnused
    % Initialize columns of D that couldn't be optimized
    initQ=randperm(m);
    D(:,~Qind)=X(:,initQ(1:(p-sum(Qind))))+1e-7;
    D=bsxfun(@rdivide,D,sum(D));
end
end

function D=randomInitializationFromData(X,options,sizeD,p)
initIndices=randperm(sizeD(2),p);
D=X(:,initIndices);

if options.GPU
    D=gpuArray(D);
end
end

function [D, HD, obj, Qind]=optimizeDWithUnused(lambda,HD,optimizeD,p)
Qind=sum(abs(lambda),2)>0;
hasUnused=sum(Qind)<p;
if hasUnused                    % Some rows of lambda are always 0, corresponding columns of D cannot be optimized
    
    tmpLambda=lambda(Qind,:);
    
    [Dpart, HD, obj]=optimizeD(tmpLambda,HD);
    D(:,Qind)=Dpart;
    
else
    [D, HD, obj]=optimizeD(lambda,HD);
end
end

function D=randomInitializationD(options,sizeD,p)
D=randi(100000,sizeD(1),p);
D=bsxfun(@rdivide,D,sum(D));
if options.GPU
    D=gpuArray(D);
end
end
