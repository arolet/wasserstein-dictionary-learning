function [H, obj,gradient]=accelerated_gradient(H0,computeObjGrad,proximal,stop,t0,alpha,beta,verbose,sumFunction,computeNorm)
%% Accelerated gradient
% Function that computes the minimum of a convex function using an
% accelerated gradient with the update described in:
%   Chambolle, A., & Dossal, C. (2015). On the convergence of the iterates
%   of the gfast iterative shrinkage/thresholding algorithmh. Journal of
%   Optimization Theory and Applications, 166(3), 968-982.
%

if ~exist('t0','var')||isempty(t0)
    t0=1;
end
if ~exist('alpha','var')||isempty(alpha)
    alpha=.5;
end
if ~exist('beta','var')||isempty(beta)
    beta=0.8;
end

if ~exist('verbose','var')||isempty(verbose)
    verbose=0;
end

if ~exist('stop','var')||isempty(stop)
    stop=1e-5;
end

if ~exist('sumFunction','var')||isempty(sumFunction)
    sumFunction=@(x,t,y)x+t*y;
end

if ~exist('computeNorm','var')||isempty(computeNorm)
    computeNorm=@(matr)gather(norm(matr,'fro'));
end


H=H0;
[objective, gradient, gradNorm]=computeObjGrad(H);


obj=gather(objective);
checkObj=obj;
niter=0;
t=t0;
numberEval=1;

Glast=H;
Hlast=H;
if verbose
    display(sprintf('\tDual iteration : %d Objective : %f, Current step size %f',niter,obj(end),t));
    display(sprintf('\t\tGradnorm=%f stop=%f',gradNorm,stop));
end
% display(['    Dual initialization, time taken : ', num2str(time),' s']);

tol=stop*(1+computeNorm(H));

while gradNorm>tol
    niter=niter+1;
    last=obj(end);
    prevt=t;
    [t, objective, H, ~, ~]=backtrack(@(G)computeObjGrad(G),last,H,gradient,-alpha*gradNorm,beta,t,sumFunction);
    numberEval=numberEval+log(t/prevt)/log(beta)+1;
    
    G=proximal(H);
    H=sumFunction(G,(niter-2)/(niter+1),sumFunction(G,-1,Glast));
    [objective, gradient, gradNorm]=computeObjGrad(H);
    
    obj=[obj gather(objective)];
    Glast=G;
    numberEval=numberEval+1;
    
    tol=stop*(1+computeNorm(H));
    
    if mod(niter-1,20)==0
        if verbose
            display(sprintf('\tDual iteration : %d Objective : %f, Current step size %f',niter,obj(end),t));
            display(sprintf('\t\tGradnorm=%f tol=%f',gradNorm,tol));
        end
        Hlast=H;
    end
    
    if checkObj<obj(end)
        niter=0;
        H=Hlast;
        [objective, gradient, gradNorm]=computeObjGrad(H);
        obj=[obj gather(objective)];
    end
    checkObj=obj(end);
    
    t=min([t/sqrt(beta), t0]);
        
    
   
    
end
if verbose
    display(sprintf('\tDual iteration : %d Objective : %f, Current step size %f',niter,obj(end),t));
    display(sprintf('\t\tGradnorm=%f stop=%f',gradNorm,tol));
%     display(['    Mean eval time : ', num2str(time/numberEval), ' Number of evaluations : ', num2str(numberEval)]);
end



end
