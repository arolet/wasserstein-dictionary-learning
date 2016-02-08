function [H,obj, gradient]=linear_projected_gradient_descent(H,computeObj,projection,stop,t0,alpha,beta,verbose)


if ~exist('t0','var')||isempty(t0)
    t0=1;
end
if ~exist('alpha','var')||isempty(alpha)
    alpha=0;
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

[objective, gradient, ~]=computeObj(H);
obj=gather(objective);
last=Inf;
niter=0;
t=t0;
numberEval=1;
if verbose
    display(['    Dual initialization, Objective : ', num2str(obj(end)),' Current step size ', num2str(t)]);
end
while abs(obj(end)-last)>stop*(1+abs(obj(end)))*t || abs(obj(end)-last)>1
    last=obj(end);
    [projected, gradNorm]=projection(gradient);
    prevt=t;
    [t, objective, H, gradient, ~]=backtrack(@(x)computeObj(x),last,H,projected,-alpha*gradNorm,beta,t,@(a,coeff,b)a+coeff*b);
    numberEval=numberEval+log(t/prevt)/log(beta)+1;
    obj=[obj gather(objective)];
    
    if verbose && mod(niter,20)==0
        display(['    Dual iteration : ', num2str(niter+1), ' Objective : ', num2str(obj(end)),' Current step size ', num2str(t)]);
    end
    
    t=t/sqrt(beta);
    niter=niter+1;
        
    
   
    
end
if verbose
    display(['    Finished, number of iterations : ', num2str(niter), ' Objective : ', num2str(obj(end)),' Current step size ', num2str(t)]);
%     display(['    Mean eval time : ', num2str(time/numberEval), ' Number of evaluations : ', num2str(numberEval)]);
end






end