function [t, obj, H, grad, gradNorm]=backtrack(phi,f,U,dir,alpha,beta,t,sumFunction)
%% Backtracking line search
% Performs a backtracking line search on convex function phi with initial
% position U in the decrease direction dir. The initial value is f.

if nargin<6
    t=1;
end
H=sumFunction(U,-t,dir);
[obj, grad, gradNorm]=phi(H);
objs=obj;
test=obj+gradNorm;
while ~isreal(obj)||~isreal(test)||isnan(test)||isinf(test)||obj>f+alpha*t
    t=beta*t;
    H=sumFunction(U,-t,dir);
    [obj, grad, gradNorm]=phi(H);
    objs=[objs obj];
    test=obj+gradNorm;
end

end