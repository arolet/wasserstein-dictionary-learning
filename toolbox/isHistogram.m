function isHist=isHistogram(X)
%% Check whether columns of a matrix are histograms
% Checks whether all columns of X are real valued, non-negative and sum to
% one

isHist=~any(isnan(X(:))|isinf(X(:)))&&allPositive(X)&&sumsToOne(X);

end

function allPos=allPositive(X)
allPos=isreal(X)&&all(X(:)>=0);
end

function sums=sumsToOne(X)
n=size(X,1);

sums=all(abs(1-sum(X,1))<eps(1.0)*n*2);
end