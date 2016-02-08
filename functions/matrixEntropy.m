function pX=matrixEntropy(X)
%% compute the entropy of a matrix

pX=X.*log(X);
pX(isnan(pX))=0;
pX=-sum(pX(:));
end