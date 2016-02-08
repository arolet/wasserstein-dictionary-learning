function M=gaussianConv(M,filters)

for i=1:numel(filters)
    M=imfilter(M,filters{i},'same','replicate');
end

end