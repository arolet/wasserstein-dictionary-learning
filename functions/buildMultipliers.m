function [multiplyK, multiplyKt]=buildMultipliers(K,gamma,options,sizeX)
%% Prepare the multipliers for Wasserstein Dictionary Learning
% Fonction that creates handlers for multiplication by K and K^T for
% computing a regularized wasserstein distance
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


if strcmp(options.Kmultiplication,'symmetric')
    multiplyK=@(M)K*M;
    multiplyKt=multiplyK;
elseif strcmp(options.Kmultiplication,'asymmetric')
    multiplyK=@(M)K*M;
    if options.bigMatrices
        multiplyKt=@(M)K'*M;
    else
        Kt=K';
        multiplyKt=@(M)Kt*M;
    end
elseif strcmp(options.Kmultiplication,'convolution')
    filters=cell(1,numel(K.kernelSize));
    for dimIndex=1:numel(K.kernelSize)      % A gaussian convolution can be done one dimension at a time
        kSize=ceil(K.kernelSize(dimIndex)/2);
        if dimIndex==1
            filters{dimIndex}=exp(-((-kSize:kSize).^2)/gamma)';
        else
            filters{dimIndex}=reshape(exp(-((-kSize:kSize).^2)/gamma),[ones(1,dimIndex-1), 2*kSize+1]);
        end
    end
    multiplyK=@(M)reshape(gaussianConv(reshape(M,[K.grid_dimensions, sizeX(2)]),filters),sizeX);
    multiplyKt=multiplyK;
else
    error('Unknown multiplication type')
end
end