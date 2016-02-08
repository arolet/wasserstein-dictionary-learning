function [obj, grad]=computeWassersteinLegendreWeighted(X,H,gamma,pX,multiplyK,multiplyKt,weights)
%% Legendre transform of Wasserstein distance
% This function computes \sum W^(x_i,h_i*weights_i) and its gradient where
% W^ is the Legendre transform of the regularized wasserstein distance
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

alphaTmp=exp(bsxfun(@rdivide,H,weights)/gamma);
grad=multiplyK(alphaTmp);
if any(grad(:)==0)
    grad(grad==0)=min(grad(grad>0));
end
if any(isinf(grad(:)))
    grad(isinf(grad))=max(grad(~isinf(grad)));
end
obj=(sum(sum(X.*log(grad),1).*weights)+pX)*gamma;
grad=alphaTmp.*(multiplyKt(X./grad));


end