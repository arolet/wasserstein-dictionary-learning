function options=checkOptionsWasserteinProjection(options)
%% Check the optional inputs of Wasserstein projection
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

if ~isfield(options,'t0')||isempty(options.t0)
    options.t0=1;
end
if ~isfield(options,'alpha')
    options.alpha=[];
end
if ~isfield(options,'beta')||isempty(options.beta)
    options.beta=.8;
end

if ~isfield(options,'verbose')||isempty(options.verbose)
    options.verbose=0;
end

if ~isfield(options,'dual_descent_stop')||isempty(options.dual_descent_stop)
    options.dual_descent_stop=1e-5;
end
if ~isfield(options,'bigMatrices')||isempty(options.bigMatrices)
    options.bigMatrices=false;
end
if ~isfield(options,'weights')
    options.weights=[];
end
if isfield(options,'GPU')&&options.GPU
    options.createZeroArray=@(d)zeros(d,'gpuArray');
else
    options.createZeroArray=@(d)zeros(d);
end
end