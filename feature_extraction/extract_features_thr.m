function [feature_map, img_samples] = extract_features_thr(image, pos, scales, features, gparams, extract_info, scales_p)

% Sample image patches at given position and scales. Then extract features
% from these patches.
% Requires that cell size and image sample size is set for each feature.

if ~iscell(features)
    error('Wrong input');
end

if ~isfield(gparams, 'use_gpu')
    gparams.use_gpu = false;
end
if ~isfield(gparams, 'data_type')
    gparams.data_type = zeros(1, 'single');
end
if nargin < 6
    % Find used image sample size
    extract_info = get_feature_extract_info(features);
end

num_features = length(features);

if length(scales)>2 
    num_scales = length(scales)/2+length(scales_p);
    scas_x=scales(1:length(scales)/2);
    scas_y=scales((length(scales)/2+1):length(scales));
    scas_x_p=scales_p(1:length(scales_p)/2);
    scas_y_p=scales_p((length(scales_p)/2+1):length(scales_p));
elseif length(scales)==2
    num_scales=1;
    scas_x=scales(1);
    scas_y=scales(2);
elseif length(scales)==1
    num_scales = 1;
end
num_sizes = length(extract_info.img_sample_sizes);

% Extract image patches
img_samples = cell(num_sizes,1);
for sz_ind = 1:num_sizes
    img_sample_sz = extract_info.img_sample_sizes{sz_ind};
    img_input_sz = extract_info.img_input_sizes{sz_ind};
    img_samples{sz_ind} = zeros(img_input_sz(1), img_input_sz(2), size(image,3), num_scales, 'uint8');
    for scale_ind = 1:num_scales
        if num_scales == 1
            if length(scales) == 1
            img_scales=[img_sample_sz(1)*scales,img_sample_sz(2)*scales];     
            img_scales=fliplr(img_scales);
            img_samples{sz_ind}(:,:,:,scale_ind) = sample_patch(image, pos, img_scales, img_input_sz, gparams);                
            elseif length(scales) == 2                
            img_scales=[img_sample_sz(1)*scas_x(scale_ind),img_sample_sz(2)*scas_y(scale_ind)];     
            img_scales=fliplr(img_scales);
            img_samples{sz_ind}(:,:,:,scale_ind) = sample_patch(image, pos, img_scales, img_input_sz, gparams);
            end
        elseif scale_ind < length(scales)/2+1
            img_scales=[img_sample_sz(1)*scas_x(scale_ind),img_sample_sz(2)*scas_y(scale_ind)];     
            img_scales=fliplr(img_scales);
            img_samples{sz_ind}(:,:,:,scale_ind) = sample_patch(image, pos, img_scales, img_input_sz, gparams);
        elseif scale_ind < length(scales)/2+length(scales_p)/2+1
            scas_y_fli=fliplr(scas_y_p);
            img_scales=[img_sample_sz(1)*scas_x_p(scale_ind-length(scales)/2),img_sample_sz(2)*scas_y_fli(scale_ind-length(scales)/2)];
            img_scales=fliplr(img_scales);
            img_samples{sz_ind}(:,:,:,scale_ind) = sample_patch(image, pos, img_scales, img_input_sz, gparams);
        elseif scale_ind < length(scales)/2+length(scales_p)+1
            scas_x_fli=fliplr(scas_x_p);
            img_scales=[img_sample_sz(1)*scas_x_fli(scale_ind-(length(scales)/2+length(scales_p)/2)),img_sample_sz(2)*scas_y_p(scale_ind-(length(scales)/2+length(scales_p)/2))];           
            img_scales=fliplr(img_scales);
            img_samples{sz_ind}(:,:,:,scale_ind) = sample_patch(image, pos, img_scales, img_input_sz, gparams);
            
%         elseif scale_ind<length(scales)/2*3+2
%             img_scales=[img_sample_sz(1)*scas_x((length(scales)/2+1)/2)*2,img_sample_sz(2)*scas_y((length(scales)/2+1)/2)/2];           
%             img_scales=fliplr(img_scales);
%             img_samples{sz_ind}(:,:,:,scale_ind) = sample_patch(image, pos, img_scales, img_input_sz, gparams);           
%         elseif scale_ind<length(scales)/2*3+3
%             img_scales=[img_sample_sz(1)*scas_x((length(scales)/2+1)/2)*2.5,img_sample_sz(2)*scas_y((length(scales)/2+1)/2)/2];           
%             img_scales=fliplr(img_scales);
%             img_samples{sz_ind}(:,:,:,scale_ind) = sample_patch(image, pos, img_scales, img_input_sz, gparams);            
        end    
    end
end
%     save('C:\Users\Dominik\Desktop\lea\img_sample\img_sample.mat','img_samples');
% Find the number of feature blocks and total dimensionality
num_feature_blocks = 0;
total_dim = 0;
for feat_ind = 1:num_features
    num_feature_blocks = num_feature_blocks + length(features{feat_ind}.fparams.nDim);
    total_dim = total_dim + sum(features{feat_ind}.fparams.nDim);
end

feature_map = cell(1, 1, num_feature_blocks);

% Extract feature maps for each feature in the list
ind = 1;
for feat_ind = 1:num_features
    feat = features{feat_ind};
%     gparams.cell_size = feat.fparams.cell_size;
    
    % get the image patch index
    img_sample_ind = cellfun(@(sz) isequal(feat.img_sample_sz, sz), extract_info.img_sample_sizes);
    
    % do feature computation
    if feat.is_cell
        num_blocks = length(feat.fparams.nDim);
        feature_map(ind:ind+num_blocks-1) = feat.getFeature(img_samples{img_sample_ind}, feat.fparams, gparams);
    else
        num_blocks = 1;
        feature_map{ind} = feat.getFeature(img_samples{img_sample_ind}, feat.fparams, gparams);
    end
    
    ind = ind + num_blocks;
end
              
% Do feature normalization
if ~isempty(gparams.normalize_power) && gparams.normalize_power > 0
    if gparams.normalize_power == 2
        feature_map = cellfun(@(x) bsxfun(@times, x, ...
            sqrt((size(x,1)*size(x,2))^gparams.normalize_size * size(x,3)^gparams.normalize_dim ./ ...
            (sum(reshape(x, [], 1, 1, size(x,4)).^2, 1) + eps))), ...
            feature_map, 'uniformoutput', false);
    else
        feature_map = cellfun(@(x) bsxfun(@times, x, ...
            ((size(x,1)*size(x,2))^gparams.normalize_size * size(x,3)^gparams.normalize_dim ./ ...
            (sum(abs(reshape(x, [], 1, 1, size(x,4))).^gparams.normalize_power, 1) + eps)).^(1/gparams.normalize_power)), ...
            feature_map, 'uniformoutput', false);
    end
end
if gparams.square_root_normalization
    feature_map = cellfun(@(x) sign(x) .* sqrt(abs(x)), feature_map, 'uniformoutput', false);
end

end