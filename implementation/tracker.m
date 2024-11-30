function results = tracker(params)

%% Initialization
% Get sequence info
[seq, im] = get_sequence_info(params.seq);
params = rmfield(params, 'seq');

if isempty(im)
    seq.rect_position = [];
    [seq, results] = get_sequence_results(seq);
    return;
end

% Init regularizer
if strcmpi(seq.format, 'vot')
    if numel(seq.region) > 4,
        seq.rect8 = round(seq.region(:));
        rect8 = seq.rect8;
        x1 = round(min(rect8(1:2:end)));
        x2 = round(max(rect8(1:2:end)));
        y1 = round(min(rect8(2:2:end)));
        y2 = round(max(rect8(2:2:end)));
        seq.init_rect = round([x1, y1, x2 - x1, y2 - y1]);
        seq.target_mask = single(poly2mask(rect8(1:2:end)-seq.init_rect(1), ...
            rect8(2:2:end)-seq.init_rect(2), seq.init_rect(4), seq.init_rect(3)));
        seq.t_b_ratio = sum(seq.target_mask(:))/prod(seq.init_rect([4,3]));
    else
        r = seq.region(:);
        seq.rect8 = [r(1),r(2),r(1)+r(3),r(2),r(1)+r(3),r(2)+r(4),r(1),r(2)+r(4)];
        seq.target_mask = single(ones(seq.region([4,3])));
        seq.t_b_ratio = 1;
    end
end

% Init position
pos = seq.init_pos(:)';
target_sz = seq.init_sz(:)';
params.init_sz = target_sz;

% Feature settings
features = params.t_features;

% Set default parameters
params = init_default_params(params);

% Global feature parameters
if isfield(params, 't_global')
    global_fparams = params.t_global;
else
    global_fparams = [];
end

global_fparams.use_gpu = params.use_gpu;
global_fparams.gpu_id = params.gpu_id;
global_fparams.augment = 0;

% Define data types
if params.use_gpu
    params.data_type = zeros(1, 'single', 'gpuArray');
else
    params.data_type = zeros(1, 'single');
end
params.data_type_complex = complex(params.data_type);

global_fparams.data_type = params.data_type;

init_target_sz = target_sz;

if size(im,3) == 3
    if all(all(im(:,:,1) == im(:,:,2)))
        is_color_image = false;
    else
        is_color_image = true;
    end
else
    is_color_image = false;
end

if size(im,3) > 1 && is_color_image == false
    im = im(:,:,1);
end

params.use_mexResize = true;
global_fparams.use_mexResize = true;

% Calculate search area and initial scale factor
search_area = prod(init_target_sz * params.search_area_scale);
if search_area > params.max_image_sample_size
    currentScaleFactor = sqrt(search_area / params.max_image_sample_size);
elseif search_area < params.min_image_sample_size
    currentScaleFactor = sqrt(search_area / params.min_image_sample_size);
else
    currentScaleFactor = 1.0;
end

% target size at the initial scale
base_target_sz = target_sz / currentScaleFactor;

% window size, taking padding into account
switch params.search_area_shape
    case 'proportional'
        img_sample_sz = floor(base_target_sz * params.search_area_scale);     % proportional area, same aspect ratio as the target
    case 'square'
        img_sample_sz = repmat(sqrt(prod(base_target_sz * params.search_area_scale)), 1, 2); % square area, ignores the target aspect ratio
    case 'fix_padding'
        img_sample_sz = base_target_sz + sqrt(prod(base_target_sz * params.search_area_scale) + (base_target_sz(1) - base_target_sz(2))/4) - sum(base_target_sz)/2; % const padding
    case 'custom'
        img_sample_sz = [base_target_sz(1)*2 base_target_sz(2)*2];
end

[features, global_fparams, feature_info] = init_features_res(features, global_fparams, is_color_image, img_sample_sz, 'odd_cells');

% Set feature info
img_support_sz = feature_info.img_support_sz;
feature_sz = feature_info.data_sz;
num_feature_blocks = size(feature_sz, 1);

% Get feature specific parameters
feature_extract_info = get_feature_extract_info(features);

% Size of the extracted feature maps
feature_sz_cell = permute(mat2cell(feature_sz, ones(1,num_feature_blocks), 2), [2 3 1]);
filter_sz = feature_sz + mod(feature_sz+1, 2);
filter_sz_cell = permute(mat2cell(filter_sz, ones(1,num_feature_blocks), 2), [2 3 1]);

% The size of the label function DFT. Equal to the maximum filter size
[output_sz, k1] = max(filter_sz, [], 1);
params.output_sz = output_sz;
k1 = k1(1);

% Get the remaining block indices
block_inds = 1:num_feature_blocks;
block_inds(k1) = [];

% Pre-computes the grid that is used for socre optimization
ky = circshift(-floor((filter_sz_cell{1}(1) - 1)/2) : ceil((filter_sz_cell{1}(1) - 1)/2), [1, -floor((filter_sz_cell{1}(1) - 1)/2)]);
kx = circshift(-floor((filter_sz_cell{1}(2) - 1)/2) : ceil((filter_sz_cell{1}(2) - 1)/2), [1, -floor((filter_sz_cell{1}(2) - 1)/2)])';
newton_iterations = params.newton_iterations;
params.small_filter={floor(base_target_sz/4) floor(base_target_sz/6) floor(base_target_sz/8)};
% Construct the Gaussian label function
yf = cell(num_feature_blocks, 1);
for i = 1:num_feature_blocks
    sz = filter_sz_cell{i};
    output_sigma_factor = params.output_sigma_factor(feature_info.feature_is_deep(i)+1);
    output_sigma = sqrt(prod(floor(base_target_sz)))*feature_sz_cell{i}./img_support_sz* output_sigma_factor;
    rg           = circshift(-floor((sz(1)-1)/2):ceil((sz(1)-1)/2), [0 -floor((sz(1)-1)/2)]);
    cg           = circshift(-floor((sz(2)-1)/2):ceil((sz(2)-1)/2), [0 -floor((sz(2)-1)/2)]);
    [rs, cs]     = ndgrid(rg,cg);
    y            = exp(-0.5 * (((rs.^2 + cs.^2) / mean(output_sigma)^2)));


end

params.sz=filter_sz_cell;

if params.use_gpu
    params.data_type = zeros(1, 'single', 'gpuArray');
else
    params.data_type = zeros(1, 'single');
end
params.data_type_complex = complex(params.data_type);

global_fparams.data_type = params.data_type;
% Compute the cosine windows
cos_window = cellfun(@(sz) hann(sz(1)+2)*hann(sz(2)+2)', feature_sz_cell, 'uniformoutput', false);
cos_window = cellfun(@(cos_window) cast(cos_window(2:end-1,2:end-1), 'like', params.data_type), cos_window, 'uniformoutput', false);

% Define initial regularizer
mask_window = cell(1,1,num_feature_blocks);
mask_search_window = ones(round(currentScaleFactor*img_support_sz)) * params.mask_window_min;
target_mask = 1.1*seq.target_mask;
target_mask_range = zeros(2, 2);
for j = 1:2
    target_mask_range(j,:) = [0, size(target_mask,j) - 1] - floor(size(target_mask,j) / 2);
end
mask_center = floor((size(mask_search_window) + 1)/ 2) + mod(size(mask_search_window) + 1,2);
target_h = (mask_center(1)+ target_mask_range(1,1)) : (mask_center(1) + target_mask_range(1,2));
target_w = (mask_center(2)+ target_mask_range(2,1)) : (mask_center(2) + target_mask_range(2,2));
mask_search_window(target_h, target_w) = target_mask;
for i = 1:num_feature_blocks
    mask_window{i} = mexResize(mask_search_window, filter_sz_cell{i}, 'auto');
end
params.mask_window = mask_window;

% Use the translation filter to estimate the scale
nScales = params.number_of_scales;
scale_step = params.scale_step;
scale_exp = (-floor((nScales-1)/2):ceil((nScales-1)/2));
scaleFactors = scale_step .^ scale_exp;

scaleFactors_p= params.scale_step_p .^ (-floor((params.number_of_scales_p-1)/2):ceil((params.number_of_scales_p-1)/2));

if nScales > 0
    min_scale_factor = scale_step ^ ceil(log(max(5 ./ img_support_sz)) / log(scale_step));
    max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));
end

seq.time = 0;
det_sample_pos = pos;
scores_fs_feat = cell(1,1,num_feature_blocks);
template_fs_feat = cell(1,1,num_feature_blocks);
fix_ind=2;

resolu_target=init_target_sz(1)*init_target_sz(2);
if resolu_target>1200
ratio_thresh1=0.45;
ratio_thresh2=0.45;
ratio_thresh3=0.70;
oversize_flag=1;
else
ratio_thresh1=1.1;
ratio_thresh2=1.1;
ratio_thresh3=1.1;
oversize_flag=0;
end
% ratio_thresh1=0.7;
% ratio_thresh2=0.7;
% ratio_thresh3=0.7;

goo_index=0;
gbb_index=0;
gee_index=0;
ggg_index=0;
act_ind=0;
lost_flag=0;
while true
    % Read image
    if seq.frame > 0
        [seq, im] = get_sequence_frame(seq);
        if isempty(im)
            break;
        end
%         im=cv.cvtColor(im,'RGB2GRAY');
        if size(im,3) > 1 && is_color_image == false
            im = im(:,:,1);
        end
    else
        seq.frame = 1;
    end
    
    tic();
    %% Target localization step
    if seq.frame > 1
        global_fparams.augment = 0;
        old_pos = inf(size(pos));
        iter = 1;
        
        %translation search
        while iter <= params.refinement_iterations && any(old_pos ~= pos)
%         while iter <= params.refinement_iterations
             % Extract features at multiple resolutions
             det_sample_pos = pos;
             sample_pos = round(pos);   
             if seq.frame ==2
                 currentScaleFactor =[currentScaleFactor currentScaleFactor];
             end
             
             sample_scale = [currentScaleFactor(1)*scaleFactors currentScaleFactor(2)*scaleFactors];    
             sample_scale_p = [currentScaleFactor(1)*scaleFactors_p currentScaleFactor(2)*scaleFactors_p];
             
          [xl, img_samples] = extract_features_six(im, sample_pos, sample_scale, features, global_fparams, feature_extract_info, sample_scale_p,target_sz);
    
            % Do windowing of features
            xtw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl, cos_window, 'uniformoutput', false);
            
            % Compute the fourier series
            xtf = cellfun(@fft2, xtw, 'uniformoutput', false);
            %xtf = project_sample(xtf, projection_matrix);
            
            scores_fs_sum_handcrafted = 0;
            scores_fs_sum_deep = 0;
            dim_handcrafted = 0;
            dim_deep = 0;
                     
            for k = [k1 block_inds]
                if feature_info.feature_is_deep(k) == 0
                    %                     temp = bsxfun(@times, conj(filter_model_f{k}), xtf{k});
                    %                     scores_fs_feat{k} = gather(sum(temp(:,:,b{k}(1:ceil(params.channel_rate*numel(b{k}))),:), 3))/params.channel_rate;
                    scores_fs_feat{k} = gather(sum(bsxfun(@times, conj(filter_model_f{k}), xtf{k}), 3));
                    scores_fs_feat{k} = resizeDFT2(scores_fs_feat{k}, output_sz);
                    scores_fs_sum_handcrafted = scores_fs_sum_handcrafted +  scores_fs_feat{k};
                    dim_handcrafted = dim_handcrafted + feature_info.dim(k);
                else
                    %                     temp = bsxfun(@times, conj(filter_model_f{k}), xtf{k});
                    %                     scores_fs_feat{k} = gather(sum(temp(:,:,b{k}(1:ceil(0.9*numel(b{k}))),:), 3));
                    scores_fs_feat{k} = gather(sum(bsxfun(@times, conj(filter_model_f{k}), xtf{k}), 3));
                    scores_fs_feat{k} = resizeDFT2(scores_fs_feat{k}, output_sz);
                    scores_fs_sum_deep = scores_fs_sum_deep +  scores_fs_feat{k};
                    dim_deep = dim_deep + feature_info.dim(k);
                end
            end
            
            scores_fs_handcrafted = permute(gather(scores_fs_sum_handcrafted), [1 2 4 3]);
            scores_fs_deep = permute(gather(scores_fs_sum_deep), [1 2 4 3]);
            response_handcrafted = ifft2(scores_fs_handcrafted, 'symmetric');
            response_deep = ifft2(scores_fs_deep, 'symmetric');    
            
            
            [disp_row, disp_col, sind, max_res, response_out] = resp_newton(response_handcrafted, response_deep,...
                scores_fs_handcrafted, scores_fs_deep, newton_iterations, ky, kx, output_sz);
            
      
                        
            max_res_old = max_res;
            
            % Compute the translation vector in pixel-coordinates and round
            % to the closest integer pixel.        
%*******************************************Scal_dou******************************************%            
            if sind < length(sample_scale)/2+1
                tra_sca1=scaleFactors(sind);
                tra_sca2=scaleFactors(sind);
            elseif sind < length(sample_scale)/2+length(sample_scale_p)/2+1
                tra_sca1=scaleFactors_p(sind-length(sample_scale)/2);
                sca_fac_y=circshift(scaleFactors_p,[0,floor(length(scaleFactors_p)/2)]);
                tra_sca2=sca_fac_y(sind-length(sample_scale)/2);                      
            elseif sind < length(sample_scale)/2+length(sample_scale_p)+1
                sca_fac_x=circshift(scaleFactors_p,[0,floor(length(scaleFactors_p)/2)]);
                tra_sca1=sca_fac_x(sind-length(sample_scale)/2-length(sample_scale_p)/2);
                tra_sca2=scaleFactors_p(sind-length(sample_scale)/2-length(sample_scale_p)/2);               
            end
            trans_row_v=[disp_row, disp_col] .* (img_support_sz./output_sz) * currentScaleFactor(1) * tra_sca1;
            trans_col_v=[disp_row, disp_col] .* (img_support_sz./output_sz) * currentScaleFactor(2) * tra_sca2;
            translation_vec = [trans_col_v(1), trans_row_v(2)];%[y x]
            scale_change_factor = [tra_sca1 tra_sca2];
            
            % update position
            old_pos = pos;
            if sum(isnan(translation_vec))
                pos = sample_pos;
            else
                pos = sample_pos + translation_vec;
            end
            
            if params.clamp_position
                pos = max([1 1], min([size(im,1) size(im,2)], pos));
            end
            
            if seq.frame<10
                scale_change_factor=[1 1];
            end

            % Update the scale
            currentScaleFactor = currentScaleFactor .* scale_change_factor;
            
            % Adjust to make sure we are not to large or to small
            currentScaleFactor(find(currentScaleFactor<min_scale_factor))=min_scale_factor;
            currentScaleFactor(find(currentScaleFactor>max_scale_factor))=max_scale_factor;
            iter = iter + 1;
 %*******************************************************************************************************************% 
            if length(currentScaleFactor)>1
                target_sz_1 = base_target_sz * currentScaleFactor(2);
                target_sz_2 = base_target_sz * currentScaleFactor(1);
                target_sz=[target_sz_1(1) target_sz_2(2)];%[h w]
            else
                target_sz = base_target_sz * currentScaleFactor;
            end 
            
    
%*********************************************%      
            if act_ind == 1         
                [match_numr] = state_predict(mob_set{1},result_now,ratio_thresh1,0);
                [match_numb,geee_match,kpts3,kpts4] = state_predict(mob_set{1},search_rig,ratio_thresh2,119);
                [match_numg,gee_match,kpts5,kpts6] = state_predict(mob_one,search_rig,ratio_thresh3,0);
                  
                figure(331);
                hold on;
                x_max(seq.frame)=seq.frame;
                r_max(seq.frame)=match_numr;
                b_max(seq.frame)=match_numb;
                g_max(seq.frame)=match_numg;
                
                plot(x_max,r_max,'-r*'); 
                plot(x_max,b_max,'-b*');
                plot(x_max,g_max,'-g*');
                match_numr=0; 
                match_numb=0;
                match_numg=0;
            end
 %*****************************************************************************************************************%
            if act_ind == 1
                if (r_max(seq.frame)==0 && b_max(seq.frame)==0 && g_max(seq.frame)==0 && lost_flag==0 && seq.frame>20 && r_max(seq.frame-1)==0 && b_max(seq.frame-1)==0 && g_max(seq.frame-1)==0) 
                    %           pos=pos_tru_rel(col_min,:);
                    %           currentScaleFactor=currentScaleFactor1; 
                    if r_max(3)>10 
                    lost_flag=1;
                    lost_frame=seq.frame;
                    lost_pos=pos_2;
                    lost_scla=currentScaleFactor2;
                    end
                end
                
                if ~isempty(geee_match)
                match_pos=match_location(mob_set{1},search_rig,kpts3,kpts4,geee_match,crop_pos);
                
                dis_obs=match_pos-pos;
                our_dis=sqrt(dis_obs(1)^2+dis_obs(2)^2);
                m_max(seq.frame)=our_dis;
                plot(x_max,m_max,'--r*');
                else
                    our_dis=50;
                    m_max(seq.frame)=our_dis;
                    plot(x_max,m_max,'--r*');               
                end 
                %********************%
%                 if r_max(seq.frame)>b_max(seq.frame) && b_max(seq.frame)>50 && our_dis<2 %coupon
                if r_max(seq.frame)>5 && b_max(seq.frame)>5 && (our_dis<5) && (g_max(seq.frame)>0)
                    good_fra=result_now;
                    good_pos=pos;                   
                    good_sca=currentScaleFactor;
                end
                if b_max(seq.frame)>r_max(seq.frame) && b_max(seq.frame)<30 && r_max(3)>60
                    [match_numa,ggg_match,kpts7,kpts8] = state_predict(good_fra,search_rig,ratio_thresh2,187);
                    %////////////////////////////////////////////////
                    
                    if ~isempty(ggg_match)
%                     num_g1=length(ggg_match);
                    match_pos1=match_location(good_fra,search_rig,kpts7,kpts8,ggg_match,crop_pos);
                    pos=match_pos1;
                    currentScaleFactor=good_sca;
                    end
                end
 %********************%
%  if (z_max(seq.frame)> y_max(seq.frame) && z_max(seq.frame-1)> y_max(seq.frame-1) && z_max(seq.frame-2)> y_max(seq.frame-2))

                if lost_flag==1 && exist('good_sca','var')
                    pos=lost_pos;
                    currentScaleFactor=lost_scla;
                    
                    %//////////////////////////////////////////////    
                    [match_numa,ggg_match,kpts7,kpts8] = state_predict(good_fra,search_rig,ratio_thresh2,0);
                    %////////////////////////////////////////////////
                    if ~isempty(ggg_match)
                    num_g1=length(ggg_match);
                    match_pos1=match_location(good_fra,search_rig,kpts7,kpts8,ggg_match,crop_pos);
                    pos=match_pos1;
                    
                    if num_g1 >5
                        pos=match_pos1;
                        currentScaleFactor=good_sca;
                        lost_flag=0;
                        mob_set{1}=good_fra;
                        mob_set{2}=good_fra;
                    end
                    end
                    %/////////////////////////////////////////////
                end
            end
 %****************************%
        end
    end
    
    %% Model update step
    if seq.frame == 1
        
        num_g1=0;
        gap_u=0;
        no_upd=0;
        no_work=0;
        yf_flag=0;
        shadow=0;
%         res_arr(1)=0;
        
        no_train=0;
        response_out=0;
        global_fparams.augment = 1;
        sample_pos = round(pos);
%         [xl, img_samples] = extract_features(im, sample_pos, currentScaleFactor, features, global_fparams, feature_extract_info);
        [xl, img_samples] = extract_features_six(im, sample_pos, currentScaleFactor, features, global_fparams, feature_extract_info, shadow, init_target_sz);        
        
        % do windowing of features
        xlw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl, cos_window, 'uniformoutput', false);
        % compute the fourier series
        xlf = cellfun(@fft2, xlw, 'uniformoutput', false);
        xlf_o_old=xlf;
        
        [filter_model_f, filter_ref_f, channel_selection] = train_filter(xlf, feature_info, yf, seq, params);
        
        filter_model_old_f=filter_model_f;
        filter_model_pre= filter_model_f;
        scal_fla=0;

    else
        % extract image region for training sample
        global_fparams.augment = 1;
        sample_pos = round(pos);
        
        
        if act_ind ==1           
            if r_max(seq.frame)==0 && b_max(seq.frame)==0 && g_max(seq.frame)==0 
                 no_upd=1;
                 if exist('good_sca','var')
                     currentScaleFactor=good_sca;
                 end
            else
                no_upd=0;
            end
        end
        
        

          [xl, img_samples] = extract_features_six(im, sample_pos, currentScaleFactor, features, global_fparams, feature_extract_info, shadow,target_sz);

        % do windowing of features
        xlw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl, cos_window, 'uniformoutput', false);
        % compute the fourier series
        xlf = cellfun(@fft2, xlw, 'uniformoutput', false);
        
        xlf_o_old=xlf;

   
    
    % Update the target size (only used for computing output box)
%     target_sz = base_target_sz * currentScaleFactor;
    if length(currentScaleFactor)>1
    target_sz_1 = base_target_sz * currentScaleFactor(2);
    target_sz_2 = base_target_sz * currentScaleFactor(1);
    target_sz=[target_sz_1(1) target_sz_2(2)];%[h w]
    else
        target_sz = base_target_sz * currentScaleFactor;     
    end    
    
    
    tracking_result.center_pos = double(pos);
    tracking_result.target_size = double(target_sz);
    seq = report_tracking_result(seq, tracking_result);
    seq.time = seq.time + toc();
    if params.visualization
        rect_position_vis = [pos([2,1]) - (target_sz([2,1]) - 1)/2, target_sz([2,1])];
        im_to_show = double(im)/255;
        if size(im_to_show,3) == 1
            im_to_show = repmat(im_to_show, [1 1 3]);
        end
        if seq.frame == 1  %first frame, create GUI
            fig_handle = figure('Name', 'Tracking');
            set(gca, 'Units', 'normalized', 'Position', [0 0 1 1]);
            axis off;
            imagesc(im_to_show);
            hold on;
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(10, 10, int2str(seq.frame), 'color', [0 1 1]);
            hold off;        
            axis image;
%             set(gca, 'Units', 'normalized', 'Position', [0 0 1 1]);
            
            
            figure(fig_handle);
            imagesc(im_to_show);
            hold on;

%222222222222222222222222222222222222222
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(10, 10, int2str(seq.frame), 'color', [0 1 1]);
            hold off;
        end
        
        drawnow
        
    end
    
end

[seq, results] = get_sequence_results(seq);

if params.objectwise
    try
    base0_path=extractBefore(seq.path,'\img');
    seq_name=base0_path(max(strfind(base0_path,'\'))+1:length(base0_path));
    object_name=seq.name;
    [none_inf, gt_boxes] = load_video(base0_path,seq_name,object_name);

    pd_boxes = results.res;
    thresholdSetOverlap = 0: 0.05 : 1;
    success_num_overlap = zeros(1, numel(thresholdSetOverlap));
    if numel(gt_boxes(1,:))>4
        temp = zeros(size(gt_boxes,1),4);  
        for i = 1:size(gt_boxes,1)
            bb8 = round(gt_boxes(i,:));
            x1 = round(min(bb8(1:2:end)));
            x2 = round(max(bb8(1:2:end)));
            y1 = round(min(bb8(2:2:end)));
            y2 = round(max(bb8(2:2:end)));
            temp(i,:) = round([x1, y1, x2 - x1, y2 - y1]);
        end
        gt_boxes = temp;
    end

    thresholdSetPre = 0: 1 : 50;
success_num_pre = zeros(1, numel(thresholdSetPre));
res = calcRectInt(gt_boxes, pd_boxes);
p_gt = [gt_boxes(:,2),gt_boxes(:,1)]+([gt_boxes(:,4),gt_boxes(:,3)]-1)/2;
p_res = [pd_boxes(:,2),pd_boxes(:,1)]+([pd_boxes(:,4),pd_boxes(:,3)]-1)/2;
dis = sqrt(sum((p_gt-p_res).^2,2));
for t = 1: length(thresholdSetOverlap)
    success_num_overlap(1, t) = sum(res > thresholdSetOverlap(t));
end
for t = 1: length(thresholdSetPre)
    success_num_pre(1, t) = sum(dis <= thresholdSetPre(t));
end
Pre = success_num_pre(21) / size(gt_boxes, 1);
cur_AUC = mean(success_num_overlap) / size(gt_boxes, 1);
FPS_vid = results.fps;
 infor=[object_name  '---->' ' FPS: ' num2str(FPS_vid)   '  op: '   num2str(cur_AUC)  '  pr: '   num2str(Pre)];
 
 fprintf(wrt_inf,'%s\n',infor);
 fclose(wrt_inf); 
    catch
    end
end
