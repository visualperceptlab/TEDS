function resized_patch = sample_patch_dou(im, pos, x_scal, y_scal, output_sz, gparams)

if nargin < 5
    output_sz = [];
end
if nargin < 6 || ~isfield(gparams, 'use_mexResize')
    gparams.use_mexResize = false;
end

% Pos should be integer when input, but floor in just in case.
pos_y = floor(pos);
im_y=im;
% Downsample factor
resize_factor = min(y_scal ./ output_sz);
df = max(floor(resize_factor - 0.1), 1);
if df > 1
    % pos = 1 + of + df * (npos - 1)
    
    % compute offset and new center position
    os = mod(pos_y - 1, df);
    pos_y = (pos_y - 1 - os) / df + 1;
    
    % new sample size
    y_scal = y_scal / df;
    
    % donwsample image
    im_y = im(1+os(1):df:end, 1+os(2):df:end, :);
end

% make sure the size is not too small and round it
y_scal = max(round(y_scal), 2);

%***********************
pos_x = floor(pos);
im_x=im;
% Downsample factor
resize_factor = min(x_scal ./ output_sz);
df = max(floor(resize_factor - 0.1), 1);
if df > 1
    % pos = 1 + of + df * (npos - 1)
    
    % compute offset and new center position
    os = mod(pos_x - 1, df);
    pos_x = (pos_x - 1 - os) / df + 1;
    
    % new sample size
    x_scal = x_scal / df;
    
    % donwsample image
    im_x = im(1+os(1):df:end, 1+os(2):df:end, :);
end

% make sure the size is not too small and round it
x_scal = max(round(x_scal), 2);
%***********************


xs = pos_x(2) + (1:x_scal(2))  - floor((x_scal(2)+1)/2);
ys = pos_y(1) + (1:y_scal(1))  - floor((y_scal(1)+1)/2);


% xs = pos(2) - (1:y_scal(2)) + floor((y_scal(2)+1)/2);
% ys = pos(1);

%check for out-of-bounds coordinates, and set them to the values at
%the borders
xs(xs < 1) = 1;
ys(ys < 1) = 1;
xs(xs > size(im_x,2)) = size(im_x,2);
ys(ys > size(im_y,1)) = size(im_y,1);

%extract image
im_patch = im(ys, xs, :);

if isempty(output_sz) || (isequal(y_scal(:), output_sz(:)) && isequal(x_scal(:), output_sz(:)))
    resized_patch = im_patch;
else
    if gparams.use_mexResize
        resized_patch = mexResize(im_patch, output_sz, 'linear');
    else
        resized_patch = imresize(im_patch, output_sz, 'bilinear', 'Antialiasing',false);
    end
end

end

