function descriptors = do_descriptor(octave, ... % 一组的高斯尺度空间
                                      oframes, ...  % frames 包含关键点相关的尺度和主方向
                                      sigma0, ...   % 基本 sigma 值
                                      S, ...        % 该组的尺度层数
                                      smin, ...     
                                      varargin)

for k=1:2:length(varargin)
	switch lower(varargin{k})
      case 'magnif'
        magnif = varargin{k+1} ;
        
      case 'numspatialbins'
        NBP = varargin{k+1} ;  
        
      case  'numorientbins'
        NBO = varargin{k+1} ;   
        
      otherwise
        error(['Unknown parameter ' varargin{k} '.']) ;
     end
end 
      
                               
num_spacialBins = NBP;
num_orientBins = NBO;
key_num = size(oframes, 2);
% 计算该图的向量和方向
[M, N, s_num] = size(octave); % M 是图像的高度, N 是图像的宽度; num_level is the number of scale level of the octave
descriptors = [];
magnitudes = zeros(M, N, s_num);
angles = zeros(M, N, s_num);
% compute image gradients
for si = 1: s_num
    img = octave(:,:,si);
    dx_filter = [-0.5 0 0.5];
    dy_filter = dx_filter';
    gradient_x = imfilter(img, dx_filter);
    gradient_y = imfilter(img, dy_filter);
    magnitudes(:,:,si) =sqrt( gradient_x.^2 + gradient_y.^2);
%     if sum( gradient_x == 0) > 0
%         fprintf('00');
%     end
    angles(:,:,si) = mod(atan(gradient_y ./ (eps + gradient_x)) + 2*pi, 2*pi);
end

x = oframes(1,:);
y = oframes(2,:);
s = oframes(3,:);
% round off
x_round = floor(oframes(1,:) + 0.5);
y_round = floor(oframes(2,:) + 0.5);
scales =  floor(oframes(3,:) + 0.5) - smin;

for p = 1: key_num  %对各个关键点处理

    s = scales(p);
    xp= x_round(p);
    yp= y_round(p);
    theta0 = oframes(4,p);%关键点的主方向
    sinth0 = sin(theta0) ;
    costh0 = cos(theta0) ;
    sigma = sigma0 * 2^(double (s / S)) ;
    SBP = magnif * sigma;
    %W =  floor( sqrt(2.0) * SBP * (NBP + 1) / 2.0 + 0.5);
    W =   floor( 0.8 * SBP * (NBP + 1) / 2.0 + 0.5);
    
    descriptor = zeros(NBP, NBP, NBO);
    
    % within the big square, select the pixels with the circle and put into
    % the histogram. no need to do rotation which is very expensive
    %在大正方形中用高斯加权圆选择像素点放入方向直方图中，不需要做昂贵的图像旋转
    for dxi = max(-W, 1-xp): min(W, N -2 - xp)
        for dyi = max(-W, 1-yp) : min(+W, M-2-yp)
            mag = magnitudes(yp + dyi, xp + dxi, s); % 当前点(yp + dyi, xp + dxi)的梯度幅值
            angle = angles(yp + dyi, xp + dxi, s) ;  % 当前点(yp + dyi, xp + dxi)的梯度幅角
%           angle = mod(-angle + theta0, 2*pi);      % 用关键点的主方向调整角度 并且 mod it with 2*pi
            angle = mod(angle - theta0, 2*pi);
             dx = double(xp + dxi - x(p));            % x(p) 是关键点的精确位置 (浮点数). dx 相对于该关键点当前像素的位置
            dy = double(yp + dyi - y(p));            % dy 相对于该关键点当前像素的位置
            
            nx = ( costh0 * dx + sinth0 * dy) / SBP ; % nx 是旋转(dx, dy)后的规格化位置 with the major orientation angle. this tells which x-axis spatial bin the pixel falls in 
            ny = (-sinth0 * dx + costh0 * dy) / SBP ; 
            nt = NBO * angle / (2* pi) ;
            wsigma = NBP/2 ;
            wincoef =  exp(-(nx*nx + ny*ny)/(2.0 * wsigma * wsigma)) ;
            
            binx = floor( nx - 0.5 ) ;
            biny = floor( ny - 0.5 ) ;
            bint = floor( nt );
            rbinx = nx - (binx+0.5) ;
            rbiny = ny - (biny+0.5) ;
            rbint = nt - bint ;
             
            for(dbinx = 0:1) 
               for(dbiny = 0:1) 
                   for(dbint = 0:1) 
                        % if condition limits the samples within the square
                        % width W. binx+dbinx is the rotated x-coordinate.
                        % therefore the sampling square is effectively a
                        % rotated one
                        %如果条件限制在用宽度W设定的方形内的样点，binx+dbinx是旋转后的x坐标
                        %因此采样方形的旋转是有效的。
         
                        if( binx+dbinx >= -(NBP/2) && ...
                            binx+dbinx <   (NBP/2) && ...
                            biny+dbiny >= -(NBP/2) && ...
                            biny+dbiny <   (NBP/2) &&  isnan(bint) == 0) 
                              
                              weight = wincoef * mag * abs(1 - dbinx - rbinx) ...
                                  * abs(1 - dbiny - rbiny) ...
                                  * abs(1 - dbint - rbint) ;
   
                              descriptor(binx+dbinx + NBP/2 + 1, biny+dbiny + NBP/2+ 1, mod((bint+dbint),NBO)+1) = ...
                                  descriptor(binx+dbinx + NBP/2+ 1, biny+dbiny + NBP/2+ 1, mod((bint+dbint),NBO)+1 ) +  weight ;
                        end
                   end
               end
            end

        end
            
    end
    descriptor = reshape(descriptor, 1, NBP * NBP * NBO);%用一维向量表示各梯度值
    descriptor = descriptor ./ norm(descriptor); %归一化处理梯度值
            
            %Truncate at 0.2
    indx = find(descriptor > 0.2);%找出幅值大于0.2的梯度值
    descriptor(indx) = 0.2;       %大于0.2的梯度值直接取0.2
    descriptor = descriptor ./ norm(descriptor);  %再次归一化梯度值
    
    descriptors = [descriptors, descriptor'];
end
