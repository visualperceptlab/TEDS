function feature_map = get_cnn_layers(im, fparams, gparams, object_size)

% Get layers from a cnn.

if size(im,3) == 1
    im = repmat(im, [1 1 3]);
end

im_sample_size = size(im);

if gparams.augment == 0;
    
    %preprocess the image
    if ~isequal(im_sample_size(1:2), fparams.net.meta.normalization.imageSize(1:2))
        im = imresize(single(im), fparams.net.meta.normalization.imageSize(1:2));
    else
        im = single(im);
    end
    
    % Normalize with average image
    im = bsxfun(@minus, im, fparams.net.meta.normalization.averageImage);
    
    if gparams.use_gpu
        im = gpuArray(im);
        fparams.net.eval({'data', im});
    else
        fparams.net.eval({'data', im});
    end
    
    feature_map = cell(1,1,length(fparams.output_var));
    
    for k = 1:length(fparams.output_var)
        if fparams.downsample_factor(k) == 1
            feature_map{k} = fparams.net.vars(fparams.output_var(k)).value(fparams.start_ind(1,1)...
                :fparams.end_ind(1,1), fparams.start_ind(1,2):fparams.end_ind(1,2), :, :);
        else
            feature_map{k} = vl_nnpool(fparams.net.vars(fparams.output_var(k)).value(fparams.start_ind(1,1)...
                :fparams.end_ind(1,1), fparams.start_ind(1,2):fparams.end_ind(1,2), :, :),...
                fparams.downsample_factor(k), 'stride', fparams.downsample_factor(k), 'method', 'avg');
        end
    end
else
    
     %preprocess the image
    if ~isequal(im_sample_size(1:2), fparams.net.meta.normalization.imageSize(1:2))
        im = imresize(single(im), fparams.net.meta.normalization.imageSize(1:2));
    else
        im = single(im);
    end
        
    % Normalize with average image
    im = bsxfun(@minus, im, fparams.net.meta.normalization.averageImage);
    
    im_aug = im;
%   imwrite(im,'C:\Users\Dominik\Desktop\lea\myfig\im.jpg');
%     im_def=single(imread('C:\Users\Dominik\Desktop\lea\myfig\im.jpg'));
    if fparams.augment.blur == 1
        im_aug = cat(4,im_aug,imgaussfilt(im,1));
        im_aug = cat(4,im_aug,imgaussfilt(im,2));
%         im_aug = cat(4,im_aug,imgaussfilt(im,3));
%         im_aug = cat(4,im_aug,imgaussfilt(im,4));
%         im_aug = cat(4,im_aug,imgaussfilt(im,5));
    end
    if fparams.augment.rotation == 1
        im_aug = cat(4,im_aug,imrotate(im,15,'bilinear','crop'));
        im_aug = cat(4,im_aug,imrotate(im,-15,'bilinear','crop'));
        im_aug = cat(4,im_aug,imrotate(im,7,'bilinear','crop'));
        im_aug = cat(4,im_aug,imrotate(im,-7,'bilinear','crop'));
        im_aug = cat(4,im_aug,imrotate(im,22,'bilinear','crop'));
        im_aug = cat(4,im_aug,imrotate(im,-22,'bilinear','crop'));
        im_aug = cat(4,im_aug,imrotate(im,30,'bilinear','crop'));
        im_aug = cat(4,im_aug,imrotate(im,-30,'bilinear','crop'));
        im_aug = cat(4,im_aug,imrotate(im,37,'bilinear','crop'));
        im_aug = cat(4,im_aug,imrotate(im,-37,'bilinear','crop'));
        
%         im_aug = cat(4,im_aug,imrotate(im,16,'bilinear','crop'));
%         im_aug = cat(4,im_aug,imrotate(im,-16,'bilinear','crop'));
%         im_aug = cat(4,im_aug,imrotate(im,8,'bilinear','crop'));
%         im_aug = cat(4,im_aug,imrotate(im,-8,'bilinear','crop'));
%         im_aug = cat(4,im_aug,imrotate(im,24,'bilinear','crop'));
%         im_aug = cat(4,im_aug,imrotate(im,-24,'bilinear','crop'));
%         im_aug = cat(4,im_aug,imrotate(im,32,'bilinear','crop'));
%         im_aug = cat(4,im_aug,imrotate(im,-32,'bilinear','crop'));
%         im_aug = cat(4,im_aug,imrotate(im,40,'bilinear','crop'));
%         im_aug = cat(4,im_aug,imrotate(im,-40,'bilinear','crop'));
%******************************************************************************              
%     try 
%         x1=round(1/2*(size(im,1)-1/2*object_size(2)));
%         x2=round(x1+object_size(2));
%         y1=round(1/2*size(im,2)-1/2*object_size(1));
%         y2=round(y1+1/2*object_size(1));  
%         mask1=im;
%         mask1(y1:y2,x1:x2,1:3)=0;       
%         im_aug = cat(4,im_aug,mask1);
% %         save('C:\Users\Dominik\Desktop\lea\mask\mask_1.mat','mask1');
%         y1=round(1/2*size(im,2));
%         y2=round(y1+1/2*object_size(1));  
%         mask1=im;
%         mask1(y1:y2,x1:x2,1:3)=0;       
%         im_aug = cat(4,im_aug,mask1);   
% %         save('C:\Users\Dominik\Desktop\lea\mask\mask_2.mat','mask1');      
% 
% 
%         y1=round(1/2*size(im,2)-1/2*object_size(1));
%         y2=round(y1+object_size(1));  
%         x1=round(1/2*size(im,1)-1/2*object_size(2));
%         x2=round(x1+1/2*object_size(2));       
%         mask1=im;
%         mask1(y1:y2,x1:x2,1:3)=0;       
%         im_aug = cat(4,im_aug,mask1);  
% %         save('C:\Users\Dominik\Desktop\lea\mask\mask_3.mat','mask1');       
%         x1=round(1/2*size(im,1));
%         x2=round(x1+1/2*object_size(2));        
%         mask1=im;
%         mask1(y1:y2,x1:x2,1:3)=0;       
%         im_aug = cat(4,im_aug,mask1);       
% %         save('C:\Users\Dominik\Desktop\lea\mask\mask_4.mat','mask1');
%   catch 
%     end
%******************************************************************************        
%         sca_im=imresize(im,0.65,'Antialiasing',true);
%         pad_im=padarray(sca_im,(size(im,[1,2])-size(sca_im,[1,2]))/2);
%         im_aug =cat(4,im_aug,pad_im);
%         
%         scl_im=imresize(im,1.35);
%         rec_im=[round((size(scl_im,[1,2])-size(im,[1,2]))/2) size(im,[1,2])-1];
%         crop_im=imcrop(scl_im,rec_im);
%         im_aug =cat(4,im_aug,crop_im);

%         im_aug =cat(4,im_aug,im(:,:,:)*0.9);
%         im_aug =cat(4,im_aug,im(:,:,:)*0.7);
%         im_aug =cat(4,im_aug,im(:,:,:)*0.5);
%         im_aug =cat(4,im_aug,im(:,:,:)*0.3);
%         im_aug =cat(4,im_aug,im(:,:,:)*0.1);
%         im_aug =cat(4,im_aug,im(:,:,:)*0.05);
%         im_aug =cat(4,im_aug,im(:,:,:)*0.01);
% %         
%         im_aug =cat(4,im_aug,im(:,:,:)*1.5);
%         im_aug =cat(4,im_aug,im(:,:,:)*3.5);
%         im_aug =cat(4,im_aug,im(:,:,:)*5.5);

%         V2 = imadjustn(im/255,[0.4 0.7],[]);        
% save('C:\Users\Dominik\Desktop\lea\V2\V2.mat','V2');
% save('C:\Users\Dominik\Desktop\lea\V2\im.mat','im');

%         im_aug =cat(4,im_aug,imadjustn(im/255,[0.49 0.51],[])*255);
%         im_aug =cat(4,im_aug,imadjustn(im/255,[0.30 0.70],[])*255);
%         im_aug =cat(4,im_aug,imadjustn(im/255,[0.20 0.80],[])*255);
%         im_aug =cat(4,im_aug,imadjustn(im/255,[0.10 0.90],[])*255);
%         
%         
%         im_aug =cat(4,im_aug,imadjustn(im/255,[0.3 0.7],[0.4 0.6])*255);
%         im_aug =cat(4,im_aug,imadjustn(im/255,[0.2 0.8],[0.4 0.6])*255);
%         
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%         
%         im1=imadjustn(im/255,[0.45 0.55],[])*255;
%         im_aug =cat(4,im_aug,imadjustn(im1/255,[0.45 0.55],[])*255);
        
%         im2=imadjustn(im/255,[0.05 0.15],[])*255;
%         im_aug =cat(4,im_aug,imadjustn(im2/255,[0.035 0.3],[])*255);  
        
%         im3=imadjustn(im/255,[0.02 0.08],[])*255;
%         im_aug =cat(4,im_aug,imadjustn(im3/255,[0.02 0.11],[])*255);  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%         


%           im_aug =cat(4,im_aug,imadjustn(im/255,[0.5 0.55],[0 1])*255);
%           im_aug =cat(4,im_aug,imadjustn(im/255,[0.4 0.55],[0 1])*255);
          
%           salien=cv.StaticSaliencyFineGrained();
% %           salien=cv.StaticSaliencySpectralResidual();
%           im112=single(computeSaliency(salien,im(:,:,1)));
%           im112=cat(3,im112,single(computeSaliency(salien,im(:,:,2))));
%           im112=cat(3,im112,single(computeSaliency(salien,im(:,:,3))));
%           im_aug =cat(4,im_aug,im112);
%           save('C:\Users\Dominik\Desktop\lea\V2\im112.mat','im112');
%         
%         im_aug =cat(4,im_aug,im(:,:,:)*7.5);
%         im_aug =cat(4,im_aug,im(:,:,:)*9.5);
%         im_aug =cat(4,im_aug,im(:,:,:)*11.5);
%         im_aug =cat(4,im_aug,im(:,:,:)*15.5);
        
        
        
        
%         im_aug =cat(4,im_aug,im+10);
% %         im_aug =cat(4,im_aug,im+20);
%         im_aug =cat(4,im_aug,im+30);
% %         im_aug =cat(4,im_aug,im+40);
%         im_aug =cat(4,im_aug,im+50);
% %         im_aug =cat(4,im_aug,im+60);
%         im_aug =cat(4,im_aug,im+70);
% %         im_aug =cat(4,im_aug,im+80);
%         im_aug =cat(4,im_aug,im+90);
% %         im_aug =cat(4,im_aug,im+100);
%         
%         im_aug =cat(4,im_aug,im-10);
% %         im_aug =cat(4,im_aug,im-20);
%         im_aug =cat(4,im_aug,im-30);
% %         im_aug =cat(4,im_aug,im-40);
%         im_aug =cat(4,im_aug,im-50);
% %         im_aug =cat(4,im_aug,im-60);
%         im_aug =cat(4,im_aug,im-70);
% %         im_aug =cat(4,im_aug,im-80);
%         im_aug =cat(4,im_aug,im-90);
% %         im_aug =cat(4,im_aug,im-100);
%******************************************************************************        
    end 
    if fparams.augment.flip == 1              
        im_aug = cat(4,im_aug,flipdim(im,2)); 
        im_aug = cat(4,im_aug,flipdim(im,1)); 
        im_aug = cat(4,im_aug,flipdim(flipdim(im,1),2)); 
        
        
%         im_aug = cat(4,im_aug,flipdim(im,2));
%         im_aug = cat(4,im_aug,flipdim(imgaussfilt(im,2),2));
%         try
%         x1=round( ( size(im,1) - object_size(2) )/2);
%         x2=round( ( size(im,1) + object_size(2) )/2);
%         y1=round( ( size(im,2) - object_size(1) )/2);
%         y2=round( ( size(im,2) + object_size(1) )/2);
%         
%         im1_0=flipdim(im(y1:y2,x1:x2,:),1);
%         im1=im;
%         im1(y1:y2,x1:x2,:)=im1_0;
%         im_aug = cat(4,im_aug,im1);        
% %         save('C:\Users\Dominik\Desktop\lea\V2\im1.mat','im1');       
%                            
%         im2_0=flipdim(im(y1:y2,x1:x2,:),2);
%         im2=im;
%         im2(y1:y2,x1:x2,:)=im2_0;
%         im_aug = cat(4,im_aug,im2);        
% %         save('C:\Users\Dominik\Desktop\lea\V2\im2.mat','im2');
%         
%         im3_0=flipdim(flipdim(im(y1:y2,x1:x2,:),1),2);
%         im3=im;
%         im3(y1:y2,x1:x2,:)=im3_0;
%         im_aug = cat(4,im_aug,im3);        
% %         save('C:\Users\Dominik\Desktop\lea\V2\im3.mat','im3');      
%         catch
%         end
    end

    if gparams.use_gpu
        fparams.net.eval({'data', gpuArray(im_aug)});
    else
        fparams.net.eval({'data', im_aug});
    end
    
    
    feature_map = cell(1,1,length(fparams.output_var));
    
    for k = 1:length(fparams.output_var)
        if fparams.downsample_factor(k) == 1
            temp = fparams.net.vars(fparams.output_var(k)).value(fparams.start_ind(1,1)...
                :fparams.end_ind(1,1), fparams.start_ind(1,2):fparams.end_ind(1,2), :, :);
            feature_map{k} = mean(temp,4);
            
        else
            temp = vl_nnpool(fparams.net.vars(fparams.output_var(k)).value(fparams.start_ind(1,1)...
                :fparams.end_ind(1,1), fparams.start_ind(1,2):fparams.end_ind(1,2), :, :),...
                fparams.downsample_factor(k), 'stride', fparams.downsample_factor(k), 'method', 'avg');
            feature_map{k} = mean(temp,4);
        end
    end
    
    
    
end
