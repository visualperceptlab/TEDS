function [seq, ground_truth] = load_video(video_path,sequence_name)


    if exist('object_name','var')
        str_object_num=object_name(strfind(object_name,'-')+1);
    end
    if exist('str_object_num','var') && ~isempty(str_object_num)   
        if exist([video_path '/groundtruth_rect.' str_object_num '.txt'])
            ground_truth = dlmread([video_path '/groundtruth_rect.' str_object_num '.txt']);
            seq.format = 'otb';
        elseif exist([video_path '/groundtruth.' str_object_num '.txt'])
            ground_truth = dlmread([video_path '/groundtruth.' str_object_num '.txt']);
            seq.format = 'otb_8';
        end     
    else
    if exist([video_path '/groundtruth_rect.txt'])
        ground_truth = dlmread([video_path '/groundtruth_rect.txt']);
        seq.format = 'otb';
    elseif exist([video_path '/groundtruth.txt'])
        ground_truth = dlmread([video_path '/groundtruth.txt']);
        seq.format = 'otb_8';
    end
    end
    
    seq.len = size(ground_truth, 1);
    seq.init_rect = ground_truth(1,:);

    if strcmp(sequence_name, 'David')
        start_frame = 300;end_frame = 770;
    elseif strcmp(sequence_name, 'Football1')
        start_frame = 1;end_frame = 74;
    elseif strcmp(sequence_name, 'Freeman3')
        start_frame = 1;end_frame = 460;
    elseif strcmp(sequence_name, 'Freeman4')
        start_frame = 1;end_frame = 283;
    else
        start_frame = 1; end_frame = seq.len;
    end

    if strcmp(sequence_name, 'BlurCar1')
        nn = 247;
    elseif strcmp(sequence_name, 'BlurCar3')
        nn = 3;
    elseif strcmp(sequence_name, 'BlurCar4')
        nn = 18;
    else
        nn = 1;
    end

    switch seq.format 
        case 'otb'
            img_path = [video_path '/img/'];
            if exist([img_path num2str(nn, '%04i.png')], 'file'),
                img_files = num2str((start_frame+nn-1:end_frame+nn-1)', [img_path '%04i.png']);
            elseif exist([img_path num2str(nn, '%04i.jpg')], 'file'),
                img_files = num2str((start_frame+nn-1:end_frame+nn-1)', [img_path '%04i.jpg']);
            elseif exist([img_path num2str(nn, '%04i.bmp')], 'file'),
                img_files = num2str((start_frame+nn-1:end_frame+nn-1)', [img_path '%04i.bmp']);
            elseif exist([img_path num2str(nn, '%05i.jpg')], 'file'),
                img_files = num2str((start_frame+nn-1:end_frame+nn-1)', [img_path '%05i.jpg']);
            else
                error('No image files to load.')
            end
        case 'otb_8';
            img_path = [video_path '/'];
            img_files = num2str((start_frame:end_frame)', [img_path '%08i.jpg']);
    end
        seq.s_frames = cellstr(img_files);
        seq.start_frame = start_frame;
        seq.end_frame = end_frame;
end

