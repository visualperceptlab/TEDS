function [match_num,good_match,kpts1,kpts2]=state_predict_orb(img1,img2,ratio_thresh,draw_number) 
match_num=0;
sift=cv.SURF();
[kpts1, desc1] = sift.detectAndCompute(img1);
[kpts2, desc2] = sift.detectAndCompute(img2); 
matcher = cv.DescriptorMatcher('BruteForce');
try
knn_matches=matcher.knnMatch(desc1,desc2,2);   
for i=1:length(knn_matches)
    if knn_matches{1,i}(1).distance < ratio_thresh*knn_matches{1,i}(2).distance
        match_num=match_num+1;
        good_match(match_num)=knn_matches{1,i}(1);
    end
end
catch
    disp('Error!');
end

if match_num==0
    good_match=[];
end

clear knn_matches;
if draw_number>0 && match_num>0
    draw_img=cv.drawMatches(img1,kpts1,img2,kpts2,good_match,'NotDrawSinglePoints',true);
    figure(draw_number);
    imshow(draw_img);   
end