function [match_pos]=match_location(im1,im2,kpts1,kpts2,good_match,crop_pos)
pos_rlt_fom=size(im1);
pos_rlt_fom=pos_rlt_fom([2,1]);
pos_rlt_now=size(im2);  
pos_rlt_now=pos_rlt_now([2,1]);
num_g1=length(good_match);
pos_tru_rel=zeros(num_g1,2);

for m_i=1:length(good_match)  
    key_1=kpts1(good_match(m_i).queryIdx+1).pt;
    key_2=kpts2(good_match(m_i).trainIdx+1).pt;
    dis_cet_tem=key_1-pos_rlt_fom/2;
    dis_cet(m_i)=dis_cet_tem(1)^2+dis_cet_tem(2)^2;    
    pos_tru_rel(m_i,:)=round(fliplr(key_2-key_1+pos_rlt_fom/2+crop_pos));
    dis_mod(m_i,:)=pos_tru_rel(m_i,1)^2+pos_tru_rel(m_i,2)^2;
end

[most_dis num_occ]=mode(dis_mod);
col_min=min(find(dis_mod == most_dis));
if num_occ==1
    [out_val col_min]=min(dis_cet);
end
match_pos=pos_tru_rel(col_min,:);