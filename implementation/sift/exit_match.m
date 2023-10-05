function matches=exit_match(I1,I2,frame)

% if ndims(I1)==3
% I1=rgb2gray(I1);
% I2=rgb2gray(I2);
% end

I1=imresize(I1, [240 320]);
I2=imresize(I2, [240 320]);


I1=I1-min(I1(:)) ;
I1=I1/max(I1(:)) ;
I2=I2-min(I2(:)) ;
I2=I2/max(I2(:)) ;

%fprintf('CS5240 -- SIFT: Match image: Computing frames and descriptors.\n') ;
[frames1,descr1,gss1,dogss1] = do_sift( I1, 'Verbosity', 1, 'NumOctaves', 4, 'Threshold',  0.1/3/2 ) ; %0.04/3/2
[frames2,descr2,gss2,dogss2] = do_sift( I2, 'Verbosity', 1, 'NumOctaves', 4, 'Threshold',  0.1/3/2 ) ;

descr1 = descr1';
descr2 = descr2';
if frame>2
close('XXX');
end
matches=do_match(I1, descr1, frames1',I2, descr2, frames2' ) ;
