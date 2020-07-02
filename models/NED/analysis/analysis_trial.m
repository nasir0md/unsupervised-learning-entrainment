
feat_dir = '/media/nasir/xData/newdata/Fisher/ldc2004s13/fe_03_p1_sph1/feats/000/';

feat_filelist =dir([feat_dir  '*_func_feat.csv']);

for d=1:size(featmat,2)
    Pitch1=[]; Pitch2=[];
    for i=1:length(feat_filelist)
        
        feat_file = feat_filelist(i).name;
        featmat = load([feat_dir feat_file]);
        
        
        nn=length(featmat(:,1));
        nn=floor(nn/2)*2;
        
        
        pitch1 = featmat(1:2:nn,d);
        pitch2 = featmat(2:2:nn,d);
        
        Pitch1 = [Pitch1; pitch1];
        Pitch2 = [Pitch2; pitch2];
        
        nt = floor(nn/2);
        
        
    end
    
    disp([d corr(Pitch1,Pitch2)])
end


%     figure(d)
%     plot(nt, pitch1, nt , pitch2);