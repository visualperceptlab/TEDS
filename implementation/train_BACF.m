function [filter_model_f, filter_ref_f, channel_selection, params] = train_BACF(xlf, feature_info, yf, seq, params, filter_model_f, filter_ref_f)
% exch_frame=seq.frame;
% seq.frame=1;
if seq.frame == 1
    filter_model_f = cell(size(xlf));
    filter_ref_f = cell(size(xlf));
    channel_selection = cell(size(xlf));
end
sz=params.sz;
params.admm_lambda=0.01;
small_filter=params.small_filter;

%//////////////////////////////////////////////////////////////////
for k = 1: numel(xlf)
    if seq.frame == 1
        params.filter_model{k} = xlf{k};
    else
        params.filter_model{k} = ((1 - feature_info.learning_rate(k)) * params.filter_model{k}) + (feature_info.learning_rate(k) * xlf{k});
    end 
    % intialize the variables
    g_f = single(zeros(size(xlf{k})));
    h_f = g_f;
    l_f = g_f;
    mu    = 1;
    betha = 10;
    mumax = 10000;
    i = 1;
    dwdw=sz{k};
    
    T = prod(sz{k});
    S_xx = sum(conj(params.filter_model{k}) .* params.filter_model{k}, 3);
    params.admm_iterations = 2;    
    
    %   ADMM
    while (i <= params.admm_iterations)
        %   solve for G- please refer to the paper for more details
        B = S_xx + (T * mu);
        S_lx = sum(conj(params.filter_model{k}) .* l_f, 3);
        S_hx = sum(conj(params.filter_model{k}) .* h_f, 3);
        g_f = (((1/(T*mu)) * bsxfun(@times, yf{k}, params.filter_model{k})) - ((1/mu) * l_f) + h_f) - ...
            bsxfun(@rdivide,(((1/(T*mu)) * bsxfun(@times, params.filter_model{k}, (S_xx .* yf{k}))) - ((1/mu) * bsxfun(@times, params.filter_model{k}, S_lx)) + (bsxfun(@times, params.filter_model{k}, S_hx))), B);
        
        %   solve for H
        h = (T/((mu*T)+ params.admm_lambda))* ifft2((mu*g_f) + l_f);
        [sx,sy,h] = get_subwindow_no_window(h, floor(sz{k}/2) , small_filter{k});
        t = single(zeros(sz{k}(1), sz{k}(2), size(h,3)));
        t(sx,sy,:) = h;
        h_f = fft2(t);
        
        %   update L
        l_f = l_f + (mu * (g_f - h_f));
        
        %   update mu- betha = 10.
        mu = min(betha * mu, mumax);
        i = i+1;
    end 
    filter_model_copy{k}=g_f;
%     filter_ref_f=1;
%     channel_selection=2;
end
%//////////////////////////////////////////////////////////////////



%*********************************************************************
for k = 1: numel(xlf)
    model_xf = gather(xlf{k});
    
    if (seq.frame == 1)
        filter_ref_f{k} = zeros(size(model_xf));
        filter_model_f{k} = zeros(size(model_xf));
        lambda2 = 0;
    else
        lambda2 = params.temporal_consistency_factor(feature_info.feature_is_deep(k)+1);
    end
    
    % intialize the variables
    f_f = single(zeros(size(model_xf)));
    f_prime_f = f_f;
    Gamma_f = f_f;
    mu  = params.init_penalty_factor;
    mu_scale_step = params.penalty_scale_step;
    
    % pre-compute the variables
    T = feature_info.data_sz(k)^2;
    S_xx = sum(conj(model_xf) .* model_xf, 3);
    Stheta_pre_f = sum(conj(model_xf) .* filter_ref_f{k}, 3);
    Sfx_pre_f = bsxfun(@times, model_xf, Stheta_pre_f);
    
    % solve via ADMM algorithm
    iter = 1;
    while (iter <= params.max_iterations)
        
        % subproblem f
        B = S_xx + T * (mu + lambda2);
        Sgx_f = sum(conj(model_xf) .* f_prime_f, 3);
        Shx_f = sum(conj(model_xf) .* Gamma_f, 3);   
        
        f_f = ((1/(T*(mu + lambda2)) * bsxfun(@times,  yf{k}, model_xf)) - ((1/(mu + lambda2)) * Gamma_f) +(mu/(mu + lambda2)) * f_prime_f) + (lambda2/(mu + lambda2)) * filter_ref_f{k} - ...
            bsxfun(@rdivide,(1/(T*(mu + lambda2)) * bsxfun(@times, model_xf, (S_xx .*  yf{k})) + (lambda2/(mu + lambda2)) * Sfx_pre_f - ...
            (1/(mu + lambda2))* (bsxfun(@times, model_xf, Shx_f)) +(mu/(mu + lambda2))* (bsxfun(@times, model_xf, Sgx_f))), B);
        
        %   subproblem f_prime
        X = real(ifft2(mu * f_f+ Gamma_f));
        if (seq.frame == 1)
            T1 = zeros(size(X));
            channel_selection{k} = ones(size(X));
            for i = 1:size(X,3)
                T1(:,:,i) = params.mask_window{k} .* X(:,:,i);
            end
        else
            T1=X;
            
            channel_selection{k} = max(0,1-5./(numel(X)*sqrt(sum(sum(T1.^2,1),2))));
            
            [~,b] = sort(channel_selection{k}(:),'descend');
            
            if numel(b)>1
                channel_selection{k}(b(ceil(feature_info.channel_selection_rate(k)*numel(b)):end)) = 0;
            end
            
            T1 = repmat(channel_selection{k},size(T1,1),size(T1,2),1) .* T1;
        end
        
        f_prime_f = fft2(T1);
        
        %   update Gamma
        Gamma_f = Gamma_f + (mu * (f_f - f_prime_f));
        
        %   update mu
        mu = min(mu_scale_step * mu, 0.1);
        
        iter = iter+1;
    end
    
    % save the trained filters
    filter_ref_f{k} = f_f+randn(size(f_f))*mean(f_f(:))*params.stability_factor;
    
    if seq.frame == 1
        filter_model_f{k} = f_f;
    else
        filter_model_f{k} = feature_info.learning_rate(k)*f_f + (1-feature_info.learning_rate(k))*filter_model_f{k};
    end
end

filter_model_f{1}=filter_model_copy{1};
filter_model_f{2}=filter_model_copy{2};
%*********************************************************************
% seq.frame=exch_frame;

end