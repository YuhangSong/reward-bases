function sea_salt

    % general parameters
    alpha = 0.1;
    trials = 100;
    % parameters of reward basis model
    w_norm = [-1, 1];
    w_depl = [0.5, 1];
    
    %simulating models
    VTD = TD_learner (alpha, trials);
    VRB_bases = RB_learner (alpha, trials, w_norm);
    VRB_norm = w_norm * VRB_bases;
    VRB_depl = w_depl * VRB_bases;
    
    % experimental data
    EXnorm_mean = [1, 4.2];
    EXnorm_std = [0.1, 0.8];
    EXdepl_mean = [3.3, 4.5];
    EXdepl_std = [0.5, 0.2];
    
    % linear regression 
    coefTD = polyfit (VTD, EXnorm_mean, 1);
    coefRB = polyfit (VRB_norm, EXnorm_mean, 1);
    TDnorm = coefTD(1) * VTD + coefTD(2);
    TDdepl = coefTD(1) * VTD + coefTD(2);
    RBnorm = coefRB(1) * VRB_norm + coefRB(2);
    RBdepl = coefRB(1) * VRB_depl + coefRB(2);

    % plotting
    subplot (1,3,1);
    plot_data (EXnorm_mean, EXdepl_mean, EXnorm_std, EXdepl_std)
    title ('Data');
    subplot (1,3,2);
    plot_data (TDnorm, TDdepl);
    title ('TD-learning');
    subplot (1,3,3);
    plot_data (RBnorm, RBdepl);
    title ('Reward Bases');
end

function V = TD_learner (alpha, trials)
    r = [-1, 1];
    n = length(r);
    V = zeros(1,n);
    for i = 1:n
        for t = 1:trials
            V(i) = V(i) + alpha * (r(i)-V(i));
        end
    end
end

function V = RB_learner (alpha, trials, w)
    r = [1, 0; 0, 1];
    n = 2;
    V = zeros(n);
    for k = 1:n
        for t = 1:trials
            for i = 1:n
                V(i,k) = V(i,k) + alpha * w(i)*(w(i)*r(i,k)-w(i)*V(i,k));
            end
        end
    end
end 

function plot_data (norm_mean, depl_mean, norm_std, depl_std)
    bar([norm_mean; depl_mean]);
    if nargin > 2
        hold on
        errorbar ([0.85, 1.15, 1.85, 2.15], [norm_mean(1), norm_mean(2), depl_mean(1), depl_mean(2)], ...
            [norm_std(1), norm_std(2), depl_std(1), depl_std(2)], 'k.');
    end
    legend ('Salt', 'Juice');
    ylim ([0,6]);
    xticklabels ({'Normal', 'Depleted'});
    ylabel ('Approach behaviour');
end
