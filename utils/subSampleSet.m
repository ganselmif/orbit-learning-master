function r = subSampleSet(X, N, subScheme)

r = [];
sizeSet = size(X, 2);

switch subScheme
    case 'decor'
        %% choose most decorrelated templates for nOrbits
        r(1) = randi(sizeSet);
        Xp = project_unit_norm(X);
        for i=1:N-1
            
            % [~, ind_min] = sort(Xp'*project_unit_norm(X(:,r(end))));
            [~, ind_min] = sort(sum(Xp'*project_unit_norm(X(:,r)), 2));
            
            c = 1;
            while ismember(ind_min(c), r)
                c = c + 1;
            end
            
            r(end+1) = ind_min(c);
        end
    case 'rand'
        %% random
        r = randperm(sizeSet, N);
end