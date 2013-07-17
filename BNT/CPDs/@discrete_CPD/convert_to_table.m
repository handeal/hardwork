function T = convert_to_table(CPD, domain, evidence)
% CONVERT_TO_TABLE Convert a discrete CPD to a table
% T = convert_to_table(CPD, domain, evidence)
%
% We convert the CPD to a CPT, and then lookup the evidence on the discrete parents.
% The resulting table can easily be converted to a potential.


domain = domain(:);
CPT = CPD_to_CPT(CPD);
odom = domain(~isemptycell(evidence(domain)));

if isa(CPD,'bernoulli_CPD')
    n = length(size(CPT));
    index = cell(1,n);
    for i=1:n
        index{i} = ':';
    end
    
    idx = 1;
    for i=odom'
        if size(evidence{i},1) > 1
           T = squeeze(CPT(index{:}));
           A = cat(2, T(:,find(evidence{i} == 1),1), T(:,find(evidence{i} == 2),2));
           %T = prod(A,2);
           T = A;
        else
           index{idx} = find_equiv_posns(odom(idx), domain);
        end
        idx = idx + 1;
    end
else
    vals = cat(1, evidence{odom}); % vertical concatenation
    map = find_equiv_posns(odom, domain);
    index = mk_multi_index(length(domain), map, vals);
    T = CPT(index{:});
    T = T(:);
end


