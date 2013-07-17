function sz = dom_sizes(CPD)
% DOM_SIZES Return the size of each node in the domain
% sz = dom_sizes(CPD)
if isa(CPD,'bernoulli_CPD')
    sz = CPD.dom_sizes([1 2 4]);
else
    sz = CPD.dom_sizes;
end
