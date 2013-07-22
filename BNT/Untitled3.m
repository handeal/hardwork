testHMMScript(5, 15, 14);

eclass = bnet_learned.equiv_class;
Prior=struct(bnet_learned.CPD{eclass(1,1)});
Trans=struct(bnet_learned.CPD{1,eclass(1,2)});

for i = 1:14
    Obs{i}=struct(bnet_learned.CPD{eclass(i+1,1)});
end


Prior.CPT
Trans.CPT


for i = 1:14
    Obs{i}.CPT
end
