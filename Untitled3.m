engine_init = jtree_dbn_inf_engine(bnet);
%engine = smoother_engine(jtree_dbn_inf_engine(bnet));
[bnet_learned, LL, engine_learned] = ...
    learn_params_dbn_em(engine_init, data, 'max_iter', 1, 'thresh', 1e-2);

eclass = bnet_learned.equiv_class;
CPDQ1_t1=struct(bnet_learned.CPD{eclass(Q1,1)});
CPDQ1_tn=struct(bnet_learned.CPD{eclass(Q1,2)});
CPDQ1_t1.CPT
squeeze(CPDQ1_tn.transprob)

CPDQ2_t1=struct(bnet_learned.CPD{eclass(Q2,1)});
CPDQ2_tn=struct(bnet_learned.CPD{eclass(Q2,2)});
CPDQ2_t1.CPT
CPDQ2_tn.transprob
CPDQ2_tn.startprob

CPDF2=struct(bnet_learned.CPD{eclass(F2,1)});
CPDF2.CPT

CPDFO=struct(bnet_learned.CPD{eclass(Onode,1)});
CPDFO.CPT


data1 = data; %{1 1 1 1 2 2 2 1 2 1;}
data1{1}(4,:) ={2 1 2 1 1 2 1 1 1 2;} ;

[bnet_learned, LL, engine_learned] = ...
    learn_params_dbn_em(engine_init, data1, 'max_iter', 1, 'thresh', 1e-2);
eclass = bnet_learned.equiv_class;
CPDQ1_t1=struct(bnet_learned.CPD{eclass(Q1,1)});
CPDQ1_tn=struct(bnet_learned.CPD{eclass(Q1,2)});
CPDQ1_t1.CPT
squeeze(CPDQ1_tn.transprob)

CPDQ2_t1=struct(bnet_learned.CPD{eclass(Q2,1)});
CPDQ2_tn=struct(bnet_learned.CPD{eclass(Q2,2)});
CPDQ2_t1.CPT
CPDQ2_tn.transprob
CPDQ2_tn.startprob

CPDF2=struct(bnet_learned.CPD{eclass(F2,1)});
CPDF2.CPT

CPDFO=struct(bnet_learned.CPD{eclass(Onode,1)});
CPDFO.CPT
