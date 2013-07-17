Qsize = [2 3]; % Number of states in each level 
D = 2;  % depth of the HHMM
transprob = cell(1,D);
termprob  = cell(1,D);
startprob = cell(1,D);
clear A;

% LEVEL 1
startprob{1} = 'unif';
transprob{1} = 'rnd';
termprob{1}  = 'rnd';

% LEVEL 2
startprob{2} = 'unif';
transprob{2} = 'rnd';
termprob{2}  = 'rnd';

% OBS LEVEl
chars = ['a', 'b'];
Osize = length(chars);

Q1 = 1; Q2 = 2; F2 = 3; Onode = 4;
Qnodes = [Q1 Q2]; Fnodes = [F2];

hande = 1;

if hande
    senseSize = 2;
    obsprob = rand([Qsize senseSize Osize]);
    obsprob = mk_stochastic(obsprob);
    %squeeze(obsprob(1,1,:,:))
    Oargs = {'CPT', obsprob};
    bnet = mk_hhmm_hande('Qsizes', Qsize, 'Osize', Osize, 'discrete_obs', 1, ...
	       'Oargs', Oargs, 'Ops', Qnodes, 'senseSize' , senseSize, ...
	       'startprob', startprob, 'transprob', transprob, 'termprob', termprob);
else
    Oargs = {'CPT', 'rnd'};
    bnet = mk_hhmm('Qsizes', Qsize, 'Osize', Osize, 'discrete_obs', 1, ...
	       'Oargs', Oargs, 'Ops', Qnodes,  ...
	       'startprob', startprob, 'transprob', transprob, 'termprob', termprob);
end

bnet.observed = [Q1 Onode];


clear data;
L=1;
SeqLength = 10;
data = cell(1,L);
% Generate Training Data
for l=1:L
    evidence = sample_dbn(bnet,SeqLength);    
    T = size(evidence, 2);
    data{l} = cell(Onode, T);
    data{l}(Onode,:) = evidence(Onode,:);
    data{l}(Q1,:) = evidence(Q1,:);
end

% for l=1:L
%     evidence = sample_dbn(bnet,SeqLength);    
%     T = size(evidence, 2);
%     data{l} = cell(Onode, T);
%     ev = cell2num(evidence);
%     A = rand(1,5);
%     A( A >= 0.5 ) = 1;
%     A( A < 0.5  ) = 2;
%     data{l}(Onode,:) = {A'};
%     %data{l}(Onode,:) = num2cell(ev(end,:));
%     data{l}(Q1,:) = num2cell(ev(Q1,:));
% end

%engine_init = smoother_engine(jtree_2TBN_inf_engine(bnet));
engine_init = jtree_dbn_inf_engine(bnet);
%engine = smoother_engine(jtree_dbn_inf_engine(bnet));
[bnet_learned, LL, engine_learned] = ...
    learn_params_dbn_em(engine_init, data, 'max_iter', 100, 'thresh', 1e-2);

% % examine the params
% eclass = bnet_learned.equiv_class;
% CPDQ1_true=struct(bnet_learned.CPD{eclass(Q1,2)});
% CPDQ2_true=struct(bnet_learned.CPD{eclass(Q2,2)});
% CPDF2_true=struct(bnet_learned.CPD{eclass(F2,1)});
% CPDFO_true=struct(bnet_learned.CPD{eclass(Onode,1)});
% CPDFO_true.CPT
% 
% 
% [bnet_learned1, LL1, engine_learned1] = ...
%     learn_params_dbn_em(engine_init, data1, 'max_iter', 100, 'thresh', 1e-2);
% eclass = bnet_learned.equiv_class;
% CPDQ1=struct(bnet_learned1.CPD{eclass(Q1,2)});
% CPDQ2=struct(bnet_learned1.CPD{eclass(Q2,2)});
% CPDF2=struct(bnet_learned1.CPD{eclass(F2,1)});
% CPDFO=struct(bnet_learned1.CPD{eclass(Onode,1)});
% CPDFO_true.CPT
% 
% L=5;
% test = data(1:L);
% 
% for i = 1:L
%     [mpe, ll] = calc_mpe_dbn(engine_learned, test{i});
%     % bu da kullanýlabilir
%     % mpe = find_mpe(engine_learned, data{i});
%     %pretty_print_hhmm_parse(mpe, Qnodes, Fnodes, Onode, chars);
% end
