Qsize = [2 2]; % Number of states in each level 
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
%     obsprob = rand([Qsize senseSize Osize]);
%     obsprob = mk_stochastic(obsprob);
%     %squeeze(obsprob(1,1,:,:))
%     Oargs = {'CPT', obsprob};
    Oargs = {'CPT', 'rnd'};
    bnet = mk_hhmm_hande('Qsizes', Qsize, 'Osize', Osize, 'discrete_obs', 1, ...
	       'Oargs', Oargs, 'Ops', Qnodes, 'senseSize' , senseSize, ...
	       'startprob', startprob, 'transprob', transprob, 'termprob', termprob);
       
%    bnet = mk_hhmm('Qsizes', Qsize, 'Osize', Osize, 'discrete_obs', 1, ...
% 	       'Oargs', Oargs, 'Ops', Qnodes(1:3), ...
% 	       'startprob', startprob, 'transprob', transprob, 'termprob', termprob);
       
else
    Oargs = {'CPT', 'rnd'};
    bnet = mk_hhmm('Qsizes', Qsize, 'Osize', Osize, 'discrete_obs', 1, ...
	       'Oargs', Oargs, 'Ops', Qnodes,  ...
	       'startprob', startprob, 'transprob', transprob, 'termprob', termprob);
end

bnet.observed = [Q1 bnet.observed];

clear data;
L=5;
SeqLength = 10;
data = cell(1,L);
Onodes = 4:5;
% Generate Training Data
for l=1:L
    evidence = sample_dbn(bnet,SeqLength);    
    T = size(evidence, 2);
    data{l}(Onodes,:) = evidence(Onodes,:);
    data{l}(Q1,:) = evidence(Q1,:);
end

%engine_init = jtree_dbn_inf_engine(bnet);
engine_init = smoother_engine(jtree_2TBN_inf_engine(bnet));
%engine_init =  smoother_engine(hmm_2TBN_inf_engine(bnet));   % cannot use : observed node 1 is not allowed children

% 
%engine_init = bk_inf_engine(bnet, 'clusters', 'ff');
% engine_init = bk_inf_engine(bnet, 'clusters' 'exact');

[bnet_learned, LL, engine_learned] = ...
    learn_params_dbn_em(engine_init, data, 'max_iter', 100, 'thresh', 1e-2);

% Generate Test Data
L=5;
test = cell(1,L);
for l=1:L
    evidence = sample_dbn(bnet,5);     
    T = size(evidence, 2);
    test{l}(Onodes,:) = evidence(Onodes,:);
    Gtruth{l} = cell2num(evidence(Q1,:));
end

bnet_learned.observed = [Onodes];
engine_learned2 = smoother_engine(jtree_2TBN_inf_engine(bnet_learned));

for i = 1:L
    %[mpe, ll] = calc_mpe_dbn(engine_learned, test{i});
    % bu da kullanýlabilir
    mpe = find_mpe(engine_learned2, test{i});
    %pretty_print_hhmm_parse(mpe, Qnodes, Fnodes, Onode, chars);
end

% eclass = bnet_learned.equiv_class;
% CPDQ1_t1=struct(bnet_learned.CPD{eclass(Q1,1)});
% CPDQ1_tn=struct(bnet_learned.CPD{eclass(Q1,2)});
% CPDQ1_t1.CPT
% squeeze(CPDQ1_tn.transprob)
% 
% CPDQ2_t1=struct(bnet_learned.CPD{eclass(Q2,1)});
% CPDQ2_tn=struct(bnet_learned.CPD{eclass(Q2,2)});
% CPDQ2_t1.CPT
% CPDQ2_tn.transprob
% CPDQ2_tn.startprob
% 
% CPDF2=struct(bnet_learned.CPD{eclass(F2,1)});
% CPDF2.CPT
% 
% for z=Onodes
%     CPDFO=struct(bnet_learned.CPD{eclass(z,1)});
%     CPDFO.CPT
% end