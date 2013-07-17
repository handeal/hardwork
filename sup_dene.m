Qsize = [10 3]; % Number of states in each level 
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
chars = ['a', 'b', 'c', 'd', 'e', 'f'];
Osize = length(chars);

Oargs = {'CPT', 'rnd'};

% bnet = mk_hhmm('Qsizes', Qsize, 'Osize', Osize, 'discrete_obs', 1, ...
% 	       'Oargs', Oargs,  ...
% 	       'startprob', startprob, 'transprob', transprob, 'termprob', termprob);
bnet = mk_hhmm('Qsizes', Qsize, 'Osize', Osize, 'discrete_obs', 1, ...
	       'Oargs', Oargs);
Q1 = 1; Q2 = 2; F2 = 3; Onode = 4;
Qnodes = [Q1 Q2]; Fnodes = [F2];

bnet.observed = [Q1 Onode];

% Generate Training Data
clear data;
L=10;
SeqLength = 20;
data = cell(1,L);
for l=1:L
    evidence = sample_dbn(bnet,SeqLength);    
    T = size(evidence, 2);
    data{l} = cell(Onode, T);
    ev = cell2num(evidence);
    data{l}(Onode,:) = num2cell(ev(end,:));
    data{l}(Q1,:) = num2cell(ev(Q1,:));
end

%engine_init = hmm_inf_engine(bnet);
engine_init = jtree_dbn_inf_engine(bnet);

[bnet_learned, LL, engine_learned] = ...
    learn_params_dbn_em(engine_init, data, 'max_iter', 100, 'thresh', 1e-2);

% Generate Test Data
L=5;
test = cell(1,L);
for l=1:L
    evidence = sample_dbn(bnet,SeqLength);     
    T = size(evidence, 2);
    test{l} = cell(Onode, T);
    ev = cell2num(evidence);
    test{l}(Onode,:) = num2cell(ev(end,:));
    Gtruth{l} = num2cell(ev(Q1,:));
end

% CALC_MPE Computes the most probable explanation of the evidence
for i = 1:L
    [mpe, ll] = calc_mpe_dbn(engine_learned, test{i});
    % bu da kullanýlabilir
    % mpe = find_mpe(engine_learned, data{i});
    pretty_print_hhmm_parse(mpe, Qnodes, Fnodes, Onode, chars);
end
