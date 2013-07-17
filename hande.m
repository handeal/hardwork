
Qsize = [10 3]; % Number of states in each level 
D = 2;  % depth of the HHMM
transprob = cell(1,D);
termprob  = cell(1,D);
startprob = cell(1,D);
clear A;

% LEVEL 1
%       1 2 3 4 5 6 7 8 9 10 e
% A{1}=[0 0 0 0 0 0 0 0 0 0 1;
% 	    0 0 0 0 0 0 0 0 0 0 1;
%       0 0 0 0 0 0 0 0 0 0 1;
% 	    0 0 0 0 0 0 0 0 0 0 1;
%       0 0 0 0 0 0 0 0 0 0 1;
% 	    0 0 0 0 0 0 0 0 0 0 1;
%       0 0 0 0 0 0 0 0 0 0 1;
% 	    0 0 0 0 0 0 0 0 0 0 1;
%       0 0 0 0 0 0 0 0 0 0 1;
% 	    0 0 0 0 0 0 0 0 0 0 1];
% [transprob{1}, termprob{1}] = remove_hhmm_end_state(A{1});
startprob{1} = 'unif';
transprob{1} = 'rnd';
termprob{1}  = 'rnd';

% LEVEL 2
A{2} = zeros(Qsize(2), Qsize(1), Qsize(2)+1);

% for i=1:10
% %              1 2 3 e
% A{2}(:,i,:) =[0 1 0 0  
% 	            0 0 1 0
% 	            0 0 0 1];
% end
% [transprob{2}, termprob{2}] = remove_hhmm_end_state(A{2});	       
startprob{2} = 'rnd';
transprob{2} = 'rnd';
termprob{2}  = 'rnd';

% OBS LEVEl
chars = ['a', 'b', 'c', 'd', 'e', 'f'];
Osize = length(chars);

%obsprob = zeros([Qsize Osize]);
%obsprob = normalise(rand([Qsize Osize]),3);

Oargs = {'CPT', 'rnd'};
%'Ops', Qnodes(1:3),
bnet = mk_hhmm('Qsizes', Qsize, 'Osize', Osize, 'discrete_obs', 1, ...
	       'Oargs', Oargs,  ...
	       'startprob', startprob, 'transprob', transprob, 'termprob', termprob);

Q1 = 1; Q2 = 2; F2 = 3; Onode = 4;
Qnodes = [Q1 Q2]; Fnodes = [F2];

% data{l}{i,t} = value of node i in slice t of time-series l, or [] if hidden.
%   Suppose you have L time series, each of length T, in an O*T*L array D, 
%   where O is the num of observed scalar nodes, and N is the total num nodes per slice.
%   Then you can create data as follows, where onodes is the index of the observable nodes:
clear data;
L=10;
data = cell(1,L);
for l=1:L
    evidence = sample_dbn(bnet,10);    
    %evidence = sample_dbn(bnet, 'stop_test', 'is_F2_true_D3');   
    T = size(evidence, 2);
    data{l} = cell(Onode, T);
    ev = cell2num(evidence);
    %chars(ev(end,:))
    data{l}(Onode,:) = num2cell(ev(end,:));
end

%data{1}(Q2,:) = num2cell(1:10); 
%data{1}(Onode,:) = num2cell(1); 

engine_init = hmm_inf_engine(bnet);
engine_init = jtree_dbn_inf_engine(bnet);
% engine_init = jtree_unrolled_dbn_inf_engine(bnet, T);

[bnet_learned, LL, engine_learned] = ...
    learn_params_dbn_em(engine_init, data, 'max_iter', 100, 'thresh', 1e-2);

Q1 = 1; Q2 = 2; F2 = 3; Onode = 4;


% CALC_MPE Computes the most probable explanation of the evidence
for i = 1:L
    [mpe, ll] = calc_mpe_dbn(engine_learned, data{i});
    mpe = find_mpe(engine_learned, data{i});
    pretty_print_hhmm_parse(mpe, Qnodes, Fnodes, Onode, chars);
end
