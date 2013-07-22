
%% Generate True Model
intra = zeros(3);
intra(1,2) = 1; 
intra(1,3) = 1;

inter = zeros(3);
inter(1,1) = 1; 

Q = 2; % num hidden states
O = 2; % num observable symbols
ns = [Q O O];%number of states
dnodes = 1:3;
onodes = [2:3]; % only possible with jtree, not hmm
%onodes = [2]; 
bnet = mk_dbn(intra, inter, ns, 'discrete', dnodes, 'observed', onodes);

prior0 = [1;0]; % always start from state 1
bnet.CPD{1} = tabular_CPD(bnet, 1, 'CPT', prior0);

obsmat1 = [1 0 ; 0 1]; % first obs node in state 1 generates 1 
                       %                         2 generates 2
bnet.CPD{2} = tabular_CPD(bnet, 2, 'CPT', obsmat1);

obsmat2 = [0 1 ; 1 0]; % second obs node in state 1 generates 2 
                       %                          2 generates 1
bnet.CPD{3} = tabular_CPD(bnet,3, 'CPT', obsmat2);

trans = [0.1 0.9 ; 0.9 0.1]; % uniform transition matrix
bnet.CPD{4} = tabular_CPD(bnet, 4, 'CPT', trans);

bnet.observed = [1 2 3];

ss = 3;       %slice size(ss)
ncases = 10;  %number of examples
T=10;
cases = cell(1, ncases);
for i=1:ncases
  ev = sample_dbn(bnet, T);
  cases{i} = cell(ss,T);
  cases{i}(onodes,:) = ev(onodes, :);
  cases{i}(1,:) = ev(1, :);
end

% generate test data without the states
test = cell(1, ncases);
for i=1:ncases
  ev = sample_dbn(bnet, T);
  test{i} = cell(ss,T);
  test{i}(onodes,:) = ev(onodes, :);
  gt{i} = ev(1, :); % ground truth
end

%% Initialize Parameters
bnet_init = mk_dbn(intra, inter, ns, 'discrete', dnodes, 'observed', onodes);

worst = 2;
stdVal = 0.001;

if worst == 1
    % Initialize with impossible parameters
    prior0 = 1-prior0; % always start from state 1
    bnet_init.CPD{1} = tabular_CPD(bnet_init, 1, 'CPT',  prior0);

    obsmat1 = 1-obsmat1;
    bnet_init.CPD{2} = tabular_CPD(bnet_init, 2, 'CPT', obsmat1);

    obsmat2 = 1-obsmat2;
    bnet_init.CPD{3} = tabular_CPD(bnet_init,3, 'CPT', obsmat2);

    trans = 1-trans; 
    bnet_init.CPD{4} = tabular_CPD(bnet_init, 4, 'CPT', trans);
elseif worst == 2
    % Initialize with nearly impossible parameters
    prior0 = mk_stochastic( abs((1-prior0) + normrnd(0,stdVal, size(prior0))) );
    bnet_init.CPD{1} = tabular_CPD(bnet_init, 1, 'CPT',  prior0);

    obsmat1 = mk_stochastic( abs((1-obsmat1) + normrnd(0,stdVal, size(obsmat1))) );
    bnet_init.CPD{2} = tabular_CPD(bnet_init, 2, 'CPT', obsmat1);

    obsmat2 = mk_stochastic( abs((1-obsmat2) + normrnd(0,stdVal, size(obsmat2))) );
    bnet_init.CPD{3} = tabular_CPD(bnet_init,3, 'CPT', obsmat2);

    trans = mk_stochastic( abs((1-trans) + normrnd(0,stdVal, size(trans))) );
    bnet_init.CPD{4} = tabular_CPD(bnet_init, 4, 'CPT', trans);
else
    for i=1:4
        bnet_init.CPD{i} = tabular_CPD(bnet_init, i, 'CPT', 'unif');
        %bnet_init.CPD{i} = tabular_CPD(bnet_init, i, 'CPT', 'rnd');
    end
end

bnet_init.observed = [1 2 3];
engine = smoother_engine(jtree_2TBN_inf_engine(bnet_init));
[bnet2, LLtrace] = learn_params_dbn_em(engine, cases, 'max_iter', 10);

%% look at what we've learnt
eclass = bnet2.equiv_class;
Prior=struct(bnet2.CPD{eclass(1,1)});
Trans=struct(bnet2.CPD{eclass(1,2)});
Obs1=struct(bnet2.CPD{eclass(2,1)});
Obs2=struct(bnet2.CPD{eclass(3,1)});

Prior.CPT
Trans.CPT
Obs1.CPT
Obs2.CPT


%% Make inference on the test data 

for i=1:length(bnet2.CPD)
    curCPD = struct(bnet2.CPD{i});
    after = mk_stochastic( abs((curCPD.CPT) + normrnd(0,stdVal, size(curCPD.CPT))) );
    bnet2.CPD{i} = tabular_CPD(bnet2, i, 'CPT',  after );
end

bnet2.observed = [onodes];
engine_learned2 = smoother_engine(jtree_2TBN_inf_engine(bnet2));

for i = 1:ncases
    mpe{i} = find_mpe(engine_learned2, test{i});
    A = [ cell2num(mpe{i}(1,:)) ; cell2num(gt{i}(1,:))];
    disp(A);
end
