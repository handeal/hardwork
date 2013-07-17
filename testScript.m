function  testScript(foldNum, numAct, numSense, numActions)

Qsize = [numAct numActions]; % Number of states in each level 
D = 2;  % depth of the HHMM

transprob = cell(1,D);
termprob  = cell(1,D);
startprob = cell(1,D);

% LEVEL 1
startprob{1} = 'unif';
transprob{1} = 'rnd';
termprob{1}  = 'rnd';

% LEVEL 2
startprob{2} = 'unif';
transprob{2} = 'rnd';
termprob{2}  = 'rnd';

% OBS LEVEl
Osize = 2;

Q1 = 1; Q2 = 2; F2 = 3; Onode = 4;
Qnodes = [Q1 Q2]; 


senseSize = numSense;
Oargs = {'CPT', 'rnd'};
bnet = mk_hhmm_hande('Qsizes', Qsize, 'Osize', Osize, 'discrete_obs', 1, ...
    'Oargs', Oargs, 'Ops', Qnodes, 'senseSize' , senseSize, ...
    'startprob', startprob, 'transprob', transprob, 'termprob', termprob);

bnet.observed = [Q1 bnet.observed];

clear data;
load folds;
L=length(folds{foldNum}.trainFeatMat);
data = cell(1,L);
Onodes = Onode:(senseSize+3);
% Generate Training Data
for l=1:L
    evidence = folds{foldNum}.trainFeatMat{l}+1;
    data{l}(Onodes,:) = num2cell(evidence);
    evidence = folds{foldNum}.trainLabels{l};
    data{l}(Q1,:) =  num2cell(evidence);
end

engine_init = smoother_engine(jtree_2TBN_inf_engine(bnet));

[bnet_learned, LL, engine_learned] = ...
    learn_params_dbn_em(engine_init, data, 'max_iter', 10, 'thresh', 1e-2);

testData = cell(1,1);
testData{1}(Onodes,:) = num2cell(folds{foldNum}.testFeatMat{1}+1); 

bnet_learned.observed = [Onodes];
engine_learned2 = smoother_engine(jtree_2TBN_inf_engine(bnet_learned));

InferredLabels = find_mpe(engine_learned2, testData{1});
% A = cell2num(mpe) ; 
% B = folds{foldNum}.testLabels{1};
% find ( folds{foldNum}.testLabels{1} ~= A(1,:) );

%% Save results to a mat file
timestamp = datestr(now, 'dd-mm-yyyy_HH.MM.SS');
outFile = sprintf('InferredLabels_F%d_A%d_%s.mat', foldNum, numActions, timestamp);
save(outFile, 'InferredLabels','-v7.3');   

output.bnet_learned = bnet_learned;
output.LL = LL;
output.engine_learned = engine_learned;

outFile = sprintf('output_F%d_A%d_%s.mat', foldNum, numActions, timestamp);
save(outFile, 'output','-v7.3');   
end

