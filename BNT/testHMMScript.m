function  testHMMScript(foldNum, numAct, numSense)
disp(sprintf('Fold %d',foldNum ));
disp(sprintf('Number of Activities %d, Number of Sensors %d',numAct,numSense ));

sliceSize = numSense+1;

intra = zeros(sliceSize);
intra(1,(2:end)) = 1; 

inter = zeros(sliceSize);
inter(1,1) = 1; 

Q = numAct; % num hidden states
O = 2; % num observable symbols
ns = [Q repmat(O,1,numSense)];%number of states

dnodes = 1:sliceSize;
onodes = dnodes(2:end); % only possible with jtree, not hmm

bnet = mk_dbn(intra, inter, ns, 'discrete', dnodes, 'observed', onodes);

for i=1:sliceSize+1
    bnet.CPD{i} = tabular_CPD(bnet,i);
end


clear data;
load folds;
L=length(folds{foldNum}.trainFeatMat);
data = cell(1,L);
% Generate Training Data
for l=1:L
    evidence = folds{foldNum}.trainFeatMat{l}+1;
    data{l}(onodes,:) = num2cell(evidence);
    evidence = folds{foldNum}.trainLabels{l};
    data{l}(1,:) =  num2cell(evidence);
end

bnet.observed = dnodes;
engine_init = smoother_engine(jtree_2TBN_inf_engine(bnet));

[bnet_learned, LL, engine_learned] = ...
    learn_params_dbn_em(engine_init, data, 'max_iter', 50, 'thresh', 1e-2);

testData = cell(1,1);
testData{1}(onodes,:) = num2cell(folds{foldNum}.testFeatMat{1}+1); 

bnet_learned.observed = [onodes];
engine_learned2 = smoother_engine(jtree_2TBN_inf_engine(bnet_learned));

InferredLabels = find_mpe(engine_learned2, testData{1});
% A = cell2num(mpe) ; 
% B = folds{foldNum}.testLabels{1};
% find ( folds{foldNum}.testLabels{1} ~= A(1,:) );

%% Save results to a mat file
timestamp = datestr(now, 'dd-mm-yyyy_HH.MM.SS');
outFile = sprintf('InferredLabels_F%d_%s.mat', foldNum, timestamp);
save(outFile, 'InferredLabels','-v7.3');   

output.bnet_learned = bnet_learned;
output.LL = LL;
output.engine_learned = engine_learned;

outFile = sprintf('output_F%d_%s.mat', foldNum, timestamp);
save(outFile, 'output','-v7.3');   
end