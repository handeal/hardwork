[bnet_learned, LL, engine_learned] = testScript(1, 15, 14, 2);

load folds;
Onodes = 4:17;
foldNum = 1;
testData = cell(1,1);
testData{1}(Onodes,:) = num2cell(folds{foldNum}.testFeatMat{1}+1); 

bnet_learned.observed = [Onodes];
engine_learned2 = smoother_engine(jtree_2TBN_inf_engine(bnet_learned));

mpe = find_mpe(engine_learned2, testData{1});
A = cell2num(mpe) ; 
B = folds{foldNum}.testLabels{1};
find ( folds{foldNum}.testLabels{1} ~= A(1,:) );

InferredLabels = mpe;
timestamp = datestr(now, 'dd-mm-yyyy_HH.MM.SS');
outFile = sprintf('mpe_F%d_%s.mat', numFold, timestamp);
save(outFile, 'InferredLabels','-v7.3');   
