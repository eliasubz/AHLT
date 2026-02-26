#! /bin/bash

AHLT=../..
rm -rf *.out *.stats *.mem *.svm *.idx

for C in 0.1 1 10 100 1000; do
  # train MEM model
  echo "Training MEM model $C ..."
  python3 train.py train.feat model-$C.mem C=$C
done
  
for C in 0.1 1 10 100 1000; do
  echo "Training SVM model $C ..."
  python3 train.py train.feat model-$C.svm C=$C  &
done
wait

for C in 0.1 1 10 100 1000; do  
  # run MEM model
  echo "Running MEM model $C ..."
  python3 predict.py devel.feat model-$C.mem > devel-MEM.out
  python3 predict.py test.feat model-$C.mem > test-MEM.out
  # evaluate MEM results
  echo "Evaluating MEM results $C ..."
  python3 $AHLT/util/evaluator.py DDI $AHLT/data/devel devel-MEM.out > devel-MEM-$C.stats
  python3 $AHLT/util/evaluator.py DDI $AHLT/data/test test-MEM.out > test-MEM-$C.stats

  # run SVM model
  echo "Running SVM model $C ..."
  python3 predict.py devel.feat model-$C.svm > devel-SVM.out
  python3 predict.py test.feat model-$C.svm > test-SVM.out
  # evaluate SVM results
  echo "Evaluating SVM results $C..."
  python3 $AHLT/util/evaluator.py DDI $AHLT/data/devel devel-SVM.out > devel-SVM-$C.stats
  python3 $AHLT/util/evaluator.py DDI $AHLT/data/test test-SVM.out > test-SVM-$C.stats
done
