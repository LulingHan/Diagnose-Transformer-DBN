./svm_multiclass_learn.bin -c 5000 -t 1 $1 ./tmp.model
./svm_multiclass_classify.bin $2 ./tmp.model ./predictions
