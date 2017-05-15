./svm_multiclass_learn.bin -c 5000 ../data/svm_trainfull.txt ./tmp.model
./svm_multiclass_classify.bin ../data/svm_testfull.txt ./tmp.model ./predictions
