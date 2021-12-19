Navigate the python terminal to the CSE 8803 Code submission directory: 

Then, to train the models
1. Run python train_gen_model.py (--gpu) --save
2. Run python train_classifier_model.py (--gpu) --save
3. Run python train_all_modules.py (--gpu) --save

This will generate learning curves, accuracy plots and some metrics like accuracy, precision and recall (in case of train_classifier_model.py). Will
also produce some generated comments during the training itself.

(Make sure to keep all dataset and model related hyperparameters the same across all files. Just better to not change anything in the files.) 

To evaluate, use the following files after the above steps have been completed. 

1. train_perplexity_gen_model.py: To evaluate perplexity of the generator over training set.
2. valid_perplexity_gen_model.py: To evaluate perplexity of the generator over validation set.
3. valid_all_modules.py: To generate comments using comments from validation set. 
4. valid_classifier_modle.py: To evaluate classifier accuracy, precision, recall on the validation set. 

Please email pranjalg96@gmail.com if you have any problems/questions in running the code. Thanks!

