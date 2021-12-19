# DISQUS-Adversarial-Text-Generation
 Adversarial text generation to fool the DISQUS toxic comment filter. Please look at 'CSE8803_Final_Report.pdf'
 for more details about the model.

## Dataset
The data is present in DISQUS/disqus_dataset
1. disqus_data_filtered.csv: This is a subset of the Kaggle toxic comment classification dataset, with sentences only limited to 60 words. Contains
only the text with no labels. Used for training the conditional language model.
2. disqus_data_sup.csv: This is a supervised version of disqus_data_filtered.csv, with labels of 'toxic' or 'non-toxic' for each sentence. Used for training
the Realism and Adversarial modules.
3. disqus_data_valid.csv: Similar format as disqus_data_sup.csv, except these sentences were not seen during training. Used to perform all the evaluations listed in the evaluation section.

## Training

To train the models
1. Run python train_gen_model.py (--gpu) --save
2. Run python train_classifier_model.py (--gpu) --save
3. Run python train_all_modules.py (--gpu) --save

This will generate learning curves, accuracy plots and some metrics like accuracy, precision and recall (in case of train_classifier_model.py). Will
also produce some generated comments during the training itself.

(Make sure to keep all dataset and model related hyperparameters the same across all files. To run the default case, don't change anything.) 

## Evaluation
To evaluate, use the following files after the above steps have been completed.

1. train_perplexity_gen_model.py: To evaluate perplexity of the generator over training set.
2. valid_perplexity_gen_model.py: To evaluate perplexity of the generator over validation set.
3. gen_adversarial_comments_valid.py: To generate adversarial comments using comments from validation set.
4. valid_classifier_model.py: To evaluate classifier accuracy, precision, recall on the validation set. 

## Contact
Please email pranjalg96@gmail.com if you have any problems/questions in running the code. Thanks!
