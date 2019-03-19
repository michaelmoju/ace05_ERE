# ace05_ERE
Entity/Relation/Event extraction on ACE 2005 corpus 

Some codes are refereced from emnlp2017-relation-extraction 
https://github.com/michaelmoju/emnlp2017-relation-extraction

---------------
Train the model
---------------
`python relation_train.py model_LSTMbaseline train "../resource/data/ace-2005/relationMention/*/adj/*.json" --epoch 100 --models_folder ./trainedmodels/exp3-units1-128/ --checkpoint True --tensorboard True`

