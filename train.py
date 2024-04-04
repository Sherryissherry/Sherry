from finbert.finbert import *
from transformers import AutoModelForSequenceClassification
import argparse
import os

model = AutoModelForSequenceClassification.from_pretrained("models/classifier_model/finbert-sentiment",
                                                           num_labels=3,
                                                           cache_dir=None)

config = Config(data_dir="data/train",
                bert_model=model,
                num_train_epochs=4,
                model_dir="models/classifier_model/finbert-sentiment/finetuned",
                max_seq_length = 48,
                train_batch_size = 32,
                learning_rate = 2e-5,
                output_mode='classification',
                warm_up_proportion=0.2,
                local_rank=-1,
                discriminate=True,
                gradual_unfreeze=True)

finbert = FinBert(config)
finbert.base_model = 'bert-base-uncased'
finbert.config.discriminate=True
finbert.config.gradual_unfreeze=True

finbert.prepare_model(label_list=['positive','negative','neutral'])

train_data = finbert.get_data('train')
model = finbert.create_the_model()

freeze = 6

for param in model.bert.embeddings.parameters():
    param.requires_grad = False
    
for i in range(freeze):
    for param in model.bert.encoder.layer[i].parameters():
        param.requires_grad = False

trained_model = finbert.train(train_examples = train_data, model = model)