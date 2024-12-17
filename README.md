# Project description: NLP task with RuBERT and LoRA for RuCoLa
In this project the goal was to fine-tune the RuBERT model for the RuCoLa task, a binary classification task predicting cognitive load in Russian texts. The model was evaluated using accuracy and MCC metrics. Additionally, LoRA was applied to the model to optimize training and reduce the number of trainable parameters.


###Input:

in_domain_train.csv: Training dataset with labeled text samples.

in_domain_dev.csv: Evaluation dataset for testing the model's performance.

###Output:

The model predicts the cognitive load class for each text.

###Approach:

Fine-tuning the RuBERT model using the DeepPavlov/rubert-base-cased version, augmented with LoRA for parameter efficiency. LoRA was used to reduce the number of trainable parameters by applying low-rank updates to certain layers.

##Model Description. 
The model utilized:

RuBERT: Pretrained BERT model for the Russian language.

LoRA: A technique applied to reduce training time and model size by applying low-rank updates to the attention layers. The LoraConfig was configured with r=16, lora_alpha=32, and lora_dropout=0.2.

###Steps:

1) Data Preprocessing: text was cleaned, lemmatized, and augmented using synonym replacement. Additional features like sentence length and punctuation count were also generated.

2) LoRA Integration: LoRA was applied to the pretrained RuBERT model to make the training process more efficient by adapting low-rank matrices in specific transformer layers. This resulted in a more compact model with fewer trainable parameters.

3) Training: the model was trained with class weights to handle class imbalance and used a weighted CrossEntropyLoss. A learning rate scheduler was implemented for optimal training performance.

##Data Analysis
Basic statistics were calculated, including class distribution, average sentence length, and the number of unique words in the dataset.

##Model Training
The model was fine-tuned with LoRA, and training was conducted for 60 epochs using CrossEntropyLoss with class weights. The model was deployed on a GPU for faster computation. During training, model were logged for each epoch.

##Results
The model was evaluated on in_domain_dev.csv using accuracy and MCC. LoRA helped achieve efficient training while maintaining high performance.

##Model logging with MLflow

The model was logged for each epoch.

Also at the end of the project model performance and configuration were logged using MLflow. 

This included: 

Logging LoRA configuration parameters.

Recording key model metrics such as accuracy and MCC.
