import tensorflow as tf
import transformers

import data

model_name = "FacebookAI/roberta-base"
tokenizer = transformers.RobertaTokenizer.from_pretrained(model_name)
# out = tokenizer("Hello world")
# input_ids = out['input_ids']
# attention_mask = out['attention_mask']
train, test = data.load_train_test_set()
type(train)
#
bert_model = transformers.TFRobertaModel.from_pretrained(model_name)
x = bert_model(input_ids=tf.convert_to_tensor(train.data['input_ids']),
               attention_mask=tf.convert_to_tensor(train.data[
                                                       'attention_mask']))

x1 = tf.keras.layers.Dropout(0.1)(x[0])
x1 = tf.keras.layers.Conv1D(768, 2,padding='same')(x1)
x1 = tf.keras.layers.LeakyReLU()(x1)
x1 = tf.keras.layers.Dense(1)(x1)
x1 = tf.keras.layers.Flatten()(x1)
x1 = tf.keras.layers.Activation('softmax')(x1)
