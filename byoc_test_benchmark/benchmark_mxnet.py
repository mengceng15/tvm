import mxnet as mx
import gluonnlp
import numpy as np

batch_size = 1
seq_length = 128

# Instantiate a BERT classifier using GluonNLP
model_name = "bert_12_768_12"
dataset = "book_corpus_wiki_en_uncased"
model, _ = gluonnlp.model.get_model(
    name=model_name,
    dataset_name=dataset,
    pretrained=True,
    use_pooler=True,
    use_decoder=False,
    use_classifier=False,
)

# Convert the MXNet model into TVM Relay format
shape_dict = {
    "data0": (batch_size, seq_length),
    "data1": (batch_size, seq_length),
    "data2": (batch_size,),
}
input_shape = (shape_dict["data0"], shape_dict["data1"], shape_dict["data2"])

# Feed input data
data = np.random.uniform(size=input_shape[0])
token_types = np.random.uniform(size=input_shape[1])
valid_length = np.array([seq_length] * batch_size)

import time

def warmup():
    for i in range(200):
        model(mx.nd.array(data), mx.nd.array(token_types), mx.nd.array(valid_length))

def x():
    for i in range(1000):
        model(mx.nd.array(data), mx.nd.array(token_types), mx.nd.array(valid_length))

warmup()
start = time.time()
x()
end = time.time()
print("time:", (end-start)/1000)