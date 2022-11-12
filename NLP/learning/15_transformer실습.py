path = 'C:/pytest'

import pandas as pd
df = pd.read_csv('C:/pytest/chatdata_small.csv', encoding='cp949')
df.shape

inputs = df.iloc[:,0]
outputs = df.iloc[:,1]
inputs, outputs = list(inputs), list(outputs)

outputs_input = df.A.apply(lambda x: '<SOS> '+x+' <EOS>')
outputs_target = df.A.apply(lambda x: x+' <EOS>')

from tensorflow.keras.preprocessing.text import Tokenizer
inputs_series = pd.Series(inputs)
inputs_outputs = pd.concat([inputs_series, outputs_input], axis = 0)
tokenizer = Tokenizer(lower=False)
tokenizer.fit_on_texts(inputs_outputs)
word_index = tokenizer.word_index

import os
import pickle

if os.path.exists('C:/pytest/data/model_name'):
    print('이미 경로에 모델이 있습니다.')
else:
    os.makedirs('C:/pytest/data/model_name', exist_ok=True)
    print('경로에 폴더 만들었습니다.')

with open('C:/pytest/data/model_name/transformer.pickle','wb')as file:
    pickle.dump(tokenizer, file, protocol = pickle.HIGHEST_PROTOCOL)

encoder_input = tokenizer.texts_to_sequences(list(inputs))

decoder_input = tokenizer.texts_to_sequences(list(outputs_input))
decoder_target = tokenizer.texts_to_sequences(list(outputs_target))

print('\nResult of decoder_input sequencing: ')
print(outputs_input[0], decoder_input[0])
print(outputs_input[1], decoder_input[1])
print(outputs_input[2], decoder_input[2])

print('\nResult of decoder_target sequencing: ')
print(outputs_target[0], decoder_target[0])
print(outputs_target[1], decoder_target[1])
print(outputs_target[2], decoder_target[2])

sentence_max_length = inputs_outputs.apply(lambda x: len(x.split())).max()
print(sentence_max_length)

from tensorflow.keras.preprocessing.sequence import pad_sequences
encoder_input_pad = pad_sequences(encoder_input, maxlen=sentence_max_length, padding = 'post')
decoder_input_pad = pad_sequences(decoder_input, maxlen=sentence_max_length, padding = 'post')
decoder_target_pad = pad_sequences(decoder_target, maxlen=sentence_max_length, padding = 'post')

print('\nencoder_input_pad shape: ', encoder_input_pad.shape)
print("inputs: ", inputs[1])
print("encoder_input: ", encoder_input[1])
print("encoder_input_pad: ", encoder_input_pad[1])
print('\ndecoder_input_pad shape: ', decoder_input_pad.shape)
print("outputs_input: ", outputs_input[1])
print("decoder_input: ", decoder_input[1])
print("decoder_input_pad: ", decoder_input_pad[1])
print('\ndecoder_target_pad shape: ', decoder_target_pad.shape)
print("outputs_target: ", outputs_target[1])
print("decoder_target: ", decoder_target[1])
print("decoder_target_pad: ", decoder_target_pad[1])

import tensorflow as tf
import numpy as np
import enum
import re
import os
import json

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

SEED_NUM = 1234
tf.random.set_seed(SEED_NUM)

PAD_INDEX = 0
STD_INDEX = 1
END_INDEX = 2

index_inputs = encoder_input_pad
index_outputs = decoder_input_pad
index_targets = decoder_target_pad

char2idx_dict = word_index
idx2char_dict = {y: x for x,y in word_index.items()}

char2idx_dict['<PAD>'] = 0
char2idx_dict['<SOS>'] = char2idx_dict['SOS']
del char2idx_dict['SOS']
char2idx_dict['<END>'] = char2idx_dict['EOS']
del char2idx_dict['EOS']

idx2char_dict[0] = '<PAD>'
idx2char_dict[1] = '<SOS>'
idx2char_dict[2] = '<END>'


prepro_configs = dict({'char2idx':char2idx_dict, 'idx2char':idx2char_dict, 
                       'vocab_size':len(word_index), 'pad_symbol': '<PAD>', 
                       'std_symbol': '<SOS>', 'end_symbol': '<END>'})
print(prepro_configs)

char2idx = prepro_configs['char2idx']
end_index = prepro_configs['end_symbol'] 
vocab_size = prepro_configs['vocab_size']
BATCH_SIZE = 2
MAX_SEQUENCE = 25 
EPOCHS = 10
VALID_SPLIT = 0.1



kargs = {'model_name': 'transfomer',
'num_layers': 2, 
'd_model': 512,
'num_heads': 8, 
'dff': 2048,
'input_vocab_size': vocab_size, 
'target_vocab_size': vocab_size,
'maximum_position_encoding': MAX_SEQUENCE,
'end_token_idx': char2idx[end_index], 
'rate': 0.1
}

#패딩 0으로 된 부분을 1로 변환시킴
def create_padding_mask(seq):
    mask = tf.cast(tf.math.equal(seq, 0), tf.float32)

    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp) # 인코더 패딩 마스크
    dec_padding_mask = create_padding_mask(inp)
    
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask

enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(index_inputs, index_outputs)

def get_angles(pos, i, d_model):
# PE = sin(pos/(10000^(2i/d_model))) 수식의 pos/(10000^(2i/d_model)) 부분
# pos는 포지션에 대한 인덱스 위치 리스트, i는 차원 리스트, d_model은 단어의 차원 수
    angle_rates = 1 / np.power(10000, (2 * i//2) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
# 단어의 위치 정보를 생성
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :],d_model)

# 인덱스가 짝수(2i)인 경우는 sin 함수를, 홀수(2i+1)인 경우는 cos 함수를 사용하여 구분
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...] # 차원을 늘린다 shape: (3, 512) --> (1, 3, 512)
    
    return tf.cast(pos_encoding, dtype=tf.float32)

def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True) # Q행렬과 전치된 K행렬을 내적연산하여 Attention Score를 구한다
    dk = tf.cast(tf.shape(k)[-1], tf.float32) # K행렬의 차원 수(열의 수)을 구한다
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk) # Key 벡터의 차원 수의 제곱근으로 나눠 크기를 줄인다
    if mask is not None:
        scaled_attention_logits += (mask * -1e9) # -10의 9승이라는 매우 작은 값을 mask와 곱한 뒤, 더한다
# softmax 함수를 거치면서 매우 작은 값은 0으로 마스킹 된다(우삼각), 이것으로 자신보다 뒤에 나오는 단어는 참조되지 못한다
# 그 외의 양의 값은 확률 정보가 된다(하삼각)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
# 확률값 Attention Score에 Value 벡터로 가중합을 수행한다
    output = tf.matmul(attention_weights, v)
    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, **kargs): # 초기화 함수
        super(MultiHeadAttention, self).__init__() # layers.Layer의 __init__() 메소드 호출하여 클래스에서 layer 생성 준비
        self.num_heads = kargs['num_heads'] # 어텐션 헤드 수. 8
        self.d_model = kargs['d_model'] # 단어의 차원 수. 512
        assert self.d_model % self.num_heads == 0 # d_model의 차원 수는 헤드의 개수로 나머지 없이 나뉘어야 함
        self.depth = self.d_model // self.num_heads # 각 헤드에 입력될 벡터의 차원 수를 둘을 나눈 몫으로 결정
        # query, key, value 가중치 레이어 생성. input 결과를 받을 수 있도록 차원 수를 동일하게 맞춘다
        self.wq = tf.keras.layers.Dense(kargs['d_model'])
        self.wk = tf.keras.layers.Dense(kargs['d_model'])
        self.wv = tf.keras.layers.Dense(kargs['d_model'])
        self.dense = tf.keras.layers.Dense(kargs['d_model'])
        
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
        
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0] # batch size를 구한다
        q = self.wq(q) # (batch_size, seq_len, d_model)
        k = self.wk(k) # (batch_size, seq_len, d_model)
        v = self.wv(v) # (batch_size, seq_len, d_model)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3]) # (batch_size, num_heads, seq_len_q, depth) → (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model)) # 4D → 3D 변환. (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention) # 출력층 (batch_size, seq_len_q, d_model)
        return output, attention_weights


def feed_forward_network(**kargs):
    return tf.keras.Sequential([tf.keras.layers.Dense(kargs['dff'], activation='relu'), tf.keras.layers.Dense(kargs['d_model'])])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, **kargs):                               # 초기화
        super(EncoderLayer, self).__init__() # layers.Layer의 __init__() 메소드 호출하여 클래스에서 layer 생성 준비

        self.mha = MultiHeadAttention(**kargs) # 멀티 헤드 어텐션 레이어 생성
        self.ffn = feed_forward_network(**kargs) # 피드 포워드 네트워크 생성

        # 층 정규화(Layer Normalizaion)
        # LayerNormalization은 같은 층별로 평균을 0, 표준편차 1로 정규화한다
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Dropout 레이어 생성
        self.dropout1 = tf.keras.layers.Dropout(kargs['rate'])
        self.dropout2 = tf.keras.layers.Dropout(kargs['rate'])

    def call(self, x, mask):
        attn_output, _ = self.mha(x, x, x, mask) # 멀티 헤드 어텐션 레이어 수행
        attn_output = self.dropout1(attn_output) # 드롭아웃 수행
        out1 = self.layernorm1(x + attn_output) # 리지듀얼 커넥션 & 층 정규화 수행

        ffn_output = self.ffn(out1) # out1에 대해 피드포워드 연산 수행
        ffn_output = self.dropout2(ffn_output) # 드롭아웃 수행
        out2 = self.layernorm2(out1 + ffn_output) # 리지듀얼 커넥션 & 층 정규화 수행

        return out2

# 디코더레이어 준비
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, **kargs):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(**kargs) # 멀티 헤드 어텐션 레이어 1 생성
        self.mha2 = MultiHeadAttention(**kargs) # 멀티 헤드 어텐션 레이어 2 생성
        self.ffn = feed_forward_network(**kargs) # 피드 포워드 네트워크 생성

        # 층 정규화(Layer Normalizaion)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Dropout 레이어 생성
        self.dropout1 = tf.keras.layers.Dropout(kargs['rate'])
        self.dropout2 = tf.keras.layers.Dropout(kargs['rate'])
        self.dropout3 = tf.keras.layers.Dropout(kargs['rate'])
    
    
    def call(self, x, enc_output, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask) # 멀티 헤드 어텐션 레이어 1 수행
        attn1 = self.dropout1(attn1) # 드롭아웃 수행
        out1 = self.layernorm1(attn1 + x) # 층 정규화 및 리지듀얼 커넥션 수행

        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask) # 멀티 헤드 어텐션 레이어 2 수행
        attn2 = self.dropout2(attn2) # 드롭아웃 수행
        out2 = self.layernorm2(attn2 + out1) # 층 정규화 및 리지듀얼 커넥션 수행

        ffn_output = self.ffn(out2) # out2에 대해 피드포워드 연산 수행
        ffn_output = self.dropout3(ffn_output) # 드롯아웃 수행
        out3 = self.layernorm3(ffn_output + out2) # 층 정규화 및 리지듀얼 커넥션 수행

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, **kargs):
        super(Encoder, self).__init__()
        self.d_model = kargs['d_model'] # 단어의 임베딩 차원
        self.num_layers = kargs['num_layers'] # 사용할 인코더 레이어 개수
        # 워드 임베딩 레이어 생성 (1)
        self.embedding = tf.keras.layers.Embedding(input_dim=kargs['input_vocab_size'], output_dim=self.d_model)
        # 포지셔널 인코딩 레이어 생성 (2)
        self.pos_encoding = positional_encoding(position=kargs['maximum_position_encoding'], d_model=self.d_model)

        # 인코더 레이어 생성. num_layers 수만큼 리스트 배열로 만든다 (3)
        self.enc_layers = [EncoderLayer(**kargs) for _ in range(self.num_layers)]
        # 드롭아웃 레이어 생성
        self.dropout = tf.keras.layers.Dropout(kargs['rate'])

    def call(self, x, mask):
        seq_len = tf.shape(x)[1] # 입력한 벡터의 seq_len를 받는다

        x = self.embedding(x) # 워드 임베딩 수행 (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32)) # 가중치 곱하기(옵션). 각 워드 임베딩에 대해 스케일을 맞추는 과정
        x += self.pos_encoding[:, :seq_len, :] # 입력 벡터의 seq_len까지 포지션 임베딩 정보를 더하는 포지셔널 인코딩
        x = self.dropout(x) # 드롭아웃 수행(옵션)

        # 인코더 레이어 연산을 반복하는 부분. __init__() 함수에서 생성된 복수의 인코더 레이어에 대하여 실제 수행이 이루어진다

        for i in range(self.num_layers): # 이제까지의 과정이 적용된 입력 벡터를 num_layers 수만큼
            x = self.enc_layers[i](x, mask) # 인코더 레이어의 i번째 리스트 배열에 패딩 마스크와 함께 입력
        return x

class Decoder(tf.keras.layers.Layer):
    def __init__(self, **kargs):
        super(Decoder, self).__init__()
        self.d_model = kargs['d_model']
        self.num_layers = kargs['num_layers']
        # 워드 임베딩 레이어 생성 (1)
        self.embedding = tf.keras.layers.Embedding(input_dim=kargs['target_vocab_size'], output_dim=self.d_model)
        # 포지셔널 인코딩 레이어 생성 (2)
        self.pos_encoding = positional_encoding(position=kargs['maximum_position_encoding'], d_model=self.d_model)
        # 디코더 레이어 생성. num_layers 수만큼 리스트 배열로 만든다 (3)
        self.dec_layers = [DecoderLayer(**kargs) for _ in range(self.num_layers)]
        # 드롭아웃 레이어 생성
        self.dropout = tf.keras.layers.Dropout(kargs['rate'])
        
    def call(self, x, enc_output, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {} # 딕셔너리 초기화
        x = self.embedding(x) # 워드 임베딩 수행 (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32)) # 가중치 곱하기(옵션). 각 워드 임베딩에 대해 스케일을 맞춘다
        x += self.pos_encoding[:, :seq_len, :] # 입력 벡터의 seq_len까지 포지션 임베딩 정보를 더하는 포지셔널 인코딩
        x = self.dropout(x) # 드롭아웃 수행(옵션)
        
        for i in range(self.num_layers): # 이제까지의 과정이 적용된 입력 벡터를 num_layers 수만큼 아래를 진행
            x, block1, block2 = self.dec_layers[i](x, enc_output, look_ahead_mask, padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1 # 첫번째 어텐션의 가중치
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2 # 두번째 어텐션의 가중치
        
        return x, attention_weights

class Transformer(tf.keras.Model):
    def __init__(self, **kargs):
        super(Transformer, self ).__init__(name=kargs['model_name'])
        self.end_token_idx = kargs['end_token_idx'] # 종료 표지 숫자 '2' 저장

        self.encoder = Encoder(**kargs) # 인코더 생성
        self.decoder = Decoder(**kargs) # 디코더 생성

        self.final_layer = tf.keras.layers.Dense(kargs['target_vocab_size'])

    def call(self, x):
        inp, tar = x
        enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(inp, tar)

        # 인코더 결과 출력 (batch_size, inp_seq_len, d_model)
        enc_output = self.encoder(inp, enc_padding_mask)

        # 디코더 결과 출력 (batch_size, tar_seq_len, d_model)
        dec_output, _ = self.decoder(tar, enc_output, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output) # (batch_size, tar_seq_len, target_vocab_size)

        return final_output

    def inference(self, x):
        inp = x
        tar = tf.expand_dims([STD_INDEX], axis=0)
        enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(inp, tar)
        enc_output = self.encoder(inp, enc_padding_mask)

        predict_tokens = list()
        for t in range(0, MAX_SEQUENCE):
            dec_output, _ = self.decoder(tar, enc_output, look_ahead_mask, dec_padding_mask)
            final_output = self.final_layer(dec_output)
            outputs = tf.argmax(final_output, axis=-1).numpy()
            pred_token = outputs[0][-1]
            if pred_token == self.end_token_idx:
                break
            predict_tokens.append(pred_token)
            tar = tf.expand_dims([STD_INDEX]+predict_tokens, axis=0)
            _, look_ahead_mask, dec_padding_mask = create_masks(inp, tar)
        return predict_tokens


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')

def loss(real,pred):
    mask = tf.math.logical_not(tf.math.equal(real,0))
    loss_ = loss_object(real,pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    
    return tf.reduce_mean(loss_)



def accuracy(real,pred):
    mask = tf.math.logical_not(tf.math.equal(real,0))
    mask = tf.expand_dims(tf.cast(mask, dtype=pred.dtype), axis = -1)
    pred *= mask
    acc = train_accuracy(real,pred)
    
    return tf.reduce_mean(acc)

model = Transformer(**kargs)
model.compile(optimizer = tf.keras.optimizers.Adam(1e-4), loss=loss, metrics=[accuracy])

earlystop_callback = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=10)

checkpoint_path = 'C:/pytest/data/' + 'transformer' + '/weights'
checkpoint_dir = os.path.dirname(checkpoint_path)

if os.path.exists(checkpoint_dir):
    print(f'{checkpoint_dir} -- Folder already exists \n')
else:
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f'{checkpoint_dir}--Folder create complete \n')
    
checkpointer = ModelCheckpoint(checkpoint_path, monitor = 'val_accuracy', verbose = 1,
                               save_best_only=True, save_weights_only=True)

history = model.fit([index_inputs, index_outputs], index_targets, batch_size = BATCH_SIZE, 
                    epochs = EPOCHS, validation_split= VALID_SPLIT,
                    callbacks = [earlystop_callback,checkpointer])

import matplotlib.pyplot as plt

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()
    
plot_graphs(history, 'accuracy')
print('hi')

pred = model.inference(np.array([decoder_input_pad[1]]))

result=''
for i in pred:
    result+=idx2char_dict[i]+' '
print("A:",result)