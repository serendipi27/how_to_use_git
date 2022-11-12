path = 'C:/pytest/data/'

# !ls path

from tokenize import Token
import pandas as pd

data = pd.read_csv(path+'eng-kor/eng-kor_small.txt',names=['source','target'],sep='\t', encoding='utf-8')
print(data.shape)

data.target_input = data.target.apply(lambda x: '\t'+x+'\n')
data.target_target = data.target.apply(lambda x : x+'\n')
print('\ndata.target_input:\n',data.target_input)

max_src_len = data.source.apply(lambda x: len(x)).max()
print(max_src_len)

max_tar_len = data.target_input.apply(lambda x: len(x)).max()-2
print(max_tar_len)

from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer_source = Tokenizer(num_words = None, char_level=True, lower=False)
tokenizer_source.fit_on_texts(data.source)
word_index_source = tokenizer_source.word_index

print(word_index_source)

tokenizer_target = Tokenizer(num_words=None, char_level=True, lower=False)
tokenizer_target.fit_on_texts(data.target_input)
word_index_target = tokenizer_target.word_index

print(word_index_target)

encoder_input = tokenizer_source.texts_to_sequences(data.source)

decoder_input = tokenizer_target.texts_to_sequences(data.target_input)
decoder_target = tokenizer_target.texts_to_sequences(data.target_target)

from tensorflow.keras.preprocessing.sequence import pad_sequences
encoder_input = pad_sequences(encoder_input, maxlen=max_src_len, padding='post')
decoder_input = pad_sequences(decoder_input, maxlen=max_tar_len, padding='post')
decoder_target = pad_sequences(decoder_target, maxlen=max_tar_len, padding='post')

print(data.target_input[10])

from tensorflow.keras.utils import to_categorical
encoder_input = to_categorical(encoder_input, num_classes=len(word_index_source)+1)
decoder_input = to_categorical(decoder_input, num_classes=len(word_index_target)+1)
decoder_target = to_categorical(decoder_target, num_classes=len(word_index_target)+1)

# to_categorical & texts_to_matrix

# texts = ['우리 소망 꿈 통일']
# tokenizer = Tokenizer(num_words=4)
# tokenizer.fit_on_texts(texts)
# data = tokenizer.texts_to_sequences(texts)
# print(data)
# print(to_categorical(data, num_classes=4))

# print(decoder_input.shape)

#훈련용 Encoder

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
encoder_inputs = Input(shape=(None, len(word_index_source)+1))

#encoder의 의미를 decoder로 전달하기위해(은닉,셀) return_state=True
encoder_lstm = LSTM(256, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c] #encoder_상태 = [은닉, 셀]

# Decoder-input
decoder_inputs = Input(shape=(None,len(word_index_target)+1))
decoder_lstm = LSTM(units=256, return_sequences=True, return_state= True)
decoder_outputs,_,_=decoder_lstm(decoder_inputs, initial_state = encoder_states)
decoder_dense = Dense(len(word_index_target)+1, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss = 'categorical_crossentropy')
model.fit(x = [encoder_input, decoder_input], y = decoder_target, batch_size=64, epochs=1, validation_split=0.2)

encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states)

decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))
decoder_states_inputs = [decoder_state_input_h,decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(inputs = [decoder_inputs]+decoder_states_inputs, outputs=[decoder_outputs]+decoder_states)

index_to_src = dict((i,char) for char, i in word_index_source.items())
index_to_tar = dict((i, char) for char, i in word_index_target.items())
print(index_to_src)

import numpy as np

def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1,1,len(word_index_target)+1))
    target_seq[0,0, word_index_target['\t']]=1
    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens,h,c = decoder_model.predict([target_seq]+ states_value)
        sampled_token_index = np.argmax(output_tokens)

        if sampled_token_index ==0:
            sampled_token_index = 1

        sampled_char = index_to_tar[sampled_token_index]
        decoded_sentence += sampled_char
        
        if sampled_char == '\n' or len(decoded_sentence)>max_tar_len:
            stop_condition = True

        target_seq = np.zeros((1,1, len(word_index_target)+1))
        target_seq[0,0, sampled_token_index] = 1.
        states_value = [h,c]

    return decoded_sentence

for seq_index in [1,2,3]:
    input_seq = encoder_input[seq_index:seq_index+1]
    decoded_sentence = decode_sequence(input_seq)

    print(35 * "-")
    print('입력 문장:', data.source[seq_index])
    print('정답 문장:', data.target[seq_index][0:len(data.target[seq_index])])
    print('번역기가 번역한 문장:', decoded_sentence[:len(decoded_sentence)-1])



type(data)
data.source[1:2]