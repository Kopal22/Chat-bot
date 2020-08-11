# building a chatbot with deep NLP

#PART-1 DATA PRE-PROCESSING

#importing libraries
import numpy as np
import tensorflow as tf
import re
import time


#importing the dataset
lines=open('movie_lines.txt',encoding='utf-8',errors='ignore').read().split('\n')
conversations=open('movie_conversations.txt',encoding='utf-8',errors='ignore').read().split('\n')


#creating a dictionary that maps each line to its id
id2line={}
for line in lines:
    _line=line.split(' +++$+++ ')
    if len(_line)==5:
        id2line[_line[0]]=_line[4]
        
#creating a list of all conversations
conversation_ids=[]
for conversation in conversations:
    _conversation=conversation.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    conversation_ids.append(_conversation.split(','))
    
    
#getting separately the questions and the answers
questions=[]
answers=[]
for conversation in conversation_ids:
    for i in range(len(conversation)-1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])
        
        
#doing a first cleaning of the texts
def cleantext(text):
    text=text.lower()
    text=re.sub(r"i'm","i am",text)
    text=re.sub(r"he's","he is",text)
    text=re.sub(r"she's","she is",text)
    text=re.sub(r"he's","he is",text)
    text=re.sub(r"that's","that is",text)
    text=re.sub(r"what's","what is",text)
    text=re.sub(r"where's","where is",text)
    text=re.sub(r"\'ll","will",text)
    text=re.sub(r"\'ve","have",text)
    text=re.sub(r"\'re","are",text)
    text=re.sub(r"\'d","would",text)
    text=re.sub(r"won't","will not",text)
    text=re.sub(r"can't","cannot",text)
    text=re.sub(r"[-()~\"# ;:+={}<>|,?.]"," ",text)
    return text               
     

#cleaning questions
clean_questions=[]
for question in questions:
    clean_questions.append(cleantext(question))
    
    
#cleaning answers
clean_answers=[]
for answer in answers:
    clean_answers.append(cleantext(answer))
    
    
#creating a dictionary that maps each word to its occurences
word2count={}
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word]=1
        else:
            word2count[word]+=1
for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word]=1
        else:
            word2count[word]+=1
            
            
#creating dictionaries that map questions words and answers words to a unique integer
threshold=20
wordnumber=0
questionswords2int={}
for word,count in word2count.items():
    if count>=threshold:
        questionswords2int[word]=wordnumber
        wordnumber+=1

wordnumber=0
answerswords2int={}
for word,count in word2count.items():
    if count>=threshold:
        answerswords2int[word]=wordnumber
        wordnumber+=1
        
        
#Adding the last tokens to these dictionaries
tokens=['<PAD>','<EOS>','<OUT>','<SOS>']
for token in tokens:
    questionswords2int[token]=len(questionswords2int)+1
    
for token in tokens:
    answerswords2int[token]=len(answerswords2int)+1
        
    
#Creating the inverse dictionary of the answerswords2int dictionary
answersint2words={w_i:w for w,w_i in answerswords2int.items()}

#Adding the end of string tokens to the end of every answer
for i in range(len(clean_answers)):
    clean_answers[i]+=' <EOS>'
    

#Translating all the questions and answers into integers
#and replacing all the words that are filtered out by <OUT>
questions_into_int=[]
for question in clean_questions:
    ints=[]
    for word in question.split():
        
        if word not in questionswords2int:
            ints.append(questionswords2int['<OUT>'])
        else:
            ints.append(questionswords2int[word])
    questions_into_int.append(ints)
    
answers_into_int=[]
for answer in clean_answers:
    ints=[]
    for word in answer.split():
        if word not in answerswords2int:
            ints.append(answerswords2int['<OUT>'])
        else:
            ints.append(answerswords2int[word])
    answers_into_int.append(ints)
    
    
#sorting questions and answers by the length of questions
sorted_clean_questions=[]
sorted_clean_answers=[]
for length in range(1,25+1):
    for i in enumerate(questions_into_int):
        if len(i[1])==length:
            sorted_clean_questions.append(questions_into_int[i[0]])
            sorted_clean_answers.append(answers_into_int[i[0]])


#PART-2 BUILDING THE SEQ-2SEQ MODEL

# Creating placeholders for the inputs and targets
def model_inputs():
    inputs=tf.placeholder(tf.int32,[None,None],name='input')
    targets=tf.placeholder(tf.int32,[None,None],name='target')
    lr=tf.placeholder(tf.float32,name='learning_rate')
    keep_prob=tf.placeholder(tf.float32,name='keep_prob')
    return inputs,targets,lr,keep_prob


#preprocessing the targets
def preprocess_targets(targets,word2int,batch_size):
    left_side=tf.fill([batch_size,1],word2int['<SOS>'])
    right_side=tf.strided_slice(targets,[0,0],[batch_size,-1],[1,1])
    preprocessed_targets=tf.concat([left_side,right_side],1)
    return preprocessed_targets
    

#Creating the encoder Rnn layer
def encoder_rnn(rnn_inputs,rnn_size,num_layers,keep_prob,sequence_length):
    lstm=tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout=tf.contrib.rnn.DropoutWrapper(lstm,input_keep_prob=keep_prob)
    encoder_cell=tf.contrib.rnn.MultiRNNCell([lstm_dropout]*num_layers)
    _,encoder_state=tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,cell_bw=encoder_cell,
                                                    sequence_length=sequence_length,inputs=rnn_inputs,
                                                    dtype=tf.float32)
    return encoder_state


#Decoding the training set
def decode_training_set(encoder_state,decoder_cell,decoder_embedded_input,sequence_length,decoding_scope,output_function,keep_prob,batch_size):
    attention_states=tf.zeros([batch_size,1,decoder_cell.output_size])
    attention_keys,attention_values,attention_score_function,attention_construct_function=tf.contrib.seq2seq.prepare_attention(attention_states,attention_option="bahdanau",num_units=decoder_cell.output_size)
    training_decoder_function=tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                            attention_keys,
                                                                            attention_values,
                                                                            attention_score_function,
                                                                            attention_construct_function,
                                                                            name="attn_dec_train")
    decoder_output,decoder_final_state,decoder_final_context_state=tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                          training_decoder_function,
                                                                                                          decoder_embedded_input,
                                                                                                          sequence_length,
                                                                                                          scope=decoding_scope)
    decoder_output_dropout=tf.nn.dropout(decoder_output,keep_prob)
    return output_function(decoder_output_dropout)



























        