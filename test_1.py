import numpy as np
import os
from datasets import load_dataset
import matplotlib.pyplot as plt

# 디렉토리 경로 설정
directory_path = "/Users/suminsim/Desktop/python/deep_running/homework_2/1.데이터/Training/02.라벨링데이터/TL_1.발화단위평가_경제활동_상품상거래"

# 디렉토리 내 JSON 파일 경로 리스트 생성
json_files = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.json')]

# JSON 파일을 datasets 라이브러리로 불러오기
dataset = load_dataset('json', data_files=json_files, split='train')

# 예시 데이터
data = [
    {
        'conversation_id': 60523,
        'metadata': {'topic': '경제활동, 상품/상거래', 'speakers': [{'id': 1842, 'speaker_type': 'human'}, {'id': 0, 'speaker_type': 'bot'}]},
        'utterances': [
            {'exchange_id': 'c60523.e1', 'utterance_id': 'c60523.u1', 'speaker_id': 1842, 'utterance_text': '이 기업의 공시제도는 어떤 내용을 알리는 거야?'},
            {'exchange_id': 'c60523.e1', 'utterance_id': 'c60523.u2', 'speaker_id': 0, 'utterance_text': '공시제도는 기업의 사업내용이나 재무상황, 영업실적 등을 투자자 등 이해관계자에게 알리는 제도입니다.'},
            {'exchange_id': 'c60523.e2', 'utterance_id': 'c60523.u3', 'speaker_id': 1842, 'utterance_text': '그렇군. 그럼 외국어 표기 공시는 뭐야?'},
            {'exchange_id': 'c60523.e2', 'utterance_id': 'c60523.u4', 'speaker_id': 0, 'utterance_text': '외국어 표기 공시는 사업내용이나 재무상황, 영업실적 등 기업의 경영 내용을 투자자 등 이해관계자에게 알리는 제도입니다.'},
            {'exchange_id': 'c60523.e3', 'utterance_id': 'c60523.u5', 'speaker_id': 1842, 'utterance_text': '공시제도는 어떤 목적을 가지고 있어?'},
            {'exchange_id': 'c60523.e3', 'utterance_id': 'c60523.u6', 'speaker_id': 0, 'utterance_text': '공시제도는 정보의 투명성과 공정성을 증진시키기 위해 도입되었습니다.'}
        ],
        'conversation_summary': '기업의 공시제도와 외국어 표기 공시에 대한 질문과 설명이 오가는 대화.',
        'conversation_evaluation': {
            'likeability': ['yes', 'yes', 'yes'],
            'sensibleness': ['yes', 'yes', 'yes']
        }
    }
]

# 전처리 함수
def preprocess(data):
    processed_data = []
    for conv in data:
        conversation_info = {
            'conversation_id': conv['conversation_id'],
            'topic': conv['metadata']['topic'],
            'speakers': [speaker['speaker_type'] for speaker in conv['metadata']['speakers']],
            'utterances': []
        }
        
        for utt in conv['utterances']:
            conversation_info['utterances'].append({
                'utterance_text': utt['utterance_text'],
                'speaker_id': utt['speaker_id']
            })
        
        conversation_info['conversation_summary'] = conv['conversation_summary']
        conversation_info['evaluation'] = conv['conversation_evaluation']
        
        processed_data.append(conversation_info)
    
    return processed_data

# 전처리 실행
preprocessed_data = preprocess(data)

# 단어 인덱싱
def build_vocab(data):
    word2index = {}
    index2word = {}
    index = 0
    for conv in data:
        for utt in conv['utterances']:
            for word in utt['utterance_text'].split():
                if word not in word2index:
                    word2index[word] = index
                    index2word[index] = word
                    index += 1
    return word2index, index2word

# 단어 인덱스 구축
word2index, index2word = build_vocab(preprocessed_data)

# 텍스트를 인덱스로 변환하는 함수
def text_to_sequence(text, word2index):
    return [word2index[word] for word in text.split() if word in word2index]

# 텍스트를 인덱스 배열로 변환
input_sequences = []
output_sequences = []

for conv in preprocessed_data:
    for utt in conv['utterances']:
        input_text = utt['utterance_text']
        output_text = input_text  # 예시로 입력과 출력이 동일하도록 설정
        input_sequences.append(text_to_sequence(input_text, word2index))
        output_sequences.append(text_to_sequence(output_text, word2index))

# 시퀀스의 최대 길이 찾기
max_length = max(len(seq) for seq in input_sequences)

# X_train과 y_train을 (배치 크기, 시퀀스 길이) 형태로 변환하기 위해 패딩 추가
input_sequences_padded = [np.pad(seq, (0, max_length - len(seq)), mode='constant') for seq in input_sequences]
output_sequences_padded = [np.pad(seq, (0, max_length - len(seq)), mode='constant') for seq in output_sequences]

# 리스트를 NumPy 배열로 변환
X_train_padded = np.array(input_sequences_padded)
y_train_padded = np.array(output_sequences_padded)
X_train_padded = np.expand_dims(X_train_padded, axis=-1)
# SimpleLSTM 모델 정의 (이전 코드 그대로 사용)

class SimpleLSTM:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 가중치 초기화
        # 가중치 초기화 수정
        self.Wf = np.random.randn(input_size + hidden_size, hidden_size) * 0.01  # (input_size + hidden_size, hidden_size)
        self.Wi = np.random.randn(input_size + hidden_size, hidden_size) * 0.01  # (input_size + hidden_size, hidden_size)
        self.Wo = np.random.randn(input_size + hidden_size, hidden_size) * 0.01  # (input_size + hidden_size, hidden_size)
        self.Wc = np.random.randn(input_size + hidden_size, hidden_size) * 0.01  # (input_size + hidden_size, hidden_size)



        print("input_size:", input_size)
        print("hidden_size:", hidden_size)
        
        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))

        self.Wh = np.random.randn(output_size, hidden_size) * 0.01
        self.bh = np.zeros((output_size, 1))

    def forward(self, X):
        batch_size, sequence_length, _ = X.shape

        h = np.zeros((batch_size, self.hidden_size))
        c = np.zeros((batch_size, self.hidden_size))
        y_hat = []

        for t in range(sequence_length):
            x_t = X[:, t, :]
            combined = np.concatenate((h, x_t), axis=1)  # (batch_size, input_size + hidden_size)

            print("combined.shape:", combined.shape)
            print("self.Wf.shape:", self.Wf.shape)
            # 여기에 np.dot의 연산을 제대로 확인
            ft = self.sigmoid(np.dot(combined, self.Wf) + self.bf.T)  # 수정된 부분: bf.T로 전치
            it = self.sigmoid(np.dot(combined, self.Wi) + self.bi.T)  # 수정된 부분: bi.T로 전치
            ot = self.sigmoid(np.dot(combined, self.Wo) + self.bo.T)  # 수정된 부분: bo.T로 전치
            c_tilde = np.tanh(np.dot(combined, self.Wc) + self.bc.T)  # 수정된 부분: bc.T로 전치

            c = ft * c + it * c_tilde
            h = ot * np.tanh(c)

            y = np.dot(h, self.Wh.T) + self.bh.T  # 수정된 부분: bh.T로 전치
            y_hat.append(y)

        return np.array(y_hat)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

# 모델 객체 생성
model = SimpleLSTM(input_size=31, hidden_size=128, output_size=len(word2index))

# 모델 훈련 및 손실 계산
epochs = 100
losses = []

for epoch in range(epochs):
    y_hat = model.forward(X_train_padded)  # 모델 예측
    loss = np.mean((y_hat - y_train_padded) ** 2)  # 손실 계산
    losses.append(loss)

# 손실 그래프 그리기
plt.plot(range(epochs), losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
