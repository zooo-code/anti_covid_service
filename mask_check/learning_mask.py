import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from matplotlib import pyplot as plt
import pandas as pd


# class CustomCallback(tf.keras.callbacks.Callback):
#
#
#     def on_train_batch_end(self, batch, logs=None):
#         keys = list(logs.keys())
#         print("...Training: end of batch {}; got log keys: {}".format(batch, keys))
#
#
#
#     def on_test_batch_end(self, batch, logs=None):
#         keys = list(logs.keys())
#         print("...Evaluating: end of batch {}; got log keys: {}".format(batch, keys))


path_dir1 = 'D:/data/without_mask/'
path_dir2 = 'D:/data/with_mask/'

file_list1 = os.listdir(path_dir1)  # path에 존재하는 파일 목록 가져오기
file_list2 = os.listdir(path_dir2)

file_list1_num = len(file_list1)
file_list2_num = len(file_list2)

file_num = file_list1_num + file_list2_num

# %% 이미지 전처리
num = 0;
all_img = np.float32(np.zeros((file_num, 224, 224, 3)))
all_label = np.float64(np.zeros((file_num, 1)))

for img_name in file_list1:
    img_path = path_dir1 + img_name
    img = load_img(img_path, target_size=(224, 224))

    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    all_img[num, :, :, :] = x

    all_label[num] = 0  # nomask
    num = num + 1

for img_name in file_list2:
    img_path = path_dir2 + img_name
    img = load_img(img_path, target_size=(224, 224))

    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    all_img[num, :, :, :] = x

    all_label[num] = 1  # mask
    num = num + 1

# 데이터셋 섞기(적절하게 훈련되게 하기 위함)
n_elem = all_label.shape[0]
indices = np.random.choice(n_elem, size=n_elem, replace=False)

all_label = all_label[indices]
all_img = all_img[indices]

# 훈련셋 테스트셋 분할
num_train = int(np.round(all_label.shape[0] * 0.8))
num_test = int(np.round(all_label.shape[0] * 0.2))

train_img = all_img[0:num_train, :, :, :]
test_img = all_img[num_train:, :, :, :]

train_label = all_label[0:num_train]
test_label = all_label[num_train:]

# %%
# create the base pre-trained model
IMG_SHAPE = (224, 224, 3)

base_model = ResNet50(input_shape=IMG_SHAPE, weights='imagenet', include_top=False)
base_model.trainable = False
base_model.summary()
print("Number of layers in the base model: ", len(base_model.layers))

flatten_layer = Flatten()
dense_layer1 = Dense(128, activation='relu')
bn_layer1 = BatchNormalization()
dense_layer2 = Dense(1, activation=tf.nn.sigmoid)

model = Sequential([
    base_model,
    flatten_layer,
    dense_layer1,
    bn_layer1,
    dense_layer2,
])

base_learning_rate = 0.001

from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# 일반 적인 학습 성능 지표
# model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# recall f1 등 추가 한 지표
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy',f1_m,precision_m, recall_m])

model.summary()

history = model.fit(train_img, train_label, epochs=10, batch_size=100, validation_data=(test_img, test_label))


history.history.keys()

# save model
model.save("mask_model1103.h5")


# def vis(history, name):
#     plt.title(f"{name.upper()}")
#     plt.xlabel('epochs')
#     plt.ylabel(f"{name.lower()}")
#     value = history.history.get(name)
#     val_value = history.history.get(f"val_{name}", None)
#     epochs = range(1, len(value) + 1)
#     plt.plot(epochs, value, 'b-', label=f'training {name}')
#     if val_value is not None:
#         plt.plot(epochs, val_value, 'r:', label=f'validation {name}')
#     plt.legend(loc='upper center', bbox_to_anchor=(0.05, 1.2), fontsize=10, ncol=1)
#
#
# def plot_history(history):
#     key_value = list(set([i.split("val_")[-1] for i in list(history.history.keys())]))
#     plt.figure(figsize=(12, 4))
#     for idx, key in enumerate(key_value):
#         plt.subplot(1, len(key_value), idx + 1)
#         vis(history, key)
#     plt.tight_layout()
#     plt.show()

# acc
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
# loss
loss = history.history['loss']
val_loss = history.history['val_loss']

# f1 스코어
f1_m = history.history['f1_m']
val_f1_m = history.history['val_f1_m']

#  precision_m
precision_m = history.history['precision_m']
val_precision_m = history.history['val_precision_m']

# recall_m
recall_m = history.history['recall_m']
val_recall_m = history.history['val_recall_m']

epochs = range(len(acc))

col  = ["precision_m","val_precision_m","recall_m","val_recall_m","f1_m","val_f1_m","mean"]

data = {"precision_m":precision_m,"val_precision_m":val_precision_m,"recall_m":recall_m,"val_recall_m":val_recall_m,"f1_m":f1_m,"val_f1_m":val_f1_m}
data_frame = pd.DataFrame(data)
data_frame=data_frame.append({"precision_m":np.mean(np.array(precision_m)),"val_precision_m":np.mean(np.array(val_precision_m)),"recall_m":np.mean(np.array(recall_m)),"val_recall_m":np.mean(np.array(val_recall_m)),"f1_m":np.mean(np.array(f1_m)),"val_f1_m":np.mean(np.array(val_f1_m))},ignore_index=True)
print(data_frame)

# plot_history(history)

# _loss, _acc, _precision, _recall, _f1score = model.evaluate(data.X_test, data.y_test, batch_size=100, verbose=1)
# print('loss: {:.3f}, accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1score: {:.3f}'.format(_loss, _acc, _precision, _recall, _f1score))



plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='testing acc')
plt.title('Training and testing accuracy')
plt.xlabel('epochs')
plt.ylabel('acc')
# plt.xlim([0, 10])      # X축의 범위: [xmin, xmax]
plt.ylim([0.97, 1])     # Y축의 범위: [ymin, ymax]
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='testing loss')
plt.title('Training and testing loss')
plt.xlabel('epochs')
plt.ylabel('loss')
# plt.xlim([0, 10])      # X축의 범위: [xmin, xmax]
plt.ylim([0, 0.07])     # Y축의 범위: [ymin, ymax]
plt.legend()
plt.figure()
#
#
# # f1 스코어
# plt.plot(epochs, f1_m, 'r', label='Training f1_m')
# plt.plot(epochs, val_f1_m, 'b', label='testing f1_m')
# plt.title('Training and testing f1_m')
# plt.xlabel('epochs')
# plt.ylabel('f1_m')
# # plt.xlim([0, 10])      # X축의 범위: [xmin, xmax]
# # plt.ylim([0.945, 1])     # Y축의 범위: [ymin, ymax]
# plt.legend()
# plt.figure()
#
# #precision_m
# plt.plot(epochs, precision_m, 'r', label='Training precision_m')
# plt.plot(epochs, val_precision_m, 'b', label='testing precision_m')
# plt.title('Training and testing precision_m')
# plt.xlabel('epochs')
# plt.ylabel('precision_m')
# # plt.xlim([0, 10])      # X축의 범위: [xmin, xmax]
# # plt.ylim([0, 0.055])     # Y축의 범위: [ymin, ymax]
# plt.legend()
# plt.figure()
#
# # recall_m
# plt.plot(epochs, recall_m, 'r', label='Training recall_m')
# plt.plot(epochs, val_recall_m, 'b', label='testing recall_m')
# plt.title('Training and testing recall_m')
# plt.xlabel('epochs')
# plt.ylabel('recall_m')
# # plt.xlim([0, 10])      # X축의 범위: [xmin, xmax]
# # plt.ylim([0.945, 1])     # Y축의 범위: [ymin, ymax]
# plt.legend()
# plt.figure()
#
plt.show()

print("Saved model to disk")