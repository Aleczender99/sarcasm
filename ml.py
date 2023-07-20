import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn.naive_bayes
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from transformers import BertTokenizer, TFBertModel
import numpy as np
import pickle
from sklearn.metrics import roc_curve, auc


# Naive Bayes model
def nb(train_padded, train_labels, test_padded, test_labels):
    model = sklearn.naive_bayes.MultinomialNB()
    model.fit(train_padded, train_labels)
    print("Accuracy score:", model.score(train_padded, train_labels))

    f = open('Models/NB.pickle', 'wb')
    pickle.dump(model, f)
    f.close()

    pred = model.predict(test_padded)
    disp = confusion_matrix(test_labels, np.round(pred), normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=disp)

    disp.plot()
    plt.savefig('SS/Confusion_Matrix_NB')

    probs = model.predict_proba(test_padded)[:, 1]
    fpr, tpr, thresholds = roc_curve(test_labels, probs)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.show()

    return model


# CNN model
def cnn(train_padded, train_labels, test_padded, test_labels, vocab_size, max_len):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size+1, 16, input_length=max_len),
        tf.keras.layers.Conv1D(128, 5, activation='relu'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    ch = tf.keras.callbacks.ModelCheckpoint('Models/CNN.h5', monitor='val_accuracy',
                                            verbose=1, save_best_only=True, mode='max')
    es = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=2)
    model.summary()
    history = model.fit(train_padded, np.array(train_labels), validation_data=(test_padded, np.array(test_labels)),
                        callbacks=[ch, es], epochs=10, verbose=2)

    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel("Epochs")
    plt.ylabel('accuracy')
    plt.legend(['accuracy', 'val_accuracy'])
    plt.savefig('SS/Accuracy_CNN')

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel("Epochs")
    plt.ylabel('loss')
    plt.legend(['loss', 'val_loss'])
    plt.savefig('SS/Loss_CNN')

    pred = model.predict(test_padded)
    disp = confusion_matrix(test_labels, np.round(pred), normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=disp)

    plt.figure()
    disp.plot()
    plt.savefig('SS/Confusion_Matrix_CNN')

    return model


# Bidirectional LSTM model
def lstm(train_padded, train_labels, test_padded, test_labels, vocab_size, max_len):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size+1, 16, input_length=max_len),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    ch = tf.keras.callbacks.ModelCheckpoint('Models/LSTM.h5', monitor='val_accuracy',
                                            verbose=1, save_best_only=True, mode='max')
    es = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=2)
    model.summary()
    history = model.fit(train_padded, np.array(train_labels), validation_data=(test_padded, np.array(test_labels)),
                        callbacks=[ch, es], epochs=10, verbose=2)

    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel("Epochs")
    plt.ylabel('accuracy')
    plt.legend(['accuracy', 'val_accuracy'])
    plt.savefig('SS/Accuracy_LSTM')

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel("Epochs")
    plt.ylabel('loss')
    plt.legend(['loss', 'val_loss'])
    plt.savefig('SS/Loss_LSTM')

    pred = model.predict(test_padded)
    disp = confusion_matrix(test_labels, np.round(pred), normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=disp)

    disp.plot()
    plt.savefig('SS/Confusion_Matrix_LSTM')

    return model


# Bidirectional GRU model
def gru(train_padded, train_labels, test_padded, test_labels, vocab_size, max_len):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size+1, 16, input_length=max_len),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    ch = tf.keras.callbacks.ModelCheckpoint('Models/GRU.h5', monitor='val_accuracy',
                                            verbose=1, save_best_only=True, mode='max')
    es = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=2)
    model.summary()
    history = model.fit(train_padded, np.array(train_labels), validation_data=(test_padded, np.array(test_labels)),
                        callbacks=[ch, es], epochs=10, verbose=2)

    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel("Epochs")
    plt.ylabel('accuracy')
    plt.legend(['accuracy', 'val_accuracy'])
    plt.savefig('SS/Accuracy_GRU')

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel("Epochs")
    plt.ylabel('loss')
    plt.legend(['loss', 'val_loss'])
    plt.savefig('SS/Loss_GRU')

    pred = model.predict(test_padded)
    disp = confusion_matrix(test_labels, np.round(pred), normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=disp)

    disp.plot()
    plt.savefig('SS/Confusion_Matrix_GRU')

    return model


def bert(train_data, train_labels, test_data, test_labels, max_len):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    train_ids = encoder(train_data, tokenizer, max_len)
    test_ids = encoder(test_data, tokenizer, max_len)

    train_ids = tf.convert_to_tensor(train_ids)
    test_ids = tf.convert_to_tensor(test_ids)
    test_labels = tf.convert_to_tensor(test_labels)
    train_labels = tf.convert_to_tensor(train_labels)

    bert_encoder = TFBertModel.from_pretrained('bert-base-uncased')
    input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    embedding = bert_encoder([input_word_ids])
    dense = tf.keras.layers.Lambda(lambda seq: seq[:, 0, :])(embedding[0])
    dense = tf.keras.layers.Dense(128, activation='relu')(dense)
    dense = tf.keras.layers.Dropout(0.2)(dense)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(dense)

    model = tf.keras.Model(inputs=[input_word_ids], outputs=output)
    model.compile(tf.keras.optimizers.Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    ch = tf.keras.callbacks.ModelCheckpoint('Models/Bert.h5', include_optimizer=False, monitor='val_accuracy',
                                            verbose=1, save_best_only=True, mode='max')
    es = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=2)

    model.summary()
    history = model.fit(train_ids, train_labels, epochs=10, verbose=1,
                        validation_data=(test_ids, test_labels), callbacks=[es, ch])

    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel("Epochs")
    plt.ylabel('accuracy')
    plt.legend(['accuracy', 'val_accuracy'])
    plt.savefig('SS/Accuracy_BERT')

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel("Epochs")
    plt.ylabel('loss')
    plt.legend(['loss', 'val_loss'])
    plt.savefig('SS/Loss_BERT')

    pred = model.predict(test_ids)
    disp = confusion_matrix(test_labels, np.round(pred), normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=disp)

    disp.plot()
    plt.savefig('SS/Confusion_Matrix_BERT')

    return model


def encoder(sentences, tokenizer, max_len):
    ids = []
    for sentence in sentences:
        encoding = tokenizer.encode_plus(
            sentence,
            max_length=max_len,
            truncation=True,
            add_special_tokens=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=False)
        ids.append(encoding['input_ids'])

    return ids
