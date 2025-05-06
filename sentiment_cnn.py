# sentiment_cnn.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

# بيانات تجريبية بسيطة (استخدمي مجموعة أكبر لاحقًا)
texts = [
    "I love this movie",
    "I hate that song",
    "not bad",
    "very good",
    "terrible experience",
    "awesome work"
]
labels = [1, 0, 1, 1, 0, 1]  # 1 = إيجابي، 0 = سلبي

# تحويل النصوص إلى تسلسلات أرقام
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=10)
y = np.array(labels)

# تقسيم البيانات لتدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# بناء النموذج
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=100, input_length=10))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # تصنيف ثنائي

# تجميع النموذج
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# حفظ أفضل نموذج تلقائيًا
checkpoint = ModelCheckpoint('sentiment_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)

# تدريب النموذج
history = model.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint],
    verbose=2
)

# تحميل أفضل نسخة محفوظة
model = load_model('sentiment_model.h5')

# تقييم النموذج
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# رسم التدريب
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('training_plot.png')
plt.show()
