from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas
import numpy as np

app = Flask(__name__)

model_temp = load_model('models/dense_temp.h5')
model_kelembapan = load_model('models/dense_kelembapan.h5')
model_lpm = load_model('models/dense_lpm.h5')

train_file = open('data/train_df.pkl', 'rb')
val_file = open('data/val_df.pkl', 'rb')
test_file = open('data/test_df.pkl', 'rb')

train_df = pickle.load(train_file, encoding='bytes')
val_df = pickle.load(val_file, encoding='bytes')
test_df = pickle.load(test_file, encoding='bytes')

# print(train_df)
# print(val_df)
# print(test_df)

class WindowGenerator():
    def __init__(self, input_width, label_width, shift, 
                 train_df=train_df, val_df=val_df, test_df=test_df, 
                 label_columns=None):
        
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}
        
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        
        self.total_window_size = input_width + shift
        
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
        
    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'
        ])
    
    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:,:,self.column_indices[name]] for name in self.label_columns],
                axis=-1
            )
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        
        return inputs, labels
    
    def plot(self, model=None, plot_col='', max_subplots=3):
        inputs, labels = self.example
        plot_col = self.label_columns[0]
        
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        
        for n in range(max_n):
            plt.subplot(3, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
              label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
              label_col_index = plot_col_index

            if label_col_index is None:
              continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
              predictions = model(inputs)
              plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                          marker='X', edgecolors='k', label='Predictions',
                          c='#ff7f0e', s=64)

            if n == 0:
              plt.legend()

        plt.xlabel('Month')
        
    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32
        )
        
        ds = ds.map(self.split_window)
        return ds
    
    @property
    def train(self):
        return self.make_dataset(self.train_df)
    
    @property
    def val(self):
        return self.make_dataset(self.val_df)
    
    @property
    def test(self):
        return self.make_dataset(self.test_df)
    
    @property
    def example(self):
        result = getattr(self, '_example', None)
        if result is None:
            result = next(iter(self.train))
            self._example = result
        return result

@app.route('/')
def index():

    single_step_window_temp = WindowGenerator(input_width=1, label_width=1, shift=0, label_columns=['temp_rataan'])
    single_step_window_kelembapan = WindowGenerator(input_width=1, label_width=1, shift=0, label_columns=['kelembapan'])
    single_step_window_lpm = WindowGenerator(input_width=1, label_width=1, shift=0, label_columns=['lpm'])

    result_temp = model_temp.predict(single_step_window_temp.test, verbose=0)
    # print(result_temp)

    result_kelembapan = model_kelembapan.predict(single_step_window_kelembapan.test, verbose=0)
    # print(result_kelembapan)

    result_lpm = model_lpm.predict(single_step_window_lpm.test, verbose=0)
    # print(result_lpm)
    bulan = ['Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember']

    return render_template('monitor.html', temp=result_temp, kelembapan=result_kelembapan, lpm=result_lpm, bulan=bulan)

if __name__ == '__main__':
    app.run(debug=True)