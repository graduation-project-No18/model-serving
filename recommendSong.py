#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import openpyxl
import warnings
warnings.filterwarnings(action='ignore')


class Recommendation:

    res = dict()
    songs = list()
    octave_max = ""
    tone = ""

    def print_column_info(self, filename, column_number):

        wb = openpyxl.load_workbook(filename)
        sheet = wb.active

        column = sheet[column_number]
        column_values = [cell.value for cell in column]
        print(column_values)
        row_info = ''.join(str(value+",") for value in column_values)
        print(row_info)
        wb.close()
        return row_info

    def __init__(self, member_nickname):
        self.res = dict()
        self.songs = list()
        y1, sr = librosa.load(member_nickname + ".wav", sr=44000)

        data = pd.read_csv("DF.csv")
        data = data.fillna(data.mean())

        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        rfm = RandomForestClassifier(n_estimators=200, max_depth=3)
        rfm.fit(X_train, y_train)
        y_pred = rfm.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        ##############################################################################################

        mfcc = librosa.feature.mfcc(y=y1, sr=sr, n_mfcc=20)

        spectral_centroids = librosa.feature.spectral_centroid(y=y1, sr=sr)

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y1, sr=sr)

        spectral_contrast = librosa.feature.spectral_contrast(y=y1, sr=sr)

        spectral_rolloff = librosa.feature.spectral_rolloff(y=y1, sr=sr)

        meanfreq = np.mean(spectral_centroids)

        maxfreq = np.max(spectral_centroids)

        mfcc_mean = np.mean(mfcc[:, 1:], axis=1).tolist()
        spectral_centroids = np.mean(spectral_centroids[spectral_centroids > 0])
        spectral_bandwidth = np.mean(spectral_bandwidth[spectral_bandwidth > 0])
        spectral_contrast = np.mean(spectral_contrast[spectral_contrast > 0], axis=0).tolist()
        spectral_rolloff = np.mean(spectral_rolloff[spectral_rolloff > 0])
        meanfreq = np.mean(meanfreq[meanfreq > 0])
        maxfreq = np.mean(maxfreq[maxfreq > 0])

        input_features = np.array(
            [mfcc_mean[0], mfcc_mean[1], mfcc_mean[2], mfcc_mean[3], mfcc_mean[4], mfcc_mean[5], mfcc_mean[6],
             mfcc_mean[7], mfcc_mean[8], mfcc_mean[9], mfcc_mean[10], mfcc_mean[11], mfcc_mean[12], mfcc_mean[13],
             mfcc_mean[14], mfcc_mean[15], mfcc_mean[16], mfcc_mean[17], mfcc_mean[18], mfcc_mean[19],
             spectral_centroids,
             spectral_bandwidth, spectral_contrast, spectral_rolloff, meanfreq,
             maxfreq])

        prediction = rfm.predict(input_features.reshape(1, -1))
        if prediction == 1:
            print("톤==여성적")
            # 주파수 데이터
            data = pd.read_csv('doremi_fre_girl.csv')

            X = data.iloc[:, 0].values.reshape(42, 1)
            y = data.iloc[:, 1]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

            dt = DecisionTreeClassifier()
            dt.fit(X_train, y_train)

            start = (0 * sr)
            end = (2 * sr)
            f0, voiced_flag, voiced_probs = librosa.pyin(y1[start:end], fmin=librosa.note_to_hz('C1'),
                                                         fmax=librosa.note_to_hz('B4'))
            valid_pitch = f0[~np.isnan(f0)]

            pitch_mean = np.nanmean(valid_pitch)
            pitch_mean_reshaped = pitch_mean.reshape(-1, 1)

            predicted = dt.predict(pitch_mean_reshaped)  # 분류 결과 출력
            print(pitch_mean_reshaped, '-> 예상 최고 옥타브:', predicted)
            self.octave_max = predicted
            self.tone = "여성적"
        else:
            print("톤==남성적")
            # 주파수 데이터
            data = pd.read_csv('doremi_fre_boy.csv')

            X = data.iloc[:, 0].values.reshape(42, 1)
            y = data.iloc[:, 1]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

            dt = DecisionTreeClassifier()
            dt.fit(X_train, y_train)

            start = (0 * sr)
            end = (2 * sr)
            f0, voiced_flag, voiced_probs = librosa.pyin(y1[start:end], fmin=librosa.note_to_hz('C1'),
                                                         fmax=librosa.note_to_hz('B4'))
            valid_pitch = f0[~np.isnan(f0)]

            pitch_mean = np.nanmean(valid_pitch)
            pitch_mean_reshaped = pitch_mean.reshape(-1, 1)

            predicted = dt.predict(pitch_mean_reshaped)  # 분류 결과 출력
            print(pitch_mean_reshaped, '-> 예상 최고 옥타브:', predicted)
            self.octave_max = predicted
            self.tone = "남성적"
        tone_result = prediction
        oc_result = pitch_mean_reshaped

        # ---------------------------------------------------------------------------

        data = pd.read_csv("Song.csv", encoding="euc-kr")
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        X_train, y_train = X, y

        rf = RandomForestClassifier(n_estimators=200, max_depth=3)
        rf_clt = rf.fit(X_train, y_train)

        mfcc = librosa.feature.mfcc(y=y1, sr=sr, n_mfcc=20)

        spectral_centroids = librosa.feature.spectral_centroid(y=y1, sr=sr)

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y1, sr=sr)

        spectral_contrast = librosa.feature.spectral_contrast(y=y1, sr=sr)

        spectral_rolloff = librosa.feature.spectral_rolloff(y=y1, sr=sr)

        meanfreq = np.mean(spectral_centroids)

        maxfreq = np.max(spectral_centroids)

        mfcc_mean = np.mean(mfcc[:, 1:], axis=1).tolist()
        spectral_centroids = np.mean(spectral_centroids[spectral_centroids > 0])
        spectral_bandwidth = np.mean(spectral_bandwidth[spectral_bandwidth > 0])
        spectral_contrast = np.mean(spectral_contrast[spectral_contrast > 0], axis=0)
        spectral_rolloff = np.mean(spectral_rolloff[spectral_rolloff > 0])
        meanfreq = np.mean(meanfreq[meanfreq > 0])
        maxfreq = np.mean(maxfreq[maxfreq > 0])
        tone_results = np.mean(tone_result)
        oc_results = np.mean(oc_result)

        tone_results = np.array(tone_results)
        oc_results = np.array(oc_results)

        input_features = np.array(
            [mfcc_mean[0], mfcc_mean[1], mfcc_mean[2], mfcc_mean[3], mfcc_mean[4], mfcc_mean[5], mfcc_mean[6],
             mfcc_mean[7], mfcc_mean[8], mfcc_mean[9], mfcc_mean[10], mfcc_mean[11], mfcc_mean[12], mfcc_mean[13],
             mfcc_mean[14], mfcc_mean[15], mfcc_mean[16], mfcc_mean[17], mfcc_mean[18], mfcc_mean[19],
             spectral_centroids, spectral_bandwidth, spectral_contrast, spectral_rolloff, meanfreq,
             maxfreq, tone_results, oc_results])

        result1 = rf.predict(input_features.reshape(1, -1))
        row_number1 = data.index[data['label'] == result1[0]].tolist()

        X_updated = X[~y.isin(result1)]
        y_updated = y[~y.isin(result1)]

        rf_clt_updated = rf.fit(X_updated, y_updated)
        result2 = rf_clt_updated.predict(input_features.reshape(1, -1))
        row_number2 = data.index[data['label'] == result2[0]].tolist()

        X_updated2 = X_updated[~y.isin(result2)]
        y_updated2 = y_updated[~y.isin(result2)]
        rf_clt_updated2 = rf.fit(X_updated2, y_updated2)
        result3 = rf_clt_updated2.predict(input_features.reshape(1, -1))
        row_number3 = data.index[data['label'] == result3[0]].tolist()

        row_number1 = list(map(int, row_number1))
        row_number2 = list(map(int, row_number2))
        row_number3 = list(map(int, row_number3))

        self.songs.append(self.print_column_info('Song_details.xlsx', row_number1[0] + 2))
        self.songs.append(self.print_column_info('Song_details.xlsx', row_number2[0] + 2))
        self.songs.append(self.print_column_info('Song_details.xlsx', row_number3[0] + 2))

    def get_result(self):
        self.res["tone"] = self.tone
        self.res["octave"] = self.octave_max[0]
        self.res["songs"] = self.songs
        return self.res



