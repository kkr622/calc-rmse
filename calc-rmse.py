import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# CSVファイルからデータを読み込む
file_path = 'csv-raw/test.csv'  # ファイルのパスを指定
data = pd.read_csv(file_path)

# 計算したいphaseを選択
phase = 2
# 計算したい開始時間を選択
time_start = 5
time_end = 9
# csvファイルで結果出力の場合 True, グラフを確認する場合 False
pub_csv = False


# 条件に合致するデータを選択
filtered_data_1N = data[(data['phase'] == phase) & (data['Time'] >= time_start) & (data['Time'] <= time_end)]
filtered_data_0_5N = data[(data['phase'] == phase) & (data['Time'] >= time_start + 10) & (data['Time'] <= time_end + 10)]

# セット数
sets = list(range(1, 8))

# 各SetごとにRMSEを計算（1N）
print("RMSE：1N")
rmse_values_1N = []
for set_number in sets:
    # 取得した「Sensor Fz」のデータ数分の目標値（1N）が入ったリストを用意
    true_values = np.ones_like(filtered_data_1N[filtered_data_1N['Set'] == set_number][' Sensor Fz'].values)
    #「Sensor Fz」のデータを取得
    predicted_values = filtered_data_1N[filtered_data_1N['Set'] == set_number][' Sensor Fz'].values
    #rmseを計算
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    rmse_values_1N.append(rmse)

    print(f"Set {set_number}: RMSE = {rmse}")

# RMSEの平均を計算、出力
ave_1N = sum(rmse_values_1N) / len(rmse_values_1N)
print(f"The average of RMSE: {ave_1N}")

# 各SetごとにRMSEを計算（0.5N）
print("RMSE：0.5N")
rmse_values_0_5N = []
for set_number in sets:
    # 取得した「Sensor Fz」のデータ数分の目標値（0.5N）が入ったリストを用意
    true_values = np.full_like(filtered_data_0_5N[filtered_data_0_5N['Set'] == set_number][' Sensor Fz'].values , fill_value=0.5)
    predicted_values = filtered_data_0_5N[filtered_data_0_5N['Set'] == set_number][' Sensor Fz'].values
    
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    rmse_values_0_5N.append(rmse)
    
    print(f"Set {set_number}: RMSE = {rmse}")

# RMSEの平均を計算、出力
ave_0_5N = sum(rmse_values_0_5N) / len(rmse_values_0_5N)
print(f"The average of RMSE: {ave_0_5N}")

# 各SetごとにRMSEを計算（diff）
print("RMSE：diff")
rmse_values_diff = []
for set_number in sets:
    diff = []
    # 1Nについての「Sensor Fz」と0.5Nについての「Sensor Fz」のうち、データ数が少ない方に長さに合わせる
    for oneN, halfN in zip(filtered_data_1N[filtered_data_1N['Set'] == set_number][' Sensor Fz'].values, filtered_data_0_5N[filtered_data_0_5N['Set'] == set_number][' Sensor Fz'].values):
        diff.append(oneN - halfN)
    
    # 取得したデータ数分の目標値（1.0N - 0.5N）が入ったリストを用意
    true_values = np.full_like(diff , fill_value=0.5)
    predicted_values = diff
    
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    rmse_values_diff.append(rmse)
    
    print(f"Set {set_number}: RMSE = {rmse}")

ave_diff = sum(rmse_values_diff) / len(rmse_values_diff)
print(f"The average of RMSE: {ave_diff}")


if pub_csv:
    # CSVファイル系
    # 平均値をそれぞれの列の最後に追加
    sets.append("average")
    rmse_values_1N.append(ave_1N)
    rmse_values_0_5N.append(ave_0_5N)
    rmse_values_diff.append(ave_diff)

    # 結果をCSVファイルとして保存
    result_df = pd.DataFrame({'Set': sets, 'RMSE_1N': rmse_values_1N, 'RMSE_0.5N': rmse_values_0_5N, 'RMSE_diff': rmse_values_diff,})
    result_df.to_csv('csv-rmse/test.csv', index=False)
else:
    # グラフ表示
    plt.figure(figsize=(6, 6))
    plt.bar(sets, rmse_values_1N, color='blue')
    plt.xlabel('Set')
    plt.ylabel('RMSE')
    plt.title('RMSE for each Set (1N)')

    plt.figure(figsize=(6, 6))
    plt.bar(sets, rmse_values_0_5N, color='green')
    plt.xlabel('Set')
    plt.ylabel('RMSE')
    plt.title('RMSE for each Set (0.5N)')

    plt.figure(figsize=(6, 6))
    plt.bar(sets, rmse_values_diff, color='red')
    plt.xlabel('Set')
    plt.ylabel('RMSE')
    plt.title('RMSE for each Set (diff)')

    # plt.show(block=False)
    plt.show()

