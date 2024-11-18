import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
try:
    df = pd.read_csv(r"搜广推\code_bymyself\losses\wideNdeep_val_losses.csv")
except FileNotFoundError:
    print("文件未找到，请检查路径是否正确。")
    exit()
except Exception as e:
    print(f"读取CSV文件出错：{e}")
    exit()


# 提取epoch列
epochs = df['Epoch']

# 获取除'epoch'列以外的所有列名
loss_columns = df.columns[1:].drop('PNN')  # Assuming 'epoch' is the first column
# 绘制图像
plt.figure(figsize=(12, 6))  # 设置图像大小

for column in loss_columns:
    plt.plot(epochs, df[column], label=column)

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Validation Losses for Wide & Deep")
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格
plt.show()