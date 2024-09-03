import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.metrics.pairwise import cosine_distances

# 随机生成形态特征数据
np.random.seed(42)
X = np.random.rand(100, 50)  # 100个样本，每个样本50个特征

# 随机生成城市形态类型标签
labels = np.random.choice(['AM1', 'AM2', 'AM3', 'AM4', 'AM5', 'SF1', 'SF2', 'SF3', 'SF4', 'SG1', 'SG2', 'SG3', 'SG4', 'SG5', 'BC1', 'BC2', 'BC3'], 100)

# 生成 t-SNE 图
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(10, 8))
for label in np.unique(labels):
    indices = labels == label
    plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1], label=label)
plt.legend()
plt.title("t-SNE visualisation of morphology types")
plt.show()

# 计算余弦距离
dist_matrix = cosine_distances(X)
plt.figure(figsize=(10, 8))
sns.heatmap(dist_matrix, xticklabels=labels, yticklabels=labels, cmap='viridis')
plt.title("Morphology pairwise cosine distance")
plt.show()
