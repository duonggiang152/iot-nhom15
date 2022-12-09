# import dependencies
import matplotlib.pyplot as plt
import numpy as np


def corrcoef(x, y):
    """
    Tính toán hệ số tương quan giữa tập hợp dữ liệu x và y.
    Args:
        x: tập hợp dữ liệu x (ma trận 1 chiều n_samples x 1)
        y: tập hợp dữ liệu y (ma trận 1 chiều n_samples x 1)
    Returns:
        Hệ số tương quan giữa x và y
    """
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_x_sq = sum([pow(i, 2) for i in x])
    sum_y_sq = sum([pow(j, 2) for j in y])
    p_sum = sum([x[i] * y[i] for i in range(n)])
    num = p_sum - (sum_x * sum_y / n)
    den = math.sqrt((sum_x_sq - pow(sum_x, 2) / n)
                    * (sum_y_sq - pow(sum_y, 2) / n))
    if den == 0:
        return 0
    return num / den


def corr_matrix(data):
    """
    Tính toán ma trận tương quan giữa các cột của ma trận dữ liệu.
    Args:
        data: ma trận dữ liệu (n_samples x n_features)
    Returns:
        Ma trận tương quan (n_features x n_features)
    """
    n = len(data[0])
    corr_mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            if i == j:
                corr_mat[i][j] = 1.0
            else:
                corr = corrcoef(data[:, i], data[:, j])
                corr_mat[i][j] = corr
                corr_mat[j][i] = corr
    return corr_mat


def plot_corr_matrix(corr_mat, labels):
    """
    Vẽ biểu đồ nhiệt ma trận tương quan.
    Args:
        corr_mat: ma trận tương quan (n_features x n_features)
        labels: danh sách tên các cột (n_features x 1)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr_mat, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = [i for i in range(len(labels))]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.show()


def train_test_split(data, test_size=0.2, shuffle=True, random_seed=1):
    """
    Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra.
    Args:
        data: tập dữ liệu (n_samples x n_features)
        test_size: tỷ lệ phần trăm của tập dữ liệu dùng làm tập kiểm tra
        shuffle: có xáo trộn dữ liệu trước khi chia hay không
        random_seed: giá trị khởi tạo cho bộ sinh số ngẫu nhiên
    Returns:
        Tập huấn luyện và tập kiểm tra
    """
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(data)
    n_train = int(len(data) * (1 - test_size))
    return data[:n_train], data[n_train:]
