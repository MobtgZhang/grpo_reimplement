import os
import numpy as np
import matplotlib.pyplot as plt

def draw_loss_curve(losses_list, save_path, save_name, title="Training Loss Curve", xlabel="Steps", ylabel="Loss"):
    """
    绘制Loss曲线：原始折线 + 平滑局部均值曲线，并保存为JPG和PDF。

    Args:
        losses_list (list or np.ndarray): loss值列表
        save_path (str): 保存路径（不包含文件名）
        title (str): 图表标题
        xlabel (str): x轴标签
        ylabel (str): y轴标签
    """
    losses = np.array(losses_list)
    epochs = np.arange(1, len(losses) + 1)

    # === 平滑处理（20%点的滑动平均） ===
    window_size = max(1, len(losses) // 5)  # 保留约20%的点
    smooth_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
    smooth_epochs = np.linspace(epochs[0], epochs[-1], len(smooth_losses))

    # === 开始绘图 ===
    plt.figure(figsize=(10, 6))
    
    # 原始点（绿色线）
    plt.plot(epochs, losses, color='#58A27C', linewidth=1.5, alpha=0.6, label="Original Loss")

    # 平滑曲线（深蓝线）
    plt.plot(smooth_epochs, smooth_losses, color='#2E86AB', linewidth=2.5, label="Smoothed (Mean)")

    # 自动padding
    # y_max = y_max + 1.0
    # y_min = max(y_min-0.5,0)
    
    y_min, y_max = losses.min(), losses.max()
    y_range = y_max - y_min
    plt.ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)

    # 美化样式
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend(frameon=False, fontsize=11)
    plt.tight_layout()

    # === 保存图片 ===
    os.makedirs(save_path, exist_ok=True)
    save_file_name = os.path.join(save_path, save_name)
    plt.savefig(f"{save_file_name}.jpg", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_file_name}.pdf", bbox_inches='tight')
    plt.close()
