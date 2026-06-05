import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes



class Attention:
    def __init__(self, cmap, attn=None, path=None):
        self.cmap = cmap
        self.attn = attn
        self.path = path

# ...existing code...
    def save_heatmap(self, path: str = None, attn: np.ndarray = None, cmap: str = None, dpi: int = 300):
        """
        支持两种输入：
          - 单张热图：arr 为 2-D (C, H)，行为原来行为；
          - 多张热图：arr 为 3-D 且第0维为 9，形状 (9, C, L)，在一张 3x3 画布上绘制 9 张热力图（共享 vmin/vmax）。
        """
        arr = self.attn if attn is None else attn
        if arr is None:
            raise ValueError('No attention array provided')

        arr = np.asarray(arr)
        save_path = self.path if path is None else path
        if save_path is None:
            raise ValueError('No path provided to save the heatmap')

        cmap = self.cmap if cmap is None else cmap

        # 单张热图（向后兼容）
        if arr.ndim == 2:
            C, H = arr.shape
            cell_size = 0.4
            fig_w = max(3.0, H * cell_size)
            fig_h = max(3.0, C * cell_size)

            # Guard against excessively large pixel dimensions
            max_pix = 2 ** 16 - 1
            pix_w = fig_w * dpi
            pix_h = fig_h * dpi
            if pix_w > max_pix or pix_h > max_pix:
                scale = min(max_pix / pix_w, max_pix / pix_h)
                scale = min(scale, 1.0)
                fig_w *= scale
                fig_h *= scale

            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            im = ax.imshow(arr, origin='lower', interpolation='nearest', aspect='auto', cmap=cmap)
            ax.set_xticks([])
            ax.set_yticks([])
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            return

        # 多张热图：要求形状为 (9, C, L)
        if arr.ndim != 3 or arr.shape[0] != 9:
            raise ValueError(f'attn must be 2-D (C,H) or 3-D with first dim 9 (9,C,L), got shape {arr.shape}')

        n, C, L = arr.shape
        # 每个子图单元的建议尺寸
        cell_w = max(2.0, L * 0.12)   # 根据列数调整单元格宽度
        cell_h = max(2.0, C * 0.12)   # 根据行数调整单元格高度
        fig_w = cell_w * 3 + 0.6      # 多留一点宽度给 colorbar 列
        fig_h = cell_h * 3

        # Guard against excessively large pixel dimensions
        max_pix = 2 ** 16 - 1
        pix_w = fig_w * dpi
        pix_h = fig_h * dpi
        if pix_w > max_pix or pix_h > max_pix:
            scale = min(max_pix / pix_w, max_pix / pix_h)
            scale = min(scale, 1.0)
            fig_w *= scale
            fig_h *= scale

        # 共享 vmin/vmax，保证颜色一致
        vmin = float(arr.min())
        vmax = float(arr.max())

        # 使用 GridSpec：3x4 网格，最后一列专门给 colorbar
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = gridspec.GridSpec(3, 4, width_ratios=[1, 1, 1, 0.08], wspace=0.05, hspace=0.05)

        axes = []
        im = None
        for i in range(9):
            r = i // 3
            c = i % 3
            ax = fig.add_subplot(gs[r, c])
            im = ax.imshow(arr[i], origin='lower', interpolation='nearest', aspect='auto',
                           cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            axes.append(ax)

        # 在右侧单独列创建颜色条轴（跨三行）
        cax = fig.add_subplot(gs[:, 3])
        # 使用 colorbar 时不要关闭该轴
        cb = fig.colorbar(im, cax=cax, orientation='vertical')
        # 根据需要可以微调 colorbar 标签样式
        cax.yaxis.set_ticks_position('right')
        cax.yaxis.set_label_position('right')

        # 紧凑布局，但避免 colorbar 被压缩
        plt.tight_layout()
        # 保存并关闭
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
# ...existing code...