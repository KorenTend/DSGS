"""Research defaults. Legacy network/batch fields are documented in readme.md."""


class Config:
    # 随机种子
    SEED = 42

    # 数据相关
    NUM_POINTS_GUIDE = 2048
    NUM_PATCHES = 16
    PATCH_POINTS = 4096
    OVERLAP_RATIO = 1.3

    # 网络相关
    INPUT_DIM = 8              # x,y,z, nx,ny,nz, curvature, relative z-height
    HIDDEN_DIM = 128
    DROPOUT = 0.3

    # 训练相关
    EPOCHS = 100
    BATCH_SIZE = 4
    LR = 0.001

    # 简化相关
    TARGET_RATIO = 0.1 # 这里的默认值不影响 benchmark 脚本
