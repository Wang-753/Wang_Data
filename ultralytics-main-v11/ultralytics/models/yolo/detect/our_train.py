from ultralytics import YOLO
# import os
# import wandb
#
# os.environ["WANDB_API_KEY"] = 'KEY'
# os.environ["WANDB_MODE"] = "offline"
if __name__ == '__main__':
    # 加载模型
    model = YOLO(r'')  # 不使用预训练权重训练
    # 训练参数 ----------------------------------------------------------------------------------------------
    model.train(
        data=r'',
        epochs=200,  # (int) 训练的周期数
        patience=0,  # (int) 等待无明显改善以进行早期停止的周期数
        batch=16,  # (int) 每批次的图像数量（-1 为自动批处理）
        imgsz=640,  # (int) 输入图像的大小，整数或w，h
        save=True,  # (bool) 保存训练检查点和预测结果
        save_period=-1,  # (int) 每x周期保存检查点（如果小于1则禁用）
        cache=False,  # (bool) True/ram、磁盘或False。使用缓存加载数据
        device='',  # (int | str | list, optional) 运行的设备，例如 cuda device=0 或 device=0,1,2,3 或 device=cpu
        project='',  # (str, optional) 项目名称
        name='',  # (str, optional) 实验名称，结果保存在'project/name'目录下
        optimizer='',  # (str) 要使用的优化器，选择=[SGD，Adam，Adamax，AdamW，NAdam，RAdam，RMSProp，auto]
        resume=True,  # (bool) 从上一个检查点恢复训练
        amp=True,  # (bool) 自动混合精度（AMP）训练，选择=[True, False]，True运行AMP检查
    )


