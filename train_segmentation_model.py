import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import tensorflow.keras.backend as K  
import numpy as np
import matplotlib.pyplot as plt
import os
import math

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def build_mobile_unet(input_shape=(128, 128, 3), alpha=0.35):
    """
    构建轻量级 MobileNetV2 + UNet 解码器的分割模型

    参数:
        input_shape: 输入图像形状 (H, W, C)
        alpha: MobileNetV2 的宽度乘数，越小越轻量

    返回:
        model: Keras 模型
    """
    # 使用预训练的 MobileNetV2 作为编码器（不包含顶部全连接层）
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        alpha=alpha,
        include_top=False,
        weights=None,  # 从头开始训练，不使用预训练权重
        pooling=None
    )

    # 获取不同层的输出用于跳跃连接
    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project'       # 4x4
    ]

    # 创建新的模型，提取多层输出
    outputs = []
    for name in layer_names:
        layer = base_model.get_layer(name)
        if layer is not None:
            outputs.append(layer.output)

    encoder_model = Model(inputs=base_model.input, outputs=outputs)
    encoder_model.trainable = True  # 可以微调

    # 定义输入
    inputs = keras.Input(shape=input_shape)

    # 编码器前向传播
    enc_outputs = encoder_model(inputs)

    # 解码器：上采样 + 跳跃连接
    # 从最深层开始 (4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128)

    # block_16_project output (假设是 4x4)
    x = enc_outputs[-1]

    # 上采样到 8x8，并与 block_13_expand_relu 拼接
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Concatenate()([x, enc_outputs[-2]])
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 上采样到 16x16，并与 block_6_expand_relu 拼接
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Concatenate()([x, enc_outputs[-3]])
    x = layers.Conv2D(48, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 上采样到 32x32，并与 block_3_expand_relu 拼接
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Concatenate()([x, enc_outputs[-4]])
    x = layers.Conv2D(32, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 上采样到 64x64，并与 block_1_expand_relu 拼接
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Concatenate()([x, enc_outputs[-5]])
    x = layers.Conv2D(24, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 最后上采样到 128x128
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(16, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 输出层：单通道掩码，使用 sigmoid 激活函数
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def load_data(data_path='plant_data.npz'):
    """加载数据，保持 uint8 格式以节省 4 倍内存"""
    print(f"正在加载数据: {data_path}")
    data = np.load(data_path)
    images = data['images']
    masks = data['masks']

    print(f"数据加载完成:")
    print(f"  图像形状: {images.shape}, 数据类型: {images.dtype}")
    print(f"  掩码形状: {masks.shape}, 数据类型: {masks.dtype}")

    return images, masks


def dice_loss(y_true, y_pred):
    """
    自定义 Dice Loss，专门对抗类别不平衡。
    直接优化预测掩码与真实掩码的重合度 (IoU 替代品)
    """
    smooth = 1e-6 # 防止分母为0
    # 展平张量
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    # 计算交集
    intersection = K.sum(y_true_f * y_pred_f)
    # 计算 Dice 系数并转为 Loss
    return 1.0 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def train_model(model, images, masks, epochs=30, batch_size=32, learning_rate=1e-3):
    """
    终极省内存版训练函数：使用生成器(Generator)懒加载数据
    """
    num_samples = len(images)
    
    # 1. 仅仅打乱“索引”，绝对不复制庞大的图像数据本身
    indices = np.random.permutation(num_samples)
    split_idx = int(0.8 * num_samples)

    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    print(f"\n数据集划分:")
    print(f"  训练集: {len(train_indices)} 张")
    print(f"  验证集: {len(val_indices)} 张")

    # 2. 定义数据生成器：每次只提取1张图，拒绝一次性吞下所有数据
    def data_generator(idx_list):
        for idx in idx_list:
            img = images[idx]
            # 为 mask 扩充最后的一个通道维度 (128, 128) -> (128, 128, 1)
            mask = np.expand_dims(masks[idx], axis=-1)
            yield img, mask

    # 3. 构建懒加载的 tf.data 流水线
    train_ds = tf.data.Dataset.from_generator(
        lambda: data_generator(train_indices),
        output_signature=(
            tf.TensorSpec(shape=(128, 128, 3), dtype=tf.uint8),
            tf.TensorSpec(shape=(128, 128, 1), dtype=tf.uint8)
        )
    )

    val_ds = tf.data.Dataset.from_generator(
        lambda: data_generator(val_indices),
        output_signature=(
            tf.TensorSpec(shape=(128, 128, 3), dtype=tf.uint8),
            tf.TensorSpec(shape=(128, 128, 1), dtype=tf.uint8)
        )
    )

    # 4. 数据增强与动态归一化 (直接使用之前版本的逻辑)
    def augment(image, mask):
        image = tf.cast(image, tf.float32) / 255.0
        mask = tf.cast(mask, tf.float32)
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_up_down(image)
            mask = tf.image.flip_up_down(mask)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image, mask

    def process_val(image, mask):
        image = tf.cast(image, tf.float32) / 255.0
        mask = tf.cast(mask, tf.float32)
        return image, mask

    # 5. 配置批次预取 
    # (由于生成器提取时已经使用了打乱的 train_indices，这里无需再调用 ds.shuffle 浪费内存)
    train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = val_ds.map(process_val, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # 6. 编译与训练
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=dice_loss, 
        metrics=['accuracy', keras.metrics.MeanIoU(num_classes=2)]
    )
    # 自定义一个防焦虑的提示器
    class AntiAnxietyCallback(tf.keras.callbacks.Callback):
        def on_train_batch_begin(self, batch, logs=None):
            if batch == 0:
                print("\n[系统提示] 底层计算图编译终于完成了！马上弹出进度条...")
    
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
    ]
    steps_per_epoch = math.ceil(len(train_indices) / batch_size)
    validation_steps = math.ceil(len(val_indices) / batch_size)

    print(f"\n开始使用 懒加载流水线 + Dice Loss 进行稳健训练...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    return history

def plot_training_history(history, save_path='training_history.png'):
    """
    绘制训练过程的 Loss 曲线和其他指标

    参数:
        history: 训练历史记录
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 绘制 Loss 曲线
    axes[0].plot(history.history['loss'], label='训练损失', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='验证损失', linewidth=2)
    axes[0].set_title('训练过程 - 损失函数', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # 绘制 Accuracy 曲线
    axes[1].plot(history.history['accuracy'], label='训练准确率', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='验证准确率', linewidth=2)
    axes[1].set_title('训练过程 - 准确率', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend(loc='lower right')
    axes[1].grid(True, alpha=0.3)

    # 绘制 Mean IoU 曲线
    if 'mean_io_u' in history.history:
        axes[2].plot(history.history['mean_io_u'], label='训练 IoU', linewidth=2)
        axes[2].plot(history.history['val_mean_io_u'], label='验证 IoU', linewidth=2)
        axes[2].set_title('训练过程 - 交并比 (IoU)', fontsize=14)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('IoU')
        axes[2].legend(loc='lower right')
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'IoU 数据不可用', ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('训练过程 - 交并比 (IoU)', fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n训练曲线已保存到: {save_path}")
    plt.show()

    # 打印最佳结果
    best_epoch = np.argmin(history.history['val_loss'])
    print(f"\n最佳训练结果 (Epoch {best_epoch + 1}):")
    print(f"  训练损失: {history.history['loss'][best_epoch]:.4f}")
    print(f"  验证损失: {history.history['val_loss'][best_epoch]:.4f}")
    print(f"  训练准确率: {history.history['accuracy'][best_epoch]:.4f}")
    print(f"  验证准确率: {history.history['val_accuracy'][best_epoch]:.4f}")
    if 'mean_io_u' in history.history:
        print(f"  训练 IoU: {history.history['mean_io_u'][best_epoch]:.4f}")
        print(f"  验证 IoU: {history.history['val_mean_io_u'][best_epoch]:.4f}")


def visualize_predictions(model, images, masks, num_samples=5, save_path='predictions.png'):
    """
    可视化模型的预测结果

    参数:
        model: 训练好的模型
        images: 测试图像
        masks: 真实掩码
        num_samples: 展示的样本数量
        save_path: 保存路径
    """
    # 随机选择样本
    indices = np.random.choice(len(images), min(num_samples, len(images)), replace=False)

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    for i, idx in enumerate(indices):
        img = images[idx:idx+1]  # 保持批次维度
        true_mask = masks[idx]

        # 预测
        pred_mask = model.predict(img, verbose=0)[0, :, :, 0]

        # 二值化预测结果
        pred_binary = (pred_mask > 0.5).astype(np.float32)

        # 显示原图
        axes[i, 0].imshow(img[0])
        axes[i, 0].set_title('原图', fontsize=12)
        axes[i, 0].axis('off')

        # 显示真实掩码
        axes[i, 1].imshow(true_mask, cmap='gray')
        axes[i, 1].set_title('真实掩码', fontsize=12)
        axes[i, 1].axis('off')

        # 显示预测掩码
        axes[i, 2].imshow(pred_binary, cmap='gray')
        axes[i, 2].set_title('预测掩码', fontsize=12)
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n预测结果已保存到: {save_path}")
    plt.show()


def main():
    """主函数"""
    # ==================== 配置参数 ====================
    DATA_PATH = 'plant_data.npz'          # 数据文件路径
    MODEL_PATH = 'segmentation_model.h5'  # 模型保存路径
    ALPHA = 0.35                          # MobileNetV2 宽度乘数 (越小越轻量)
    EPOCHS = 30                           # 训练轮数
    BATCH_SIZE = 32                       # 批次大小
    LEARNING_RATE = 1e-3                  # 初始学习率
    INPUT_SHAPE = (128, 128, 3)           # 输入形状

    # ==================== 加载数据 ====================
    print("=" * 60)
    print("阶段二：模型搭建与训练")
    print("=" * 60)

    images, masks = load_data(DATA_PATH)

    # ==================== 构建模型 ====================
    print("\n" + "=" * 60)
    print("构建 MobileNetV2 + UNet 分割模型")
    print(f"  Backbone: MobileNetV2 (alpha={ALPHA})")
    print(f"  输入形状: {INPUT_SHAPE}")
    print("=" * 60)

    model = build_mobile_unet(input_shape=INPUT_SHAPE, alpha=ALPHA)

    # ==================== 训练模型 ====================
    print("\n" + "=" * 60)
    print("开始训练")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print("=" * 60)

    history = train_model(
        model,
        images,
        masks,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )

    # ==================== 绘制训练曲线 ====================
    print("\n" + "=" * 60)
    print("绘制训练曲线")
    print("=" * 60)

    plot_training_history(history)

    # ==================== 保存模型 ====================
    print("\n" + "=" * 60)
    print("保存模型")
    print("=" * 60)

    model.save(MODEL_PATH)
    print(f"模型已保存到: {MODEL_PATH}")

    # 保存为 TensorFlow Lite 格式（适合单片机部署）
    print("\n转换为 TensorFlow Lite 格式...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    tflite_path = 'segmentation_model.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite 模型已保存到: {tflite_path}")

    # 打印模型大小
    import os
    h5_size = os.path.getsize(MODEL_PATH) / 1024 / 1024
    tflite_size = os.path.getsize(tflite_path) / 1024 / 1024
    print(f"\n模型大小:")
    print(f"  .h5 格式: {h5_size:.2f} MB")
    print(f"  .tflite 格式: {tflite_size:.2f} MB")

    # ==================== 可视化预测结果 ====================
    print("\n" + "=" * 60)
    print("可视化预测结果")
    print("=" * 60)

    # 加载最佳模型进行预测
    best_model = keras.models.load_model('best_model.h5')
    visualize_predictions(best_model, images, masks)

    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()