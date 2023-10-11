# Identification-and-Extraction-of-DFS

这个项目包括了使用Seg-ASPP网络进行大图像的语义分割预测，以及使用重叠投票方法来提高预测结果的准确性。

## 准备工作

确保你已经安装了以下依赖：

- Python 3.x
- NumPy
- PyTorch
- OpenCV
- Matplotlib
- GDAL
- PIL
- torchvision

你可以使用以下命令安装这些依赖：
```bash
pip install numpy torch opencv-python matplotlib gdal pillow torchvision
```

# 使用方法

1. **准备大图像文件：** 确保你的大图像文件的格式为 .bmp, .jpeg, .jpg, .png, .tif, 或 .tiff。

2. **修改参数：** 打开脚本文件并修改以下参数，以适应你的需求：
   - `image_name`：大图像文件的路径。
   - `images_path`：Seg-ASPP模型的路径。
   - `model_path`：已经训练好的Seg-ASPP模型的权重路径。
   - `row` 和 `col`：设置每个图像切块的大小。

3. **运行脚本：** 运行脚本文件，它会自动对大图像进行分块处理，并使用Seg-ASPP网络进行语义分割预测。

4. **查看结果：** 预测结果将会保存为一个 .tif 文件。

## 注意事项

- 请确保模型文件和大图像文件的路径设置正确。
- 根据你的需求，你还可以修改其他参数，例如 `out_threshold` 和 `scale_factor`。

## 作者

- 作者：叶茂林
- 电子邮件：maolinye00@gmail.com

如果你有任何问题或建议，请随时联系作者。
