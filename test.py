import os
#import nibabel as nib
import numpy
import numpy as np
import torch.nn.functional as F
# import SimpleITK as sitk
import torch
#from medpy import metric
from scipy.ndimage.interpolation import zoom
# from networks.efficientunet import UNet
# from networks.net_factory import net_factory
import cv2
import matplotlib.pyplot as plt
from osgeo import gdal
from PIL import Image
Image.MAX_IMAGE_PIXELS = 2300000000
import time
# from unetplus import UnetPlusPlus
# from model import  unet
import torchvision.transforms as transform
from nets.segformer import SegFormer
from utils.utils import cvtColor, preprocess_input, resize_image, show_config

from utils.utils import cvtColor, preprocess_input, resize_image, show_config

# CELL_X = 256
# CELL_Y = 256
# row_cell = 5120
# col_cell = 5120
# NetName = 'FCN8'
# CLASS_NUM = 7
CELL_X = 256
CELL_Y = 256
row_cell = 10000
col_cell = 10000
# NetName = 'mobilenet'
NetName = 'segformer'
CLASS_NUM = 4
# def tans_img_rgb(img_A, ds_array_B, img_newB):
#     img_tif = Image.open(img_A)
#     width = img_tif.size[0]
#     height = img_tif.size[1]
#     img = np.asarray(img_tif).copy()
#     # img = np.zeros([height, width , 3], dtype=np.uint8)
#     ds_array_B_r = img[:,:,0]
#     ds_array_B_g = img[:,:,1]
#     ds_array_B_b = img[:,:,2]
#
#     ds_array_B_r[ds_array_B == 0] = 0
#     ds_array_B_g[ds_array_B == 0] = 127
#     ds_array_B_b[ds_array_B == 0] = 204
#
#     # ds_array_B_r[ds_array_B == 1] = 123
#     # ds_array_B_g[ds_array_B == 1] = 59
#     # ds_array_B_b[ds_array_B == 1] = 24
#     #
#     # ds_array_B_r[ds_array_B == 2] = 255
#     # ds_array_B_g[ds_array_B == 2] = 192
#     # ds_array_B_b[ds_array_B == 2] = 0
#     #
#     # ds_array_B_r[ds_array_B == 3] = 217 #255,255,0粉砂
#     # ds_array_B_g[ds_array_B == 3] = 217
#     # ds_array_B_b[ds_array_B == 3] = 217
#
#     # ds_array_B_r[ds_array_B == 4] = 56
#     # ds_array_B_g[ds_array_B == 4] = 168
#     # ds_array_B_b[ds_array_B == 4] = 0
#     img[:, :, 0] = ds_array_B_r
#     img[:, :, 1] = ds_array_B_g
#     img[:, :, 2] = ds_array_B_b
#     # 写入影像数据
#     im = Image.fromarray(img)
#     im = im.convert("RGB")
#     im.save(img_newB)
#     del img
#
#
# def tans_img_ys(img_A, ds_array_B, img_newB):
#     datasetA = gdal.Open(img_A)  # 打开文件，用这个tif图的投影信息
#     datasetA_geotrans = datasetA.GetGeoTransform()  # 仿射矩阵
#
#     datasetA_proj = datasetA.GetProjection()  # 地图投影信息
#
#     im_width = datasetA.RasterXSize  # 栅格矩阵的列数
#     im_height = datasetA.RasterYSize  # 栅格矩阵的行数
#     # im_bands = datasetA.RasterCount
#     im_bands = 1
#     # datatype = gdal.GDT_Byte
#     # list2 = [gdal.GDT_Byte,gdal.GDT_Byte,gdal.GDT_UInt16,gdal.GDT_Int16,gdal.GDT_UInt32,gdal.GDT_Int32,gdal.GDT_Float32,gdal.GDT_Float64,gdal.GDT_CInt16,gdal.GDT_CInt32,gdal.GDT_CFloat32,gdal.GDT_CFloat64]
#     datatype = gdal.GDT_Byte
#     # datatype = gdal.GDT_Float32
#     driver = gdal.GetDriverByName("GTiff")  # 创建文件驱动
#     dataset = driver.Create(img_newB, im_width, im_height, im_bands, datatype)
#     dataset.SetGeoTransform(datasetA_geotrans)  # 写入仿射变换参数
#     dataset.SetProjection(datasetA_proj)  # 写入投影1
#     # dataset.GetRasterBand(1).WriteArray(ds_array_B)
#     # 写入影像数据
#
#
#     for i in range(im_bands):
#         dataset.GetRasterBand(i+1).WriteArray(ds_array_B)
#     del dataset
# def tans_single_img(img_A, ds_array_B, img_newB):
#     datasetA = gdal.Open(img_A)  # 打开文件，用这个tif图的投影信息
#     datasetA_geotrans = datasetA.GetGeoTransform()  # 仿射矩阵
#
#     datasetA_proj = datasetA.GetProjection()  # 地图投影信息
#
#     im_width = datasetA.RasterXSize  # 栅格矩阵的列数
#     im_height = datasetA.RasterYSize  # 栅格矩阵的行数
#     # im_bands = datasetA.RasterCount
#     im_bands = 1
#     # datatype = gdal.GDT_Byte
#     # list2 = [gdal.GDT_Byte,gdal.GDT_Byte,gdal.GDT_UInt16,gdal.GDT_Int16,gdal.GDT_UInt32,gdal.GDT_Int32,gdal.GDT_Float32,gdal.GDT_Float64,gdal.GDT_CInt16,gdal.GDT_CInt32,gdal.GDT_CFloat32,gdal.GDT_CFloat64]
#     datatype = gdal.GDT_Byte
#     # datatype = gdal.GDT_Float32
#     driver = gdal.GetDriverByName("GTiff")  # 创建文件驱动
#     dataset = driver.Create(img_newB, im_width, im_height, im_bands, datatype)
#     dataset.SetGeoTransform(datasetA_geotrans)  # 写入仿射变换参数
#     dataset.SetProjection(datasetA_proj)  # 写入投影1
#     # dataset.GetRasterBand(1).WriteArray(ds_array_B)
#     # 写入影像数据
#     img_tif = Image.open(img_A)
#
#     img = np.asarray(img_tif).copy()
#
#     # img = np.zeros([im_height, im_width, 3], dtype=np.uint8)
#     ds_array_B_r = img[:, :, 0]
#     ds_array_B_g = img[:, :, 1]
#     ds_array_B_b = img[:, :, 2]
#     # ds_array_B_r = img[0,:, :]
#     # ds_array_B_g = img[1,:, :]
#     # ds_array_B_b = img[2,:, :]
#     ds_array_B_r[ds_array_B == 1] = 180
#     ds_array_B_g[ds_array_B == 1] = 180
#     ds_array_B_b[ds_array_B == 1] = 180  # ni yan
#
#     # ds_array_B_r[ds_array_B == 2] = 255
#     # ds_array_B_g[ds_array_B == 2] = 255
#     # ds_array_B_b[ds_array_B == 2] = 128
#     #
#     # ds_array_B_r[ds_array_B == 3] = 240
#     # ds_array_B_g[ds_array_B == 3] = 116
#     # ds_array_B_b[ds_array_B == 3] = 0
#     #
#     # ds_array_B_r[ds_array_B == 4] = 0  # 255,255,0粉砂
#     # ds_array_B_g[ds_array_B == 4] = 255
#     # ds_array_B_b[ds_array_B == 4] = 0
#     #
#     # ds_array_B_r[ds_array_B == 5] = 0  # 255,255,0粉砂
#     # ds_array_B_g[ds_array_B == 5] = 255
#     # ds_array_B_b[ds_array_B == 5] = 0
#     #
#     # ds_array_B_r[ds_array_B == 6] = 100  # 255,255,0粉砂
#     # ds_array_B_g[ds_array_B == 6] = 100
#     # ds_array_B_b[ds_array_B == 6] = 100
#
#     img[:, :, 0] = ds_array_B_r
#     img[:, :, 1] = ds_array_B_g
#     img[:, :, 2] = ds_array_B_b
#     dataset.GetRasterBand(1).WriteArray(ds_array_B_r)
#     dataset.GetRasterBand(2).WriteArray(ds_array_B_g)
#     dataset.GetRasterBand(3).WriteArray(ds_array_B_b)
#     # for i in range(im_bands):
#     #     dataset.GetRasterBand(i+1).WriteArray(img[i])
#     del dataset
# def tans_img(img_ij,mask):
#
#
#     img = img_ij.copy()
#
#     # img = np.zeros([img_ij.shape[0], img_ij.shape[1], 3], dtype=np.uint8)
#     ds_array_B_r = img[:, :, 0]
#     ds_array_B_g = img[:, :, 1]
#     ds_array_B_b = img[:, :, 2]
#     # ds_array_B_r = img[0,:, :]
#     # ds_array_B_g = img[1,:, :]
#     # ds_array_B_b = img[2,:, :]
#     ds_array_B_r[mask == 1] = 180
#     ds_array_B_g[mask == 1] = 180
#     ds_array_B_b[mask == 1] = 180 #ni yan
#
#     # ds_array_B_r[mask == 2] = 255
#     # ds_array_B_g[mask == 2] = 255
#     # ds_array_B_b[mask == 2] = 128
#     #
#     # ds_array_B_r[mask == 3] = 240
#     # ds_array_B_g[mask == 3] = 116
#     # ds_array_B_b[mask == 3] = 0
#     #
#     # ds_array_B_r[mask == 4] = 0  # 255,255,0粉砂
#     # ds_array_B_g[mask == 4] = 255
#     # ds_array_B_b[mask == 4] = 0
#     #
#     # ds_array_B_r[mask == 5] = 0  # 255,255,0粉砂
#     # ds_array_B_g[mask == 5] = 255
#     # ds_array_B_b[mask == 5] = 0
#     #
#     # ds_array_B_r[mask == 6] = 100  # 255,255,0粉砂
#     # ds_array_B_g[mask == 6] = 100
#     # ds_array_B_b[mask == 6] = 100
#     img[:, :, 0] = ds_array_B_r
#     img[:, :, 1] = ds_array_B_g
#     img[:, :, 2] = ds_array_B_b
#     return img
#
#
#



# def plot_img_and_mask(img, mask):
#     classes = mask.shape[2] if len(mask.shape) > 2 else 1
#     fig, ax = plt.subplots(1, classes + 1)
#     ax[0].set_title('Input image')
#     ax[0].imshow(img)
#     if classes > 1:
#         for i in range(classes):
#             ax[i+1].set_title(r'Output mask (class {i+1})')
#             ax[i+1].imshow(mask[:, :, i])
#     else:
#         ax[1].set_title(r'Output mask')
#         ax[1].imshow(mask)
#     plt.xticks([]), plt.yticks([])
#     plt.show()
# def predict_img_old(net,
#                 full_img,
#                 scale_factor=1,
#                 out_threshold=0.5):
#     x, y = full_img.shape[0], full_img.shape[1]
#     slice = zoom(full_img, (256 / x, 256 / y,1), order=0)
#     slice = np.transpose(slice, (2, 0, 1))
#     input = torch.from_numpy(slice).unsqueeze(
#         0).float().cuda()
#     net.eval()
#     with torch.no_grad():
#         out_main = net(input)
#         out = torch.argmax(torch.softmax(
#             out_main, dim=1), dim=1).squeeze(0)
#         out = out.cpu().detach().numpy()
#         pred = zoom(out, (x / 256, y / 256), order=0)
#     return pred
# def Nom01(img):
#     min = np.min(img)
#     max = np.max(img)+0.000001
#     imgNew = (img-min)/(max-min)
#     # img = imgNew.astype('uint8')
#     return imgNew
# def RGB01(img):
#     img = img.astype('int16')
#     r = img[:,:,0:1]
#     g = img[:, :, 1:2]
#     b = img[:, :, 2:3]
#
#     r01 = r.copy()
#     g01 = g.copy()
#     b01 = b.copy()
#
#     rgb = r+ g +b
#     r01 = r01/(rgb+0.000001)
#     g01 = g01/(rgb+0.000001)
#     b01 = b01 / (rgb+0.000001)
#     img_rgb01 = np.concatenate((r01,g01,b01),axis = 2)
#     return img_rgb01
# def BGR01(img):
#     img = img.astype('int16')
#     r = img[:,:,0:1]
#     g = img[:, :, 1:2]
#     b = img[:, :, 2:3]
#
#     r01 = r.copy()
#     g01 = g.copy()
#     b01 = b.copy()
#
#     # rgb = r+ g +b
#     # r01 = r01/(rgb+0.000001)
#     # g01 = g01/(rgb+0.000001)
#     # b01 = b01 / (rgb+0.000001)
#     img_rgb01 = np.concatenate((b01,g01,r01),axis = 2)#注意CV2顺序是BGR,需要转换下
#     return img_rgb01
# def Nom255(img):
#     min = np.min(img)
#     max = np.max(img) + 1e-6
#     imgNew = img*255/(max-min)
#     img = imgNew.astype('uint8')
#     return img
def predict_img(net,
                full_img,#400,400,3
                scale_factor=1,
                out_threshold=0.5,
                transform=None):
    x, y = full_img.shape[0], full_img.shape[1]
    slice = zoom(full_img, (400 / x, 400 / y,1), order=0)

    #slice = transform(slice)#【3,400，400】#把transform给注释了，因为看psp里没有（改）
    # slice = slice/255.0
    # slice = np.transpose(slice, (2, 0, 1))
    #input = slice[None].to('cuda')  # [1,3,400,400]#相应的这段也给注释了（改）
    input = np.expand_dims(np.transpose(preprocess_input(np.array(slice, np.float32)), (2, 0, 1)), 0)
    # input = torch.from_numpy(slice).unsqueeze(
    #     0).float().cuda()#[1,3,256,256]


    net.eval()
    with torch.no_grad():
        input = torch.from_numpy(input)#创建张量tensor
        # input = torch.Tensor(input)#在此处进行了一项修改，在pspnet的预测网络里，此处有该变化
        input = input.cuda()
        # out_main = net(input)[0]#[1,3,256,256]#这里为什么变成【1，2，400，400】】了呢，但是经过相同操作pspnet变成了【400，400，2】
        #这里outmain变成【1，2,400,400】
        out_main = net(input)[0]#[1,3,256,256]#这里为什么变成【1，2，400，400】】了呢，但是经过相同操作pspnet变成了【400，400，2】------这里现在还是
        # plt.imshow(out_main)
        # plt.show()
        #经过argmax和softmax操作，这里的out变成了【400,400】的张量形式，但该过程是原来代码中被注释掉，我直接解除注释使用的，如果不成功，应该再注释掉，按照原来预想重新设计代码
        # out_main = out_main.permute(1, 2, 0)#[400,400,2]
        # out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)#【400,400】
        out = F.softmax(out_main.permute(1, 2, 0), dim=-1).cpu().numpy()
       #增加掩膜乘法部分
        # out [:50, :, :] *= 0.5
        # out[350:, :, :] *= 0.5
        # out[:, :50, :] *= 0.5
        # out[:, 350:, :] *= 0.5
        out = out.argmax(axis=-1)
        # out = out.cpu().detach().numpy()
        # out_main = out.squeeze(0)
        # out = out_main.permute(1, 2, 0)
        # out = out_main.permute(1, 2, 0)

        # out = out.cpu().numpy()#out变成数组形式
        # out = 0.5 * (out + 1)
        # out = out.cpu().detach().numpy()
        # pred = np.transpose(out, (1, 2, 0))
        # out = zoom(out, (x / 400, y / 400,1), order=0)
        out = zoom(out, (x / 400, y / 400), order=0)
        # out = out[int((net.input_shape[0] - x) // 2): int((net.input_shape[0] - x) // 2 + y), \
        #      int((net.input_shape[1] - y) // 2): int((net.input_shape[1] - y) // 2 + y)]
        # out = 255*abs(out)#传进来的full_img为（256，256，3），现在这个out变为（256，256，2）#这里是将0,1二值图放大到0,255二值中
        # out = out.astype(np.uint8)

        # name_classes = ["background", "river"]
        # classes_nums        = np.zeros([2])
        # total_points_num    = x * y
        # print('-' * 63)
        # print("|%25s | %15s | %15s|"%("Key", "Value", "Ratio"))
        # print('-' * 63)
        # for i in range(2):
        #     num     = np.sum(out == i)
        #     ratio   = num / total_points_num * 100
        #     if num > 0:
        #         print("|%25s | %15s | %14.2f%%|"%(str(name_classes[i]), str(num), ratio))
        #         print('-' * 63)
        #     classes_nums[i] = num
        # print("classes_nums:", classes_nums)
        # plt.figure()
        # plt.imshow(out)
        # plt.show()
        # X = Image.fromarray(out)
        # X.save("clip6.jpg")
        # pred = np.transpose(pred, (2, 1, 0))
        # plt.imshow(np.where(out > 85, 255, 0), cmap="gray")

    return out
# def  result_Normalization(result):
#     array = result.cpu().data.numpy()
#
#     return array
# def writeTiff(image, fileName,geotrans,proj):
# def writeTiff(image, image_name, geotrans, proj):
#     bandCount = image.shape[2]
#     # bandCount = 1
#     col = image.shape[1]
#     row = image.shape[0]
#     image = np.transpose(image,(1, 0, 2))
#     driver = gdal.GetDriverByName("GTiff")
#     dataset_result = driver.Create(image_name, col, row, bandCount, gdal.GDT_Byte)
#     dataset_result.SetGeoTransform(geotrans)  # 写入仿射变换参数
#     dataset_result.SetProjection(proj)  # 写入投影1
#     # for i in range(bandCount):
#     #   dataset_result.GetRasterBand(i+1).WriteArray(image[:,:,i])
#     for i in range(bandCount):
#             dataset_result.GetRasterBand(i+1).WriteArray(image[i])
#     del dataset_result
#

# def identify_datu(image_name,model_path):
#
#     dataset_img = Image.open(image_name)
#     # dataset_dsm = gdal.Open(dsm_name)
#     # head = dataset_img.GetGeoTransform()
#     geotrans = dataset_img.GetGeoTransform()  # 仿射矩阵
#     proj = dataset_img.GetProjection()  # 地图投影信息o
#     col = dataset_img.RasterXSize  # 栅格矩阵的列数shape2 7363
#     row = dataset_img.RasterYSize  # 栅格矩阵的行数shape1 453
#     # output = os.path.dirname(image_name) + '/result/' +NetName+'_'+ os.path.basename(image_name).split('.')[1] + '.tif'
#     # output ='G:/pix2pix/result/' + NetName + '_' + os.path.basename(image_name).split('.')[
#     #     1] + '.tif'
#     output ='G:/yemaolin/1/pspnet-pytorch-master/result' + NetName + '_' + os.path.basename(image_name).split('.')[
#         1] + '.tif'
#     print(output)
#     if col > col_cell or row > row_cell:
#         colnum = int(col / col_cell) + 1
#         rownum = int(row / row_cell) + 1
#         # result = np.zeros([row,col, 3], dtype=np.uint8)
#         result = np.zeros([row, col,3], dtype=np.uint8)
#         for r in range(rownum):
#             for c in range(colnum):
#                 row_i = row_cell
#                 col_i = col_cell
#                 c_min = c * row_cell
#                 r_min = r * col_cell
#                 if c_min > 0:
#                     c_min = c_min -128
#                 if r_min > 0:
#                     r_min = r_min - 128
#                 ci_max = c_min + col_cell
#                 ri_max = r_min + row_cell
#                 if ci_max > col:
#                     col_i = col - c_min
#                 if ri_max > row:
#                     row_i = row - r_min
#                 if c_min < col and r_min < row:
#                     img_ij = dataset_img.ReadAsArray(c_min, r_min, col_i, row_i)
#                     # dsm_ij = dataset_dsm.ReadAsArray(c_min, r_min, col_i, row_i)
#                     result_ij = identify_img(img_ij, model_path, CELL_X, CELL_Y)
#                     ci = ci_max-c_min
#                     ri = ri_max-r_min
#                     result[r_min:ri_max,c_min:ci_max,:] = result_ij
#         print(result.shape)
#         # img = Image.fromarray(result).convert('RGB')
#         # img.save("G:/pix2pix/result/segtif1.tif")
#         writeTiff(result, output,geotrans,proj)
#     else :
#         result = np.zeros([row, col, 3], dtype=np.uint8)
#         result = identify_single_img(image_name, model_path, CELL_X, CELL_Y)
#         # writeTiff(result, output, geotrans, proj)
#


def identify_single_img(image_name, model_path, row, col):
    # image1 = cv2.imread(image_name)
    image = Image.open(image_name).convert("RGB")
    transforms = transform.Compose([
        transform.ToTensor(),
        # transform.Resize((256, 256)),
        transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # image = RGB01(image)
    # image = image / 255.0

    rowheight = int(row)#256
    colwidth = int(col)#256

    # h_img1 = int(image1.shape[0])
    # w_img1 = int(image1.shape[1])
    h_img = int(image.size[1])
    w_img = int(image.size[0])
    image = np.array(image) #[22458,27348,3]
    buchang = 0.125
    rownum = int(h_img / (rowheight * buchang)) +1#分块行数
    colnum = int(w_img / (colwidth * buchang)) + 1#分块列数


    # # 定义一个400*400的全0矩阵
    # matrix = np.zeros((400, 400))
    #
    # # 中间300*300的值赋为1
    # matrix[50:350, 50:350] = 1
    #
    # # 剩余的值赋为0.5
    # matrix[matrix != 1] = 0.5
    # net = net_factory(net_type='unet', in_chns=3,
    #                  class_num=4)
    # net = UnetPlusPlus(num_classes=3, deep_supervision=False).to('cuda')
    # net = pspnet(num_classes=2,downsample_factor="mobilenet").to('cuda')
    # net = pspnet(num_classes=2,downsample_factor="16",backbone='mobilenet').to('cuda')
    net = SegFormer(num_classes=5,pretrained=False).to('cuda')
    # ckpt = torch.load(model_path)
    # net.load_state_dict(ckpt['G_model'], strict=False)
    net.load_state_dict(torch.load(model_path),strict=False)
    # mask = np.zeros([h_img, w_img, 2], dtype=np.uint8)
    mask = np.zeros([h_img, w_img], dtype=np.uint8)
    maska = np.zeros([h_img, w_img], dtype=np.uint8)
    masksum = np.zeros([h_img, w_img], dtype=np.uint8)

    '''
     mask[r1:r2, c1:c2,:] = maski[ri1:ri2, ci1:ci2,:]
IndexError: too many indices for array: array is 2-dimensional, but 3 were indexed
在初始代码中mask和maski是三维的，当我将其变为【400,400】来输出后，会报该错误，故，我现在尝试把mask和maski后面删掉
    '''
    # mask = np.zeros([h_img, w_img, 3], dtype=np.uint8)
    for r in range(rownum):
        for c in range(colnum):
          # if c==57:
          # if r == 2:
          #   strXmin = float(c * colwidth * buchang)
          #   strYmin = float(r * rowheight * buchang)
            r1 = r * rowheight * buchang
            r2 = (r * buchang + 1) * rowheight
            if r2 > h_img:
                r2 = h_img - 1
            c1 = c * colwidth * buchang
            c2 = (c * buchang + 1) * colwidth
            if c2 > w_img:
                c2 = w_img - 1
            # c2 = (c * buchang + 1) * colwidth
            c1 = int(c1)
            c2 = int(c2)
            r1 = int(r1)
            r2 = int(r2)
            # print(c1,c2,r1,r2)
            if r1 >= (h_img - 1) or c1 >= (w_img - 1):
                continue#作用，不执行后续循环内容，直接下一次循环
            img = image[r1:r2, c1:c2,:]
          #把该区域的图片传递给img以用来传入到预测img里对其进行预测
            # img = BGR01(img)
            # img = img / 255.0
            maski = predict_img(net, img, transform=transforms)
            # for x in range(len(maski)):
            #     for y in range(len(maski[x])):
            #         if x<29 or x>370 or y<29 or y>370:
            #             maski[x, y] = maski[x, y]*0.5
                    #加这个掩膜加权效果不好，很明显的分割线
            # maski = maski * matrix
            # plt.imshow(maski)
            # plt.show()
            # if r1 > 0:
            #     r1 = int(r1 + rowheight / 4)
            # if c1 > 0:
            #     c1 = int(c1 + colwidth / 4)
            # if r1 == 0:
            #     ri1 = 0
            # else:
            #     ri1 = int(rowheight / 4)
            # if c1 == 0:
            #     ci1 = 0
            # else:
            #     ci1 = int(colwidth / 4)
            #
            # ri2 = int(0.75 * rowheight)#这里为什么要乘0.75
            # ci2 = int(0.75 * colwidth)
            # if c2 > (w_img):
            #     c2 = (w_img)
            # if r2 > (h_img):
            #     r2 = (h_img)
            # if c2 <= (w_img - colwidth / 4):
            #     c2 = int(c2 - colwidth / 4)
            # else:
            #     ci2 = int(ci1 + c2 - c1)
            # if r2 <= (h_img - rowheight / 4):
            #     r2 = int(r2 - rowheight / 4)
            # else:
            #     ri2 = int(ri1 + r2 - r1)
            # mask[r1:r2, c1:c2] = maski[ri1:ri2, ci1:ci2] + mask[ri1:ri2, ci1:ci2]#这里我自认为mask需要新的预测结果和上次的预测值相加，步长为0.5所以肯定有重叠
            # masksum[ri1:ri2,ci1:ci2] = masksum[ri1:ri2,ci1:ci2] + 1
            # mask[r1:r2, c1:c2] = maski[r1:r2, c1:c2] + mask[r1:r2, c1:c2]#这里我自认为mask需要新的预测结果和上次的预测值相加，步长为0.5所以肯定有重叠
            mask[r1:r2, c1:c2] = maski + mask[r1:r2, c1:c2]#这里我自认为mask需要新的预测结果和上次的预测值相加，步长为0.5所以肯定有重叠
            masksum[r1:r2,c1:c2] = masksum[r1:r2,c1:c2] + 1

        # mask = mask.argmax()
        #这里做了修改，把这些内容从循环里抛了出来尝试一下
        # X = Image.fromarray(mask)#该方法是将数组转换为图片
        # X.save("clip-pspnet1.tif")
        # print("1")
        # masksum[:721]=1
        # masksum[965:]=1
    for m in range(len(mask)):
        for n in range(len(mask[m])):
            if m == len(mask) - 1 or n == len(mask[m]) - 1:  # 判断该位置是否在最后一列或最后一行
                masksum[m][n] = 1  # 如果是，则设置该位置的值为1
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            maska[i, j] = mask [i, j] / masksum[i, j]
            if maska[i, j] <= 0.5:
                maska[i,j] = 0
                # print ("该处值为0")
            else:
                maska[i, j] = 255
                # print("该处值为1")
        print(1)
        #该过程为改动后的重叠预测方法，其中加入了masksum数组，来记录重叠次数，然后除以该重叠次数，大于0.5的设置为1，否则为0，就是个简单的投票环节了

    # X = Image.fromarray(maska)  # 该方法是将数组转换为图片
    # X.save("clip-pspnet2.tif") # 该过程把保存图片放在循环外，保证不是直接保存
    # mask = mask / masksum
    X = Image.fromarray(maska)
    X.save("clip-pspnet1.tif")
    # output = "pix2pix/result/"+NetName+'_0.1p.tif'
    # X = Image.fromarray(mask)
    # X.save("clip-pspnet2.tif")
    # mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
    # cv2.imwrite('clip6.jpg', mask)
    return mask
    # writeTiff(mask, output)
        # os.path.dirname(image_name) + '/result/'  + os.path.basename(image_name).split('.')[0] + '.TIF'
    # tans_single_img(image_name, mask, output)

# ###多波段##################
# def identify_img(image,model_path,row,col):
#      # image = np.transpose(image, (1, 2, 0))
#      # dsm = np.transpose(dsm, (1, 2, 0))
#      image = Image.open(image)
#      # image = np.transpose(image, (1, 2, 0))
#      transforms = transform.Compose([
#          transform.ToTensor(),
#          # transform.Resize((256, 256)),
#          transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#      ])
#
#      rowheight = int(row)
#      colwidth = int(col)
#      h_img = int(image.size[0])
#      w_img = int(image.size[1])
#      buchang=0.5
#      rownum=int(h_img/(rowheight*buchang)) +1
#      colnum=int(w_img/(colwidth*buchang)) +1
#      # net = net_factory(net_type='unet', in_chns=3,
#      #                  class_num=4)
#
#      # net = UnetPlusPlus(num_classes=3, deep_supervision=False).to('cuda')
#      net = pspnet(num_classes=5
#                   ,downsample_factor="16",backbone="mobilenet",pretrained=False).to('cuda')
#      # ckpt = torch.load(model_path)
#      # net.load_state_dict(torch.load(model_path))
#      net.load_state_dict(torch.load(model_path), strict=False)
#      mask=np.zeros((h_img, w_img, 2),dtype=np.uint8)
#      timeall=0
#      for r in range(rownum):
#             for c in range(colnum):
#                 strXmin=float(c*colwidth*buchang)
#                 strYmin=float(r*rowheight*buchang)
#                 r1= r * rowheight*buchang
#                 r2= (r*buchang + 1) * rowheight
#                 if r2>h_img:
#                     r2=h_img-1
#                 c1=c * colwidth*buchang
#                 c2=(c*buchang + 1) * colwidth
#                 if c2>w_img:
#                     c2=w_img-1
#                 c2=(c*buchang + 1) * colwidth
#                 c1=int(c1)
#                 c2=int(c2)
#                 r1=int(r1)
#                 r2=int(r2)
#                 # print(c1,c2,r1,r2)
#                 if r1>=(h_img-1) or c1>=(w_img-1):
#                     continue
#                 img= image[r1:r2,c1:c2,:]
#                 img = BGR01(img)
#                 time0 = time.time()
#                 maski = predict_img(net,img,transform =transforms )
#                 time1 = time.time()
#                 timed = time1 - time0
#                 timeall = timeall + timed
#                 if r1>0:
#                     r1=int(r1+ rowheight/4)
#
#                 if c1>0:
#                     c1=int(c1+ colwidth/4)
#
#                 if r1==0:
#                     ri1=0
#                 else:
#                     ri1= int(rowheight/4)
#                 if c1==0:
#                     ci1=0
#                 else:
#                     ci1= int(colwidth/4)
#
#                 ri2= int(0.75*rowheight)
#                 ci2= int(0.75*colwidth)
#                 if c2 > (w_img ):
#                     c2 = (w_img)
#                 if r2 > (h_img ):
#                     r2 = (h_img)
#                 if c2<=(w_img- colwidth/4):
#                     c2 = int(c2- colwidth/4)
#                 else:
#                     ci2=int(ci1+c2-c1)
#                 if r2<=(h_img-rowheight/4):
#                     r2 = int(r2- rowheight/4)
#                 else:
#                     ri2=int(ri1+r2-r1)
#                 mask[r1:r2,c1:c2,:]=maski[ri1:ri2,ci1:ci2,:]
#             X = Image.fromarray(mask)
#             X.save("clip-pspnet1.tif")
#             print("1")
#      result = tans_img(image, mask)
#      X = Image.fromarray(result)
#      X.save("clip-pspnet2.tif")
#      return mask
#      # return mask

if __name__ == '__main__':
    # image_name='h:/yx2021/zhunnan17/zhunnan1-7_Mesh50020.jpg'#'E:/yx2021/sample/data/1810test.tif'
    # model_path = 'F:/yx2021/sample/202204/Fully_Supervised_3_labeled/FCN8/iter_14000_dice_0.1408.pth'
    # # Uncertainty_Aware_Mean_Teacher_3_labeled
    # # Fully_Supervised_3_labeled
    # images_path = 'F:/yx2021/sample/point1-test/dom'
    # dsms_path= 'F:/yx2021/sample/point1-test/dsm'
    image_name = 'anotherDFS.tif'
    # image_name = 'dfs_png.png'
    images_path= 'G:\yemaolin\segformer-pytorch-master'#'E:/yx2021/sample/data/1810test.tif'
    # model_path = 'G:\pix2pix\clipweight_unet\piximg2img_97.pth'
    # model_path = 'G:\yemaolin\\1\pspnet-pytorch-master\model_data\pspnet_mobilenetv2.pth'
    model_path = 'logs/best_epoch_weights.pth'
    # model_path = 'G:\yemaolin\segformer-pytorch-master\logs\ep160-loss0.126-val_loss0.135.pth'
    # Uncertainty_Aware_Mean_Teacher_3_labeled
    # Fully_Supervised_3_labeled
    # images_path = 'F:/yx2021/sample/point1-test/dom'
    # dsms_path= 'F:/yx2021/sample/point1-test/dsm'a
    row=400
    col=400
    timeall = 0
    # for idx, img_name in enumerate(sorted(os.listdir(images_path))):
    for idx, image_name in enumerate(sorted(os.listdir(images_path))):
        time0 = time.time()
        if not image_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
        # if not image_name.lower().endswith(('.bmp', '.jpeg',  '.png', '.tif', '.tiff')):
            continue
        # img_path = os.path.join(images_path, img_name)
        image_path = os.path.join(images_path, image_name)
        # dsm_path = os.path.join(dsms_path, img_name)
        print(image_path)

        # identify_img(filepath, weights_path, row, col, openMode)
        # identify_datu(img_path,dsm_path,model_path)
        # identify_datu(image_path,model_path)
        identify_single_img(image_path, model_path, row, col)
        # identify_img(image_path, model_path, row, col)
        time1 = time.time()
        timed = time1 - time0
        timeall = timeall + timed
    print("time: ", timeall, " s")

