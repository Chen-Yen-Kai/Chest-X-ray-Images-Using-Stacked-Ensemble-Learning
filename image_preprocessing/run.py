import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import cv2
import numpy as np
import torch.nn as nn
from models.VAE import uVAE
import time
from glob import glob
import pdb
import argparse
torch.manual_seed(42)
np.random.seed(42)
from pydicom import dcmread
from skimage.transform import resize
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from skimage.exposure import equalize_hist as equalize
from skimage.io import imread,imsave
from skimage import img_as_ubyte
from skimage.color import rgb2gray
from utils.postProcess import postProcess
from utils.tools import dice,binary_accuracy
from torchvision.utils import save_image
plt.gray()

min_x=0
max_x=0
min_y=0
max_y=0
def loadDCM(f, no_preprocess=False,dicom=False):
	wLoc = 448
	### Load input dicom
	if dicom:
		dcmFile = dcmread(f)
		dcm = dcmFile.pixel_array
		dcm = dcm/dcm.max()
		if dcmFile.PhotometricInterpretation == 'MONOCHROME1':
			### https://dicom.innolitics.com/ciods/ct-image/image-pixel/00280004 ###
			### When MONOCHROME1, 0->bright, 1->dark intensities
			dcm = 1-dcm 
	else:
		## Load input image
		dcm = imread(f)
		dcm = dcm/dcm.max()
	if not no_preprocess:
		dcm = equalize(dcm)

	if len(dcm.shape) > 2:
		dcm = rgb2gray(dcm[:,:,:3])
	
	### Crop and resize image to 640x512 
	hLoc = int((dcm.shape[0]/(dcm.shape[1]/wLoc)))
	if hLoc > 576:
		hLoc = 576
		wLoc = int((dcm.shape[1]/(dcm.shape[0]/hLoc)))

	img = resize(dcm,(hLoc,wLoc))
	img = torch.Tensor(img)
	pImg = torch.zeros((640,512))
	h = (int((576-hLoc)/2))+p
	w = int((448-wLoc)/2)+p
	roi = torch.zeros(pImg.shape)
	if w == p:
		pImg[np.abs(h):(h+img.shape[0]),p:-p] = img
		roi[np.abs(h):(h+img.shape[0]),p:-p] = 1.0
	else:
		pImg[p:-p,np.abs(w):(w+img.shape[1])] = img	
		roi[p:-p,np.abs(w):(w+img.shape[1])] = 1.0

	imH = dcm.shape[0]
	imW = dcm.shape[1]
	pImg = pImg.unsqueeze(0).unsqueeze(0)
	return pImg, roi, h, w, hLoc, wLoc, imH, imW

def saveMask(f,img,h,w,hLoc,wLoc,imH,imgW,no_post=False):
	
	img = img.data.numpy()
	imgIp = img.copy()
	
	if w == p:
		img = resize(img[np.abs(h):(h+hLoc),p:-p],
					(imH,imW),preserve_range=True)
	else:
		img = resize(img[p:-p,np.abs(w):(w+wLoc)],
					(imH,imW),preserve_range=True)
    #print(img_as_ubyte(img>0.5))
	#cv2.imwrite('./_mask.tif', img_as_ubyte(img>0.5))


	if not no_post:
		imgPost = postProcess(imgIp)
		if w == p:
			imgPost = resize(imgPost[np.abs(h):(h+hLoc),p:-p],
							(imH,imW))
		else:
			imgPost = resize(imgPost[p:-p,np.abs(w):(w+wLoc)],
							(imH,imW))
        
		imsave(f.replace('.PNG','.PNG'),img_as_ubyte(imgPost > 0.5))
        #cv2.imwrite('./10_mask.tif', img_as_ubyte(imgPost > 0.5))
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='./test/', help='Path to input files')
parser.add_argument('--model', type=str, default='saved_models/lungVAE.pt', help='Path to trained model')
parser.add_argument('--hidden', type=int, default=16, help='Hidden units')
parser.add_argument('--latent', type=int, default=8, help='Latent dim')
parser.add_argument('--saveLoc', type=str, default='', help='Path to save predictions')
parser.add_argument('--unet',action='store_true', default=False,help='Use only U-Net.')
parser.add_argument('--dicom',action='store_true', default=False,help='DICOM inputs.')
parser.add_argument('--no_post',action='store_true', default=False,help='Do not post process predictions')
parser.add_argument('--no_preprocess',action='store_true', 
						default=False,help='Do not preprocess input images')
parser.add_argument('--padding', type=int, default=32, help='Zero padding')


args = parser.parse_args()
p = args.padding
print("Loading "+args.model)
if 'unet' in args.model:
	args.unet = True
	args.hidden = int(1.5*args.hidden)
else:
	args.unet = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#定義肺部分割模型(define lung segmentation model)
net = uVAE(nhid=args.hidden,nlatent=args.latent,unet=args.unet)
net.load_state_dict(torch.load(args.model,map_location=device))
net.to(device)
t = time.strftime("%Y%m%d_%H_%M")

if args.saveLoc is '':
	save_dir = args.data+'pred_'+t+'/'
else:
	save_dir = args.saveLoc+'pred_'+t+'/'
if not os.path.exists(save_dir):
	os.mkdir(save_dir)

nParam = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("Model "+args.model.split('/')[-1]+" Number of parameters:%d"%(nParam))

if args.dicom:
	filetype = 'DCM'
else:
	filetype= 'png'
print(filetype)
files = list(set(glob(args.data+'*.'+filetype)) \
			- set(glob(args.data+'*_mask*.'+filetype)) \
			- set(glob(args.data+'*label*.'+filetype)))

files = sorted(files)
print(files)
for fIdx in range(len(files)):
		coordinates_List = [];
		f = files[fIdx]
		fName = f.split('/test')[-1]
		print(fName)
		img, roi, h, w, hLoc, wLoc, imH, imW = loadDCM(f,
													no_preprocess=args.no_preprocess,
													dicom=args.dicom)
		img = img.to(device)
		_,mask = net(img)
		mask = torch.sigmoid(mask)
        
		f = save_dir+fName.replace('.'+filetype,'.png')
		print(f)
		saveMask(f,mask.squeeze().cpu(),h,w,hLoc,wLoc,imH,imW,args.no_post)
		print("Segmenting %d/%d"%(fIdx,len(files)))
        
		image = cv2.imread(save_dir+fName.strip('\.png')+'.png',0)
		print(save_dir+fName.strip('\.png')+'.png')
		gray = image
		thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)[1]
        # 尋找肺部分割結果的座標點
		cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = cnts[0] if len(cnts) == 2 else cnts[1]
		c = max(cnts, key=cv2.contourArea)
        #找座標點 
		for i in range(len(cnts)):
		# Obtain outer coordinates
			c=cnts[i]
			left = tuple(c[c[:, :, 0].argmin()][0])
			right = tuple(c[c[:, :, 0].argmax()][0])
			top = tuple(c[c[:, :, 1].argmin()][0])
			bottom = tuple(c[c[:, :, 1].argmax()][0])
			image_result=cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
			print('left:'+str(left)+', right:'+str(right)+', top:'+str(top)+', bottom:'+str(bottom))
			coordinates_List.append([left,right,top,bottom])
		print(coordinates_List)
        
		if coordinates_List[0][0][0] < coordinates_List[1][0][0]:
			min_x=coordinates_List[0][0][0]
		else:
			min_x=coordinates_List[1][0][0]
		print (min_x)
		if coordinates_List[0][1][0] > coordinates_List[1][1][0]:
			max_x=coordinates_List[0][1][0]
		else:
			max_x=coordinates_List[1][1][0]
		print (max_x)
		if coordinates_List[0][2][1] < coordinates_List[1][2][1]:
			min_y=coordinates_List[0][2][1]
		else:
			min_y=coordinates_List[1][2][1]
		print (min_y)
		if coordinates_List[0][3][1] > coordinates_List[1][3][1]:
			max_y=coordinates_List[0][3][1]
		else:
			max_y=coordinates_List[1][3][1]
		print ("min_x:"+str(min_x)+",max_x:"+str(max_x)+"min_y:"+str(min_y)+",max_y:"+str(max_y))
		#cropped = cv2.cuda_GpuMat(image_result, (min_y, min_x, max_y,max_x))
		new_img=image_result[min_y:max_y,min_x:max_x]
		#cv2.imshow('image', new_img)
		#cv2.waitKey()
        #兩個肺部比較找出最終的4角
		if max_x == min_x and max_y != min_x:
			Width=max_x
			Height=max_y-min_y
			center_x=max_x
			center_y=(max_y+min_y)/2
		elif max_y == min_x and max_x != min_x:
			Width=max_x-min_x
			Height=max_y
			center_x=(max_x+min_x)/2
			center_y=max_y
		else:
			Width=max_x-min_x
			Height=max_y-min_y
			center_x=(max_x+min_x)/2
			center_y=(max_y+min_y)/2
		if Width >= Height:
			Height=Width
			crop_size_with = center_x - Width / 2
			crop_size_Height = center_y - Width / 2
		else:
			Width=Height
			crop_size_with = center_x - Height / 2
			crop_size_Height = center_y - Height / 2
		if crop_size_with<=0:
			crop_size_with=0
		if crop_size_Height<=0:
			crop_size_Height=0
        #裁切
		crop_img = image[int(crop_size_Height):int(crop_size_Height+Height), int(crop_size_with):int(crop_size_with+Width)]
        #肺部mask
		mark_label = cv2.imread("./test/pred_"+t+"/"+fName.strip('\.png')+".png")
		print("./test/pred_"+t+"/"+fName.strip('\.png')+".png")
		crop_img = mark_label[int(crop_size_Height):int(crop_size_Height+512), int(crop_size_with):int(crop_size_with+512)]
		crop_img = mark_label[int(crop_size_Height):int(crop_size_Height+Height), int(crop_size_with):int(crop_size_with+Width)]
		print(crop_img.shape)
        #裁切完作RSIZE至512
		lung_mask = cv2.resize(crop_img, (512, 512), interpolation=cv2.INTER_AREA)
		#####################
        
		img = cv2.imread("./test/"+fName.strip('\.png')+".png")
		print("./test/"+fName.strip('\.png')+".png")
        
		img_crop_img_or = img[int(crop_size_Height):int(crop_size_Height+Height), int(crop_size_with):int(crop_size_with+Width)]
# 		cv2.imwrite("./result/image_or_size/"+fName.strip('\.png') img_crop_img_or)
		img_crop_img = cv2.resize(img_crop_img_or, (512, 512), interpolation=cv2.INTER_AREA)
 		#512影像
		cv2.imwrite("./result/image_or_size/"+fName.strip('\.png')+"_"+str(int(crop_size_Height))+"_"+str(int(crop_size_Height+Height))+"_"+str(int(crop_size_with))+"_"+str(int(crop_size_with+Width))+"_"+str(img_crop_img_or.shape[0])+"_.png", img_crop_img)
		cv2.imwrite("./result/lung_mask/"+fName.strip('\.png')+"_"+str(int(crop_size_Height))+"_"+str(int(crop_size_Height+Height))+"_"+str(int(crop_size_with))+"_"+str(int(crop_size_with+Width))+"_"+str(img_crop_img_or.shape[0])+"_.png", lung_mask)
		read_mask = cv2.imread("./result/lung_mask/"+fName.strip('\.png')+"_"+str(int(crop_size_Height))+"_"+str(int(crop_size_Height+Height))+"_"+str(int(crop_size_with))+"_"+str(int(crop_size_with+Width))+"_"+str(img_crop_img_or.shape[0])+"_.png",0)
		
        
		read_image = cv2.imread("./result/image_or_size/"+fName.strip('\.png')+"_"+str(int(crop_size_Height))+"_"+str(int(crop_size_Height+Height))+"_"+str(int(crop_size_with))+"_"+str(int(crop_size_with+Width))+"_"+str(img_crop_img_or.shape[0])+"_.png",0)       
		print("./result/image_or_size/"+fName.strip('\.png')+"_"+str(int(crop_size_Height))+"_"+str(int(crop_size_Height+Height))+"_"+str(int(crop_size_with))+"_"+str(int(crop_size_with+Width))+"_"+str(img_crop_img_or.shape[0])+"_.png")
		image=cv2.add(read_image, np.zeros(np.shape(read_image), dtype=np.uint8), mask=read_mask)
        # 肺部分割影像
		cv2.imwrite("./result/mask_image_area/"+fName.strip('\.png')+"_"+str(int(crop_size_Height))+"_"+str(int(crop_size_Height+Height))+"_"+str(int(crop_size_with))+"_"+str(int(crop_size_with+Width))+"_"+str(img_crop_img_or.shape[0])+"_.png", image)
		
        ##################### 如果沒有mask，mask裡面的東西要註解掉
# 		mask = cv2.imread("./result/mask/"+fName.strip('\.jpg')+".jpg")#.jpg要改成marked
# 		print("./result/mask/"+fName.strip('\.jpg')+".jpg") #.jpg要改成marked
# 		mask_crop_img = mask[int(crop_size_Height):int(crop_size_Height+Height), int(crop_size_with):int(crop_size_with+Width)]
# 		mask_crop_img = cv2.resize(mask_crop_img, (512, 512), interpolation=cv2.INTER_AREA)
# 		cv2.imwrite("./result/label/"+fName.strip('\.jpg')+"_"+str(int(crop_size_Height))+"_"+str(int(crop_size_Height+Height))+"_"+str(int(crop_size_with))+"_"+str(int(crop_size_with+Width))+"_"+str(img_crop_img_or.shape[0])+"_.png", mask_crop_img)
        #####################
        

		box_result = cv2.imread("./test/"+fName.strip('\.png')+".png")
		box_result=cv2.rectangle(box_result, (int(crop_size_with), int(crop_size_Height)), (int(crop_size_with+Width), int(crop_size_Height+Height)), (0, 0, 255), 2)
		cv2.imwrite("./result/box_result/"+fName.strip('\.png')+".png"+"_"+str(int(crop_size_Height))+"_"+str(int(crop_size_Height+Height))+"_"+str(int(crop_size_with))+"_"+str(int(crop_size_with+Width))+"_"+str(img_crop_img_or.shape[0])+"_.png", box_result)
        
        
        

        
        

            
            
            
        

