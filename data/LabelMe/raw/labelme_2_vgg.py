import numpy as np
from PIL import Image
from optparse import OptionParser

op = OptionParser()
op.add_option("-p", "--path", type="string", default='data/', help="path for data of Label me")
op.add_option("-s", "--set", type="string", default='train', help="set used to pass VGG (train/valid/test)")
op.add_option("-m", "--poolm", type="string", default='', help="pooling mode used on VGG (None or empty/avg/max)")

(opts, args) = op.parse_args()
folder = opts.path #"./Deep Learning from Crowds/LabelMe/"
set_name = opts.set 
pool_mo = opts.poolm

def read_texts(filename):
    f = open(filename)
    data = [line.strip() for line in f]
    f.close()
    return data

names = read_texts(folder+"/labels_"+set_name+"_names.txt")
f_names = read_texts(folder+"/filenames_"+set_name+".txt")
f_2_read = ["/"+folder+"/"+name for folder,name in zip(names,f_names)]

X_images = []
for values in f_2_read:
    I = np.asarray(Image.open(folder+"/"+set_name+values)) #rodrigues resize images to 150x150 (but VGG was trained with 224x224)
    X_images.append(I)
X_images = np.asarray(X_images)
print("Images shapes: ",X_images.shape)

if pool_mo == "":
    pool_mo = None

#now pass through VGG
import repackage
repackage.up()
from code.learning_models import through_VGG
new_X = through_VGG(X_images.astype('float32'),pooling_mode=pool_mo)
print("New shape through VGG: ",new_X.shape)
if pool_mo == None:
    np.save('LabelMe_VGG_'+set_name+".npy",new_X) #none pooling
else:
    np.save('LabelMe_VGG_'+pool_mo+'_'+set_name+".npy",new_X) #avg/max pooling
