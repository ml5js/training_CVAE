import urllib.request
import os

with open("./data/categories.txt", 'r') as f:
  classes = f.readlines()

classes = [c.replace('\n','').replace(' ','_') for c in classes]

# function to retrieve quick_draw dataset
def download():
  base = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'
  for c in classes:
    cls_url = c.replace('_', '%20')
    path = base+cls_url+'.npy'
    print(path)
    os.mkdir("data/"+c)
    urllib.request.urlretrieve(path, 'data/'+c+'/'+c+'.npy')

download()