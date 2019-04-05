# Training Conditional Variational Autoencoders
This is the repo for training CVAE on images. The original paper for traditional VAE and CVAE are here: [VAE](https://arxiv.org/pdf/1312.6114.pdf), [CVAE](https://pdfs.semanticscholar.org/3f25/e17eb717e5894e0404ea634451332f85d287.pdf).

## Requirments
* Make sure you have set up a python environment for tensorflow. More details of instructions can be found [here](https://ml5js.org/docs/training-setup).
* The version of the modules we use is listed in `requirements.txt`.

## Usage

### 1) Download this repository
Start by [downloading](https://github.com/ml5js/training-cvae.git) or clone this repository:
```bash
git clone https://github.com/ml5js/training-cvae.git
cd training-cvae
```

### 2) Collect data
This CVAE model can generate new image based on the training data and you could tune the latent vectors to change its shape.

When you have your image data and labels of each of them, make sure put them in a folder in the `root` of this project like this:
```
---data_folder
   |__ car
   |   |__ car1.png
   |   |__ car2.png
   |   |__ ...
   |
   |__ plane
   |   |__ plane1.png
   |   |__ plane2.png
   |   |__ ...
   |__ ...
   |__ ...
```
The type of file can be either numpy array like `.npy` or image files like `.png`, `.jpg` or `.jpeg`.
If you do not have a dataset on hand, you could use the `download.py` to download the [quick draw dataset](https://quickdraw.withgoogle.com/) provided by Google. Please make sure make a `data` folder before you run:
```bash
python download.py
```
### 3) Train

Run the training script with default settings:
```bash
python train.py --data_dir=./folder_with_my_custom_data
```
Or you could specify your preferred hyperparameters settings like this:
```bash
# This are the hyperparameters you can change to fit your data
python train.py --data_dir=./quick_draw \
--n_dim=16 \
--num_layers 2 \
--image_size=28 \
--image_depth=1 \
--filters=8 \
--learning_rate=0.0005 \
--decay_rate=0.01 \
--batch_size=128 \
--epochs=30
```
Or you could simply run the bash script `run.sh`:
```bash
bash run.sh
```

### 4) Use it


