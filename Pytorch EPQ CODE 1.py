#List of reference's that I have used, numbered linking to the appropriate source
# https://docs.google.com/document/d/1ZHGUSP32DNwzl13a6Zb2NQqVrBpcJce3Cu9dqpv_tyE/edit?tab=t.0
#We will use Object-Oreintated Programming when creating our DcGAN
#Importing Function and Libraries to use
from __future__ import print_function
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
#Allows use to use downloaded datasets
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
import torchvision.datasets as dset 
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm 
import shutil


#Used by me to check my Pytorch library has downloaded and is running properly
x = torch.rand(3)
print(x)

start = time.time()
print("Hello")

# Speed ups to make my Pytorch Code to run Faster
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)
torch.backends.cudnn.benchmark = True 
#Lines 42-45 originally came from 
#[7]https://github.com/christianversloot/machine-learning-articles/blob/main/creating-dcgan-with-pytorch.md Github

#Resize's any photos from the dataset to be 128x128
transform = transforms.Compose([transforms.Resize(64),transforms.CenterCrop(64),
                                transforms.ToTensor(),
                                #Makes the image look normal, centre on etc 
                                transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
#Lines 50-53 originally came from [1]https://www.kaggle.com/code/jesucristo/gan-introduction Kaggle
#Downloading the Dog dataset from my folders and stores it in the variable of "TrainingData"
TrainingData = datasets.ImageFolder(r'c:\Pytorch EPQ\images', transform=transform)
#Sets batch size variable which i will use for the number of the images produced at 64
BatchSize = 64
#Sets image size variable which i will use for the size of the images produced at 64
ImageSize = 64

num_workers=0

torch.set_num_threads(4)
#Loads the Dog Dataset using BatchSize
TrainLoad = torch.utils.data.DataLoader(TrainingData,batch_size=BatchSize,shuffle=True,num_workers=num_workers,pin_memory=True)
imgs, label = next(iter(TrainLoad))
num_workers=8
#Imgs.nump().transpose turns some of images into Numpy array which is an Array 
# with Height, Width and Channel, representing the image.
imgs = imgs.numpy().transpose(0, 2, 3, 1)
print(".........................")
print(".........................")
print("  Loading Training Data")
print(".........................")
print(".........................")
time.sleep(3)
print(".........................")
print(".........................")
print ("This is the Training data in Numpy Form")
print(".........................")
print(".........................")
time.sleep(1.5)
print(".........................")
print(".........................")
print(imgs)
print(".........................")
print(".........................")

#Define's what part of the device is running the program and training of Generator and Discriminator
#keeping it at Cpu for now, might switch to Gpu later if need be    `
# Define device to use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


real_batch = next(iter(TrainLoad))
#Prints a training image being used
print(".........................")
GenTrainingImage = input("Please input Y if you wish to generate an image from the Training Data: ")
print(".........................")
if GenTrainingImage == "Y" or GenTrainingImage =="y":
    # Plots some training images
    # Aka shows some of the Training Image's that are being used
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images Used By Generator")
    plt.imshow(np.transpose(torchvision.utils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()
    
#Creating the weights that will be used for the Generation of New Images
#Made it a Subroutine so I can pull it from anywhere in the code
#The variable of m is going to be used 
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
#Lines 114-120 originally came from [1]https://www.kaggle.com/code/jesucristo/gan-introduction Kaggle
# Number of channels in the outputted image, for color images this is 3
# Color images have multiple channels, typically one for each color channel, such as red, green, and blue
NC = 3
# Size of Z latent vector/ Size of the data at the start of Generator Input 
SD = 100
# Size of feature maps in generator/ 
SFM = 64
#Generating the noise to be used by the Generator
fixedNoise = torch.randn(25, 100, 1, 1, device=device)
print(fixedNoise)


print(".........................")
#Creating the Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        #Noise = Random Set of Data 
        #Here I am Creating the Convolution Layers
        #Convolution Layers extract features from the data, with the data being the Original noise
        #It does this by pooling layers to reduce/increase the size of the data and fully connected the layers 
        #This allows the GAN to make predictions, on what the data should look like, 
        #allowing us to fill in the gaps to create a new image
        #This + the Weights allows us to get a dog based of the data used
        #However due to using prediction, the dog can be seen as odd or incomplete
        #Tweaking this prediction over generations using the weights + discriminator is how we get good new image's produced 
        self.main = nn.Sequential(
            #First Convultion Layer
            nn.ConvTranspose2d(SD, SFM * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(SFM * 8),
            nn.ReLU(True),
            #Second Convultion Layer = 4 x 4`
            nn.ConvTranspose2d(SFM * 8, SFM * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(SFM * 4),
            nn.ReLU(True),
            #Third Conultion Layer = 8 x 8
            nn.ConvTranspose2d(SFM * 4, SFM * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(SFM * 2),
            nn.ReLU(True),
            #Fourth Convultion Layer = 16 x 16
            nn.ConvTranspose2d(SFM * 2, SFM, 4, 2, 1, bias=False),
            nn.BatchNorm2d(SFM),
            nn.ReLU(True),
            #Final Convultion Layer 32 x 32
            nn.ConvTranspose2d(SFM, NC, 4, 2, 1, bias=False),
            nn.Tanh()
            #Results in Image Size of 64 x 64
        )
#Lines 148-169 originally 
#from [3]https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html?highlight=dcgan Pytorch

    def forward(self, input):
        output = self.main(input)
        return output

#Applying the weights to the Generator
netGen = Generator()
netGen.apply(weights_init)
print("The Generator with weights applied looks like:")
print(netGen)

print(".........................")
#Defining the discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
               #LeakyReLU used in the discriminator to introduce a small negative slope, 
               #helping prevent the vanishing gradient problem during training. 
                nn.Conv2d(3, 64, 4, stride=2, padding=1, bias=False),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(512, 1, 4, stride=1, padding=0, bias=False),
                #Sigmoid activation used produce a probability score
                nn.Sigmoid())
#Lines 159-172 originally from [1]https://www.kaggle.com/code/jesucristo/gan-introduction Kaggle
    def forward(self, input):
        output = self.main(input)
        #The output.view(-1) ouputs the image as a 1d iamge,by flattening it from a 2d image
        return output.view(-1)

#Applying the weights to the Discriminator
netDis = Discriminator()
netDis.apply(weights_init)
print("The Discriminator with weights applied looks like:")
print(netDis)

# Assigns labels during training for the real/dataset and fake/generator images
RealLabel = 1
FakeLabel = 0

# Create the directory for saving images
def make_directory_for_run():
    # Ensure the directory exists
    if not os.path.exists('./runs'):
        os.mkdir('./runs')
    if not os.path.exists('./runs/Images'):
        os.mkdir('./runs/Images')
    if not os.path.exists('./runs/GridImages'):
        os.mkdir('./runs/GridImages')
        
def save_generated_images(epoch, Images, output_dir='./runs/Images'):
    make_directory_for_run()
    # Save each image in the batch as a PNG file 
    for i, img in enumerate(Images):
        if isinstance(img, torch.Tensor):
            img = (img + 1) / 2 # Rescale image from [-1, 1] to [0, 1]
            save_image(img, os.path.join(output_dir, f"epoch_{epoch}_img_{i+1}.png"))
    # Saves giant images in a grid 
    img = (img + 1) / 2 # Rescale image from [-1, 1] to [0, 1]
    save_image(img, os.path.join(output_dir, f"epoch_{epoch}_img_{i+1}.png"))
        
    

# Function to save checkpoints
def save_checkpoint(epoch, Generator, Discriminator, OptimizerGen, OptimizerDis, filename):
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': netGen.state_dict(),
        'discriminator_state_dict': netDis.state_dict(),
        'optimizerG_state_dict': OptimizerGen.state_dict(),
        'optimizerD_state_dict': OptimizerDis.state_dict()}
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at epoch {epoch}")
    
# Function to load checkpoints
def load_checkpoint(filename, Generator, Discriminator, OptimizerGen, OptimizerDis):
    checkpoint = torch.load(filename)
    epoch = checkpoint['epoch']
    Generator.load_state_dict(checkpoint['generator_state_dict'])
    Discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    OptimizerGen.load_state_dict(checkpoint['optimizerG_state_dict'])
    OptimizerDis.load_state_dict(checkpoint['optimizerD_state_dict'])
    print(f"Checkpoint loaded. Resuming training from epoch {epoch}")
    return epoch

lr=0.0002
beta1 = 0.5
#Makes the Discriminator and Generator run better/more optimised
OptimizerDis = optim.Adam(netDis.parameters(), lr=lr, betas=(beta1, 0.999))
OptimizerGen = optim.Adam(netGen.parameters(), lr=lr, betas=(beta1, 0.999))

Criteria = nn.BCELoss()

#Epoch = What loop we are on
#Generator Loss measures how well the Generator is creating data
#Low Generator Loss = successful and realistic data and vice versa
#Discriminator Loss measures how well the Discriminator,
#can distinguish the real data from generated data
#Low Discriminator Loss indicates that Discriminator
#is effectively identifying fake data/images from the real data/images and vice versa
print(".........................")
print(" Starting Training Loop")
print(".........................")
time.sleep(2)
GenLoss = []
DisLoss = []
Images = []
retain_graph=True
x=0
netGen = netGen.to(device)
netDis = netDis.to(device)
iterations = len(TrainingData) // BatchSize
print(iterations)
#Change start epoc to the epoc you want to start the training at
start_epoch = 0
LoadCheck = input("Put Y if you want to resume training or don't if you want to do a new training loop: ")
# Load checkpoint if it exists and if we want to

checkpoint_file = f'./runs/checkpoint_start_epoch_{start_epoch}.pth'
filename = f'./runs/checkpoint_start_epoch_{start_epoch}.pth'
    
if os.path.exists(checkpoint_file) and LoadCheck == "Y" or LoadCheck == "y":
    print(f"Checkpoint found! Loading checkpoint {checkpoint_file}...")
    start_epoch = load_checkpoint(checkpoint_file, netGen, netDis, OptimizerGen, OptimizerDis)
    start_epoch += 1  # We increment by 1 because we want to start from the next epoch if we resume training
else:
    print("No checkpoint found, starting training from scratch.")

#Use TrainLen to list how long the training loop should go on for/Just the Number of Epochs
TrainLen = start_epoch + 1
    
    
def generate_image(netGen, Epoch=0, Batch=0, Images=None, device=device, Criteria=Criteria,iterations=iterations,start_epoch=start_epoch,filename=filename):
    if Images is None:
        Images = []

    for epoch in range(start_epoch,TrainLen):
        print("We are now on epoch",epoch)
        for i, data in enumerate(TrainLoad, 0):
            if i >= iterations:
                break
            if i % 25 == 0:
                print("We are now on loop",i)
            ############################
            # Update Discriminator
            ############################
            #Train the Discriminator with the real data
            # Format batch of real data

            netDis.zero_grad()
            real_cpu = data[0].to(device, memory_format=torch.contiguous_format)
            b_size = real_cpu.size(0)

            real_labels = torch.full((b_size,), RealLabel, dtype=torch.float, device=device)
            fake_labels = torch.full((b_size,), FakeLabel, dtype=torch.float, device=device)
            
            # Forward pass through D
            output_real = netDis(real_cpu).view(-1)
            # Calculate loss on real batch of data
            errD_real = Criteria(output_real, real_labels)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output_real.mean().item()

            noise = torch.randn(64, SD, 1, 1, device=device)
            
            #Train the Discriminator with the fake data from the Generator
            fake = netGen(noise)
            output_fake = netDis(fake.detach()).view(-1)
            errD_fake = Criteria(output_fake, fake_labels)
            errD_fake.backward()
            
            # Backpropagating the total error 
            # (total amounts of correct and false guesses by the Discrimintor)
            errD = errD_real + errD_fake
            OptimizerDis.step()
            ############################
            # Update Generator
            ############################
            #Updating the Weights of the Neural Network of the Generator
            #Zeros the Gradient of the Generator
            netGen.zero_grad()
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netDis(fake).view(-1)
            # Calculate new Gradient for Generator
            errG = Criteria(output, real_labels)  
            errG.backward()
            # Update the Generator
            OptimizerGen.step()
            # Define the ReduceLROnPlateau scheduler for both the Generator and Discriminator
            schedulerGen = torch.optim.lr_scheduler.ReduceLROnPlateau(OptimizerGen, 'min', patience=5, factor=0.5)
            schedulerDis = torch.optim.lr_scheduler.ReduceLROnPlateau(OptimizerDis, 'min', patience=5, factor=0.5)

            # Save Losses for plotting later
            GenLoss.append(errG.item())
            DisLoss.append(errD.item())
            
            #Lines 330-376 originally from [2]https://www.kaggle.com/code/abhranta/dcgan Kaggle
            # Output training stats
            if i % 100 == 0:
                print(f"[{epoch}/{TrainLen}][{i}/{len(TrainLoad)}]\tLoss_D: {errD.item():.4f}\tLoss_G: {errG.item():.4f}\tD(x): {D_x:.4f}")
            if i % 300 == 0:
                print("")
                plt.figure(figsize=(10,5))
                plt.title("Generator and Discriminator Loss During Training")
                plt.plot(GenLoss,label="G")
                plt.plot(DisLoss,label="D")
                plt.xlabel("Iterations")
                plt.ylabel("Loss")
                plt.legend()
                plt.show()
                print("")
            i+=1
        # Save and plot images at the end of each epoch
        with torch.no_grad():
            noise = torch.randn(64, SD, 1, 1, device=device)
            fake = netGen(noise).detach().cpu()
            grid = vutils.make_grid(fake, padding=2, normalize=True)
            Images.append(grid)
            gen_z = torch.randn(BatchSize, SD, 1, 1, device=device)
            gen_images = netGen(gen_z).to("cpu").clone().detach()
            Images.append(gen_images)
            save_generated_images(epoch, Images, output_dir='./runs/GridImages')
            fake = netGen(noise).detach().cpu()
            save_generated_images(epoch, fake,output_dir='./runs/Images')
        schedulerGen.step(errG)  
        schedulerDis.step(errD_real + errD_fake)
        if epoch % 2 == 0:
            # Save the model checkpoint
            torch.save(netGen.state_dict(), 'gen_checkpoint.pth')
            torch.save(netDis.state_dict(), 'dis_checkpoint.pth')
            save_checkpoint(epoch, Generator, Discriminator, OptimizerGen, OptimizerDis, filename)
            print("Saved epoch ",epoch)
    return netGen, Images
        
netGen,Images = generate_image(netGen,Images)
print(".........................")
print(".........................")
print(" Finishing Training Loop ")
print(".........................")
print(".........................")
print("")
end = time.time()
print("Type:", type(Images[0][0]))
print("Shape:", Images[0][0].shape)
print(f"Training took {(end - start)/60:.2f} minutes.")
print("")
print(".........................")
print(".........................")
print(" Plotting Data Collected ")
print(".........................")
print(".........................")

time.sleep(2)
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss Overall")
plt.plot(GenLoss,label="G")
plt.plot(DisLoss,label="D")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

x=BatchSize
RealBatch = next(iter(TrainLoad))
#Plottings Grid of Images produced
plt.figure(figsize=(5,5))
plt.axis("off")
plt.title("Fake Images Created by Generator")

# Get the first entry in the Images list
first_image_set = Images[0]

# Check if it's a grid (3D) or a batch (4D) images
if first_image_set.dim() == 3:
    img = (first_image_set + 1) / 2  # Rescale to [0, 1]
    img = img.clamp(0, 1)
    plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
elif first_image_set.dim() == 4:
    img = (first_image_set[0] + 1) / 2  # Take first image in batch
    img = img.clamp(0, 1)
    plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
else:
    print("Unexpected image shape:", first_image_set.shape)
plt.show()
