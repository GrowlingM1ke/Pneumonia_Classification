from dataset import CNNDataset
from models import CNNBasic
import config
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import cv2
from torchvision import transforms
import numpy as np

if __name__ == '__main__': 
	# load the image and mask filepaths in a sorted manner
	train_normal_paths = list(paths.list_images(config.TRAIN_PATH_NORMAL))
	train_pneumonia_paths = list(paths.list_images(config.TRAIN_PATH_PNEUMONIA))
	test_normal_paths = list(paths.list_images(config.TEST_PATH_NORMAL))
	test_pneumonia_paths = list(paths.list_images(config.TEST_PATH_PNEUMONIA))
	val_normal_paths = list(paths.list_images(config.VALIDATION_PATH_NORMAL))
	val_pneumonia_paths = list(paths.list_images(config.VALIDATION_PATH_PNEUMONIA))
	# Define the transform
	transforms = transforms.Compose([transforms.ToPILImage(),
		transforms.Resize((config.INPUT_IMAGE_HEIGHT,
			config.INPUT_IMAGE_WIDTH)),
		transforms.ToTensor()])

	# create the train and test datasets
	trainDS = CNNDataset(normalPaths=train_normal_paths, pneumoniaPaths=train_pneumonia_paths, transforms=transforms, subsample=True)
	valDS = CNNDataset(normalPaths=val_normal_paths, pneumoniaPaths=val_pneumonia_paths, transforms=transforms)
	testDS = CNNDataset(normalPaths=test_normal_paths, pneumoniaPaths=test_pneumonia_paths, transforms=transforms)
	# create the training and test data loaders
	trainLoader = DataLoader(trainDS, shuffle=True,
		batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
		num_workers=12)
	testLoader = DataLoader(testDS, shuffle=False,
		batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
		num_workers=12)
	valLoader = DataLoader(valDS, shuffle=False,
		batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
		num_workers=12)

	# initialize our CNN model
	cnn = CNNBasic().to(config.DEVICE)
	# initialize loss function and optimizer

	lossFunc = BCEWithLogitsLoss()
	opt = Adam(cnn.parameters(), lr=config.INIT_LR)
	# calculate steps per epoch for training and validation set
	trainSteps = len(trainDS) // config.BATCH_SIZE
	valSteps = np.max([len(valDS) // config.BATCH_SIZE, 1])
	# initialize a dictionary to store training history
	H = {"train_loss": [], "val_loss": []}

	# loop over epochs
	print("[INFO] training the network...")
	startTime = time.time()
	for e in tqdm(range(config.NUM_EPOCHS)):
		# set the model in training mode
		cnn.train()
		# initialize the total training and validation loss
		totalTrainLoss = 0
		totalValLoss = 0
		# loop over the training set
		for (i, (x, y)) in enumerate(trainLoader):
			# send the input to the device
			(x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
			# perform a forward pass and calculate the training loss
			pred = cnn(x)
			y = y.unsqueeze(1)
			loss = lossFunc(pred, y)
			# first, zero out any previously accumulated gradients, then
			# perform backpropagation, and then update model parameters
			opt.zero_grad()
			loss.backward()
			opt.step()
			# add the loss to the total training loss so far
			totalTrainLoss += loss
		# switch off autograd
		with torch.no_grad():
			# set the model in evaluation mode
			cnn.eval()
			# loop over the validation set
			for (x, y) in valLoader:
				# send the input to the device
				(x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
				# make the predictions and calculate the validation loss
				pred = cnn(x)
				y = y.unsqueeze(1)
				totalValLoss += lossFunc(pred, y)
		# calculate the average training and validation loss
		avgTrainLoss = totalTrainLoss / trainSteps
		avgValLoss = totalValLoss / valSteps
		# update our training history
		H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
		H["val_loss"].append(avgValLoss.cpu().detach().numpy())
		# print the model training and validation information
		print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
		print("Train loss: {:.6f}, Val loss: {:.4f}".format(
			avgTrainLoss, avgValLoss))
	# display the total time needed to perform the training
	endTime = time.time()
	print("[INFO] total time taken to train the model: {:.2f}s".format(
		endTime - startTime))

	# plot the training loss
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(H["train_loss"], label="train_loss")
	plt.plot(H["test_loss"], label="test_loss")
	plt.title("Training Loss on Dataset")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss")
	plt.legend(loc="lower left")
	plt.savefig(config.PLOT_PATH)
	# serialize the model to disk
	torch.save(cnn, config.MODEL_PATH)