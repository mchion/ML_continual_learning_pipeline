import matplotlib.pyplot as plt
import numpy as np 

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models

from torch.utils.data.dataset import random_split
from torch.utils.data import WeightedRandomSampler

import torch.nn as nn
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim as optim # optimzer

from PIL import Image
import shutil
import time
import copy
from mlflow import logging



def preprocessing(batch_size, num_workers):



	#Calculate Mean and Std of dataset
	def batch_mean_and_sd(loader):
		cnt = 0
		fst_moment = torch.empty(3)
		snd_moment = torch.empty(3)

		for images, _ in loader:
			b, c, h, w = images.shape
			nb_pixels = b * h * w
			sum_ = torch.sum(images, dim=[0, 2, 3])
			sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
			fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
			snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
			cnt += nb_pixels

		mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)        
		return mean,std

	transform_tmp = transforms.Compose([transforms.Resize((256,256)), 
									transforms.ToTensor()])

	# load all data
	root = 'Continents'
	tmp_dataset = torchvision.datasets.ImageFolder(root=root, transform=transform_tmp)
	tmp_dataloader = torch.utils.data.DataLoader(tmp_dataset, batch_size=8,
											num_workers=0)
 
 
	mean, std = batch_mean_and_sd(tmp_dataloader)
 
	#Use mean and std from above
	transform = transforms.Compose([transforms.Resize((256,256)), 
									transforms.ToTensor(), 
									transforms.Normalize(mean, std)])

 
	# load train data
	dataset = torchvision.datasets.ImageFolder(root='./Continents', transform=transform)

	train_split = .8

	dataset_size = dataset.__len__()
	train_count = int(dataset_size * train_split)
	val_count = dataset_size - train_count
	trainset, valset = random_split(dataset, [train_count, val_count])
 
	y_train_indices = trainset.indices
	y_train = [dataset.targets[i] for i in y_train_indices]
	class_sample_count = np.array(
		[len(np.where(y_train == t)[0]) for t in np.unique(y_train)])


	weight = 1.0 / class_sample_count
	samples_weight = np.array([weight[t] for t in y_train])
	samples_weight = torch.from_numpy(samples_weight)
 
 	sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
 
 
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          sampler=sampler, num_workers=num_workers)
	valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers)

	logging.info('preprocessed the data')

	return[trainloader, valloader]


# def load_current_model(model_path, file_m):

# 	model = load_model(os.getcwd()+model_path + str(file_m))
# 	model.compile(loss=keras.losses.categorical_crossentropy,
# 				  optimizer=keras.optimizers.Adadelta(),
# 				  metrics=['accuracy'])
# 	return model


def update_model(**kwargs):

	ti = kwargs['ti']
	loaded = ti.xcom_pull(task_ids='preprocessing')

	logging.info('variables successfully fetched from previous task')

	trainloader = loaded[0]
	valloader = loaded[1]

	# load new samples
	x_new = new_samples[0]
	y_new = new_samples[1]

	y_new = keras.utils.to_categorical(y_new, kwargs['num_classes'])

	# load test set
	x_test = test_set[0]
	y_test = test_set[1]

	y_test = keras.utils.to_categorical(y_test, kwargs['num_classes'])

	# get current_model
	for file_m in os.listdir(os.getcwd()+kwargs['path_current_model']):
		if 'H5' in file_m:

			mlflow.set_tracking_uri('http://mlflow:5000')

			with mlflow.start_run():

				model = load_current_model(kwargs['path_current_model'], file_m)

				# get score of current model
				current_score = model.evaluate(x_test, y_test, verbose=0)

				# update model with new data and evaluate score
				model.fit(x_new, y_new,
						  batch_size=kwargs['batch_size'],
						  epochs=kwargs['epochs'],
						  verbose=1,
						  validation_data=(x_test, y_test))

				updated_score = model.evaluate(x_test, y_test, verbose=0)

				# log results to MLFlow
				mlflow.log_metric('Epochs', kwargs['epochs'])
				mlflow.log_metric('Batch size', kwargs['batch_size'])

				mlflow.log_metric('test accuracy - current model', current_score[1])
				mlflow.log_metric('test accuracy - updated model', updated_score[1])

				mlflow.log_metric('loss - current model', current_score[0])
				mlflow.log_metric('loss - updated model', updated_score[0])

				mlflow.log_metric('Number of new samples used for training', x_new.shape[0])

				# if the updated model outperforms the current model -> move current version to archive and promote the updated model
				if updated_score[1] - current_score[1] > 0:

					logging.info('Updated model stored')
					mlflow.set_tag('status', 'the model from this run replaced the current version ')

					updated_model_name = 'model_' + str(time.strftime("%Y%m%d_%H%M"))

					model.save(os.getcwd()+kwargs['path_current_model'] + updated_model_name + '.H5')

					os.rename(os.getcwd()+kwargs['path_current_model']+file_m, os.getcwd()+kwargs['path_model_archive']+file_m)

				else:
					logging.info('Current model maintained')
					mlflow.set_tag('status', 'the model from this run did not replace the current version ')

		else:

			logging.info(file_m + ' is not a model')


def data_to_archive(**kwargs):

	# store data that was used for updating the model in archive along date + time tag

	for file_d in os.listdir(os.getcwd()+kwargs['path_new_data']):
		if 'new_samples.p' in file_d:

			os.rename(os.getcwd()+kwargs['path_new_data'] + file_d, os.getcwd()+kwargs['path_used_data'] + file_d)

			logging.info('data used for updating the model has been moved to archive')

		else:
			print('no data found')