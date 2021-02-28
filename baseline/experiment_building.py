import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm
import pickle
import os
import numpy as np
import time
from matplotlib import pyplot as plt
import matplotlib
import pickle
import os
import csv


matplotlib.rcParams.update({'font.size': 8})


def save_to_stats_pkl_file(experiment_log_filepath, filename, stats_dict):
    summary_filename = os.path.join(experiment_log_filepath, filename)
    with open("{}.pkl".format(summary_filename), "wb") as file_writer:
        pickle.dump(stats_dict, file_writer)


def load_from_stats_pkl_file(experiment_log_filepath, filename):
    summary_filename = os.path.join(experiment_log_filepath, filename)
    with open("{}.pkl".format(summary_filename), "rb") as file_reader:
        stats = pickle.load(file_reader)

    return stats


def save_statistics(experiment_log_dir, filename, stats_dict, current_epoch, continue_from_mode=False, save_full_dict=False):
    """
    Saves the statistics in stats dict into a csv file. Using the keys as the header entries and the values as the
    columns of a particular header entry
    :param experiment_log_dir: the log folder dir filepath
    :param filename: the name of the csv file
    :param stats_dict: the stats dict containing the data to be saved
    :param current_epoch: the number of epochs since commencement of the current training session (i.e. if the experiment continued from 100 and this is epoch 105, then pass relative distance of 5.)
    :param save_full_dict: whether to save the full dict as is overriding any previous entries (might be useful if we want to overwrite a file)
    :return: The filepath to the summary file
    """
    summary_filename = os.path.join(experiment_log_dir, filename)
    mode = 'a' if continue_from_mode else 'w'
    with open(summary_filename, mode) as f:
        writer = csv.writer(f)
        if not continue_from_mode:
            writer.writerow(list(stats_dict.keys()))

        if save_full_dict:
            total_rows = len(list(stats_dict.values())[0])
            for idx in range(total_rows):
                row_to_add = [value[idx] for value in list(stats_dict.values())]
                writer.writerow(row_to_add)
        else:
            row_to_add = [value[current_epoch] for value in list(stats_dict.values())]
            writer.writerow(row_to_add)

    return summary_filename


def load_statistics(experiment_log_dir, filename):
    """
    Loads a statistics csv file into a dictionary
    :param experiment_log_dir: the log folder dir filepath
    :param filename: the name of the csv file to load
    :return: A dictionary containing the stats in the csv file. Header entries are converted into keys and columns of a
     particular header are converted into values of a key in a list format.
    """
    summary_filename = os.path.join(experiment_log_dir, filename)

    with open(summary_filename, 'r+') as f:
        lines = f.readlines()

    keys = lines[0].split(",")
    stats = {key: [] for key in keys}
    for line in lines[1:]:
        values = line.split(",")
        for idx, value in enumerate(values):
            stats[keys[idx]].append(value)

    return stats


class ExperimentBuilder(nn.Module):
    def __init__(self, network_model, experiment_name, num_epochs, train_data, val_data,
                 test_data, weight_decay_coefficient, use_gpu, continue_from_epoch=-1,learning_rate=0.001,annealing=True):
        """
        Initializes an ExperimentBuilder object. Such an object takes care of running training and evaluation of a deep net
        on a given dataset. It also takes care of saving per epoch models and automatically inferring the best val model
        to be used for evaluating the test set metrics.
        :param network_model: A pytorch nn.Module which implements a network architecture.
        :param experiment_name: The name of the experiment. This is used mainly for keeping track of the experiment and creating and directory structure that will be used to save logs, model parameters and other.
        :param num_epochs: Total number of epochs to run the experiment
        :param train_data: An object of the DataProvider type. Contains the training set.
        :param val_data: An object of the DataProvider type. Contains the val set.
        :param test_data: An object of the DataProvider type. Contains the test set.
        :param weight_decay_coefficient: A float indicating the weight decay to use with the adam optimizer.
        :param use_gpu: A boolean indicating whether to use a GPU or not.
        :param continue_from_epoch: An int indicating whether we'll start from scrach (-1) or whether we'll reload a previously saved model of epoch 'continue_from_epoch' and continue training from there.
        """
        super(ExperimentBuilder, self).__init__()



        self.experiment_name = experiment_name
        self.model = network_model

        if torch.cuda.device_count() > 1 and use_gpu:
            self.device = torch.cuda.current_device()
            self.model.to(self.device)
            self.model = nn.DataParallel(module=self.model)
            print('Use Multi GPU', self.device)
        elif torch.cuda.device_count() == 1 and use_gpu:
            self.device =  torch.cuda.current_device()
            self.model.to(self.device)  # sends the model from the cpu to the gpu
            print('Use GPU', self.device)
        else:
            print("use CPU")
            self.device = torch.device('cpu')  # sets the device to be CPU
            print(self.device)


        self.model.reset_parameters()  # re-initialize network parameters
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        print(f"Learning rate: {learning_rate}")
        self.optimizer = optim.Adam(self.parameters(), amsgrad=False,
                                    weight_decay=weight_decay_coefficient,lr=learning_rate)

        self.annealing = annealing
        print(self.annealing)
        if self.annealing==1:
            print("doing annealing")
            self.learning_rate_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                            T_max=num_epochs,
                                                                            eta_min=0.00002)
        # Generate the directory names
        self.experiment_folder = os.path.abspath(experiment_name)
        self.experiment_logs = os.path.abspath(os.path.join(self.experiment_folder, "result_outputs"))
        self.experiment_saved_models = os.path.abspath(os.path.join(self.experiment_folder, "saved_models"))

        # self.pickle_dir = os.path.abspath(os.path.join(self.experiment_folder, "grad_pickle_output"))

        # self.grads = []

        # Set best models to be at 0 since we are just starting
        self.best_val_model_idx = 0
        self.best_val_model_acc = 0.

        if not os.path.exists(self.experiment_folder):  # If experiment directory does not exist
            os.mkdir(self.experiment_folder)  # create the experiment directory
            os.mkdir(self.experiment_logs)  # create the experiment log directory
            os.mkdir(self.experiment_saved_models)  # create the experiment saved models directory

            # os.mkdir(self.pickle_dir) # make grad pickle output folder

        self.num_epochs = num_epochs
        self.criterion = nn.BCELoss().to(self.device)  # send the loss computation to the GPU

        if continue_from_epoch == -2:  # if continue from epoch is -2 then continue from latest saved model
            self.state, self.best_val_model_idx, self.best_val_model_acc = self.load_model(
                model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                model_idx='latest')  # reload existing model from epoch and return best val model index
            # and the best val acc of that model
            self.starting_epoch = int(self.state['model_epoch'])

        elif continue_from_epoch > -1:  # if continue from epoch is greater than -1 then
            self.state, self.best_val_model_idx, self.best_val_model_acc = self.load_model(
                model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                model_idx=continue_from_epoch)  # reload existing model from epoch and return best val model index
            # and the best val acc of that model
            self.starting_epoch = continue_from_epoch
        else:
            self.state = dict()
            self.starting_epoch = 0

    def get_num_parameters(self):
        total_num_params = 0
        for param in self.parameters():
            total_num_params += np.prod(param.shape)

        return total_num_params
        

    def run_train_iter(self, x, y, labels):
        """
        Receives the inputs and targets for the model and runs training iterations. Returns loss and accuracy metrics.
        :param x: Batch of left images of shape batch_size, channels, height, width
        :param y: Batch of right images of shape batch_size, ch
        :param labels: Batch of labels
        """
        self.train()  # sets model to training mode (in case batch normalization or other methods have different procedures for training and evaluation)
        x, y, labels = x.float().to(device=self.device), y.float().to(device=self.device), labels.float().to(
            device=self.device)  # send data to device as torch tensors
        out_xy = self.model.forward(x,y)  # forward the data in the model
        loss = self.criterion(input=torch.squeeze(out_xy), target=labels)  # compute loss
    
        self.optimizer.zero_grad()  # set all weight grads from previous training iters to 0
        loss.backward()  # backpropagate to compute gradients for current iter loss
        
        if self.annealing==1:
            self.learning_rate_scheduler.step(epoch=self.current_epoch)
        self.optimizer.step()  # update network parameters

        predicted = torch.squeeze(torch.round(out_xy))
        accuracy = np.mean(list(predicted.eq(labels.data).cpu()))  # compute accuracy
        return loss.cpu().data.numpy(), accuracy

    def run_evaluation_iter(self, x, y, labels):
        """
        Receives the inputs and targets for the model and runs an evaluation iterations. Returns loss and accuracy metrics.
        :param x: Batch of left images of shape batch_size, channels, height, width
        :param y: Batch of right images of shape batch_size, ch
        :param labels: Batch of labels
        :return: the loss and accuracy for this batch
        """
        self.eval()  # sets the system to validation mode
        x, y, labels = x.float().to(device=self.device), y.float().to(device=self.device), labels.float().to(
            device=self.device)  # convert data to pytorch tensors and send to the computation device
        out_xy = self.model.forward(x,y)  # forward the data in the model

        loss = self.criterion(input=torch.squeeze(out_xy), target=labels)  # compute loss
        predicted = torch.squeeze(torch.round(out_xy))
        accuracy = np.mean(list(predicted.eq(labels.data).cpu()))  # compute accuracy
        return loss.cpu().data.numpy(), accuracy

    def save_model(self, model_save_dir, model_save_name, model_idx, best_validation_model_idx,
                   best_validation_model_acc):
        """
        Save the network parameter state and current best val epoch idx and best val accuracy.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :param best_validation_model_idx: The index of the best validation model to be stored for future use.
        :param best_validation_model_acc: The best validation accuracy to be stored for use at test time.
        :param model_save_dir: The directory to store the state at.
        :param state: The dictionary containing the system state.

        """
        self.state['network'] = self.state_dict()  # save network parameter and other variables.
        self.state['best_val_model_idx'] = best_validation_model_idx  # save current best val idx
        self.state['best_val_model_acc'] = best_validation_model_acc  # save current best val acc
        torch.save(self.state, f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(
            model_idx))))  # save state at prespecified filepath

    def load_model(self, model_save_dir, model_save_name, model_idx):
        """
        Load the network parameter state and the best val model idx and best val acc to be compared with the future val accuracies, in order to choose the best val model
        :param model_save_dir: The directory to store the state at.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :return: best val idx and best val model acc, also it loads the network state into the system state without returning it
        """
        state = torch.load(f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(model_idx))))
        self.load_state_dict(state_dict=state['network'])
        return state, state['best_val_model_idx'], state['best_val_model_acc']

    def run_experiment(self):
        """
        Runs experiment train and evaluation iterations, saving the model and best val model and val model accuracy after each epoch
        :return: The summary current_epoch_losses from starting epoch to total_epochs.
        """
        total_losses = {"train_acc": [], "train_loss": [], "val_acc": [],
                        "val_loss": []}  # initialize a dict to keep the per-epoch metrics
        for i, epoch_idx in enumerate(range(self.starting_epoch, self.num_epochs)):
            epoch_start_time = time.time()
            current_epoch_losses = {"train_acc": [], "train_loss": [], "val_acc": [], "val_loss": []}
            self.current_epoch = epoch_idx
            with tqdm.tqdm(total=len(self.train_data)) as pbar_train:  # create a progress bar for training
                for idx, (x, y, labels) in enumerate(self.train_data):  # get data batches
                    loss, accuracy = self.run_train_iter(x,y,labels)  # take a training iter step
                    current_epoch_losses["train_loss"].append(loss)  # add current iter loss to the train loss list
                    current_epoch_losses["train_acc"].append(accuracy)  # add current iter acc to the train acc list
                    pbar_train.update(1)
                    pbar_train.set_description("loss: {:.4f}, accuracy: {:.4f}".format(loss, accuracy))

            with tqdm.tqdm(total=len(self.val_data)) as pbar_val:  # create a progress bar for validation
                for x, y, labels in self.val_data:  # get data batches
                    loss, accuracy = self.run_evaluation_iter(x,y,labels)  # run a validation iter
                    current_epoch_losses["val_loss"].append(loss)  # add current iter loss to val loss list.
                    current_epoch_losses["val_acc"].append(accuracy)  # add current iter acc to val acc lst.
                    pbar_val.update(1)  # add 1 step to the progress bar
                    pbar_val.set_description("loss: {:.4f}, accuracy: {:.4f}".format(loss, accuracy))
            val_mean_accuracy = np.mean(current_epoch_losses['val_acc'])
            if val_mean_accuracy > self.best_val_model_acc:  # if current epoch's mean val acc is greater than the saved best val acc then
                self.best_val_model_acc = val_mean_accuracy  # set the best val model acc to be current epoch's val accuracy
                self.best_val_model_idx = epoch_idx  # set the experiment-wise best val idx to be the current epoch's idx

            for key, value in current_epoch_losses.items():
                total_losses[key].append(np.mean(
                    value))  # get mean of all metrics of current epoch metrics dict, to get them ready for storage and output on the terminal.

            save_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv',
                            stats_dict=total_losses, current_epoch=i,
                            continue_from_mode=True if (self.starting_epoch != 0 or i > 0) else False)  # save statistics to stats file.

            # load_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv') # How to load a csv file if you need to

            out_string = "_".join(
                ["{}_{:.4f}".format(key, np.mean(value)) for key, value in current_epoch_losses.items()])
            # create a string to use to report our epoch metrics
            epoch_elapsed_time = time.time() - epoch_start_time  # calculate time taken for epoch
            epoch_elapsed_time = "{:.4f}".format(epoch_elapsed_time)
            print("Epoch {}:".format(epoch_idx), out_string, "epoch time", epoch_elapsed_time, "seconds")
            self.state['model_epoch'] = epoch_idx
            self.save_model(model_save_dir=self.experiment_saved_models,
                            # save model and best val idx and best val acc, using the model dir, model name and model idx
                            model_save_name="train_model", model_idx=epoch_idx,
                            best_validation_model_idx=self.best_val_model_idx,
                            best_validation_model_acc=self.best_val_model_acc)
            self.save_model(model_save_dir=self.experiment_saved_models,
                            # save model and best val idx and best val acc, using the model dir, model name and model idx
                            model_save_name="train_model", model_idx='latest',
                            best_validation_model_idx=self.best_val_model_idx,
                            best_validation_model_acc=self.best_val_model_acc)
            
            ################################################################
            ##### Plot Gradient Flow at each Epoch during Training  ######
            # print("Generating Gradient Flow Plot at epoch {}".format(epoch_idx))
            # plt = self.plot_grad_flow(self.model.named_parameters())
            # if not os.path.exists(os.path.join(self.experiment_saved_models, 'gradient_flow_plots')):
            #     os.mkdir(os.path.join(self.experiment_saved_models, 'gradient_flow_plots'))
            #     # plt.legend(loc="best")



            # plt.savefig(os.path.join(self.experiment_saved_models, 'gradient_flow_plots', "epoch{}.pdf".format(str(epoch_idx))))
            ################################################################
        
        print("Generating test set evaluation metrics")
        self.load_model(model_save_dir=self.experiment_saved_models, model_idx=self.best_val_model_idx,
                        # load best validation model
                        model_save_name="train_model")
        current_epoch_losses = {"test_acc": [], "test_loss": []}  # initialize a statistics dict
        with tqdm.tqdm(total=len(self.test_data)) as pbar_test:  # ini a progress bar
            for x, y,labels in self.test_data:  # sample batch
                loss, accuracy = self.run_evaluation_iter(x=x,
                                                          y=y,labels=labels)  # compute loss and accuracy by running an evaluation step
                current_epoch_losses["test_loss"].append(loss)  # save test loss
                current_epoch_losses["test_acc"].append(accuracy)  # save test accuracy
                pbar_test.update(1)  # update progress bar status
                pbar_test.set_description(
                    "loss: {:.4f}, accuracy: {:.4f}".format(loss, accuracy))  # update progress bar string output

        test_losses = {key: [np.mean(value)] for key, value in
                       current_epoch_losses.items()}  # save test set metrics in dict format
        save_statistics(experiment_log_dir=self.experiment_logs, filename='test_summary.csv',
                        # save test set metrics on disk in .csv format
                        stats_dict=test_losses, current_epoch=0, continue_from_mode=False)

        return total_losses, test_losses
