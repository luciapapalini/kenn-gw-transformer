
"""This file contains the Trainer class which is used to handle the
    training procedure of HYPERION
"""

import os
import torch
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from gwskysim.gwskysim.utilities.gwlogger import GWLogger

from tensordict import TensorDict

from torchmetrics.classification import MulticlassAccuracy


class Trainer:
    def __init__(self,
                model: torch.nn.Module,
                training_dataset,
                validation_dataset,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler,
                device: str,
                checkpoint_filepath: str,
                verbose = True,
                add_noise = True,
                config = None,
                ):
        
        self.device       = device
        self.model        = model.to(device)
        self.train_dataset = training_dataset
        self.val_dataset   = validation_dataset
        self.optimizer    = optimizer
        self.scheduler    = scheduler
        self.add_noise    = add_noise
        self.config       = config
        
        self.metric = {'BBH' : MulticlassAccuracy(num_classes=config.max_num_bbh).to(device),
                       'NSBH': MulticlassAccuracy(num_classes=config.max_num_nsbh).to(device),
                       'BNS' : MulticlassAccuracy(num_classes=config.max_num_bns).to(device)
                       }
        
        self.checkpoint_filepath = checkpoint_filepath
        self.checkpoint_dir      = os.path.dirname(checkpoint_filepath)
     
        self.verbose = verbose
        return

    def _train_on_epoch(self, epoch):
        """
        Training over one epoch where an epoch is defined as when
        the model has been optimized on a number of batches equal
        to the specified training steps
        """

        avg_train_loss = 0
        fail_counter = 0
        
        avg_train_acc = {kind: 0 for kind in self.metric.keys()}
        
        
        step = 1
        total_steps = len(self.train_loader)
        
        #main loop over the epoch's batches
        for hi, n_signals, _ in self.train_loader:
            #getting the trainig batch
            #convert single signals to TensorDict and send to the gpu
            h_i = TensorDict.from_dict(hi).to(self.device)

            #whitener returns the single whitened signals, the sum of the whitened signals and their SNR
            hi_w, hw_sum, SNR = self.whitener(h_i, add_noise=self.add_noise)
            x = [hw_sum[det] for det in hw_sum.keys()]
            x = torch.stack(x, dim = 1)
            
            #plt.figure()
            #plt.plot(x[0][0].cpu().numpy())
            #plt.savefig('test.jpg')
            #plt.close()
            
            
            #training step
            self.optimizer.zero_grad()
            
            '''
            #compute model predictions
            predictions = self.model(x)
            
            accuracy = {}
            
            #compute the loss
            
            total_loss = 0
            for kind in n_signals:
                #print(kind)
                if kind == 'BBH':
                    loss = self.loss(predictions[kind], n_signals[kind].to(self.device))
                    total_loss += loss
                    #print(predictions[kind].detach(), n_signals[kind])
                    accuracy[kind] = self.metric[kind](predictions[kind], n_signals[kind].to(self.device))
                    
                    avg_train_acc[kind] += accuracy[kind].item()
            '''
            total_loss = -self.model.log_prob(n_signals.to(self.device), strain = x.to(self.device)).mean()
            
            #print(total_loss)
            
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                #do not update model's weights
                fail_counter += 1
            else:
                if self.verbose:
                    #print(f'Epoch = {epoch} |  Step = {step} / {total_steps}  |  Loss = {total_loss.item():.3f} |  BNSacc = {accuracy["BNS"].item():.3f}   |   NSBHacc = {accuracy["NSBH"].item():.3f}   |  BBHacc = {accuracy["BBH"].item():.3f}', end='\r')
                    print(f'Epoch = {epoch} |  Step = {step} / {total_steps}  |  Loss = {total_loss.item():.3f}', end='\r')

                #updating weights
                total_loss.backward()
                avg_train_loss += total_loss.item() #item() returns loss as a number instead of a tensor
            
            #perform the single step gradient descend
            self.optimizer.step()
            #if step > 3:
            #    break
            
            step+=1
            

        if fail_counter >= 0.5*total_steps:
            #something went wrong too many times during the epoch
            #better to leave the model as it is
            return np.nan
       
        #compute the mean loss
        avg_train_loss /= (total_steps-fail_counter)
        avg_train_acc = {kind: avg_train_acc[kind]/(total_steps-fail_counter) for kind in avg_train_acc.keys()}
    
        return avg_train_loss#, avg_train_acc
   

    def _test_on_epoch(self, epoch):
        """
        Validation over one epoch where an epoch is defined as when
        the model has been optimized on a number of batches equal
        to the specified training steps
        """

        avg_val_loss = 0
        fail_counter = 0
        step=0
        total_steps = len(self.val_loader)
        
        avg_val_acc = {kind: 0 for kind in self.metric.keys()}
        
        for  hi, n_signals, _ in self.val_loader:
            #getting the trainig batch
            #convert single signals to TensorDict and send to the gpu
            h_i = TensorDict.from_dict(hi).to(self.device)

            #whitener returns the single whitened signals, the sum of the whitened signals and their SNR
            hi_w, hw_sum, SNR = self.whitener(h_i, add_noise=self.add_noise)
            x = [hw_sum[det] for det in hw_sum.keys()]
            x = torch.stack(x, dim = 1)
            
            
            
            #training step
            self.optimizer.zero_grad()
            '''
            #compute model predictions
            predictions = self.model(x)
            
            accuracy = {}
            
            #compute the loss
            total_loss = 0
            for kind in n_signals:
                if kind=='BBH':
                    loss = self.loss(predictions[kind], n_signals[kind].to(self.device))
                    total_loss += loss
                    accuracy[kind] = self.metric[kind](predictions[kind], n_signals[kind].to(self.device))
                    avg_val_acc[kind] += accuracy[kind].item()
            '''
            total_loss = -self.model.log_prob(n_signals.to(self.device), strain = x.to(self.device)).mean()
            
            
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                fail_counter += 1
                #print(f'Epoch = {epoch}  |  Validation Step = {step+1} / {total_steps}  |  Loss = {total_loss.item():.3f} |  BNSacc = {accuracy["BNS"].item():.3f}   |   NSBHacc = {accuracy["NSBH"].item():.3f}   |  BBHacc = {accuracy["BBH"].item():.3f}', end='\r')
                print(f'Epoch = {epoch}  |  Validation Step = {step+1} / {total_steps}  |  Loss = {total_loss.item():.3f}', end='\r')
            
            else:
                avg_val_loss += total_loss.item() #item() returns loss as a number instead of a tensor
                if self.verbose:
                    
                    #print(f'Epoch = {epoch}  |  Validation Step = {step} / {total_steps}  |  Loss = {total_loss.item():.3f} |  BNSacc = {accuracy["BNS"].item():.3f}   |   NSBHacc = {accuracy["NSBH"].item():.3f}   |  BBHacc = {accuracy["BBH"].item():.3f}', end='\r')
                    print(f'Epoch = {epoch}  |  Validation Step = {step} / {total_steps}  |  Loss = {total_loss.item():.3f}', end='\r')
            #if step> 3:
            #    break
            step+=1
           
        if fail_counter >= 0.5*total_steps:
            return np.nan
        
        avg_val_loss /= (total_steps-fail_counter)
        avg_val_acc = {kind: avg_val_acc[kind]/(total_steps-fail_counter) for kind in avg_val_acc.keys()}
        return avg_val_loss#, avg_val_acc
  
    
    def _save_checkpoint(self, epoch):
        
        checkpoints = {
            #'configuration': self.model.configuration,
            #'prior_metadata': self.model.prior_metadata,
            'model_state_dict': self.model.state_dict(),
            #'optimizer_state_dict': self.optimizer.state_dict(),
            #'epoch': epoch,
        }
            
        torch.save(checkpoints, self.checkpoint_filepath)
        if self.verbose:
            self.log.info(f"Training checkpoint saved at {self.checkpoint_filepath}")
    
    def _make_history_plots(self):
        train_loss, val_loss, lr = np.loadtxt(self.checkpoint_dir+'/history.txt', delimiter=',',unpack = True)
        epochs = np.arange(len(train_loss))+1

        plt.figure(figsize=(20, 8))
        #history
        plt.subplot(121)
        plt.plot(epochs, train_loss, label ='train loss')
        plt.plot(epochs, val_loss, label = 'val loss')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        ymin = min(np.min(train_loss), np.min(val_loss))-0.5
        ymax = max(np.max(train_loss), np.max(val_loss)) #train_loss[0]+0.5
        plt.ylim(ymin, ymax)
        
        #learning_rate
        plt.subplot(122)
        plt.plot(epochs, lr, label ='learning rate')
        #plt.plot(epochs, val_acc_BNS, label = 'val acc BNS')
        plt.legend()
        plt.xlabel('epoch')
        #plt.ylabel('$\eta$')
        '''
        plt.subplot(223)
        plt.plot(epochs, train_acc_NSBH, label ='train acc NSBH')
        plt.plot(epochs, val_acc_NSBH, label = 'val acc NSBH')
        plt.legend()
        plt.xlabel('epoch')
        
        plt.subplot(224)
        plt.plot(epochs, train_acc_BBH, label ='train acc BBH')
        plt.plot(epochs, val_acc_BBH, label = 'val acc BBH')
        plt.legend()
        plt.xlabel('epoch')
        
        '''
        
        plt.savefig(self.checkpoint_dir+'/history_plot.jpg', dpi=200, bbox_inches='tight')
        plt.close()
    
	
    def train(self, num_epochs, overwrite_history=True):
        self.log = GWLogger('training_logger')
        self.log.setLevel('INFO')

        best_train_loss = np.inf
        best_val_loss   = np.inf
        
        best_train_acc = {'BNS': 0, 'NSBH': 0, 'BBH': 0}
        best_val_acc = {'BNS': 0, 'NSBH': 0, 'BBH': 0}
        
        train_accuracies = {'BNS': [], 'NSBH': [], 'BBH': []}
        val_accuracies = {'BNS': [], 'NSBH': [], 'BBH': []}
        

        self.history_fpath = os.path.join(self.checkpoint_dir, 'history.txt')
        
        if not overwrite_history:
            f = open(self.history_fpath, 'a')
        else:
            f = open(self.history_fpath, 'w')
            f.write('#training loss, validation loss, learning rate\n')
            f.flush()
        
        print('\n')
        self.log.info('Starting Training...\n')
        
        #main training loop over the epochs
        for epoch in tqdm(range(1,num_epochs+1)):

            #on-epoch training
            self.model.train(True) #train attribute comes from nn.Module and is used to set the weights in training mode
            train_loss = self._train_on_epoch(epoch)
            '''
            for kind in train_acc:
                train_accuracies[kind].append(train_acc[kind])
                if train_acc[kind] > best_train_acc[kind]:
                    best_train_acc[kind] = train_acc[kind]
            '''
            
            
            
            #on-epoch validation
            self.model.eval()      #eval attribute comes from nn.Module and is used to set the weights in evaluation mode
            with torch.inference_mode():
                val_loss = self._test_on_epoch(epoch)
            '''
            for kind in train_acc:
                val_accuracies[kind].append(val_acc[kind])
                if val_acc[kind] > best_val_acc[kind]:
                    best_val_acc[kind] = val_acc[kind]
            '''
            
            if np.isnan(train_loss) or np.isnan(val_loss):
                self.log.error(f'Epoch {epoch} skipped due to nan loss\n')
                continue #we skip to next iteration
        
            self.log.info(f'Epoch = {epoch}  |  avg train loss = {train_loss:.3f}  |  avg val loss = {val_loss:.3f}')
           
            #save updated model weights and update best values
            if (train_loss < best_train_loss) and (val_loss < best_val_loss):
                self._save_checkpoint(epoch+1)
                best_train_loss = train_loss
                best_val_loss   = val_loss
                print(f"best train loss = {best_train_loss:.3f} at epoch {epoch}")
                print(f"best val   loss = {best_val_loss:.3f} at epoch {epoch}\n")
           
            
            #get current learning rate
            if epoch > 1:
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.optimizer.param_groups[0]["lr"]
            
            #write history to file
            #string = f"{train_loss},{val_loss},{current_lr},{train_acc['BNS']},{train_acc['NSBH']},{train_acc['BBH']},{val_acc['BNS']},{val_acc['NSBH']},{val_acc['BBH']}\n"
            string = f"{train_loss},{val_loss},{current_lr}\n"
            f.write(string)
            f.flush()
            
            #make history plot
            try:
                self._make_history_plots()
            except Exception as e:
                #print(e)
                pass
            
        
            #perform learning rate schedule step
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss) #for ReduceLrOnPlateau
                updated_lr = self.scheduler.get_last_lr()[0]
                if updated_lr < current_lr:
                    self.log.info(f"Reduced learning rate to {updated_lr}")
            else:
                self.scheduler.step() #for CosineAnnealingLr
                
            

        f.close()
        self.log.info('Training Completed!\n')
         


    
    
    
