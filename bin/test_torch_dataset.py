import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from optparse import OptionParser
from tensordict import TensorDict
from torch.utils.data import DataLoader
from deepfastet.dataset import DatasetGenerator, WhitenNet


if __name__=='__main__':

    #set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = OptionParser()
 
    parser.add_option("-b", "--batch_size", default=32, help="batch size (default 32)")
    parser.add_option("-n", "--noise", default=False, action="store_true", help="Inject noise in the data")
    parser.add_option("-f", "--figure", default=False, action="store_true", help="Plot the data")
    parser.add_option("-p", "--nproc", default=4, help="Number of parallel workers of the dataloader (default 4)")
    parser.add_option("-d", "--duration", default=32, help="Duration in s of signals (default 32)")
    (options, args) = parser.parse_args()
    
    batch_size = int(options.batch_size)
    nproc = int(options.nproc)
    noise = options.noise
    plot = options.figure
    duration = int(options.duration)
    if plot:
        nproc=0
        batch_size=1
        print('Setting nproc=0 to plot the data')

    print('[INFO]: Using device {}'.format(device))
    print('[INFO]: Batch size {}'.format(batch_size))
    print('[INFO]: Setting up Multiprocessing with {} workers'.format(nproc))
    
    dataset = DatasetGenerator(data_dir='../../gwdata-archive/ET_mdc/training_dataset', duration=duration, 
                               max_signals={'BNS' : 3, 
                                            'NSBH': 1, 
                                            'BBH' : 2})
    
    dataset.plot = plot
    dataset.add_noise = noise
    
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=False, 
                            pin_memory=True,
                            persistent_workers=False,
                            num_workers=nproc, 
                            drop_last=True)
    
    fs = dataset.fs
    whitener = WhitenNet(fs=dataset.fs, 
                         duration=dataset.duration, 
                         device=device)
    
   
    
    for hi, n, t in tqdm(dataloader):
        
        print(n)
        
        h_i = TensorDict.from_dict(hi).to(device)

        hi_w, hw_sum, SNR = whitener(h_i, add_noise=noise)
        #h_sum = TensorDict.from_dict(hs).to(device)

        #print(SNR)

        #print(hw_sum)
        if plot:
            times = torch.linspace(0, dataset.duration, dataset.duration*fs)
            plt.figure(figsize=(10, 5))
            for det in hw_sum.keys():
                h = hw_sum[det][0].cpu().numpy()
                plt.plot(times, h)
            plt.xlabel('Time [s]')
            title = 'Sum of whitened signals'
            title += ' with noise' if noise else ''
            plt.title(title)
            plt.show()

        if plot:
            for kind in hi_w.keys():
                iplot = 1
                plt.figure(figsize=(20, 15))
                for idet, det in enumerate(hi_w[kind].keys()):
                    h = hi_w[kind][det][0].cpu().numpy()
                    for i in range(len(h)):
                        plt.subplot(3, len(h), iplot)
                        #print(SNR[kind][0])
                        if idet == 0:
                            plt.title(f'{kind} {det} {SNR[kind][0][i].item():.1f}')
                        plt.plot(h[i])
                        iplot += 1
                plt.show()                      
                    #plt.plot(h)
                    #plt.show()
    print(hw_sum)

    print('ok')

    #dataset.close_all()
