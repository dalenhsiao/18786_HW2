import matplotlib.pyplot as plt

# Train log
class Train_Log:

    def __init__(self):

         # loss across epochs
        self.ovrLoss = [] 
        self.ovrValLoss = []

        # accuracy across all epochs
        self.ovrAcc = [] 
        self.ovrValAcc = []

        # loss for single batches
        self.batchLoss = [] 

    def log_update_train(self, loss):
        self.batchLoss.append(loss)

    def log_update(self, trainLoss, validLoss, trainAcc, validAcc):
            
        self.ovrLoss.append(trainLoss)
        self.ovrValLoss.append(validLoss)
        self.ovrAcc.append(trainAcc)
        self.ovrValAcc.append(validAcc)

def Paras_Log(paras):
    
    print("\n------------ Model Summary ------------")
    
    if paras.device.type == 'cuda':
        print("\nRunning on device:", torch.cuda.get_device_name(0))
        print(' '*18, 'Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print(' '*18, 'Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    else:
        print("\nRunning on device:", paras.device)

    print("\nHyperparameters:")
    print(' '*6, "num epochs:", paras.maxEpoch)
    print(' '*6, "batch size:", paras.batchSize)
    print(' '*6, "learn rate:", paras.lr)
    print(' '*6, "early stop:", paras.earlyStop)

    print("\n---------------------------------------\n")

# plot loss 
def plotLoss(logger, epoch, bestEpoch):

    fig, ax = plt.subplots(3, 1, figsize=(12, 8))

    # Training Loss across all batches
    ax[0].plot(logger.batchLoss, label= "Training Loss")
    ax[0].set_title("Batch Loss")
    ax[0].set_xlabel("Iterations")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    
    # Training loss across epochs
    ax[1].plot(logger.ovrLoss, label = "Training Loss")
    ax[1].plot(logger.ovrValLoss, label = "Validation Loss")
    
    ax[1].axvline(x = bestEpoch , color="#F75D59", linewidth=0.8, linestyle="dashed", label="Early stop")
    ax[1].set_title("Training & Validation Loss")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")
    ax[1].legend()

    # accuracy across batches
    ax[2].plot(logger.ovrAcc, label = "Training Accuracy")
    ax[2].plot(logger.ovrValAcc, label = "Validation Accuracy")
    ax[2].axvline(x = bestEpoch , color="#F75D59", linewidth=0.8, linestyle="dashed", label="Early stop")
    ax[2].set_title("Training & Validation Accuracy")
    ax[2].set_xlabel("Epochs")
    ax[2].set_ylabel("Accuracy")
    ax[2].legend()

    plt.tight_layout()  # Adjust spacing between subplots
    plt.savefig("Q3.png")