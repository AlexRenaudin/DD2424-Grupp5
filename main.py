import matplotlib.pyplot as plt
from Project import run_model
import numpy as np
import time

# Run models in model_list and store training/test loss

model_list = ['rnn', 'lstm', '2-lstm', 'bi-lstm', 'gru']
#model_list = ['rnn', 'lstm']

training_loss_list = []
test_loss_list = []
for model in model_list:
    start_time = time.time()
    t_loss, v_loss = run_model(model)
    training_loss_list.append(t_loss)
    test_loss_list.append(v_loss)
    print("--- %s seconds ---" % (time.time() - start_time))



#Plot and save training loss

for idx, training_loss in enumerate(training_loss_list):
    plt.plot(training_loss, label= model_list[idx] + ' Training loss')
    np.savetxt(model_list[idx] +'_train_with_conv.csv', training_loss, delimiter=',')
    
plt.title('Training losses of various models on Goblet Book dataset')
plt.xlabel('Sequence')
plt.ylabel('Loss')
plt.legend()
plt.figure(1)

plt.figure(2)

# Plot and save validation loss

for idx, test_loss in enumerate(test_loss_list):
    plt.plot(test_loss, label= model_list[idx] + ' Test loss')
    np.savetxt(model_list[idx] +'_test_with_conv.csv', test_loss, delimiter=',')
plt.title('Test losses of various models on Golbet Book dataset')
plt.xlabel('Sequence')
plt.ylabel('Loss')
plt.legend()


plt.show()





