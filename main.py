import matplotlib.pyplot as plt
from Project import run_model
import numpy as np

#model_list = ['rnn', 'lstm', '2-lstm', 'bi-lstm', 'gru']
model_list = ['rnn', 'lstm']
test_loss_list = []
val_loss_list = []
for model in model_list:
    t_loss, v_loss = run_model(model)
    test_loss_list.append(t_loss)
    val_loss_list.append(v_loss)

for idx, test_loss in enumerate(test_loss_list):
    plt.plot(test_loss, label= model_list[idx] + ' test loss')
    savetxt(model_list[idx] +'_train.csv', test_loss, delimiter=',')
    
plt.title('Test losses of various models on goblet_book')
plt.xlabel('Sequence')
plt.ylabel('Loss')
plt.legend()
plt.figure(1)

plt.figure(2)
for idx, val_loss in enumerate(val_loss_list):
    plt.plot(val_loss, label= model_list[idx] + ' val loss')
    savetxt(model_list[idx] +'_test.csv', test_loss, delimiter=',')
plt.title('Validation losses of various models on goblet_book')
plt.xlabel('Sequence')
plt.ylabel('Loss')
plt.legend()


plt.show()


