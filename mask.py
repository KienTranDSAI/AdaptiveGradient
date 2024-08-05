import numpy as np
def create_gamma(grads_temp):
  gamma_t = []
  for i in range(grads_temp.shape[0]):
    if i < 5:
      gamma_t.append(0.3)
    elif i < 10:
      gamma_t.append(0.002)
    elif i < 20:
      gamma_t.append(0.001)
    else:
      gamma_t.append(0.005)
  return gamma_t
def create_mask(grads_temp, get_early_epoch = True):
  if get_early_epoch:
    mask = grads_temp[2:7,:,:,:]
    mask = mask.sum(0)
    mask[mask < 0] = 0
    mask = mask/mask.max()
    mask = np.stack([mask]*100)
  else:
    mask = np.zeros(grads_temp.shape[1:])
    gamma_t = create_gamma(grads_temp)
    avg_aggregated_grad = np.zeros(grads_temp.shape[1:])
    for i in range(grads_temp.shape[0]):
    # for i in range(0,40):

      new_grad = np.copy(grads_temp[i])
      avg_aggregated_grad = (avg_aggregated_grad * (i+1) + new_grad)/(i+2)
      diff = new_grad - avg_aggregated_grad
      mask += gamma_t[i] * new_grad
      # mask = mask/mask.max()
      mask = np.clip(mask, 0, 1)
  return mask