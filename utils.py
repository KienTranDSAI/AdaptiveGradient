import torch
import numpy as np
from AdaptiveExplainer import AdaptiveExplainer
import shap
import matplotlib.pyplot as plt
from mask import *

def normalize(image, mean = None, std = None):
    if mean == None:
       mean = [0.485, 0.456, 0.406]
    if std == None:
       std = [0.229, 0.224, 0.225]
    if image.max() > 1:
        image /= 255
    image = (image - mean) / std
    # in addition, roll the axis so that they suit pytorch
    return torch.tensor(image.swapaxes(-1, 1).swapaxes(2, 3)).float()

def pred(img,model,class_names, printOut = True): #img = np.array((B,H,W,C))
    y_pred = model(torch.from_numpy(img).permute(0,3,1,2))
    class_ = torch.argmax(y_pred, dim = 1)
    ans = torch.nn.functional.softmax(y_pred,1)
    maxScore = torch.max(ans, dim = 1)
    if printOut:
        print(f"The predicted class for this image is {class_} - {class_names[str(class_.item())][1]} with score is {maxScore[0].item()} and raw score = {y_pred[0][142]} ")
    return class_names[str(class_.item())][1]
def getTrueId(img, model):
    y_pred = model(torch.from_numpy(img).permute(0,3,1,2))
    class_ = torch.argmax(y_pred, dim = 1)
    ans = torch.nn.functional.softmax(y_pred,1)
    return class_.item()
def getPredictedScore(img,model, classId = None):
    y_pred = model(torch.from_numpy(img).permute(0,3,1,2))
    class_ = torch.argmax(y_pred, dim = 1)
    ans = torch.nn.functional.softmax(y_pred,1)
    maxScore = torch.max(ans, dim = 1)
    if classId:
        return ans[0][classId].item()
    return maxScore[0].item()


def explain_img(to_explain,model, baseline,class_names = None, local_smoothing = 0, nsamples = 100, numBaseline = 3):
  e = AdaptiveExplainer(model, normalize(baseline), local_smoothing = local_smoothing)
  shap_values, indexes, path_grads = e.shap_values(
      normalize(to_explain), ranked_outputs=1, nsamples = nsamples, numBaseline = numBaseline
  )
  # print(f"Shape of shap_values: {shap_values[0].shape}")
  # get the names for the classes
  if class_names:
    index_names = np.vectorize(lambda x: class_names[str(x)][1])(indexes)
  else:
    index_names = np.array([None])
  print(f"Predict: {index_names.item()}, score: {getPredictedScore(to_explain,model)}")
  # print(f"Shape of path_grads: {path_grads.shape
  shap_values = [np.swapaxes(s, 0, -1) for s in shap_values]
  return shap_values, path_grads, index_names

def draw_img_from_shap(shap_values, to_explain, index_names):
  shap_values = [np.swapaxes(s, 0, -1) for s in shap_values]
  shap.image_plot(shap_values, to_explain, index_names)

def plot_gradient_heatmap(step_grad):
  step_grad = step_grad.sum(0)
  print(step_grad.mean())
  plt.imshow(step_grad, cmap = 'hot', vmin = -10, vmax = 10)
  plt.colorbar(label="Color bar", orientation="horizontal")
  plt.show()

def get_important_val(grads_temp, get_early_epoch = False):
  mask = create_mask(grads_temp,get_early_epoch = get_early_epoch)
  sample = grads_temp*mask
  val = sample.sum(0)
  val_ = np.expand_dims(val,-1)
  val_ = np.swapaxes(val_, 0, -1)
  return [val_]
def get_important_point(important_val, percentile):
  val = important_val
  abs_val = [np.abs(i) for i in val]
  important_val = [np.sum(i, axis = 3) for i in abs_val]
  threshold = np.percentile(important_val, percentile)
  indexes  = np.where(important_val > threshold)
  second_dim = indexes[2]
  third_dim = indexes[3]
  datapoint = [[second_dim[i],third_dim[i]] for i in range(len(second_dim))]
  datapoint = np.array(datapoint)
  return datapoint
def draw_important_point(inp_img, point):
  plt.imshow(inp_img.squeeze())
  plt.scatter(point[:,1], point[:,0],color='red', marker='o', s = 2)
  plt.show()
def run_pipeline_explain(to_explain, baseline, percentile, local_smoothing = 0, numBaseline = 3, get_early_epoch = False):
  shap_val, grads_temp, index_names = explain_img(to_explain, baseline, local_smoothing = local_smoothing, numBaseline = numBaseline)
  import_val = get_important_val(grads_temp, get_early_epoch= get_early_epoch)
  point = get_important_point(import_val, percentile)
  draw_important_point(to_explain, point)
  return shap_val, grads_temp, point