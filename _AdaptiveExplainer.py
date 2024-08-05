import warnings

import numpy as np
import pandas as pd
from packaging import version
import torch
class AdaptiveExplainer():

    def __init__(self, model, data, batch_size=50, local_smoothing=0):

        import torch
        if version.parse(torch.__version__) < version.parse("0.4"):
            warnings.warn("Your PyTorch version is older than 0.4 and not supported.")

        # check if we have multiple inputs
        self.multi_input = False
        if isinstance(data, list):
            self.multi_input = True
        if not isinstance(data, list):
            data = [data]

        # for consistency, the method signature calls for data as the model input.
        # However, within this class, self.model_inputs is the input (i.e. the data passed by the user)
        # and self.data is the background data for the layer we want to assign importances to. If this layer is
        # the input, then self.data = self.model_inputs
        self.model_inputs = data
        self.batch_size = batch_size
        self.local_smoothing = local_smoothing

        self.data = data
        self.model = model.eval()


        self.gradients = [None]
    def gradient(self, idx, inputs):
        
        self.model.zero_grad()
        X = [x.requires_grad_() for x in inputs]
        outputs = self.model(*X)
        selected = [val for val in outputs[:, idx]]
        grads = [torch.autograd.grad(selected, x,
                                         retain_graph=True if idx + 1 < len(X) else None)[0].cpu().numpy()
                     for idx, x in enumerate(X)]
        return grads
        


    def shap_values(self, X, nsamples=200, ranked_outputs=None, output_rank_order="max", rseed=None, return_variances=False, numBaseline = 3):

        # X ~ self.model_input
        # X_data ~ self.data

        # check if we have multiple inputs
        if not self.multi_input:
            assert not isinstance(X, list), "Expected a single tensor model input!"
            X = [X]
        else:
            assert isinstance(X, list), "Expected a list of model inputs!"

        model_output_ranks = (torch.ones((X[0].shape[0], len(self.gradients))).int() *
                                  torch.arange(0, len(self.gradients)).int())

        # compute the attributions
        X_batches = X[0].shape[0]
        # print(f"X_batches is: {X_batches}")
        output_phis = []
        output_phi_vars = []
        # samples_input = input to the model
        # samples_delta = (x - x') for the input being explained - may be an interim input
        samples_input = [torch.zeros((nsamples,) + X[t].shape[1:], device=X[t].device) for t in range(len(X))]
        samples_delta = [np.zeros((nsamples, ) + self.data[t].shape[1:]) for t in range(len(self.data))]
        # print(f"len of sample input: {len(samples_input)}")
        # print(f"sample input, delta shape: {samples_input[0].shape}, {samples_delta[0].shape}")
        # use random seed if no argument given
        if rseed is None:
            rseed = np.random.randint(0, 1e6)
        # print(f"Shape of X is {X[0].shape}")
        for i in range(model_output_ranks.shape[1]): # Diễn giải từng score
            all_phis = []
            all_phis_vars = []
            for k in range(len(self.data)):
                # for each of the inputs being explained - may be an interim input
                all_phis.append(np.zeros((X_batches,) + self.data[k].shape[1:]))
                all_phis_vars.append(np.zeros((X_batches, ) + self.data[k].shape[1:]))
            grad_temp = 0
            for f in range(numBaseline):
                np.random.seed(rseed)  # so we get the same noise patterns for each output class
                phis = []
                phi_vars = []
                for k in range(len(self.data)):
                    # for each of the inputs being explained - may be an interim input
                    phis.append(np.zeros((X_batches,) + self.data[k].shape[1:]))
                    phi_vars.append(np.zeros((X_batches, ) + self.data[k].shape[1:]))
                print(f"Shape of phi and phi_vars: {phis[0].shape} - {phi_vars[0].shape}")
                for j in range(X[0].shape[0]):
                    # fill in the samples arrays
                    rind = np.random.choice(self.data[0].shape[0])
                    segment = 1/nsamples
                    for k in range(nsamples):
                        # rind = np.random.choice(self.data[0].shape[0]) #Select one random baseline
                        # t = np.random.uniform()
                        t = k*segment
                        # print(t)
                        for a in range(len(X)):
                            if self.local_smoothing > 0:
                                # local smoothing is added to the base input, unlike in the TF gradient explainer
                                x = X[a][j].clone().detach() + torch.empty(X[a][j].shape, device=X[a].device).normal_() \
                                    * self.local_smoothing
                            else:
                                x = X[a][j].clone().detach()
                            samples_input[a][k] = (t * x + (1 - t) * (self.model_inputs[a][rind]).clone().detach()).\
                                clone().detach()
                            # if self.input_handle is None:
                            #     samples_delta[a][k] = (x - (self.data[a][rind]).clone().detach()).cpu().numpy()
                            samples_delta[a][k] = (x - (self.data[a][rind]).clone().detach()).cpu().numpy()
                    # compute the gradients at all the sample points
                    find = model_output_ranks[j, i]
                    grads = []
                    for b in range(0, nsamples, self.batch_size):   #Tính gradient theo batch
                        batch = [samples_input[c][b:min(b+self.batch_size,nsamples)].clone().detach() for c in range(len(X))]
                        grads.append(self.gradient(find, batch))
                    grad = [np.concatenate([g[z] for g in grads], 0) for z in range(len(self.data))]  #Len of grad 1, shape of grad[0]: (50, 3, 224, 224)
                    for t in range(len(self.data)):
                        samples = grad[t] * samples_delta[t]    #len of sample_delta: 1, shape is: (50, 3, 224, 224)
                        phis[t][j] = samples.sum(0)
                        # temp = samples
                        phi_vars[t][j] = samples.var(0) / np.sqrt(samples.shape[0]) # estimate variance of means
                        all_phis[t][j] += phis[t][j]
                        all_phis_vars += phi_vars[t][j]
                        grad_temp += samples
            
            output_phis.append(all_phis[0] if len(self.data) == 1 else all_phis)
            output_phi_vars.append(all_phis_vars[0] if not self.multi_input else all_phis_vars)


        if isinstance(output_phis, list):
            # in this case we have multiple inputs and potentially multiple outputs
            if isinstance(output_phis[0], list):
                output_phis = [np.stack([phi[i] for phi in output_phis], axis=-1)
                               for i in range(len(output_phis[0]))]
            # multiple outputs case
            else:
                output_phis = np.stack(output_phis, axis=-1)

        if ranked_outputs is not None:
            if return_variances:
                return output_phis, output_phi_vars, model_output_ranks
            else:
                return output_phis, model_output_ranks, grad_temp
        else:
            if return_variances:
                return output_phis, output_phi_vars
            else:
                return output_phis
