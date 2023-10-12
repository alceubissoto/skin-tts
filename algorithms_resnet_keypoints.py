import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import copy
import numpy as np
import time
import matplotlib.pyplot as plt
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def masked_forward(model, mask, x):
    # See note [TorchScript super()]
    #print(x, mask)
    #x.scatter_(1, mask, 0)
    x = torch.scatter(x,1, mask, 0)
    #x = x * 10 #renormalização considerando que ficamos com 10%
    #x[:, mask, :, :] = 0.0 #mask=[B, C] x=[B, C, 7, 7] -> [B, B, C, 7, 7]
    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    #x[:, mask] = 0.0 # TO-DO
    x = model.fc(x)

    return x


def masked_extract(model, mask, x):
    # See note [TorchScript super()]
    #print(x, mask)
    #x.scatter_(1, mask, 0)
    x = torch.scatter(x,1, mask, 0)
    #x = x * 10 #renormalização considerando que ficamos com 10%
    #x[:, mask, :, :] = 0.0 #mask=[B, C] x=[B, C, 7, 7] -> [B, B, C, 7, 7]
    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    #x[:, mask] = 0.0 # TO-DO
    #x = model.fc(x)

    return x


def extract_features(model, x):
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    #x = model.fc(x)
    return x

def extract_cam(model, x): 
    # See note [TorchScript super()]
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    return x

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, num_classes, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError
        

        
class TTActivation(Algorithm):
    def __init__(self, num_classes, model, hparams, fun_extract, fun_masked, mode_up='nearest', norm='null', relu=0, img_size=224):
        super().__init__(num_classes, hparams)
        self.mode_up = mode_up
        self.norm = norm
        self.relu = relu
        self.img_size = img_size
        self.model = model
        self.hparams = hparams # mask_preprocess=0.1, mask_percentile=70, amt_pos, amt_neg, perc_feature_mask=0.4~0.6
        self.num_classes = num_classes
        self.upsample = torch.nn.UpsamplingBilinear2d(size=(224,224))
        #self.upsample = torch.nn.UpsamplingBilinear2d(scale_factor=32)
        #self.upsample = F.interpolate(size=(224,224))
        self.seed = 1111
        self.sorted_channels = []
        self.fun_masked = fun_masked
        self.fun_extract = fun_extract
        
    def forward(self, x, pos_keypoints, keypoints, adapt=False):
        
        if adapt:
            activations = self.fun_extract(self.model, x) # [B, C, 7, 7]
            #print('activations', activations.shape)
            feature_masks = self.select_features(activations, pos_keypoints, keypoints) # [C]   [B, C * H], H<1
            #print('feature masks', feature_masks.shape)
            #feature_masks = torch.from_numpy(feature_masks)
            feature_masks_reshaped = feature_masks.reshape(feature_masks.size(0), feature_masks.size(1), 1, 1).repeat(1, 1, activations.size(2), activations.size(3))
            outputs = self.fun_masked(self.model, feature_masks_reshaped, activations) # [B, N_CLASS]
        else:
            outputs = self.model(x)
         
        return outputs, feature_masks
    
    
    def forward_vit(self, x, pos_keypoints, keypoints, adapt=True):
        def reshape_transform(tensor, height=14, width=14):
            result = tensor[:, 1 :  , :].reshape(tensor.size(0),
                height, width, tensor.size(2))

            # Bring the channels to the first dimension,
            # like in CNNs.
            result = result.transpose(2, 3).transpose(1, 2)
            return result
        
        if adapt:
            x = self.model.forward_features(x)

            activations_map = reshape_transform(x)
            #print('activation_map', activations_map.size())
            feature_masks = self.select_features(activations_map, pos_keypoints, keypoints) # [C]   [B, C * H], H<1
            #print('before', feature_masks.size())
            feature_masks_reshaped = feature_masks.reshape(feature_masks.size(0),1, feature_masks.size(1)).repeat(1, x.size(1), 1)
            #print('after', feature_masks.size())
            x.scatter_(2, feature_masks_reshaped, 0)

            x = self.model.forward_head(x)
        else: 
            x = self.model(x)
            
        return x, feature_masks
    
    
    def random_masked_features(self, x, mask, adapt=None):
        
        activations = self.fun_extract(self.model, x)
        batch_size = activations.size(0)
        channel = activations.size(1)
        
        weights = torch.ones(channel).expand(batch_size, -1)
        feature_masks = torch.multinomial(weights, num_samples=int((1 - self.hparams['perc_feature_mask']) * channel), replacement=False)
        feature_masks_reshaped = feature_masks.reshape(feature_masks.size(0), feature_masks.size(1), 1, 1).repeat(1, 1, activations.size(2), activations.size(3)).to("cuda")
        outputs = self.fun_masked(self.model, feature_masks_reshaped, activations)
        return outputs, feature_masks
        
    def extract_masked_features(self, x, mask, adapt=False):
        if adapt:
            activations = self.fun_extract(self.model, x) # [B, C, 7, 7]
            #print('activations', activations.shape)
            feature_masks = self.select_features(activations, mask) # [C]   [B, C * H], H<1
            #print('feature masks', feature_masks.shape)
            #feature_masks = torch.from_numpy(feature_masks)
            feature_masks_reshaped = feature_masks.reshape(feature_masks.size(0), feature_masks.size(1), 1, 1).repeat(1, 1, activations.size(2), activations.size(3))
            outputs = self.fun_masked(self.model, feature_masks_reshaped, activations) # [B, N_CLASS]
        else:
            outputs = extract_features(self.model, x)
            feature_masks = []
        return outputs, feature_masks
    
    def sort_channels(self, upsampled, pos_coords, neg_coords): # pos_coords [B, AMT_POS, 2]
        
        samples, channel, _, _ = upsampled.shape #[B, C, 224, 224]
        if self.hparams['amt_pos'] > 0 and pos_coords is not None:
            # Combine the x- and y-coordinates into a single index
            index = pos_coords[..., 0] * upsampled.size(-1) + pos_coords[..., 1]
            index = index[..., None].repeat(1, 1, channel).transpose(1, 2).to(upsampled.device)
            # Use gather to extract the values from upsampled corresponding to the coordinates in pos_coords
            pos_sum = torch.gather(
                upsampled.view(samples, channel, -1), # flatten upsampled to have 1 row per pixel
                2, # gather along the channel dimension
                index # expand the index to select all channels
            ).view(samples, channel, self.hparams['amt_pos']).to(upsampled.device)
            
            #pos_sum = torch.sum(output, axis=-1)
            #pos_sum = self.hparams['alpha']
        else:
            pos_sum = torch.zeros(samples, channel, 1).to(upsampled.device) #[B, C, 1]
        if self.hparams['amt_neg'] > 0 and neg_coords is not None and neg_coords.sum() > 0:
            index = neg_coords[..., 0] * upsampled.size(-1) + neg_coords[..., 1]
            index = index[..., None].repeat(1, 1, channel).transpose(1, 2).to(upsampled.device)
            # Use gather to extract the values from upsampled corresponding to the coordinates in pos_coords
            neg_sum = torch.gather(
                upsampled.view(samples, channel, -1), # flatten upsampled to have 1 row per pixel
                2, # gather along the channel dimension
                index # expand the index to select all channels
            ).view(samples, channel, self.hparams['amt_neg']).to(upsampled.device)
            
            #neg_sum = torch.sum(output, axis=-1)
            neg_sum = -1* neg_sum

        else:
            neg_sum = torch.zeros(samples, channel, 1).to(upsampled.device) #[B, C, 1]
        
        # Importance of negative or positive
        pos_sum *= self.hparams['alpha']
        neg_sum *= (1-self.hparams['alpha'])

        stack = torch.cat((pos_sum, neg_sum), dim=-1) # [B, C, AMT_POS + AMT_NEG]
        both = torch.sum(stack, dim=2) # [B, C]
        #print("both", both)
        sort = torch.argsort(both) # [B, C]
        
        # [C]
        #indices_1d = torch.arange(both.size(1))

        # [B, C]
        #indices = indices_1d.repeat(both.size(0), 1)
        # [B, C]
        #mask = both > 0
        # Replace indices where mask is False (i.e., values <= 0) with -1
        #indices[~mask] = 0
        #print("indices", indices)
        #print("nonzero", torch.count_nonzero(indices, dim=1))
        self.sorted_channels.append(sort) # [B, C]
        sort = sort[:, :int((1 - self.hparams['perc_feature_mask']) * channel)]
        
        return sort

    
    def randomCordinates(self, a, n_samples, value, rng):
        output = []
        for i in range(len(a)):
            pos_coords = []
            last = None
            iter_cnt = 0
            while len(pos_coords) < n_samples:
                x0 = rng.integers(0, a.size(1))
                x1 = rng.integers(0, a.size(2))
                if a[i, x0, x1] == value and [x0, x1] not in pos_coords:
                    last = [x0,x1]
                    pos_coords.append([x0, x1])

                # control to avoid infinite loop
                iter_cnt += 1
                if iter_cnt >= a.size(1) * a.size(2) * 2:
                    print('failure when selecting points')
                    pos_coord.append(last)
                    
            output.append(pos_coords)
        return torch.tensor(np.array(output))
    
    def randomCordinates_n(self, a, n_samples, value, rng):
        
        output = None
        
        # Find indices of elements in a that are equal to the target value
        indices = torch.nonzero(a == value, as_tuple=False)
        for i in range(len(a)):
            sel = indices[indices[:,0] == i]
            if len(sel) >= n_samples:
                res = sel[rng.choice(len(sel), size=n_samples, replace=False)][:,1:]
                if output is None:
                    output = res.unsqueeze(0)
                else:
                    output = torch.cat((output, res.unsqueeze(0)),0)
            else:
                output = None
                break
                
        return output
    
    def select_features(self, activations, pos_keypoints=None, keypoints=None):
        
        
        if self.norm == 'before':
            maxs = activations.view(activations.size(0),
                                  activations.size(1), -1).max(dim=-1)[0]
            mins = activations.view(activations.size(0),
                                  activations.size(1), -1).min(dim=-1)[0]

            maxs, mins = maxs[:, :, None, None], mins[:, :, None, None]
            activations = (activations - mins) / (maxs - mins + 1e-12)
        
        if self.mode_up == 'bilinear':
            upsampled = self.upsample(activations)
        else:
            scale_factor = self.img_size // activations.size(2)
            upsampled = activations[:,:, :,None, :, None].expand(-1, -1, -1, scale_factor, -1, scale_factor).reshape(activations.size(0), activations.size(1), scale_factor * activations.size(2), scale_factor * activations.size(3))
            
        if self.relu:
            upsampled = F.relu(upsampled)
        
        if self.norm == 'after':
            maxs = upsampled.view(upsampled.size(0),
                                  upsampled.size(1), -1).max(dim=-1)[0]
            mins = upsampled.view(upsampled.size(0),
                                  upsampled.size(1), -1).min(dim=-1)[0]

            maxs, mins = maxs[:, :, None, None], mins[:, :, None, None]
            upsampled = (upsampled - mins) / (maxs - mins + 1e-12)
        
        
      
        
        #rng = np.random.default_rng(self.seed)
        #pos_coords = self.randomCordinates_n(mask, self.hparams['amt_pos'], 1.0, rng)
        #rng = np.random.default_rng(self.seed)
        #neg_coords = self.randomCordinates_n(mask, self.hparams['amt_neg'], 0.0, rng)
        
        
        not_considered_masks = self.sort_channels(upsampled, pos_keypoints, keypoints)
        del upsampled
        return not_considered_masks
    
    ### Legacy methods ###  
    def sort_channels_old(self, upsampled, pos_coords, neg_coords):
        best_channels = []
        sam, channel, _, _ = upsampled.shape
        k = 0
        considered_masks = []
        while len(considered_masks) < int(self.hparams['perc_feature_mask'] * channel):
            for i in range(channel):
                if np.sum([upsampled[0, i, coord[0], coord[1]] for coord in pos_coords[0]]) >= len(pos_coords[0]) - k:
                    if np.sum([upsampled[0, i, coord[0], coord[1]] for coord in neg_coords[0]]) <= 0 + k:
                        if i not in considered_masks and len(considered_masks) < int(self.hparams['perc_feature_mask'] * channel):
                            considered_masks.append(i)
            k += 1
                            

        not_considered_masks = []
        for i in range(channel):
            if i not in considered_masks:
                not_considered_masks.append(i)
        best_channels.append(not_considered_masks)
        return np.array(best_channels)
        
                
    def randomCordinates_old(self, a, n_samples, value, rng):
        output = []
        for m in a:
            argwhere = np.argwhere(m == value).T
            permuted = rng.permutation(argwhere, axis=0)[:n_samples]
            output.append(permuted)
        return np.array(output)        
        
class T3A(Algorithm):
    """
    Test Time Template Adjustments (T3A)
    """
    def __init__(self, num_classes, model, hparams, fun_extract):
        super().__init__(num_classes, hparams)
        
        self.model = model

        warmup_supports = self.model.fc.weight.data
        self.warmup_supports = warmup_supports
        warmup_prob = self.model.fc(self.warmup_supports)
        self.warmup_ent = softmax_entropy(warmup_prob)
        self.warmup_labels = torch.nn.functional.one_hot(warmup_prob.argmax(1), num_classes=num_classes).float()

        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data

        self.filter_K = self.hparams['filter_K']
        self.num_classes = num_classes
        self.softmax = torch.nn.Softmax(-1)
        self.fun_extract = fun_extract
    def forward(self, x, adapt=False):

        z = self.fun_extract(self.model, x)
        #z = self.model.global_pool(z)
        z = self.model.avgpool(z)
        z = torch.flatten(z, 1)
        if adapt:
            # online adaptation
            p = self.model.fc(z)
            yhat = torch.nn.functional.one_hot(p.argmax(1), num_classes=self.num_classes).float()
            ent = softmax_entropy(p)
            # prediction
            self.supports = self.supports.to(z.device)
            self.labels = self.labels.to(z.device)
            self.ent = self.ent.to(z.device)
            self.supports = torch.cat([self.supports, z])
            self.labels = torch.cat([self.labels, yhat])
            self.ent = torch.cat([self.ent, ent])
        
        supports, labels = self.select_supports()
        supports = torch.nn.functional.normalize(supports, dim=1)
        weights = (supports.T @ (labels))
        return z @ torch.nn.functional.normalize(weights, dim=0)

    def select_supports(self):
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        filter_K = self.filter_K
        if filter_K == -1:
            indices = torch.LongTensor(list(range(len(ent_s))))

        indices = []
        indices1 = torch.LongTensor(list(range(len(ent_s))))
        for i in range(self.num_classes):
            _, indices2 = torch.sort(ent_s[y_hat == i])
            indices.append(indices1[y_hat==i][indices2][:filter_K])
        indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ent = self.ent[indices]
        
        return self.supports, self.labels

    def predict(self, x, adapt=False):
        return self(x, adapt)

    def reset(self):
        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data



class TentFull(Algorithm):
    
    def __init__(self, num_classes, model, hparams):
        super().__init__( num_classes, hparams)
        self.hparams = hparams
        self.old_model = model
        self.model, self.optimizer = self.configure_model_optimizer(self.old_model, alpha=self.hparams['alpha'])
        self.steps = self.hparams['gamma']
        assert self.steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = self.hparams['episodic']
    
        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x, adapt=False):
        if adapt:
            if self.episodic:
                self.reset()

            for _ in range(self.steps):
                self.model.eval()
                outputs = self.forward_and_adapt(x, self.model, self.optimizer)
                self.model.train()
        else:
            outputs = self.model(x)
        return outputs

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        # forward
        optimizer.zero_grad()
        outputs = model(x)
        # adapt
        loss = softmax_entropy(outputs).mean(0)
        loss.backward()
        optimizer.step()
        return outputs

    def configure_model_optimizer(self, model, alpha):
        featurizer = configure_model(model)
        params, param_names = collect_params(featurizer)
        #print(param_names)
        optimizer = torch.optim.Adam(
            params, 
            lr=self.hparams["lr"]*alpha,
            weight_decay=self.hparams['weight_decay']
        )
        # adapted_algorithm.classifier.predict = lambda self, x: self(x)
        return featurizer, optimizer

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)

def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None   
    return model


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = copy.deepcopy(model.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names

class TentClf(TentFull):
    def configure_model_optimizer(self, model, alpha):
        optimizer = torch.optim.Adam(
            model.fc.parameters(), 
            lr=self.hparams["lr"]  * alpha,
            weight_decay=self.hparams['weight_decay']
        )
        #adapted_algorithm.classifier.predict = lambda self, x: self(x)
        return model, optimizer
