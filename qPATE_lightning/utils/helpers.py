### ref: https://github.com/BorealisAI/private-data-generation/blob/master/utils/helper.py

# Copyright 2019 RBC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# helper.py contains some helper utility functions
import torch
import torch.nn as nn
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score
from torch.distributions.laplace import Laplace




def pate(teacher_networks: list, data, lap_scale):
    """
    Make predictions using Noisy-max using Laplace mechanism.
    Args:
        data: Data for which predictions are to be made
    Returns:
        predictions: Predictions for the data
    ref: https://github.com/BorealisAI/private-data-generation/blob/master/utils/helper.py#L30
        
    """
    model_predictions = torch.empty(len(teacher_networks), data.size()[0], device=data.device)
    with torch.no_grad():
        for ith, discriminator_t in enumerate(teacher_networks):
            output = discriminator_t(data)
            _, predict = torch.max(output.detach(), 1)
            model_predictions[ith] = predict
    """
    clean_votes = torch.sum(model_predictions, dim=0).unsqueeze(1)  # (B, 1) 
    noise = Laplace(torch.tensor([0.0], device=data.device), torch.tensor([1/lap_scale], device=data.device))
    noisy_results = clean_votes + noise.sample()
    noisy_labels = noisy_results.max(dim=1)[1].clone().detach()
    return noisy_labels.cpu(), clean_votes.cpu()
    """
    clean_votes = torch.sum(model_predictions, dim=0).unsqueeze(1)  # (B, 1) 
    model_predictions = model_predictions.permute(1,0) # (B, num_networks)
    noise = Laplace(torch.tensor(0.0, device=data.device), torch.tensor(1/lap_scale, device=data.device))
    noisy_labels = []
    for preds in model_predictions: 
        label_counts = torch.bincount(preds.long(), minlength=2).float()
        for i in range(len(label_counts)): 
            label_counts[i] += noise.sample()
        new_label = torch.argmax(label_counts)
        noisy_labels.append(new_label)
    noisy_labels = torch.tensor(noisy_labels, device=data.device)
    return noisy_labels.cpu(), clean_votes.cpu()
        



def moments_acc(num_teachers, clean_votes, lap_scale, l_list):
    q = (2 + lap_scale * torch.abs(2*clean_votes - num_teachers)
         )/(4 * torch.exp(lap_scale * torch.abs(2*clean_votes - num_teachers)))

    update = []
    for l in l_list:
        a = 2*lap_scale*lap_scale*l*(l + 1)
        t_one = (1 - q) * torch.pow((1 - q) / (1 - math.exp(2*lap_scale) * q), l)
        t_two = q * torch.exp(2*lap_scale * l)
        t = t_one + t_two
        update.append(torch.clamp(t, max=a).sum())

    return torch.tensor(update)


def mutual_information(labels_x: pd.Series, labels_y: pd.DataFrame):

    if labels_y.shape[1] == 1:
        labels_y = labels_y.iloc[:, 0]
    else:
        labels_y = labels_y.apply(lambda x: ' '.join(x.get_values()), axis=1)

    return mutual_info_score(labels_x, labels_y)


def normalize_given_distribution(frequencies):
    distribution = np.array(frequencies, dtype=float)
    distribution = distribution.clip(0)  # replace negative values with 0
    summation = distribution.sum()
    if summation > 0:
        if np.isinf(summation):
            return normalize_given_distribution(np.isinf(distribution))
        else:
            return distribution / summation
    else:
        return np.full_like(distribution, 1 / distribution.size)


