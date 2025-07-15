import torch
import torch.nn as nn
import torch.nn.functional as F

def sensitivity(logits, targets, class_index, epsilon=1e-8):
    #probabilities = F.sigmoid(logits)
    probabilities = F.softmax(logits, dim=1)
    predicted_binary = (torch.argmax(probabilities, dim=1) == class_index).float()
    target_binary = targets[:, class_index, ...].float()
    true_positives = torch.sum(predicted_binary * target_binary)
    false_negatives = torch.sum((1 - predicted_binary) * target_binary)
    sensitivity = true_positives / (true_positives + false_negatives + epsilon)
    return sensitivity.item()


def specificity(logits, targets, class_index, epsilon=1e-8):
    #probabilities = F.sigmoid(logits)
    probabilities = F.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)
    predicted_binary = (predicted_class == class_index).float()
    target_binary = targets[:, class_index, ...].float()
    true_negatives = torch.sum((1 - predicted_binary) * (1 - target_binary))
    false_positives = torch.sum(predicted_binary * (1 - target_binary))
    specificity = true_negatives / (true_negatives + false_positives + epsilon)
    return specificity.item()



class overlapmetricdice(nn.Module):
    def __init__(self, thresh=None, size_average=True):
        super(overlapmetricdice, self).__init__()
    def forward(self, inputs, targets, weight_map, thresh):
        criterion = ThreeClassDice()
        overlapregion = torch.tensor(weight_map > thresh, dtype=torch.float32)
        #inputs = F.sigmoid(inputs)
        inputs = F.softmax(inputs,dim = 1)
        overlap_inputs = overlapregion*inputs
        overlap_targets = overlapregion*targets
        overalldice = criterion(overlap_inputs,overlap_targets,'ne')
        nucleidice = criterion(overlap_inputs,overlap_targets,'n')
        edgedice = criterion(overlap_inputs,overlap_targets,'e')
    
        return overalldice, nucleidice, edgedice

class ThreeClassDice(nn.Module): 
    def __init__(self, weight=None, epsilon=1e-8):
        super(ThreeClassDice, self).__init__()
        self.epsilon = epsilon
    def forward(self, inputs, targets, mode, activation = None):
        dice_coef = []
        if mode == 'a':
            for class_idx in range(inputs.shape[1]):
                target = (targets[:, class_idx, ...] > 0).float()
                if activation == 'sigmoid':
                    prediction = F.sigmoid(inputs[:, class_idx, ...])
                    threshold = 0.5
                    prediction = (prediction > threshold).float()            
                elif activation == "softmax":
                    probabilities = F.softmax(inputs, dim=1)
                    prediction = probabilities[:, class_idx, ...]
                    threshold = 0.5
                    prediction = (prediction > threshold).float()
                else:
  
                    prediction = (inputs[:, class_idx, ...])
                    threshold = 0.5
                    prediction = (prediction > threshold).float()   
                intersection = torch.sum(prediction * target)
                union = torch.sum(prediction) + torch.sum(target)
                dice_coefficient = (2.0 * intersection + self.epsilon) / (union + self.epsilon)     
                dice_coef.append(dice_coefficient)
            total_coef = torch.mean(torch.stack(dice_coef))
        elif mode == 'ne':
            for class_idx in range(2):
                target = (targets[:, class_idx, ...] > 0).float()
                if activation == 'sigmoid':
                    prediction = F.sigmoid(inputs[:, class_idx, ...])
                    threshold = 0.5
                    prediction = (prediction > threshold).float()
                elif activation == "softmax":
                    probabilities = F.softmax(inputs, dim=1)
                    prediction = probabilities[:, class_idx, ...]
                    threshold = 0.5
                    prediction = (prediction > threshold).float()       
                else:
                    prediction = (inputs[:, class_idx, ...])
                    threshold = 0.5
                    prediction = (prediction > threshold).float()             
                intersection = torch.sum(prediction * target)
                union = torch.sum(prediction) + torch.sum(target)
                dice_coefficient = (2.0 * intersection + self.epsilon) / (union + self.epsilon)
                dice_coef.append(dice_coefficient)
            total_coef = torch.mean(torch.stack(dice_coef))    
        elif mode == 'n':
            target = (targets[:, 0, ...] > 0).float()
            if activation == 'sigmoid':
                prediction = F.sigmoid(inputs[:, 0, ...])
                threshold = 0.5
                prediction = (prediction > threshold).float()
            elif activation == "softmax":
                probabilities = F.softmax(inputs, dim=1)
                prediction = probabilities[:, 0, ...]
                threshold = 0.5
                prediction = (prediction > threshold).float()
            else:
                prediction = (inputs[:, 0, ...])
                threshold = 0.5
                prediction = (prediction > threshold).float() 
            intersection = torch.sum(prediction * target)
            union = torch.sum(prediction) + torch.sum(target)

            total_coef = (2.0 * intersection + self.epsilon) / (union + self.epsilon)
        
        elif mode == 'e':
            target = (targets[:, 1, ...] > 0).float()
            if activation == 'sigmoid':
                prediction = F.sigmoid(inputs[:, 1, ...])
                threshold = 0.5
                prediction = (prediction > threshold).float()
            elif activation == "softmax":
                probabilities = F.softmax(inputs, dim=1)
                prediction = probabilities[:, 1, ...]
                threshold = 0.5
                prediction = (prediction > threshold).float()
            else:
                prediction = (inputs[:, 1, ...])
                threshold = 0.5
                prediction = (prediction > threshold).float()
            intersection = torch.sum(prediction * target)
            union = torch.sum(prediction) + torch.sum(target)
            total_coef = (2.0 * intersection + self.epsilon) / (union + self.epsilon)
        return total_coef
