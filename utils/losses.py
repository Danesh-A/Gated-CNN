import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, weight=None, epsilon=1e-8):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
    def forward(self, inputs, targets, mode, activation = None):
        dice_losses = []
        if mode == 'a':
            for class_idx in range(inputs.shape[1]):
                target = (targets[:, class_idx, ...] > 0).float()
                if activation == 'sigmoid':
                    prediction = F.sigmoid(inputs[:, class_idx, ...])
                elif activation == "softmax":
                    probabilities = F.softmax(inputs, dim=1)
                    prediction = probabilities[:, class_idx, ...]
                else:
                    prediction = (inputs[:, class_idx, ...])
                
                intersection = torch.sum(prediction * target)
                union = torch.sum(prediction) + torch.sum(target)

                dice_coefficient = (2.0 * intersection + self.epsilon) / (union + self.epsilon)
                dice_loss = 1.0 - dice_coefficient

                dice_losses.append(dice_loss)

            total_loss = torch.mean(torch.stack(dice_losses))
        elif mode == 'ne':
            for class_idx in range(2):
                target = (targets[:, class_idx, ...] > 0).float()
                if activation == 'sigmoid':
                    prediction = F.sigmoid(inputs[:, class_idx, ...])
                elif activation == "softmax":
                    probabilities = F.softmax(inputs, dim=1)
                    prediction = probabilities[:, class_idx, ...]
                else:
                    prediction = (inputs[:, class_idx, ...])
                
                intersection = torch.sum(prediction * target)
                union = torch.sum(prediction) + torch.sum(target)

                dice_coefficient = (2.0 * intersection + self.epsilon) / (union + self.epsilon)
                dice_loss = 1.0 - dice_coefficient

                dice_losses.append(dice_loss)

            total_loss = torch.mean(torch.stack(dice_losses))
            
        elif mode == 'n':
            target = (targets[:, 0, ...] > 0).float()
            if activation == 'sigmoid':
                prediction = F.sigmoid(inputs[:, 0, ...])
            elif activation == "softmax":
                probabilities = F.softmax(inputs, dim=1)
                prediction = probabilities[:, 0, ...]
            else:
                prediction = (inputs[:, 0, ...])
            
            intersection = torch.sum(prediction * target)
            union = torch.sum(prediction) + torch.sum(target)

            dice_coefficient = (2.0 * intersection + self.epsilon) / (union + self.epsilon)
            total_loss = 1.0 - dice_coefficient
        
        elif mode == 'e':
            target = (targets[:, 1, ...] > 0).float()
            if activation == 'sigmoid':
                prediction = F.sigmoid(inputs[:, 1, ...])
            elif activation == "softmax":
                probabilities = F.softmax(inputs, dim=1)
                prediction = probabilities[:, 1, ...]
            else:
                prediction = (inputs[:, 1, ...])
            
            intersection = torch.sum(prediction * target)
            union = torch.sum(prediction) + torch.sum(target)

            dice_coefficient = (2.0 * intersection + self.epsilon) / (union + self.epsilon)
            total_loss = 1.0 - dice_coefficient

        return total_loss
        



