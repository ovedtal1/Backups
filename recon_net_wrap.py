import torch
import torch.nn as nn
from recon_net import ReconNet


class ViTfuser(nn.Module):
    def __init__(self, net, epsilon=1e-6):
        super().__init__()
        self.device = 'cuda:3'

        # ViT layer
        self.recon_net = ReconNet(net).to(self.device)#.requires_grad_(False)
        self.recon_net_ref = ReconNet(net).to(self.device)

        # Load weights
        cp = torch.load('./checkpoints_trained_start/model_100.pt', map_location=self.device) # Try new pretrained 11.12
        self.recon_net.load_state_dict(cp['model'])
        self.recon_net_ref.load_state_dict(cp['model'])
        
        # Fusion layers
        self.epsilon = epsilon
        self.param1 = nn.Parameter(torch.normal(0.85, 0.00, size=(198,)))
        self.param2 = nn.Parameter(torch.normal(0.15, 0.05, size=(198,)))


    def printer(self, x):
        print("Current value of param1 during forward:", self.param1)
        return

    def forward(self, img,ref): 

        in_pad, wpad, hpad = self.recon_net.pad(img)
        ref_pad, wpad, hpad = self.recon_net.pad(ref)
        input_norm,mean,std = self.recon_net.norm(in_pad.float())
        ref_norm,mean_ref,std_ref = self.recon_net.norm(ref_pad.float())
       
        # Feature extract
        features = self.recon_net.net.forward_features(input_norm)
        
        
        features_ref = self.recon_net_ref.net.forward_features(ref_norm)#.permute(0,2,1)

        batch_size, num_channels, height = features.shape
        features_flat = features.reshape(batch_size, num_channels, -1)
        features_ref_flat = features_ref.reshape(batch_size, num_channels, -1)       
        
        # Reshape params to match the dimensions
        param1_expanded = self.param1.reshape(1, -1, 1)  # Shape: [1, 416, 1]
        param2_expanded = self.param2.reshape(1, -1, 1)  # Shape: [1, 416, 1]
        # Expand params to match the flattened tensor dimensions
        param1_expanded = param1_expanded.expand(batch_size, -1, height)  # Shape: [batch_size, 416, height*width]
        param2_expanded = param2_expanded.expand(batch_size, -1, height)  # Shape: [batch_size, 416, height*width]
        # Calculate weighted sum


        weighted_sum = (param1_expanded * features_flat + param2_expanded * features_ref_flat)

        
        # Calculate normalization factor
        normalization_factor = param1_expanded + param2_expanded + self.epsilon
        
        # Normalize
        features_comb = weighted_sum / normalization_factor

        # Low Resolution
        features_comb = features_comb.reshape(features_flat.shape[0], 198, 1024)
        
        
        # Recon Head
        head_out = self.recon_net.net.head(features_comb)
        
        # Low Resolution 
        head_out_img = self.recon_net.net.seq2img(head_out, (180, 110))


        # un-norm
        merged = self.recon_net.unnorm(head_out_img, mean, std) 

        # un-pad 
        im_out = self.recon_net.unpad(merged,wpad,hpad)
        
        return im_out