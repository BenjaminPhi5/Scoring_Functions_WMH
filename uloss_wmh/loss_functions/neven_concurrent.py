import torch
import torch.nn as nn

class ScaledSigmoidActivation(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
    def forward(self, x):
        x =  -x / (2 * self.scale)
        x = x.exp()
        return self.scale / (1 + x) 
    
    
class EmbeddingVectors(nn.Module):
    def __init__(self,spatial_dims, embedding_spatial_dims, rescale_factor):
        super().__init__()
        # calculate the pixel position grid used to calculate the embeddings
        # positions is a grid of pixel coordinates.
        # spatial dims denotes the size of each dimension, e.g H,W or H,W,D, or H,W,D,T etc its flexible.
        spatial_dims=list(spatial_dims)
        grid = torch.stack(
            torch.meshgrid(
                *[torch.arange(0, sdi, 1) for sdi in spatial_dims],
                indexing='ij'
            )
        ).type(torch.float32)
        # in the case that we have more embedding dims than true spatial dims:
        S = len(embedding_spatial_dims)
        if S > len(spatial_dims):
            positions = torch.zeros(S, *spatial_dims)
            positions[0:len(spatial_dims)] = grid
        else:
            positions = grid
        
        spatial_dim_scaler = [embedding_spatial_dims[i] / spatial_dims[i] for i in range(len(spatial_dims))]
        if S > len(spatial_dims):
            
            spatial_dim_scaler.extend(torch.ones(S - len(spatial_dims), dtype=torch.int16).tolist())
        self.spatial_dims_to_embedding_dims_factor = torch.Tensor(
            spatial_dim_scaler
        ).unsqueeze(-1).unsqueeze(-1)
        self.positions = positions * self.spatial_dims_to_embedding_dims_factor
        
        self.rescale_factor = rescale_factor
        
    def forward(self, offsets):
        """
        offsets is a vector of shape [B,S,<dims>] where S is the number of spatial dims.
        ei = xi + oi
        """
        if self.positions.device != offsets.device:
            self.positions = self.positions.to(offsets.device)
            self.spatial_dims_to_embedding_dims_factor = self.spatial_dims_to_embedding_dims_factor.to(offsets.device)

        # print(self.positions.shape, offsets.shape[1:])
        assert self.positions.shape == offsets.shape[1:]
        
        return (self.positions + offsets) * self.rescale_factor

    def get_grid(self):
        return self.positions


class NevenLoss(nn.Module):
    def __init__(self, base_loss, sigma_normalizer=ScaledSigmoidActivation(64), true_spatial_dims=(192,224), embedding_spatial_dims=(64, 64), embedding_rescale_factor=1/64, instance_weight=1, seed_weight=1, smooth_weight=1, debug=False):
        super().__init__()
        """
        base loss should be a loss with reduction set to sum!!
        base_loss should operate on a probability distribution over pixels
        """
        self.sigma_normalizer = sigma_normalizer
        self.true_spatial_dims = true_spatial_dims
        self.embedding_spatial_dims = embedding_spatial_dims
        self.embedding_rescale_factor = embedding_rescale_factor
        self.embedding_vectors = EmbeddingVectors(true_spatial_dims, embedding_spatial_dims, embedding_rescale_factor)
        self.base_loss = base_loss
        self.instance_weight = instance_weight
        self.smooth_weight = smooth_weight
        self.seed_weight = seed_weight
        self.debug = debug
        
    
    def _get_sigma(self, log_inv_2pres_map):
        pass
    
    def _get_instance_distributions(self, mean_sigmas):
        pass
    
    def _get_instance_maps(self, labels):
        """
        labels is an integer map containing instance labels where each instance
        has a unique number.
        labels is of shape: [<dims>]
        
        returns a map containing each instance in a separate channel (i.e an array
        of shape [I, <dims>]
        """
        with torch.no_grad():
            instance_ids = labels.unique().tolist()
            I = len(instance_ids) - 1 # 0 will be the background...
            labels_per_instance = torch.zeros((I, *labels.shape), dtype=torch.int8).to(labels.device)

            for i, iid in enumerate(instance_ids):
                if iid == 0:
                    continue
                labels_per_instance[i-1] = labels == iid
                
            return labels_per_instance
        
    def _get_sigma_and_centre_means(self, sigma_map, embeddings_map, instance_labels):
        """
        sigma_map: [1, <dims>] or [S, <dims>]
        instance_labels: [I, <dims>]
        """

        I = instance_labels.shape[0]
        S_sigma = sigma_map.shape[0]
        S = embeddings_map.shape[0]
        device = sigma_map.device
        sigma_means = torch.zeros((I, S_sigma), device=device)
        embeddings_means = torch.zeros((I, S), device=device)

        smooth_loss = 0

        sigma_map = sigma_map.view(S_sigma, -1)
        embeddings_map = embeddings_map.view(S, -1)
        instance_labels = instance_labels.view(I, -1)

        for i in range(I):
            s_at_i = sigma_map[:, instance_labels[i]==1]
            e_at_i = embeddings_map[:, instance_labels[i]==1]
            
            embedding_mean = e_at_i.mean(dim=1)
            sigma_mean = s_at_i.mean(dim=1)
            
            # sum over number of S dims, mean over
            # number of pixels in the instance
            smooth_loss += (s_at_i - sigma_mean.unsqueeze(-1).detach()).pow(2).sum(dim=0).mean()
            # print("smth:", smooth_loss)
            sigma_means[i] = sigma_mean
            embeddings_means[i] = embedding_mean

        return sigma_means.unsqueeze(-1), embeddings_means.unsqueeze(-1), smooth_loss
        
        
    def forward(self, seed_map, offset_map, labels, sigma_map=None, log_inv_2pres_map=None):
        """
        B: batch size
        C: number of semantic classes (not including background)
        <dims>: spatial dims of the input (*[H,W] or *[H,W,D])
        S: number of embedding dims learned by the model
        
        seed_map is of shape [B, C, <dims>]
        offset_map has shape [B, S, <dims>]
        sigma_map has either shape [B, 1, <dims>] or [B, S, <dims>]
        
        labels has the shape [B, C, <dims>]
        """
        
        # configure sigma map
        if sigma_map == None and log_inv_2pres == None:
            return ValueError("one of sigma map or log_ing_2pres (log(1/(2*sigma^2)) must be provided")
        
        if sigma_map == None:
            sigma_map = self._get_sigma(log_inv_2pres_map)
    
        sigma_map = self.sigma_normalizer(sigma_map)
        
        C = seed_map.shape[1]
        S = offset_map.shape[1]
        sigma_S = sigma_map.shape[1]
        B = seed_map.shape[0]
        spatial_dims = seed_map.shape[2:]
        
        smooth_loss = 0
        instance_loss = 0
        seed_loss = 0
        N = 0
        
        # compute the embedding vectors
        embedding_vectors = self.embedding_vectors(offset_map)
        
        instance_distributions = [0 for _ in range(B)]
        
        
        for b in range(B):
            embeddings_b = embedding_vectors[b]
            sigma_b = sigma_map[b]
            
            for c in range(C):
                labels_bc = labels[b,c]
                seed_bc = seed_map[b,c]
                instance_labels = self._get_instance_maps(labels_bc)
                I = instance_labels.shape[0]
                N += I
                
                instance_sigmas, instance_centres, smooth_loss_bc = self._get_sigma_and_centre_means(sigma_b, embeddings_b, instance_labels)
                smooth_loss += smooth_loss_bc
                
                if self.debug:
                    print("instance sigmas: ", instance_sigmas.squeeze().detach())
                # print("instance centres: ", instance_centres.squeeze())
                
                # compute the distributions for each instance
                
                e = embeddings_b.view(S, -1).expand(I, S, -1)
                phi = (
                    (-(e - instance_centres).pow(2) / (2*instance_sigmas.pow(2))).sum(dim=1)
                ).exp()
                
                instance_distributions[b] = phi.detach().cpu().reshape(I, *spatial_dims)
                
                # use the base loss on the instance distributions
                instance_loss += self.base_loss(phi.unsqueeze(1), instance_labels.view(I, 1, -1))
                
                # regress seed map to instance distributions, with zero's in background pixels
                seed_loss_mask = (phi.detach() * instance_labels.view(I, -1)).sum(0)
                seed_loss += (seed_bc.view(-1) - seed_loss_mask).pow(2).mean()
                
                
        
        # at the end, divide the smooth loss by the number of instances?
        instance_loss /= N
        smooth_loss /= N
        
        loss = self.instance_weight * instance_loss + self.smooth_weight * smooth_loss + self.seed_weight * seed_loss
        
        return loss / (B * C), instance_distributions, (instance_loss.detach(), smooth_loss.detach(), seed_loss.detach())
    
    def inference(self):
        pass