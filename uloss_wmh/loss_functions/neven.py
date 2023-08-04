"""

TODO complete:

[x] calculate the embedding vectors from the output
[x] get sigmas from the model outputs
[x] calculate the learnable centres
[x] calcualte the fixed centres
[x] calculate the per lesion probability distribution for fixed centres
[x]  calculate the per lesions probability distribution for learned centres
[x] compute the smooth loss for one step of the loop
[x] compute the sigma loss for one step of the loop # make sure to do the gradient masking. Nice.
[x] compute the instance loss for one step of the loop
[x] think about what is done per class and what is not
[x] build a class that does the whole loop, thinking about what we need to get for each step of the loop
[x] devise an end to end test for this loss function to check it works as expected at each step...
[ ] build the inference code for the method...
[x] do I need to detach the mean from the computation..? I need to test this.
[x] do I need to be careful about the scaling of the vectors? Thibaut suggests this is important for getting it to train...
    the problem is the small lesions will have tiny vectors which will cause havoc for the loss, so if I divide by 10
    add a rescale parameter will that work better?
[x] add a rescale parameter....
[x] check if I required grad that I can successfully backtrack the gradients to the input tensors and get something useful.
"""

from torch.nn import Module
import torch


class EmbeddingVectors(Module):
    def __init__(self,spatial_dims):
        super().__init__()
        # calculate the pixel position grid used to calculate the embeddings
        # positions is a grid of pixel coordinates.
        # spatial dims denotes the size of each dimension, e.g H,W or H,W,D, or H,W,D,T etc its flexible.
        self.positions = torch.stack(
            torch.meshgrid(
                *[torch.arange(0, sdi, 1) for sdi in spatial_dims],
                indexing='ij'
            )
        )
        # print(self.positions.shape)
        # print(torch.Tensor(spatial_dims).shape)
        self.positions = self.positions #/ torch.Tensor(spatial_dims).unsqueeze(-1).unsqueeze(-1) # rescale so that each voxel position ranges between 0 and 1
        
    def forward(self, offsets):
        """
        offsets is a vector of shape [B,S,<dims>] where S is the number of spatial dims.
        ei = xi + oi
        """
        if self.positions.device != offsets.device:
            self.positions = self.positions.to(offsets.device)
        # print(self.positions.shape, offsets.shape)
        assert self.positions.shape == offsets.shape[1:]
        return self.positions + offsets

    def get_grid(self):
        return self.positions


def get_sigmas(log_2pres_squared):
    sigma_map = (1./(2*log_2pres_squared.exp())).sqrt()
    return sigma_map

def get_sigma_mean(sigma_map, label_mask, detach):
    """
    gets the mean sigma for a specific instance
    sigma_map has shape: [S or 1, <dims>] where S is the number of spatial dims
    label_instances has shape: [<dims>] is the 
    instance_id is an integer
    """
    instance_sigmas = sigma_map[:,label_mask]
    mean_sigma = instance_sigmas.mean(dim=1) # I need to check what shape this function returns.... arrg this is getting super complicated
    # and I think it will take a lot of time for me to get it right....
    if detach:
        mean_sigma = mean_sigma.detach()
    
    return mean_sigma

def get_log2pres_mean(log2pres_sqd, label_mask, detatch):
    log2press_mean = log2pres_sqd[:,label_mask].exp().mean(dim=1).log()
    if detatch:
        log2press_mean = log2press_mean.detach()

    return log2press_mean



class SmoothLoss(Module):
    # the smooth loss from the neven loss
    # the smooth loss is applied to the sigma map
    def __init__(self, ):
        super().__init__()

    def forward(self, sigma_map, sigma_mean, instance_mask):
        """
        # redoing to be for one step of the loop
        sigma_map has shape: [S or 1, <dims>]
        sigma_mean has shape: [S or 1]
        instance_mask has shape [<dims>] and is binary
        """
        # print("shapes in smooth loss")
        # print(sigma_map[:,instance_mask].shape)
        # print(sigma_mean.shape)
        # print("smooth numerator: ", ((sigma_map[:,instance_mask] - sigma_mean.unsqueeze(-1)) ** 2.).sum())
        # print("smooth denominator: ", instance_mask.sum())
        # print("----")
        return ((sigma_map[:,instance_mask] - sigma_mean.unsqueeze(-1)) ** 2.).sum() / instance_mask.sum()

class CentreInstanceDistribution(Module):
    """
    calculates the probability distribution of pixels belonging to a particular instance
    given the centre is taken as the fixed centre of the instance
    """
    def __init__(self,is_learnable=False, position_grid=None):
        """
        learnable: if true, use the embedding vectors to calculate the instance centre
        if false, use the position grid for each pixel to cacluate the instance centre
        """
        super().__init__()
        self.is_learnable = is_learnable
        self.position_grid = position_grid

    def calculate_centre(self, instance_mask, embedding_vector=None):
        """
        calculates the fixed centre. This is the mean of voxels in the grid
        position_grid shape: [S, <dims>] S is number of spatial dims, used if learnable is false
        instance_mask shape: [<dims>]
        lesion mask needs to be of type long or bool
        embedding_vectore shape: [S, <dims>] where S is number of learnable dims
        """

        if self.is_learnable:
            positions = embedding_vector
        else:
            if self.position_grid.device != instance_mask.device:
                self.position_grid = self.position_grid.to(instance_mask.device)
            positions = self.position_grid
            
        lesion_centre = positions[:,instance_mask].type(torch.float32).mean(dim=1)

        return lesion_centre

    def forward(self, embedding_vector, log_2pres_squared_mean, instance_mask):
        """
        # embedding vector is of shape [S, <dims>], S is number of spatial or learnable dims
        # log_pres_squared is of shape [1 or S, <dims>], either fixed sigma or sigma per dims in S.
        instance_mask is of shape [<dims>]
        # instance_centre is of shape [S]
        """
        # print("embedding vector shape: ", embedding_vector.shape)
        instance_centre = self.calculate_centre(instance_mask, embedding_vector)
        # print("instance centre: ", instance_centre)
        phi_k = (
            -(embedding_vector - instance_centre.unsqueeze(-1).unsqueeze(-1))**2. * log_2pres_squared_mean.unsqueeze(-1).unsqueeze(-1).exp()  
        ).sum(dim=0).exp()

        return phi_k

class SeedLoss(Module):
    def __init__(self,):
        super().__init__()

    def forward(self, seed_map, instance_mask, instance_distribution, instance_id):
        """
        seed_map shape: [<dims>] this is the seed map for a single semantic class. Each semantic class's seed map is treated as independent...
        instance_mask shape: [<dims>]
        instance_id is an int
        lesion_instance_distribution has shape: [<dims>]
        """
        selected_seeds = seed_map[instance_mask.detach()]

        if instance_id == 0:
            return (selected_seeds ** 2.).sum()
        else:
            selected_instance_distribution = instance_distribution.detach()[instance_mask.detach()]
            return ((selected_seeds - selected_instance_distribution) ** 2.).sum()


class InstanceLoss(Module):
    def __init__(self, base_loss):
        """
        base_loss is a binary instance loss
        """
        super().__init__()
        self.base_loss = base_loss

    def forward(self, instance_distribution, instance_mask):
        """
        instance_distribution shape is: [<dims>]
        instance_mask shape is [<dims>]
        """
        # this is really inefficient because I am calculating the loss per instance
        # per element in the batch. So its going to be really slow but at least it
        # will be easier to understand and debug.
        # I can create a more efficient version later I think...
        return self.base_loss(
            instance_distribution.unsqueeze(0).unsqueeze(0),
            instance_mask.unsqueeze(0).unsqueeze(0)
        )

class NevenLoss(Module):
    def __init__(self, spatial_dims, base_loss, learnable_centre=True, instance_weight=1., seed_weight=1., smooth_weight=1., detach_means=True, l2p_tanh_scale=2, l2p_tanh_offset=4.6, apply_l2p_adjustment=True):
        super().__init__()
        self.get_embeddings = EmbeddingVectors(spatial_dims)
        positions = self.get_embeddings.get_grid()
        self.instance_loss = InstanceLoss(base_loss)
        self.seed_loss = SeedLoss()
        self.centre_instance_distribution = CentreInstanceDistribution(is_learnable=learnable_centre, position_grid=positions)
        self.smooth_loss = SmoothLoss()
        self.instance_weight = instance_weight
        self.seed_weight = seed_weight
        self.smooth_weight = smooth_weight
        self.detach_means = detach_means
        self.l2p_tanh_scale = l2p_tanh_scale
        self.l2p_tanh_offset = l2p_tanh_offset
        self.apply_l2p_adjustment = apply_l2p_adjustment

    def forward(self, offset_vectors, log_2pres_squared, seed_map, label_instances):
        """
        offset_vectors has shape: [B,S,<dims>] where S is number of embedding dims (either spatial dims or some number of learnable dims)
        log_2pres_squared has shape: [B,1 or S, <dims>] 
        seed_map has shape: [B, C_s, <dims>] where C_s is the number of semantic classes (i.e skipping the background class)
        label_instances has shape: [B, C_s, <dims>],
        <dims> are the spatial dims.
        """

        batch_size = offset_vectors.shape[0]
        semantic_classes = seed_map.shape[1]

        num_voxels = torch.Tensor([*offset_vectors.shape[2:]]).prod()

        assert label_instances.shape[1] == semantic_classes
        assert log_2pres_squared.shape[1] == 1 or log_2pres_squared.shape[1] == offset_vectors.shape[1]

        # losses_instance = []
        # losses_seed = []
        # losses_smooth = []
        losses = []
        
        # rescale the log2pres_squared to a useable range:
        if self.apply_l2p_adjustment:
            log_2pres_squared = (log_2pres_squared.tanh() * self.l2p_tanh_scale) - self.l2p_tanh_offset

        sigma_map = get_sigmas(log_2pres_squared)
        
        #print(torch.mean(sigma_map))

        # rescale the offset vectors to -1, 1 range and calculate embeddings
        # offset_vectors = offset_vectors.tanh()
        embeddings = self.get_embeddings(offset_vectors)

        for b in range(batch_size):
            offsets_b = offset_vectors[b]
            log2press_b = log_2pres_squared[b]
            sigmas_b = sigma_map[b]
            seeds_b = seed_map[b]
            labels_b = label_instances[b]
            embeddings_b = embeddings[b]
            
            seed_loss = 0 # need to divide by number of pixels (* number of classes)
            instance_loss = 0 # need to divide by number of instances end
            num_instances = 0
            smooth_loss = 0 # need to divide by nothing?

            instance_distributions = []

            for c in range(semantic_classes):
                seeds_bc = seeds_b[c]
                labels_bc = labels_b[c]

                #print("labels: ", torch.unique*labels_bc)
                
                for iid in torch.unique(labels_bc):
                    # print("#####")
                    instance_mask = labels_bc == iid # locations of the current instance
                    
                    if iid == 0: # background class
                        # seed loss for background pixels
                        sl = self.seed_loss(seeds_bc, instance_mask, instance_distribution=None, instance_id=iid)
                        seed_loss += sl
                        # print("seed loss background: ", sl)
                        continue
                    num_instances += 1

                    # the mean of the sigmas and log 1/2pres^2 within that instance
                    sigma_mean = get_sigma_mean(sigmas_b, instance_mask, self.detach_means)
                    log2press_mean = get_log2pres_mean(log2press_b, instance_mask, self.detach_means)
                    # print("sigma mean: ",sigma_mean)
                    # print("log2press mean: ", log2press_mean)
                    instance_distribution = self.centre_instance_distribution(embeddings_b, log2press_mean, instance_mask)
                    instance_distributions.append(instance_distribution.detach())
                    # print("instance distribution sum and shape: ", instance_distribution.sum(), instance_distribution.shape)

                    # compute the seed loss for instances
                    sl = self.seed_loss(seeds_bc, instance_mask, instance_distribution, iid)
                    seed_loss += sl
                    # print("seed loss instance: ", sl)
                    
                    # compute the instance loss for instances
                    il = self.instance_loss(instance_distribution, instance_mask)
                    instance_loss += il
                    # print("individual instance loss: ", il)
    
                    # compute the smoooth loss for instances
                    ml = self.smooth_loss(sigmas_b, sigma_mean, instance_mask)
                    smooth_loss += ml
                    # print("individual smooth loss: ", ml)

            seed_loss /= (num_voxels * semantic_classes)
            instance_loss /= num_instances

            # print("--------------")
            # print("seed loss: ", seed_loss)
            # print("smooth loss: ", smooth_loss)
            # print("instance loss: ", instance_loss)
            b_loss = seed_loss * self.seed_weight + instance_loss * self.instance_weight + smooth_loss * self.smooth_weight

            losses.append(b_loss)
            

        # print(losses)
        return torch.stack(losses).mean(), instance_distributions
            