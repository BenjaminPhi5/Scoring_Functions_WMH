import torchvision.transforms.functional as TF
import torch
import torch.nn.functional as F

def edge_pixels(batched_image):
    dtype = batched_image.dtype # preserve original datatype for casting at the end
    batched_image = batched_image.type(torch.float32)
    
    max_pool = F.max_pool2d(batched_image, 3, stride=1, padding=1)
    outer_edge = (max_pool != batched_image).type(torch.float32)
    
    dilated_edge = F.max_pool2d(outer_edge, 3, stride=1, padding=1)
    inner_edge = dilated_edge * batched_image
    
    return inner_edge.type(dtype)

def closing(batched_image, n_steps):
    # this works as long as all the ids are positive
    
    dtype = batched_image.dtype # preserve original datatype for casting at the end
    batched_image = batched_image.type(torch.float32)
    dilated = batched_image
    dilated = F.pad(dilated, (n_steps*2, n_steps*2, n_steps*2, n_steps*2)) # it is essential to provide all padding at the start, otherwise we get wierd behaviour at the edges
    for _ in range(n_steps):
        dilated = F.max_pool2d(dilated, 3, stride=1, padding=0)
    
    eroded = (1 - dilated)
    for _ in range(n_steps):
        eroded = F.max_pool2d(eroded, 3, stride =1, padding=1)
        
    return TF.center_crop((1-eroded), batched_image.shape[-2:]).type(dtype)

### entire batch at once version
def get_edge_cords_batch(image, closing_steps=5, unique_labels=False):
    """
    given a batch of images of shape [B, 1, H, W], containing instance segmentations, each instance has a unique ID.
    return for each element in the batch a list of x,y coordinates of the egde pixels of every istance in the image
    optionally perform closing to remove any holes in instances. Also returns the edge map for each image in the batch
    """
    bs = image.shape[0]
    
    # perform closing
    closed = closing(image, closing_steps)
    
    # get edge pixels
    edges = edge_pixels(closed)
    
    
    edge_flat = edges.flatten()
    instance_edge_locs = torch.where(edge_flat != 0)[0]
    instance_edge_values = edge_flat[instance_edge_locs]
    instance_ids = instance_edge_values.unique()
    
    shape = edges.shape
    print(shape)
    width = shape[-2]
    height = shape[-1]
    pixels = height * width
    
    batch_position = instance_edge_locs // pixels
    image_position = instance_edge_locs - (batch_position * pixels)
    row_position = image_position // height
    column_position = image_position % height
    
    batch_wheres = [[] for _ in range(bs)]
    for instance_id in instance_edge_values.unique():
        # instance_wheres = torch.where(per_instance_masks[i])
        instance_wheres = instance_edge_values == instance_id
        instance_batch = batch_position[instance_wheres]
        instance_rows = row_position[instance_wheres]
        instance_columns = column_position[instance_wheres]
        if unique_labels:
            batch_wheres[instance_batch[0]].append((instance_rows, instance_columns))
        else:
            for batch_idx in instance_batch.unique():
                batch_idx_loc = instance_batch == batch_idx
                batch_wheres[batch_idx].append((instance_rows[batch_idx_loc], instance_columns[batch_idx_loc]))
            
        
    return batch_wheres, edges

def one_hot_encoded_image(image):
    # for a 2D [H, W] shape image
    # ensure all IDs are unique
    return (image == image.unique().view(-1, 1))