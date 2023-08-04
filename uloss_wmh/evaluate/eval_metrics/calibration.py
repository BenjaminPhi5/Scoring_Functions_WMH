def calibration_over_samples(save_folder, results_file, means3d, samples3d, ys3d, do_normalize, mode="mean_conf"):
    bins = 10 + 1 # for the 0 bin
    bin_batch_accuracies = [[] for b in range(bins)]
    bin_batch_confidences = [[] for b in range(bins)]
    bin_batch_sizes = [[] for b in range(bins)]
    bin_counts = [0 for b in range(bins)]
    for batch_idx in tqdm(range(len(ys3d)), ncols=150, position=0, leave=True): # skip the last batch with a different shape
        batch_t = ys3d[batch_idx].squeeze().cuda()
        batch_samples = samples3d[batch_idx].cuda()

        if batch_t.shape[0] < 10:
            continue # skip last batch if it is very small.

        # get probabilities
        if do_normalize:
            probs = normalize_samples(batch_samples)
        else:
            probs = batch_samples
        p1s = probs[:,:,1]
        
        if mode == "all_samples":
            p1s = p1s # ie do nothing, use each sample
        elif mode == "mean_conf":
            p1s = p1s.mean(dim=0)
        elif mode == "min_conf":
            p1s = p1s.min(dim=0)[0]
        elif mode == "median_conf":
            p1s = p1s.median(dim=0)[0]
        elif mode == "max_conf":
            p1s = p1s.max(dim=0)[0]
        elif mode == "mean_only":
            if do_normalize:
                p1s = torch.softmax(means3d[batch_idx].cuda(), dim=1)[:,1]
            else:
                p1s = means3d[batch_idx][:,1]
        else:
            raise ValueError(f"mode: {mode} not accepted") 
            

        # split into bins
        bin_ids = place_in_bin(p1s)

        # compute counts
        for i in range(bins):
            is_in_bin = (bin_ids == (i / 10))
            # print(is_in_bin.shape)
            # print(batch_t.shape)

            # number of elements in each bin
            num_elem = torch.sum(is_in_bin).item()
            # if num_elem == 0:
            #     print("zero")

            # number of predictions = to class 1
            c1_acc = batch_t.expand(p1s.shape)[is_in_bin].sum() / num_elem

            # if torch.isnan(c1_acc):
            #     print("acc_nan")

            # average confidence of values in that bin
            c1_conf = p1s[is_in_bin].mean()

            # if torch.isnan(c1_conf):
            #     print("conf_nan")
                
            if torch.isnan(c1_conf) or torch.isnan(c1_acc) or num_elem == 0:
                #print("conf_nan") # just skip for this bin for this indivudal if they don't have have a prediction
                # with a confidence in this bin.
                continue

            bin_batch_accuracies[i].append(c1_acc.item())
            bin_batch_confidences[i].append(c1_conf.item())
            bin_batch_sizes[i].append(num_elem)

    bin_sizes = [torch.Tensor(bbs).sum() for bbs in bin_batch_sizes]
    bin_accuracies = [torch.Tensor([bin_batch_accuracies[i][j] * bin_batch_sizes[i][j] / bin_sizes[i] for j in range(len(bin_batch_accuracies[i]))]).sum().item() for i in range(len(bin_sizes))]
    bin_confidences = [torch.Tensor([bin_batch_confidences[i][j] * bin_batch_sizes[i][j] / bin_sizes[i] for j in range(len(bin_batch_confidences[i]))]).sum().item() for i in range(len(bin_sizes))]

    print_and_write(results_file, f"{mode} calibration curve data: ")

    print_and_write(results_file, f"{mode} bin_accuracies: ", newline=1)
    print_and_write(results_file, str(bin_accuracies))

    print_and_write(results_file, f"{mode} bin_confidences: ", newline=1)
    print_and_write(results_file, str(bin_confidences))

    total_size = torch.sum(torch.Tensor(bin_sizes)[1:])
    ece = torch.sum( (torch.Tensor(bin_sizes)[1:]/ total_size) * (torch.abs(torch.Tensor(bin_accuracies)[1:] - torch.Tensor(bin_confidences)[1:])))
    print_and_write(results_file, f"{mode} EXPECTED CALIBRATION ERROR", newline=1)
    print("note we skip the first bin due to its size")
    print_and_write(results_file, ece)

    plt.plot(bin_confidences, bin_accuracies)
    plt.plot([0,1],[0,1]);
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy");
    save(save_folder, f"{mode} calibration")
    
    return ece, bin_confidences, bin_accuracies


def compute_calibration_curve_from_confidences(predictions, labels, prediction_threshold=0., convert_to_numpy=True, num_bins=20):
    """Computes the calibration curve of a model.
    
    modified from a chat with google bard.

    Args:
    predictions: A list of pytorch tensors of shape (batch_size, height, width).
    labels: A list of pytorch tensors of shape (batch_size, height, width).
    prediction_threshold: the minimim prediction confidence to be considered

    Returns:
    list of bin values, model confidences, model accuracies (if you select the current confidence as the prediction threshold)
    
    """

    # Convert the predictions and labels to numpy arrays.
    if convert_to_numpy:
        predictions = torch.stack(predictions)
        labels = torch.stack(labels)
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()

    # Bin the predictions into 10 equally spaced bins.
    bins = np.linspace(prediction_threshold, 1, num_bins)
    bin_indices = np.digitize(predictions, bins)
    #print(bins)
    
    # Compute the accuracy in each bin.
    accuracies = np.zeros(num_bins)
    confidences = np.zeros(num_bins)
    for i in tqdm(range(num_bins)):
        accuracies[i] = np.mean(labels[bin_indices == i])
        confidences[i] = np.mean(predictions[bin_indices == i])

    # Compute the expected calibration error.
    expected_calibration_error = np.mean(np.abs(confidences[1:] - accuracies[1:]))

    # Return the calibration curve and the expected calibration error.
    return (bins, confidences, accuracies), expected_calibration_error

# calibration per individual, and calibration score per volume
def compute_calibration_curve_per_individual(predictions, labels, prediction_threshold=0., convert_to_numpy=True, num_bins=20):
    """Computes the calibration curve of a model.
    
    modified from a chat with google bard.

    Args:
    predictions: A list of pytorch tensors of shape (batch_size, height, width).
    labels: A list of pytorch tensors of shape (batch_size, height, width).
    prediction_threshold: the minimim prediction confidence to be considered

    Returns:
    list of confidences, accuracies, eces for each individual.
    """

    # Convert the predictions and labels to numpy arrays.
    if convert_to_numpy:
        predictions = torch.stack(predictions)
        labels = torch.stack(labels)
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()

    # Bin the predictions into 10 equally spaced bins.
    bins = np.linspace(prediction_threshold, 1, num_bins)
    bin_indices = np.digitize(predictions, bins)
    #print(bins)
    
    individual_accuracies = []
    individual_confidences = []
    individual_eces = []
    
    for j in tqdm(range(predictions.shape[0])):
        # Compute the accuracy in each bin.
        accuracies = np.zeros(num_bins)
        confidences = np.zeros(num_bins)
        for i in range(num_bins):
            accuracies[i] = np.mean(labels[j][bin_indices[j] == i])
            confidences[i] = np.mean(predictions[j][bin_indices[j] == i])

        # Compute the expected calibration error.
        expected_calibration_error = np.mean(np.abs(confidences[1:] - accuracies[1:]))
        
        individual_accuracies.append(accuracies)
        individual_confidences.append(confidences)
        individual_eces.append(expected_calibration_error)
        

    # Return the calibration curve and the expected calibration error.
    return individual_confidences, individual_accuracies, individual_eces
