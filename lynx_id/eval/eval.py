import torch
from oml.functional.metrics import calc_cmc, calc_map


class EvalMetrics:
    def __init__(self, candidates_nearest_neighbors: list[list[str]], lynx_id_true: list[str],
                 top_k: tuple[int] = (1, 2, 3, 4, 5)):
        self.candidates_nearest_neighbors = candidates_nearest_neighbors
        self.lynx_id_true = lynx_id_true
        self.top_k = top_k

    # Compute some metrics
    def compute_accuracy(self, lynx_id_predicted: list[str], verbose=False):
        correct_predictions = 0
        total_predictions = 0

        for i, (p, r) in enumerate(zip(lynx_id_predicted, self.lynx_id_true)):
            total_predictions += 1
            output = f"Candidate {i} | Prediction: {p} | True label: {r}"
            if p == r:
                correct_predictions += 1
                output = "\x1b[6;30;42m" + output + "\x1b[0m"

            if verbose:
                print(output)

        accuracy = correct_predictions / total_predictions

        return accuracy

    def compute_tensor_matching_candidates(self):
        candidates_acc_k_list = [
            [1 if candidate == candidate_id else 0 for candidate in candidates_row] for
            candidates_row, candidate_id in zip(self.candidates_nearest_neighbors, self.lynx_id_true)
        ]
        return torch.tensor(candidates_acc_k_list, dtype=torch.bool)

    def compute_mean_per_top_k(self, metric_output):
        metric_mean = torch.mean(torch.stack(metric_output), dim=1)
        return {k: round(v.item(), 3) for k, v in zip(self.top_k, metric_mean)}

    def compute_cmc_map_metrics(self):
        # CMC@k + mAP@k
        candidates_acc_k_tensor = self.compute_tensor_matching_candidates()

        # CMC@k
        cmc_k = calc_cmc(candidates_acc_k_tensor, self.top_k)
        cmc_k_mean = self.compute_mean_per_top_k(cmc_k)

        # mAP@k
        map_k = calc_map(candidates_acc_k_tensor, n_gt=None, top_k=self.top_k)
        map_k_mean = self.compute_mean_per_top_k(map_k)

        return cmc_k_mean, map_k_mean

    def precision_recall_individual(self, candidates_predicted: list[str], individual_name: str = "New",
                                    verbose: bool = False):
        # Initialize counters
        correct_prediction_individual = sum(p == r == "New" for p, r in zip(candidates_predicted, self.lynx_id_true))
        total_predictions_individual = candidates_predicted.count("New")
        total_references_individual = self.lynx_id_true.count("New")

        # Display information if verbose mode is enabled
        if verbose:
            print(
                f"{individual_name} individual predicted {total_predictions_individual} times (total number of images: "
                f"{len(self.lynx_id_true)}). In reality, the {individual_name} individual appears "
                f"{total_references_individual} times.")

        # Calculate precision and recall
        precision = correct_prediction_individual / total_predictions_individual \
            if total_predictions_individual > 0 else 0
        recall = correct_prediction_individual / total_references_individual \
            if total_references_individual > 0 else 0

        return {"precision": precision, "recall": recall}
