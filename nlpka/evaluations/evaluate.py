from multiprocessing import reduction
from weakref import ref
from git import Reference
import nlpka.tools.common as common
common.info(__file__,__name__,__package__)

import torch
import evaluate
from types import SimpleNamespace
from evaluate.visualization import radar_plot

import nlpka.evaluations.storage as eval_stor

import numpy as np
from nlpka.tools.enums import ModeSE, ModelArchSE, TaskCatSE, EvalTypeSE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, pair_confusion_matrix
import matplotlib.pyplot as plt

class EVALUATE:

    keys = SimpleNamespace()
    keys.labels = 'references'
    keys.predictions = 'predictions'

    stor_path = common.get_module_location(eval_stor)
    
    @staticmethod
    def get_compute_metrics(config, label_pad_id, metric_prefix, model_path, tokenizer, ds_split):

        print('Get compute_metrics function...')
        print(' metric_prefix:', metric_prefix)
        print(' label_pad_id:', label_pad_id)
        print(' tokenizer:', type(tokenizer).__name__)
        print(' ds_split:', ds_split)

        def compute_metrics(eval_pred):
            """
            This function evaluates the performance of a model 
            by comparing its predictions against the true labels.
            --
            WARNING: If the preprocess_logits_for_metrics function is used, 
            we get "predictions" instead of "logits" in the eval_pred tuple.
            """
            predictions, labels = eval_pred

            # if any of the predictions or labels is None, raise an error
            if predictions is None or labels is None: raise ValueError("predictions or labels are None")
            
            # Verify that the labels match the predictions
            EVALUATE.verify_labels_match(ds_split, labels) if \
                getattr(config.eval, 'verify_labels_match', False) else None

            # Calculate the confusion matrix
            EVALUATE.calc_confusion_matrix(predictions, labels, config, model_path) if \
                getattr(config.eval, 'calc_confusion_matrix', False) else None

            # Preprocess predictions and labels if necessary
            predictions, labels = EVALUATE.preproc_preds_labels(predictions, labels, config, label_pad_id, tokenizer, ds_split)

            # Compute the metrics
            results = EVALUATE.compute_metrics(predictions, labels, config, ds_split)

            # Postprocess the computed metrics
            results = EVALUATE.postproc_metrics(results, config, metric_prefix)
            
            # print(results); exit()
            return results
        
        return compute_metrics
    
    @staticmethod
    def verify_labels_match(ds_split, labels):
        '''
        Verify that the labels match the dataset split labels
        '''
        print('\nVerify compute_metrics labels match with ds_split labels...')
        
        # Ensure ds_split has 'labels' and it's the same length as labels
        if 'labels' not in ds_split[0]:
            raise ValueError("ds_split does not contain 'labels' key")
        
        ds_labels = [ sample['labels'] for sample in ds_split ]

        if len(ds_labels) != len(labels):
            raise ValueError(f"Mismatch in number of labels: ds_split has {len(ds_labels)}, eval_pred has {len(labels)}")

        for i, (ds_label, eval_label) in enumerate(zip(ds_labels, labels)):
            # print(type(ds_label), type(eval_label), ds_label, eval_label)
            if not (eval_label.tolist()[:len(ds_label)] == ds_label):
                raise ValueError(f"Label mismatch at index {i}: ds_split label = {ds_label}, eval_pred label = {eval_label}")
 
    @staticmethod
    def preproc_preds_labels(predictions, labels, config, label_pad_id, tokenizer, ds_split):

        # Preprocess predictions and labels
        preproc_rules = config.eval
        flatten = preproc_rules.flatten
        filter_padded = preproc_rules.filter_padded
        label_id_to_name = preproc_rules.label_id_to_name
        eval_per_task = getattr(preproc_rules, 'per_task', None)
        label_name_to_id = getattr(preproc_rules, 'label_name_to_id', False)
        label_name_to_float = getattr(preproc_rules, 'label_name_to_float', False)
        label_name_strip_lower = getattr(preproc_rules, 'label_name_strip_lower', False)
        label_pad_id = label_pad_id if label_pad_id is not None else -100
        decode = preproc_rules.decode

        def preproc_1d_preds_labels(predictions, labels, config, label_pad_id, tokenizer):
            # print(predictions.shape, labels.shape, '\n', predictions, labels)

            if decode:

                # Decode
                predictions = EVALUATE.batch_decode(predictions, tokenizer, label_pad_id)
                labels = EVALUATE.batch_decode(labels, tokenizer, label_pad_id) 

                # Strip and lower
                if label_name_strip_lower:
                    predictions = [prediction.strip().lower() for prediction in predictions]
                    labels = [label.strip().lower() for label in labels]
                # common.p(predictions, labels)
                # exit()

                # Convert label names to floating numbers or to class IDs
                if label_name_to_float:
                    predictions, labels = EVALUATE.convert_label_names_to_floats(predictions, labels)

                elif label_name_to_id:
                    predictions, labels = EVALUATE.convert_label_names_to_ids(predictions, labels)
                    
                # print(predictions.shape, labels.shape, '\n', predictions, '\n', labels)
                # exit()

            else:

                if flatten:
                    predictions = reduction.flatten()
                    labels = labels.flatten()

                if filter_padded:
                    mask = (labels != label_pad_id)
                    labels = labels[mask]
                    predictions = predictions[mask]

                if label_id_to_name:
                    label_names = np.array(config.ds.label.names)
                    predictions = label_names[predictions]
                    labels = label_names[labels]

            return predictions, labels
    
        if eval_per_task:
            # Group predictions and labels by task ID
            predictions, labels = EVALUATE.group_preds_labels(predictions, labels, ds_split, eval_per_task)
            # Preprocess predictions and labels for each task
            for task_id in labels.keys():
                # Adjust preprocess config for each task
                label_name_to_id = False if task_id in [ 4, 5, 'all' ] else True
                # Run 1D preprocess for current tasks predictions and labels
                task_preds, task_labels = predictions[task_id], labels[task_id]
                task_preds, task_labels = preproc_1d_preds_labels(task_preds, task_labels, config, label_pad_id, tokenizer)
                predictions[task_id], labels[task_id] = task_preds, task_labels

        elif config.task.category in [ TaskCatSE.LANGUAGE_MODELING, TaskCatSE.TEXT_CLASSIFICATION, TaskCatSE.TEXT_PAIR_CLASSIFICATION, TaskCatSE.TEXT_TO_TEXT ]:
            predictions, labels = preproc_1d_preds_labels(predictions, labels, config, label_pad_id, tokenizer)

        elif config.task.category == TaskCatSE.TOKEN_CLASSIFICATION:
            prep_predictions, prep_labels = [], []
            # Run over predictions and labels sentence by sentence
            for sent_preds, sent_labels in zip(predictions, labels):
                sent_preds, sent_labels = preproc_1d_preds_labels(sent_preds, sent_labels, config, label_pad_id, tokenizer)
                prep_predictions.append(sent_preds)
                prep_labels.append(sent_labels)
            predictions, labels = prep_predictions, prep_labels
        return predictions, labels

    @staticmethod
    def group_preds_labels(predictions, labels, ds_split, group_by):
        grouped_preds = {}
        grouped_labels = {}
        for p, l, s in zip(predictions, labels, ds_split):
            group_id = s[group_by]
            if group_id not in grouped_preds:
                grouped_preds[group_id] = []
                grouped_labels[group_id] = []
            grouped_preds[group_id].append(p)
            grouped_labels[group_id].append(l)
        # Convert lists to np.arrays
        for group_id in grouped_preds:
            grouped_preds[group_id] = np.array(grouped_preds[group_id])
            grouped_labels[group_id] = np.array(grouped_labels[group_id])
        grouped_preds['all'] = np.array(predictions)
        grouped_labels['all'] = np.array(labels)
        return grouped_preds, grouped_labels
     
    @staticmethod
    def convert_label_names_to_floats(predictions, labels):
        def string_to_float(string, default=-1.):
            """Converts string to float, using default when conversion not possible."""
            try:
                return float(string)
            except ValueError:
                return default
        predictions = np.array([float(prediction) for prediction in predictions])
        labels = np.array([float(label) for label in labels])
        return predictions, labels

    @staticmethod
    def convert_label_names_to_ids(predictions, labels):
        def name_to_id(string_label, label_classes, default=-1):
            """Returns index of string_label in label_classes or default if not found."""
            if string_label in label_classes:
                return label_classes.index(string_label)
            return default
        # Count and print unknown label predictions
        label_classes = sorted(set(labels))
        EVALUATE.count_decoded_unknown_label_predictions(predictions, label_classes)
        predictions = np.array([name_to_id(p, label_classes) for p in predictions])
        labels = np.array([name_to_id(l, label_classes) for l in labels])
        return predictions, labels
    
    @staticmethod
    def compute_metrics(predictions, labels, config, ds_split):
        results = {}
        if config.eval.metric_groups[0].metrics[0] == 'multirc':
            results = EVALUATE.compute_multirc(predictions, labels, ds_split)
        else:
            results = EVALUATE.compute_metrics_by_metric_groups(predictions, labels, config)
        return results
    
    @staticmethod
    def compute_multirc(predictions, labels, ds_split):

        multirc_metric = evaluate.load("super_glue", "multirc")

        structured_preds = []
        references = []

        def to_int_bool(x):
            if isinstance(x, str):
                return int(x.lower().strip() in ["true", "1"])
            return int(bool(x))

        for pred, label, sample in zip(predictions, labels, ds_split):
            try:
                structured_preds.append({
                    "idx": {
                        "question": int(sample["idx/question"]),
                        "paragraph": int(sample["idx/paragraph"]),
                        "answer": int(sample["idx/answer"]),
                    },
                    "prediction": to_int_bool(pred)
                })
                references.append(to_int_bool(label))
            except Exception as e:
                print(f"[Error] Invalid sample format in ds_split: {sample['idx'] if 'idx' in sample else sample}")
                raise e

        # Run the official scorer
        results = multirc_metric.compute(
            predictions=structured_preds,
            references=references
        )
        return results
   
    @staticmethod
    def compute_metrics_by_metric_groups(predictions, labels, config):
        eval_per_task = getattr(config.eval, 'per_task', None)
        results = {}
        
        def get_metric_args(metric_group):
            metric_args = {}
            group_preds, group_labels = predictions, labels
            if eval_per_task: 
                task_id = metric_group.task.id
                group_preds = predictions[task_id] if task_id in predictions else []
                group_labels = labels[task_id] if task_id in labels else []
            if not (len(group_preds) and len(group_labels)):
                return None
            labels_key = getattr(metric_group, 'labels_key', EVALUATE.keys.labels)
            predictions_key = getattr(metric_group, 'predictions_key', EVALUATE.keys.predictions)
            metric_args[predictions_key] = group_preds
            metric_args[labels_key] = group_labels
            metric_args.update({k: v for sn in metric_group.args for k, v in vars(sn).items()}) if hasattr(metric_group, 'args') else None
            # print('metric_group:', metric_group)
            # print('metric_args:', metric_args)
            # exit()
            return metric_args
        
        for metric_group in config.eval.metric_groups:
            metric_args = get_metric_args(metric_group)
            if metric_args:
                metrics = evaluate.combine(metric_group.metrics)
                group_results = metrics.compute(**metric_args)
                if eval_per_task: group_results = EVALUATE.add_prefix_to_metrics(group_results, f'{metric_group.task.name}/')
                results.update(group_results)
            else:
                print(f"Group {metric_group.task.id} not found in predictions or labels")
       
        if not results: raise ValueError('No metrics computed')
       
        return results
      
    @staticmethod
    def postproc_metrics(results, config, add_prefix):

        # cast np.ndarray to list and np.generic to item
        def cast_value(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            elif isinstance(value, np.generic):
                return value.item()
            else:
                return value
        results = {key: cast_value(value) for key, value in results.items()}
        
        # filter the metrics by the prefixes
        filter_by_prefixes = tuple(config.eval.filter_by_prefixes) if config.eval.filter_by_prefixes else None
        if filter_by_prefixes:
            results = {key: value for key, value in results.items() if key.startswith(filter_by_prefixes)}
            # remove the prefix from the metric names
            for prefix in filter_by_prefixes:
                results = {key.replace(prefix, ''): value for key, value in results.items()}
        
        # prefixe the metric names with the config name
        if add_prefix:
            results = EVALUATE.add_prefix_to_metrics(results, add_prefix)

        return results
    
    @staticmethod
    def add_prefix_to_metrics(results, prefix):
        return {f'{prefix}{name}': value for name, value in results.items()}

    @staticmethod
    def count_decoded_unknown_label_predictions(predictions, label_classes):
        '''
        In case of decoded predictions and labels
        '''
        total_preds = len(predictions)
        pred_unk_labels = [pred for pred in predictions if pred not in label_classes]
        pred_unk_percentage = (len(pred_unk_labels) / total_preds) * 100 if total_preds > 0 else 0
        print(f"\nLabel Classes: {label_classes}")
        print(f"Predictions > \n Total: {total_preds} \n Unknown: {len(pred_unk_labels)} ({pred_unk_percentage:.4f})%")

    @staticmethod
    def decode(texts, tokenizer, label_pad_id, skip_special_tokens=True):
        mask = (texts != label_pad_id)
        masked = texts[mask]  # texts is assumed to be a single tensor (1D)
        decoded = tokenizer.decode(masked, skip_special_tokens=skip_special_tokens)
        return decoded.strip()
    
    @staticmethod
    def batch_decode(texts, tokenizer, label_pad_id, skip_special_tokens=True):
        mask = (texts != label_pad_id)
        masked = [t[m] for t, m in zip(texts, mask)]
        decoded = tokenizer.batch_decode(masked, skip_special_tokens=skip_special_tokens)
        return [d.strip() for d in decoded]

    # @staticmethod
    # def get_preprocess_logits_for_metrics(config):
    #     print('Get preprocess_logits_for_metrics function...')
    #     pred_axis = config.eval.prediction_axis
    #     def preprocess_logits_for_metrics(logits, labels):
    #         # return logits
    #         """
    #         Shrink vocabulary size logits (class possibilities) to one prediction for each token.
    #         """
    #         if torch.isnan(logits).any():
    #             print("⚠️ NAN detected in logits!")

    #         if torch.isnan(labels).any():
    #             print("⚠️ NAN detected in labels!")

    #         # Ensures compatibility with both encoder-only models (e.g., BERT) and
    #         # encoder-decoder models (e.g., T5), which return a tuple.
    #         if isinstance(logits, tuple):
    #             # Handle models like T5 returning multiple outputs
    #             # Extract the first element, which contains the logits tensor
    #             logits = logits[0]

    #         # Apply argmax to logits for one-true-label classification
    #         predictions = torch.argmax(logits, dim=pred_axis)
            
    #         # Apply sigmoid to logits for multi-true-label classification
    #         # predictions = torch.sigmoid(logits)

    #         return predictions
    #     return preprocess_logits_for_metrics

    @staticmethod
    def get_preprocess_logits_for_metrics(config):
        print('Get preprocess_logits_for_metrics function...')
        pred_axis = config.eval.prediction_axis

        def preprocess_logits_for_metrics(logits, labels):

            # Handle models like T5 returning multiple outputs
            # Extract the first element, which contains the logits tensor
            if isinstance(logits, tuple):
                logits = logits[0]

            if torch.isnan(logits).any():
                print("⚠️ NAN detected in logits!")
            if labels is not None and torch.isnan(labels).any():
                print("⚠️ NAN detected in labels!")

            # Apply argmax to logits for one-true-label classification
            predictions = torch.argmax(logits, dim=pred_axis)

            # Apply sigmoid to logits for multi-true-label classification
            # predictions = torch.sigmoid(logits)

            # ✅ Ensure correct type for tokenizer.decode
            return predictions.to(torch.long)

        return preprocess_logits_for_metrics

    @staticmethod
    def calc_confusion_matrix(predictions, true_labels, config, model_path):
        # # Flatten the lists for confusion matrix computation
        predictions = predictions.flatten()
        true_labels = true_labels.flatten()
        # Cast label names to integers if true labels are ints
        label_names = config.ds.label.names
        labels = list(range(len(label_names))) if np.issubdtype(true_labels.dtype, np.integer) else label_names
        # Compute the confusion matrix and Save the confusion matrix plot
        # print('labels:', labels, 'true_labels:', true_labels, 'predictions:', predictions)
        cm = confusion_matrix(true_labels, predictions, labels = labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(include_values=True, cmap='viridis', ax=None, xticks_rotation='horizontal')
        cm_path = f'{model_path}/confusion_matrix.png'
        plt.savefig(cm_path)
        plt.close()

    @staticmethod
    def compute_word_similarity(model, dataset, tokenizer):
        import torch
        from scipy.stats import pearsonr, spearmanr
        from scipy.spatial.distance import cosine

        # Function to compute the embedding of a word
        def get_word_embedding(word, tokenizer, model):
            inputs = tokenizer(word, return_tensors="pt")
            inputs.pop("token_type_ids", None)
            with torch.no_grad():
                outputs = model(**inputs)
            # Use the mean of the last hidden states as the word embedding
            return outputs.last_hidden_state.mean(dim=1)

        # Function to calculate cosine similarity
        def cosine_similarity(embedding1, embedding2):
            # Convert embeddings to 1D numpy arrays
            emb1 = embedding1.squeeze().numpy()
            emb2 = embedding2.squeeze().numpy()
            return 1 - cosine(emb1, emb2)

        # Evaluate the model
        model.eval()
        similarities = []
        for row in dataset:
            word1, word2, true_scor = row['word1'], row['word2'], row['labels']/10
            embedding1 = get_word_embedding(word1, tokenizer, model)
            embedding2 = get_word_embedding(word2, tokenizer, model)
            sim_score = cosine_similarity(embedding1, embedding2)
            similarities.append((word1, word2, sim_score, true_scor))

        # Print results
        for i, (word1, word2, sim_score, true_score) in enumerate(similarities):
            print(f"{word1} - {word2}: Model Score = {sim_score}, True Score = {true_score}")

        predicted_scores = [score for _, _, score, _ in similarities]
        true_scores = [true_score for _, _, _, true_score in similarities]

        # Calculate Pearson and Spearman correlations
        pearson_corr, _ = pearsonr(predicted_scores, true_scores)
        spearman_corr, _ = spearmanr(predicted_scores, true_scores)

        print(f"Pearson Correlation: {pearson_corr}")
        print(f"Spearman Correlation: {spearman_corr}")

        # DistilBERT
        # Pearson Correlation: 0.13173808507590326
        # Spearman Correlation: 0.07167651261402776

        # Uncased BERT
        # Pearson Correlation: 0.23983917671268257
        # Spearman Correlation: 0.20366549375439327

        # Cased BERT
        # Pearson Correlation: 0.11820483512012278
        # Spearman Correlation: 0.052407640825769625

        # XLM-RoBERTa
        # Pearson Correlation: -0.056036807184180665
        # Spearman Correlation: -0.06704637938595721

        # jnz/electra-ka
        # Pearson Correlation: 0.1345049301808912
        # Spearman Correlation: 0.11931619828784085

if __name__ == '__main__':
    EVALUATE()
