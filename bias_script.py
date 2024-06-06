from datasets import load_dataset
from transformers import AutoTokenizer
from nlp_helper_scripts.getData import (prepare_label, bert_glue_encode,pretrained_bert_model)
from bias_helpers.bias_functions import tprs, calculate_gaps
from bias_helpers.bias_deepview import DeepViewBias

import tensorflow as tf
import numpy as np
import matplotlib as mpl

professions_dict = {
    0: "accountant",
    1: "architect",
    2: "attorney",
    3: "chiropractor",
    4: "comedian",
    5: "composer",
    6: "dentist",
    7: "dietitian",
    8: "dj",
    9: "filmmaker",
    10: "interior_designer",
    11: "journalist",
    12: "model",
    13: "nurse",
    14: "painter",
    15: "paralegal",
    16: "pastor",
    17: "personal_trainer",
    18: "photographer",
    19: "physician",
    20: "poet",
    21: "professor",
    22: "psychologist",
    23: "rapper",
    24: "software_engineer",
    25: "surgeon",
    26: "teacher",
    27: "yoga_teacher"
}



if __name__ == "__main__":
    dataset = load_dataset("LabHC/bias_in_bios")
    MODEL_CHECKPOINT = "bert-base-uncased"
    task = "bias_in_bios"
    def single_preprocess_function(examples):
        # glue datasets field mapping
        task_to_keys = {
            "cola": ("sentence", None),
            "mnli": ("premise", "hypothesis"),
            "mnli-mm": ("premise", "hypothesis"),
            "mrpc": ("sentence1", "sentence2"),
            "qnli": ("question", "sentence"),
            "qqp": ("question1", "question2"),
            "rte": ("sentence1", "sentence2"),
            "sst2": ("sentence", None),
            "stsb": ("sentence1", "sentence2"),
            "wnli": ("sentence1", "sentence2"),
            "bias_in_bios": ("hard_text", None),
        }

        tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
        sentence1_key, sentence2_key = task_to_keys[task]
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key],
                             max_length=128,
                             padding='max_length',
                             truncation=True)
        return tokenizer(examples[sentence1_key],
                         examples[sentence2_key],
                         max_length=128,
                         padding='max_length',
                         truncation=True)

    preprocessed_dataset = dataset.map(single_preprocess_function, batched=True)
    #x_train, y_train = bert_glue_encode(single_preprocessed_dataset["train"])
    x_val, y_val = bert_glue_encode(preprocessed_dataset["test"], "profession")

    pt_embed = pretrained_bert_model()
    saved_pretrained_classifier_only_path = "<path/to/model>"
    # saved_pretrained_classifier_only_path = './models/' + str(task) + '/{}_pretrained_BERT_Classifier'.format(
    #     task.replace('/', '_'))

    pretrained_classifier = tf.saved_model.load(saved_pretrained_classifier_only_path)

    # pt_head = classifier_model(28)
    # whole_model, ft_bert, ft_classifier = finetuned_bert_and_classifier(28)

    # train_embeddings = pt_embed.predict(x_train, batch_size=64)
    n_samples = 2500
    np.random.seed(42)
    sample_ids = np.random.choice(len(x_val[1]), n_samples, replace=False)
    x_val = (x_val[0][sample_ids], x_val[1][sample_ids], x_val[2][sample_ids])
    y_val = y_val[sample_ids]

    val_embeddings = pt_embed.predict(x_val, batch_size=64)
    all_sensitive_group = np.array(preprocessed_dataset['test']['gender'])#[:3000]


    infer = pretrained_classifier.signatures['serving_default']

    def pretrained_pred_wrapper(x):
        to_tensor = x.reshape((1, len(x), 768))
        tensor = tf.constant(to_tensor, dtype='float32')
        init_preds = infer(tensor)
        init_preds = init_preds['output'].numpy()
        preds = init_preds.reshape((len(x), 28))
        return preds


    # Get the embedded data
    # n_samples = 5000
    # sample_ids = np.random.choice(len(x_val[1]), n_samples, replace=False)

    # x_val = (x_val[0][sample_ids], x_val[1][sample_ids], x_val[2][sample_ids])
    # y_val = y_val[sample_ids]

    # pre_X1 = np.array([val_embeddings[i] for i in sample_ids])
    # pre_Y1 = np.array([y_val[i] for i in sample_ids])
    pre_X1 = val_embeddings
    pre_Y1 = y_val
    sample_sensitive_group = all_sensitive_group[sample_ids]

    preds = np.argmax(pretrained_pred_wrapper(val_embeddings), axis=1)
    n_classes = 28

    tpr = tprs(sample_sensitive_group, y_val, preds, n_classes)
    gaps = calculate_gaps(tpr, 1, 0)

    def data_viz1(sample, p, t, color):
        for id,pi,ti in zip(sample,p,t):
            input = preprocessed_dataset['test']['hard_text'][int(sample_ids[id])]
            label = professions_dict[ti]
            prediction = professions_dict[pi]
            print("-----------------------------------------------------")
            print("Description:",input)
            print("Prediction:",prediction)
            print("Label:",label)
            print("-----------------------------------------------------")


    classes = np.arange(28)
    batch_size = 64
    max_samples = 5005
    data_shape = (768,)
    resolution = 100
    N = 10
    lam = .8
    cmap = 'gist_ncar'
    # to make sure deepview.show is blocking,
    # disable interactive mode
    interactive = False
    title = 'SST2-BERT'
    disc_dist = True

    pt_deepview = DeepViewBias(pretrained_pred_wrapper, classes, max_samples, batch_size, data_shape,
                               N, lam, resolution, cmap, interactive, title,
                               data_viz=data_viz1, metrics="cosine",disc_dist=disc_dist,class_dict=professions_dict,
                               sensitive_group=sample_sensitive_group, gaps=gaps)

    pt_deepview.add_samples(pre_X1, pre_Y1)


    pt_deepview.show()
