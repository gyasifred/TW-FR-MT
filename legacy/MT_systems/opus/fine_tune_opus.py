import sys
sys.path.insert(0, 'TW_FR_EN_corpus/scripts')
import argparse
from preprocessing import Preprocessing
from huggingFace import BuildDataset
from transformers import AutoModelForSeq2SeqLM, \
    DataCollatorForSeq2Seq, AutoTokenizer, AdamW, get_scheduler
from accelerate import Accelerator
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import numpy as np
import evaluate
import warnings
warnings.simplefilter('ignore')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Fine-turned Pretrained OPUS-MT transformer model')
    parser.add_argument("pretrained_model",
                        help="Provide the Pretrained model to be Fineturned", type=str)
    parser.add_argument("input_train_data",
                        help="Provide the Path to the Input language Training  Data", type=str)
    parser.add_argument("target_train_data",
                        help="Provide the Path to the Target Language Training  Data", type=str)
    parser.add_argument(
        "input_val_data", help="Provide the Path to the Input language validation Data", type=str)
    parser.add_argument(
        "target_val_data", help="Provide the Path to the Target Language validation Data", type=str)
    parser.add_argument(
        "src_lang", help="Provide the Initial of the Source Language as used in the Pretrained model.\
            Example the source language of the pretrained model 'Helsinki-NLP/opus-mt-en-tw' is en", type=str)
    parser.add_argument(
        "targ_lang", help="Provide the Initial of the Target Language as used in the Pretrained model.\
            Example the Target language of the pretrained model 'Helsinki-NLP/opus-mt-en-tw' is tw", type=str)
    parser.add_argument(
        "--max_length", type=int, default=128, help="Enter the maximum tokens for the source and target languages")
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Give the batch size for traininf")
    parser.add_argument(
        "--epoch", type=int, default=20, help="Enter the number of epoch for training")
    parser.add_argument(
        "--warmup_steps", type=int, default=5, help="Enter the number of warmup training steps")
    parser.add_argument(
        "--lr", type=float, default=2e-5, help="Enter thr training learning  rate")
    parser.add_argument('--savedir', type=str,
                        help="Provide the path to save to the fineturned model")
    parser.add_argument("model_name", type=str,
                        help="Name for saving the fineturend model")

    args = parser.parse_args()
    print("MODEL PATH: " + args.savedir)
    print("FINED-TURNED MODEL NAME: " + args.model_name)
    print("================")

    pretrained_model = args.pretrained_model
    src_lang_train_path = args.input_train_data
    targ_lang_train_path = args.target_train_data
    src_lang_val_path = args.input_val_data
    targ_lang_val_path = args.target_val_data
    src = args.src_lang
    targ = args.targ_lang
    max_length = args.max_length
    batch_size = args.batch_size
    epoch = args.epoch
    model_name = args.model_name

    # Provide the pretrained model
    model_checkpoint = pretrained_model

    # load the metrics
    metric = evaluate.load("sacrebleu")

    # Load the datasets
    # Create an instance of tft preprocessing class
    preprocessor = Preprocessing()

    # Read training parallel dataset
    src_train_data, targ_train_data = preprocessor.read_parallel_dataset(
        filepath_1=src_lang_train_path,
        filepath_2=targ_lang_train_path
    )

    # Read validation parallel dataset
    src_val_data, targ_val_data = preprocessor.read_parallel_dataset(
        filepath_1=src_lang_val_path,
        filepath_2=targ_lang_val_path
    )

    # Build Customize Hugging Face Dataset
    dataset = BuildDataset(
        src_train_data, targ_train_data, src_val_data, targ_val_data, src, targ)

    train_dataset, val_dataset = dataset.build()
    # print(train_dataset)
    # print()
    # print(val_dataset)
    # load tokenizer of pretrained model
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Define function to preprocess the input and target data
    source_lang = src
    target_lang = targ

    def preprocess_function(examples):
        inputs = [ex[source_lang] for ex in examples["translation"]]
        targets = [ex[target_lang] for ex in examples["translation"]]
        model_inputs = tokenizer(
            inputs, text_target=targets, max_length=max_length, truncation=True
        )
        return model_inputs

    # preprocess datasets
    tokenized_train_datasets = train_dataset.map(
        preprocess_function, batched=True, remove_columns=train_dataset.column_names)
    tokenized_val_datasets = val_dataset.map(
        preprocess_function, batched=True, remove_columns=val_dataset.column_names)

    # Download model weights
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # instantiate a data_collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    tokenized_train_datasets.set_format("torch")
    tokenized_val_datasets.set_format("torch")
    train_dataloader = DataLoader(
        tokenized_train_datasets,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=batch_size,
    )
    eval_dataloader = DataLoader(
        tokenized_val_datasets, collate_fn=data_collator, batch_size=batch_size
    )

    # set up optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # instantiate accelerator object
    accelerator = Accelerator()
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    num_train_epochs = epoch
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )

    def postprocess(predictions, labels):
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()

        decoded_preds = tokenizer.batch_decode(
            predictions, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]
        return decoded_preds, decoded_labels

    # provide output directory to save the model
    output_dir = f"{args.savedir}/{model_name}"

    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_train_epochs):
        # Training
        model.train()
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        # Evaluation
        model.eval()
        for batch in tqdm(eval_dataloader):
            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_length=max_length,
                )
            labels = batch["labels"]

            # Necessary to pad predictions and labels for being gathered
            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            labels = accelerator.pad_across_processes(
                labels, dim=1, pad_index=-100)

            predictions_gathered = accelerator.gather(generated_tokens)
            labels_gathered = accelerator.gather(labels)

            decoded_preds, decoded_labels = postprocess(
                predictions_gathered, labels_gathered)
            metric.add_batch(predictions=decoded_preds,
                             references=decoded_labels)

        results = metric.compute()
        print(f"epoch {epoch}, BLEU score: {results['score']:.2f}")

        # Save the model
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)

    print("Training complete")

